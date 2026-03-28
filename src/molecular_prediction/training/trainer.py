# src/molecular_prediction/training/trainer.py

import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from src.molecular_prediction.models.base import BaseGNN
from src.molecular_prediction.training.early_stopping import EarlyStopping
from src.molecular_prediction.training.utils import save_parameters


class Trainer:
    """Generic trainer for GNN models on QM9.
    Handles training loop, evaluation, checkpointing and TensorBoard logging.

    Args:
        model: GNN model to train.
        model_name: Model identifier, used to name checkpoint and TensorBoard run.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        lr: Learning rate.
        batch_size: Number of graphs per batch.
        patience: Early stopping patience (epochs without improvement).
        delta: Minimum improvement threshold for early stopping.
        epochs: Maximum number of training epochs.
        device: Device to train on ('cpu', 'cuda', 'mps').
        target_indices: QM9 column indices for the targets to predict.
        path_weights: Directory where model checkpoints are saved (default: 'models').
            The actual file will be '{path_weights}/{model_name}.pt'.
        output_dir: Directory for TensorBoard logs (default: 'runs').
            Logs are written to '{output_dir}/{model_name}'.
    """

    def __init__(
        self,
        model: BaseGNN,
        model_name: str,
        train_dataset: Dataset,
        val_dataset: Dataset,
        lr: float,
        batch_size: int,
        patience: int,
        delta: float,
        epochs: int,
        device: str,
        target_indices: list[int],
        target_names: list[str] | None = None,
        path_weights: str = "models",
        output_dir: str = "runs",
    ) -> None:
        self.model: nn.Module = model
        self.model_name: str = model_name
        self.train_dataset: Dataset = train_dataset
        self.val_dataset: Dataset = val_dataset
        self.lr: float = lr
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.device: torch.device = device
        self.output_dir: str = output_dir

        os.makedirs(path_weights, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        self.path_weights: str = os.path.join(path_weights, f"{model_name}.pt")

        self.train_loader: DataLoader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader: DataLoader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=True
        )

        self.target_indices: list[int] = target_indices
        self.num_targets: int = len(target_indices)
        self.target_names: list[str] = (
            target_names
            if target_names is not None
            else [f"target_{i}" for i in target_indices]
        )

        self.optimizer: Adam = Adam(self.model.parameters(), lr=self.lr)
        self.criterion: nn.L1Loss = nn.L1Loss()
        self.scheduler: CosineAnnealingLR = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )
        self.early_stopping: EarlyStopping = EarlyStopping(
            patience=patience, delta=delta
        )

        self.writer: SummaryWriter = SummaryWriter(
            log_dir=f"{self.output_dir}/{self.model_name}"
        )

    def _train_epoch(self) -> tuple[float, list[float]]:
        """Run one full training epoch.

        Returns:
            Tuple of (mean combined loss, list of per-target mean MAEs).
        """
        self.model.train()
        self.model.to(self.device)
        n_batches: int = 0
        losses: list[float] = []
        per_target_losses: list[torch.Tensor] = []

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            batch = batch.to(self.device)
            output: torch.Tensor = self.model(batch)
            targets: torch.Tensor = batch.y[:, self.target_indices]
            loss: torch.Tensor = self.criterion(output, targets)

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            # Per-target MAE for this batch: shape [num_targets]
            batch_per_target: torch.Tensor = torch.abs(output - targets).mean(dim=0)
            per_target_losses.append(batch_per_target.detach().cpu())
            n_batches += 1

        # Average per-target MAEs across batches
        stacked: torch.Tensor = torch.stack(
            per_target_losses
        )  # [n_batches, num_targets]
        mean_per_target: list[float] = stacked.mean(dim=0).tolist()

        return sum(losses) / n_batches, mean_per_target

    def _evaluate(self, loader: DataLoader) -> tuple[float, list[float]]:
        """Evaluate the model on a dataset split.

        Args:
            loader: DataLoader for the split to evaluate.

        Returns:
            Tuple of (mean combined MAE, list of per-target mean MAEs).
        """
        self.model.eval()
        self.model.to(self.device)
        losses: list[float] = []
        per_target_losses: list[torch.Tensor] = []
        n_batches: int = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                output: torch.Tensor = self.model(batch)
                targets: torch.Tensor = batch.y[:, self.target_indices]
                loss: torch.Tensor = self.criterion(output, targets)

                losses.append(loss.item())
                batch_per_target: torch.Tensor = torch.abs(output - targets).mean(dim=0)
                per_target_losses.append(batch_per_target.cpu())
                n_batches += 1

        stacked: torch.Tensor = torch.stack(per_target_losses)
        mean_per_target: list[float] = stacked.mean(dim=0).tolist()

        return sum(losses) / n_batches, mean_per_target

    def fit(self) -> dict[str, list[float] | dict[str, list[float]]]:
        """Run the full training loop for all epochs.
        Logs train loss and val MAE to TensorBoard at each epoch.
        Saves the best model checkpoint based on val MAE.

        Returns:
            Dictionary with keys:
                'train_loss'            – list of combined train losses per epoch
                'val_loss'              – list of combined val MAEs per epoch
                'train_loss_per_target' – dict mapping target name -> list of per-epoch losses
                'val_loss_per_target'   – dict mapping target name -> list of per-epoch MAEs
        """
        results: dict[str, list[float] | dict[str, list[float]]] = {
            "train_loss": [],
            "val_loss": [],
            "train_loss_per_target": {name: [] for name in self.target_names},
            "val_loss_per_target": {name: [] for name in self.target_names},
        }

        for epoch in range(self.epochs):
            avg_train_loss: float
            train_per_target: list[float]
            avg_train_loss, train_per_target = self._train_epoch()

            avg_val_loss: float
            val_per_target: list[float]
            avg_val_loss, val_per_target = self._evaluate(self.val_loader)

            print(
                f"Epoch {epoch + 1}/{self.epochs} — "
                f"Train Loss: {avg_train_loss:.4f} — "
                f"Val Loss: {avg_val_loss:.4f}"
            )

            results["train_loss"].append(avg_train_loss)
            results["val_loss"].append(avg_val_loss)

            for i, name in enumerate(self.target_names):
                results["train_loss_per_target"][name].append(train_per_target[i])
                results["val_loss_per_target"][name].append(val_per_target[i])

            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            self.writer.add_scalar("Loss/val", avg_val_loss, epoch)
            for i, name in enumerate(self.target_names):
                self.writer.add_scalar(f"Loss_train/{name}", train_per_target[i], epoch)
                self.writer.add_scalar(f"Loss_val/{name}", val_per_target[i], epoch)

            self.scheduler.step()

            self.early_stopping(avg_val_loss, self.model, self.path_weights)

            if self.early_stopping.apply_early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        if not self.early_stopping.apply_early_stop:
            save_parameters(self.model, self.path_weights)

        self.writer.close()
        return results

    def evaluate_test(self, test_dataset: Dataset) -> tuple[float, list[float]]:
        """Evaluate the best saved checkpoint on the test set.

        Args:
            test_dataset: Test dataset.

        Returns:
            Tuple of (mean combined MAE, list of per-target MAEs).
        """
        test_loader: DataLoader = DataLoader(test_dataset, batch_size=self.batch_size)
        return self._evaluate(test_loader)
