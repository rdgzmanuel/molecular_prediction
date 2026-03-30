# src/molecular_prediction/experiments/per_molecule_eval.py
#
# Per-molecule evaluation utilities.
# Produces individual MAE values for each molecule in a dataset,
# optionally denormalised back to original units.

import os

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from configs.config import Config
from src.molecular_prediction.data.dataset import (
    compute_normalisation_stats,
    load_splits,
)
from src.molecular_prediction.experiments.main_comparison import (
    MODEL_NAMES,
    build_model,
)
from src.molecular_prediction.models.base import BaseGNN


def load_model(model_name: str, config: Config) -> BaseGNN:
    """Load a pretrained model from its checkpoint file.

    Args:
        model_name: One of 'GIN', 'GINDist', 'EGNN'.
        config: Full experiment config dict.

    Returns:
        Model with weights loaded, in eval mode.
    """
    model: BaseGNN = build_model(model_name, config)
    path_weights: str = config.paths.path_weights
    checkpoint_path: str = os.path.join(path_weights, f"{model_name}.pt")

    state_dict: dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_normalisation_stats(
    config: Config,
) -> dict[int, dict[str, torch.Tensor]]:
    """Compute normalisation stats from the training set.

    Args:
        config: Full experiment config dict.

    Returns:
        Dict mapping target index to {"mean": Tensor, "std": Tensor}.
    """
    from torch_geometric.datasets import QM9

    torch.manual_seed(config.data.seed)
    dataset: QM9 = QM9(config.paths.path_data)
    train_idx: int = config.data.split[0]
    train_dataset: Dataset = dataset[:train_idx]

    return compute_normalisation_stats(train_dataset, list(config.data.target_indices))


def evaluate_per_molecule(
    model: BaseGNN,
    test_dataset: Dataset,
    config: Config,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a model and return per-molecule predictions and targets.

    Both outputs are in normalised space (as the model was trained).
    Use denormalise() to convert back to original units.

    Args:
        model: Pretrained GNN model in eval mode.
        test_dataset: Test dataset with NormaliseTargets transform applied.
        config: Full experiment config dict.
        device: PyTorch device string.

    Returns:
        Tuple of:
            predictions: shape [N_test, num_targets]
            targets: shape [N_test, num_targets]
    """
    target_indices: list[int] = list(config.data.target_indices)
    batch_size: int = config.training.batch_size
    loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output: torch.Tensor = model(batch)
            targets: torch.Tensor = batch.y[:, target_indices]
            all_preds.append(output.cpu())
            all_targets.append(targets.cpu())

    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


def denormalise(
    values: torch.Tensor,
    norm_stats: dict[int, dict[str, torch.Tensor]],
    target_indices: list[int],
) -> torch.Tensor:
    """Convert normalised values back to original units.

    Args:
        values: Tensor of shape [N, num_targets] in normalised space.
        norm_stats: Dict mapping target index to {"mean": ..., "std": ...}.
        target_indices: List of target column indices.

    Returns:
        Denormalised tensor of same shape.
    """
    result: torch.Tensor = values.clone()
    for i, idx in enumerate(target_indices):
        mean: torch.Tensor = norm_stats[idx]["mean"]
        std: torch.Tensor = norm_stats[idx]["std"]
        result[:, i] = values[:, i] * (std + 1e-6) + mean
    return result


def compute_per_molecule_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute absolute error per molecule and target.

    Args:
        predictions: shape [N, num_targets].
        targets: shape [N, num_targets].

    Returns:
        Absolute errors of shape [N, num_targets].
    """
    return torch.abs(predictions - targets)


def run_per_molecule_evaluation(
    config: Config,
    device: str,
) -> dict[str, torch.Tensor]:
    """Run per-molecule evaluation for all three models.

    Returns MAE in original (denormalised) units per molecule and target.

    Args:
        config: Full experiment config dict.
        device: PyTorch device string.

    Returns:
        Dict mapping model name -> tensor of shape [N_test, num_targets]
        with absolute errors in original units.
    """
    _, _, test_dataset = load_splits(config)
    target_indices: list[int] = list(config.data.target_indices)
    norm_stats: dict[int, dict[str, torch.Tensor]] = get_normalisation_stats(config)

    results: dict[str, torch.Tensor] = {}

    for model_name in MODEL_NAMES:
        print(f"Evaluating {model_name} per molecule...")
        model: BaseGNN = load_model(model_name, config)
        preds: torch.Tensor
        targets: torch.Tensor
        preds, targets = evaluate_per_molecule(model, test_dataset, config, device)

        # Denormalise both predictions and targets, then compute error
        preds_denorm: torch.Tensor = denormalise(preds, norm_stats, target_indices)
        targets_denorm: torch.Tensor = denormalise(targets, norm_stats, target_indices)
        mae: torch.Tensor = compute_per_molecule_mae(preds_denorm, targets_denorm)

        results[model_name] = mae
        print(f"  Mean MAE per target: {mae.mean(dim=0).tolist()}")

    return results
