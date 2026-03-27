# src/molecular_prediction/experiments/main_comparison.py

import json
import os

import matplotlib.pyplot as plt
from torch_geometric.data import Dataset

from configs.config import Config
from src.molecular_prediction.data.dataset import load_splits
from src.molecular_prediction.models.base import BaseGNN
from src.molecular_prediction.models.egnn import EGNN
from src.molecular_prediction.models.gin import GIN
from src.molecular_prediction.models.gin_dist import GINDist
from src.molecular_prediction.training.trainer import Trainer

# QM9 target names at indices [0, 1, 4, 11]
TARGET_NAMES: list[str] = ["mu", "alpha", "gap", "Cv"]
MODEL_NAMES: list[str] = ["GIN", "GINDist", "EGNN"]


def build_model(model_name: str, config: Config) -> BaseGNN:
    """Instantiate a model by name using the config hyperparameters.

    Args:
        model_name: One of 'GIN', 'GINDist', 'EGNN'.
        config: Full experiment config dict (uses config['model'] section).

    Returns:
        Initialised (untrained) GNN model.
    """
    assert model_name in MODEL_NAMES, f"Invalid model name: {model_name}"

    num_targets: int = len(TARGET_NAMES)

    if model_name == "GIN":
        model: BaseGNN = GIN(
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_targets=num_targets,
        )
    elif model_name == "GINDist":
        model = GINDist(
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_targets=num_targets,
            edge_attr_dim=config.model.edge_attr_dim,
        )
    else:
        model = EGNN(
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_targets=num_targets,
            edge_attr_dim=config.model.edge_attr_dim,
        )

    return model


def build_trainer(
    model: BaseGNN,
    model_name: str,
    config: Config,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: str,
) -> Trainer:
    """Construct a Trainer for the given model and datasets.

    Args:
        model: The GNN model to train.
        model_name: Used to name the checkpoint file and TensorBoard run.
        config: Full experiment config dict.
        train_dataset: Training split.
        val_dataset: Validation split.
        device: PyTorch device string ('cpu', 'cuda', 'mps').

    Returns:
        Configured Trainer instance.
    """

    return Trainer(
        model=model,
        model_name=model_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=config.training.lr,
        batch_size=config.training.batch_size,
        patience=config.training.patience,
        delta=config.training.delta,
        epochs=config.training.epochs,
        device=device,
        target_indices=config.data.target_indices,
        path_weights=config.paths.path_weights,
        output_dir=config.paths.output_dir,
    )


def run_single_model(
    model_name: str,
    config: Config,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    device: str,
) -> dict:
    """Train one model and evaluate it on the test set.

    Args:
        model_name: One of 'GIN', 'GINDist', 'EGNN'.
        config: Full experiment config dict.
        train_dataset: Training split.
        val_dataset: Validation split.
        test_dataset: Test split.
        device: PyTorch device string.

    Returns:
        Dict with keys:
            'model_name'  (str)
            'train_loss'  (list[float]) – per-epoch training losses
            'val_loss'    (list[float]) – per-epoch validation MAEs
            'test_mae'    (float)       – final test MAE (mean over targets)
    """
    model: BaseGNN = build_model(model_name, config)
    trainer: Trainer = build_trainer(
        model, model_name, config, train_dataset, val_dataset, device
    )
    history: dict[str, list[float]] = trainer.fit()
    test_mae: float = trainer.evaluate_test(test_dataset)

    return {
        "model_name": model_name,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "test_mae": test_mae,
    }


def run_comparison(
    config: Config,
    device: str,
) -> dict[str, dict]:
    """Run the full three-model comparison experiment.

    Trains GIN, GINDist, and EGNN with identical hyperparameters and
    evaluates each on the shared test split.

    Args:
        config: Full experiment config dict.
        device: PyTorch device string.

    Returns:
        Dict mapping model name -> results dict (see run_single_model).
    """

    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    train_dataset, val_dataset, test_dataset = load_splits(config)

    results: dict[str, dict] = {}
    for name in MODEL_NAMES:
        print(f"\n{'=' * 60}")
        print(f"Training {name}...")
        print(f"{'=' * 60}")
        result: dict = run_single_model(
            name, config, train_dataset, val_dataset, test_dataset, device
        )
        results[name] = result

    print_results_table(results)
    return results


def save_results(results: dict[str, dict], output_dir: str) -> None:
    """Persist experiment results to disk.

    Saves a JSON file with test MAEs and training curves for each model.

    Args:
        results: Dict mapping model name -> results dict.
        output_dir: Directory where the JSON file will be written.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath: str = os.path.join(output_dir, "comparison_results.json")

    serialisable: dict[str, dict] = {}
    for model_name, res in results.items():
        serialisable[model_name] = {
            "model_name": res["model_name"],
            "train_loss": res["train_loss"],
            "val_loss": res["val_loss"],
            "test_mae": float(res["test_mae"]),
        }

    with open(filepath, "w") as f:
        json.dump(serialisable, f, indent=2)

    print(f"Results saved to {filepath}")


def print_results_table(results: dict[str, dict]) -> None:
    """Print a formatted table of test MAE per model.

    Args:
        results: Dict mapping model name -> results dict.
    """
    print(f"\n{'Model':<12} {'Test MAE':>10}")
    print("-" * 24)
    for model_name, res in results.items():
        print(f"{model_name:<12} {res['test_mae']:>10.4f}")
    print()


def plot_training_curves(results: dict[str, dict], output_dir: str) -> None:
    """Plot training loss and validation MAE curves for all models.

    Saves the figure to '{output_dir}/training_curves.png'.

    Args:
        results: Dict mapping model name -> results dict.
        output_dir: Directory where the plot will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_train: plt.Axes = axes[0]
    ax_val: plt.Axes = axes[1]

    for model_name, res in results.items():
        epochs: list[int] = list(range(1, len(res["train_loss"]) + 1))
        ax_train.plot(epochs, res["train_loss"], label=model_name)
        ax_val.plot(epochs, res["val_loss"], label=model_name)

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train Loss (MAE)")
    ax_train.set_title("Training Loss")
    ax_train.legend()
    ax_train.grid(True, alpha=0.3)

    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation MAE")
    ax_val.set_title("Validation MAE")
    ax_val.legend()
    ax_val.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath: str = os.path.join(output_dir, "training_curves.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {filepath}")


def plot_test_mae(results: dict[str, dict], output_dir: str) -> None:
    """Plot a bar chart of test MAE per model.

    Saves the figure to '{output_dir}/test_mae_comparison.png'.

    Args:
        results: Dict mapping model name -> results dict.
        output_dir: Directory where the plot will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    names: list[str] = list(results.keys())
    maes: list[float] = [results[n]["test_mae"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, maes, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Test MAE")
    ax.set_title("Test MAE Comparison")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, mae in zip(bars, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mae:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    filepath: str = os.path.join(output_dir, "test_mae_comparison.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Test MAE bar chart saved to {filepath}")
