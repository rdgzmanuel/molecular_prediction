# src/molecular_prediction/experiments/noise_ablation.py
#
# Noise robustness ablation study.
# For each model trained in main_comparison, we apply increasing levels of
# Gaussian noise to 3D atomic coordinates at test time and measure the
# degradation in MAE. No retraining is performed; this is purely an
# inference-time evaluation.

import json
import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from configs.config import Config
from src.molecular_prediction.data.dataset import load_splits
from src.molecular_prediction.data.transforms import AddGaussianNoise
from src.molecular_prediction.experiments.main_comparison import (
    MODEL_NAMES,
    TARGET_NAMES,
    build_model,
)
from src.molecular_prediction.models.base import BaseGNN

# Noise levels (sigma) to sweep over
DEFAULT_SIGMA_VALUES: list[float] = [0.0, 0.1, 0.25, 0.5, 1.0]

# Sigma cutoff for the main (linear-scale) plots
MAIN_PLOT_MAX_SIGMA: float = 0.5


def load_model(model_name: str, config: Config) -> BaseGNN:
    """Load a pretrained model from its checkpoint file.

    The checkpoint path is resolved as:
        {config['experiment']['path_weights']}/{model_name}.pt
    which mirrors how Trainer saves it (e.g. 'models/GIN.pt').

    Args:
        model_name: One of 'GIN', 'GINDist', 'EGNN'.
        config: Full experiment config dict (used to rebuild architecture
                and locate the checkpoint directory).

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


def apply_noise_to_dataset(test_dataset: Dataset, sigma: float) -> list:
    """Return a version of test_dataset with Gaussian noise added to coordinates.

    Uses AddGaussianNoise transform from molecular_prediction.data.transforms.
    When sigma == 0.0, no noise is added but the dataset is still materialised
    into a list for consistency.

    Args:
        test_dataset: Clean test split.
        sigma: Standard deviation of the noise in Angstroms.

    Returns:
        List of Data objects (optionally with noisy coordinates).
    """
    if sigma == 0.0:
        return [test_dataset[i] for i in range(len(test_dataset))]

    transform: AddGaussianNoise = AddGaussianNoise(sigma=sigma)

    noisy_data: list = []
    for i in range(len(test_dataset)):
        data: Data = test_dataset[i].clone()
        data = transform(data)
        noisy_data.append(data)

    return noisy_data


def evaluate_model_under_noise(
    model: BaseGNN,
    test_dataset: Dataset,
    sigma: float,
    config: Config,
    device: str,
) -> dict:
    """Evaluate a single pretrained model on a noisy test set.

    Args:
        model: Pretrained GNN model (already in eval mode).
        test_dataset: Clean test split (noise is applied inside this function).
        sigma: Noise standard deviation.
        config: Full experiment config dict (used for batch_size).
        device: PyTorch device string.

    Returns:
        Dict with keys:
            'sigma'              (float)
            'test_mae'           (float) – mean combined MAE over all targets
            'test_mae_per_target' (list[float]) – per-target MAEs
    """
    noisy_dataset: list = apply_noise_to_dataset(test_dataset, sigma)
    batch_size: int = config.training.batch_size
    target_indices: list[int] = list(config.data.target_indices)
    loader: DataLoader = DataLoader(noisy_dataset, batch_size=batch_size)

    model.to(device)
    model.eval()

    combined_losses: list[float] = []
    per_target_losses: list[torch.Tensor] = []
    n_batches: int = 0
    criterion: nn.L1Loss = nn.L1Loss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output: torch.Tensor = model(batch)
            targets: torch.Tensor = batch.y[:, target_indices]
            loss: torch.Tensor = criterion(output, targets)
            combined_losses.append(loss.item())

            batch_per_target: torch.Tensor = torch.abs(output - targets).mean(dim=0)
            per_target_losses.append(batch_per_target.cpu())
            n_batches += 1

    mean_mae: float = sum(combined_losses) / n_batches

    stacked: torch.Tensor = torch.stack(per_target_losses)
    mean_per_target: list[float] = stacked.mean(dim=0).tolist()

    return {
        "sigma": sigma,
        "test_mae": mean_mae,
        "test_mae_per_target": mean_per_target,
    }


def run_noise_ablation_for_model(
    model_name: str,
    config: Config,
    test_dataset: Dataset,
    sigma_values: list[float],
    device: str,
) -> list[dict]:
    """Run the noise sweep for a single model.

    Loads the pretrained checkpoint saved by main_comparison via load_model(),
    then calls evaluate_model_under_noise() for each sigma value.

    Args:
        model_name: One of 'GIN', 'GINDist', 'EGNN'.
        config: Full experiment config dict.
        test_dataset: Clean test split.
        sigma_values: List of noise levels to evaluate at.
        device: PyTorch device string.

    Returns:
        List of result dicts (one per sigma), each from evaluate_model_under_noise.
    """
    model: BaseGNN = load_model(model_name, config)

    results: list[dict] = []
    for sigma in sigma_values:
        print(f"  sigma={sigma:.2f} ...", end=" ", flush=True)
        result: dict = evaluate_model_under_noise(
            model, test_dataset, sigma, config, device
        )
        print(f"MAE={result['test_mae']:.4f}")
        results.append(result)

    return results


def run_noise_ablation(
    config: Config,
    device: str,
    sigma_values: list[float] = DEFAULT_SIGMA_VALUES,
) -> dict[str, list[dict]]:
    """Run the full noise ablation for all three models.

    Args:
        config: Full experiment config dict.
        device: PyTorch device string.
        sigma_values: Noise levels to sweep over.

    Returns:
        Dict mapping model name -> list of per-sigma result dicts.
    """
    _, _, test_dataset = load_splits(config)

    results: dict[str, list[dict]] = {}
    for name in MODEL_NAMES:
        print(f"\n{'=' * 60}")
        print(f"Noise ablation for {name}")
        print(f"{'=' * 60}")
        results[name] = run_noise_ablation_for_model(
            name, config, test_dataset, sigma_values, device
        )

    print_ablation_table(results)
    return results


def save_ablation_results(results: dict[str, list[dict]], output_dir: str) -> None:
    """Persist ablation results to disk as JSON.

    Args:
        results: Dict mapping model name -> list of per-sigma result dicts.
        output_dir: Directory where the JSON file will be written.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath: str = os.path.join(output_dir, "noise_ablation_results.json")

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Ablation results saved to {filepath}")


def print_ablation_table(results: dict[str, list[dict]]) -> None:
    """Print a formatted table of test MAE vs noise level for all models.

    Shows combined MAE first, then a separate table for each target.

    Args:
        results: Dict mapping model name -> list of per-sigma result dicts.
    """
    model_names: list[str] = list(results.keys())
    first_model: str = model_names[0]
    n_sigmas: int = len(results[first_model])

    # Combined table
    header: str = f"{'sigma':<10}" + "".join(f"{n:>12}" for n in model_names)
    print("\nCombined MAE")
    print(header)
    print("-" * len(header))

    for i in range(n_sigmas):
        sigma: float = results[first_model][i]["sigma"]
        row: str = f"{sigma:<10.2f}"
        for name in model_names:
            mae: float = results[name][i]["test_mae"]
            row += f"{mae:>12.4f}"
        print(row)

    # Per-target tables
    for t_idx, target_name in enumerate(TARGET_NAMES):
        print(f"\nMAE — {target_name}")
        print(header)
        print("-" * len(header))

        for i in range(n_sigmas):
            sigma = results[first_model][i]["sigma"]
            row = f"{sigma:<10.2f}"
            for name in model_names:
                mae = results[name][i]["test_mae_per_target"][t_idx]
                row += f"{mae:>12.4f}"
            print(row)

    print()


def _filter_results_by_sigma(
    results: dict[str, list[dict]],
    max_sigma: float,
) -> dict[str, list[dict]]:
    """Filter results to include only sigma values up to max_sigma.

    Args:
        results: Full results dict.
        max_sigma: Maximum sigma to include (inclusive).

    Returns:
        Filtered results dict with same structure.
    """
    filtered: dict[str, list[dict]] = {}
    for model_name, res_list in results.items():
        filtered[model_name] = [r for r in res_list if r["sigma"] <= max_sigma]
    return filtered


def _plot_ablation_combined(
    results: dict[str, list[dict]],
    output_dir: str,
    filename: str,
    title_suffix: str,
    log_scale: bool,
) -> None:
    """Plot combined test MAE vs noise level for all models.

    Args:
        results: Results dict (possibly filtered).
        output_dir: Directory where the plot will be saved.
        filename: Output filename.
        title_suffix: Appended to the plot title.
        log_scale: Whether to use logarithmic Y axis.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors: list[str] = ["#4C72B0", "#55A868", "#C44E52"]
    for idx, (model_name, res_list) in enumerate(results.items()):
        sigmas: list[float] = [r["sigma"] for r in res_list]
        maes: list[float] = [r["test_mae"] for r in res_list]
        color: str = colors[idx % len(colors)]
        ax.plot(sigmas, maes, marker="o", label=model_name, color=color)

    ax.set_xlabel("Noise σ (Å)")
    ax.set_ylabel("Test MAE")
    ax.set_title(f"Noise Robustness Ablation (Combined){title_suffix}")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath: str = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Ablation curves saved to {filepath}")


def _plot_ablation_per_target(
    results: dict[str, list[dict]],
    output_dir: str,
    filename_template: str,
    title_suffix: str,
    log_scale: bool,
) -> None:
    """Plot per-target test MAE vs noise level for all models.

    Args:
        results: Results dict (possibly filtered).
        output_dir: Directory where plots will be saved.
        filename_template: Filename template with {target} placeholder.
        title_suffix: Appended to the plot title.
        log_scale: Whether to use logarithmic Y axis.
    """
    colors: list[str] = ["#4C72B0", "#55A868", "#C44E52"]
    model_names: list[str] = list(results.keys())

    for t_idx, target_name in enumerate(TARGET_NAMES):
        fig, ax = plt.subplots(figsize=(8, 5))

        for m_idx, model_name in enumerate(model_names):
            res_list: list[dict] = results[model_name]
            sigmas: list[float] = [r["sigma"] for r in res_list]
            maes: list[float] = [r["test_mae_per_target"][t_idx] for r in res_list]
            color: str = colors[m_idx % len(colors)]
            ax.plot(sigmas, maes, marker="o", label=model_name, color=color)

        ax.set_xlabel("Noise σ (Å)")
        ax.set_ylabel("Test MAE")
        ax.set_title(f"Noise Robustness — {target_name}{title_suffix}")
        if log_scale:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        filepath: str = os.path.join(
            output_dir, filename_template.format(target=target_name)
        )
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"Per-target ablation curves saved to {filepath}")


def plot_ablation_curves(results: dict[str, list[dict]], output_dir: str) -> None:
    """Plot main combined noise ablation curves (sigma <= 0.5, linear scale).

    Args:
        results: Full results dict.
        output_dir: Directory where the plot will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    filtered: dict[str, list[dict]] = _filter_results_by_sigma(
        results, MAIN_PLOT_MAX_SIGMA
    )
    _plot_ablation_combined(
        filtered, output_dir, "noise_ablation_curves.png", "", log_scale=False
    )


def plot_ablation_curves_per_target(
    results: dict[str, list[dict]], output_dir: str
) -> None:
    """Plot main per-target noise ablation curves (sigma <= 0.5, linear scale).

    Args:
        results: Full results dict.
        output_dir: Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    filtered: dict[str, list[dict]] = _filter_results_by_sigma(
        results, MAIN_PLOT_MAX_SIGMA
    )
    _plot_ablation_per_target(
        filtered,
        output_dir,
        "noise_ablation_{target}.png",
        "",
        log_scale=False,
    )


def plot_ablation_curves_log(results: dict[str, list[dict]], output_dir: str) -> None:
    """Plot supplementary combined noise ablation curves (all sigmas, log scale).

    Includes all sigma values to show divergence behaviour at high noise.

    Args:
        results: Full results dict.
        output_dir: Directory where the plot will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    _plot_ablation_combined(
        results,
        output_dir,
        "noise_ablation_curves_log.png",
        " — Log Scale",
        log_scale=True,
    )


def plot_ablation_curves_per_target_log(
    results: dict[str, list[dict]], output_dir: str
) -> None:
    """Plot supplementary per-target noise ablation curves (all sigmas, log scale).

    Args:
        results: Full results dict.
        output_dir: Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    _plot_ablation_per_target(
        results,
        output_dir,
        "noise_ablation_{target}_log.png",
        " — Log Scale",
        log_scale=True,
    )
