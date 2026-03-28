# main.py
#
# Entry point for the molecular property prediction experiments.
#
# Usage:
#   python main.py --experiment main_comparison
#   python main.py --experiment noise_ablation

import argparse

import torch

from configs.config import Config
from src.molecular_prediction.experiments.main_comparison import (
    plot_test_mae,
    plot_test_mae_per_target,
    plot_training_curves,
    plot_training_curves_per_target,
    run_comparison,
    save_results,
)
from src.molecular_prediction.experiments.noise_ablation import (
    plot_ablation_curves,
    plot_ablation_curves_per_target,
    run_noise_ablation,
    save_ablation_results,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with at least:
            .config      (str)  – path to the YAML config file
            .experiment  (str)  – which experiment to run
            .device      (str)  – compute device ('cpu', 'cuda', 'mps')
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Molecular property prediction experiments on QM9."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["main_comparison", "noise_ablation"],
        default="main_comparison",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"],
        help="Compute device.",
    )

    return parser.parse_args()


def select_device(requested: str) -> str:
    """Resolve the compute device, falling back gracefully if unavailable.

    Args:
        requested: Device string requested by the user ('cpu', 'cuda', 'mps').

    Returns:
        A valid torch device string.
    """
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    elif requested != "cpu":
        print(f"Requested device '{requested}' not available, falling back to CPU.")
    return "cpu"


def run_main_comparison(config: Config, device: str) -> None:
    """Orchestrate the main model-comparison experiment.

    Calls run_comparison(), then save_results(), and plots.
    All outputs are saved to disk.

    Args:
        config: Full experiment config dict.
        device: PyTorch device string.
    """
    results: dict = run_comparison(config, device)
    save_results(results, config.paths.results_dir)
    plot_training_curves(results, config.paths.images_dir)
    plot_training_curves_per_target(results, config.paths.images_dir)
    plot_test_mae(results, config.paths.images_dir)
    plot_test_mae_per_target(results, config.paths.images_dir)


def run_noise_ablation_experiment(config: Config, device: str) -> None:
    """Orchestrate the noise-ablation experiment.

    Calls run_noise_ablation(), then save_ablation_results() and
    plots. All outputs are saved to disk.

    Args:
        config: Full experiment config dict.
        device: PyTorch device string.
    """
    results: dict[str, list[dict]] = run_noise_ablation(config, device)
    save_ablation_results(results, config.paths.results_dir)
    plot_ablation_curves(results, config.paths.images_dir)
    plot_ablation_curves_per_target(results, config.paths.images_dir)


def main() -> None:
    """Parse arguments, load config, and dispatch to the chosen experiment."""
    args: argparse.Namespace = parse_args()
    config: Config = Config()
    device: str = select_device(args.device)

    print(f"Experiment: {args.experiment}")
    print(f"Device:     {device}")

    if args.experiment == "main_comparison":
        run_main_comparison(config, device)
    elif args.experiment == "noise_ablation":
        run_noise_ablation_experiment(config, device)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")


if __name__ == "__main__":
    main()
