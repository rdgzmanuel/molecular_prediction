# src/molecular_prediction/data/dataset.py

import torch
import yaml
from torch_geometric.data import Dataset
from torch_geometric.datasets import QM9

from molecular_prediction.data.transforms import NormaliseTargets


def load_config(config_path: str) -> dict:
    """Load the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary with the configuration.
    """
    with open(config_path, "r") as f:
        data: dict = yaml.safe_load(f)
    return data


def compute_normalisation_stats(
    dataset: Dataset, target_indices: list[int]
) -> dict[int, dict[str, torch.Tensor]]:
    """Compute mean and std of the targets in the dataset.
    Only call this on the train set.

    Args:
        dataset: PyG dataset to compute statistics from.
        target_indices: List of target indices to normalise.

    Returns:
        Dictionary mapping target index to {"mean": ..., "std": ...}.
    """
    values: list[torch.Tensor] = [
        dataset[i].y[0, target_indices] for i in range(len(dataset))
    ]
    matrix: torch.Tensor = torch.stack(values)
    mean: torch.Tensor = matrix.mean(dim=0)
    std: torch.Tensor = matrix.std(dim=0)

    results: dict[int, dict[str, torch.Tensor]] = {}
    for i, idx in enumerate(target_indices):
        results[idx] = {"mean": mean[i], "std": std[i]}
    return results


def load_splits(config_path: str) -> tuple[Dataset, Dataset, Dataset]:
    """Load QM9 and split into train, val and test sets.
    Normalises targets using statistics computed from the train set only.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    config: dict = load_config(config_path)
    data_cfg: dict = config["data"]

    torch.manual_seed(data_cfg["seed"])
    dataset: QM9 = QM9(data_cfg["root"])

    train_idx, val_idx = data_cfg["split"]
    train_dataset: Dataset = dataset[:train_idx]
    val_dataset: Dataset = dataset[train_idx : train_idx + val_idx]
    test_dataset: Dataset = dataset[train_idx + val_idx :]

    target_indices: list[int] = data_cfg["target_indices"]
    norm_stats: dict[int, dict[str, torch.Tensor]] = compute_normalisation_stats(
        train_dataset, target_indices
    )

    transform: NormaliseTargets = NormaliseTargets(norm_stats, target_indices)
    train_dataset.transform = transform
    val_dataset.transform = transform
    test_dataset.transform = transform

    return train_dataset, val_dataset, test_dataset
