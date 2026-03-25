# src/molecular_prediction/training/metrics.py

import torch


def mae_per_target(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute Mean Absolute Error per target.

    Args:
        predictions: Predicted values of shape [num_graphs, num_targets].
        targets: Ground truth values of shape [num_graphs, num_targets].

    Returns:
        MAE per target of shape [num_targets].
    """
    return torch.abs(predictions - targets).mean(dim=0)
