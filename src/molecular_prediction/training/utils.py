# src/molecular_prediction/training/utils.py

import torch
from torch import nn


def save_parameters(model: nn.Module, path: str) -> None:
    """
    Saves the parameters of the model.

    Args:
        path: Path where the parameters of the model are saved.
    """

    torch.save(model.state_dict(), path)
