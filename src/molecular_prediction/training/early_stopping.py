# src/molecular_prediction/training/early_stopping.py


import numpy as np
from torch import nn

from molecular_prediction.training.utils import save_parameters


class EarlyStopping:
    """
    Class to implement early stopping during training.
    """

    def __init__(self, patience: int, delta: float = 1e-4) -> None:
        """
        Constructor of the class.

        Args:
            patience: Epochs allowed without improving validation loss until the
                training is stopped.
            delta: Minimum change in the validation loss to consider an improvement.
        """

        self.patience: int = patience
        self.delta: float = delta
        self.epochs_without_improvement: int = 0
        self.best_validation_loss: float = np.inf
        self.apply_early_stop: bool = False

    def __call__(self, val_loss: float, model: nn.Module, path: str) -> None:
        """
        Call method.

        Args:
            val_loss: New value of the validation loss.
            model: Model to which early stopping is applied.
            path: Path where the parameters of the model are saved if the new validation
                loss is the best one obtained.
        """

        if val_loss >= self.best_validation_loss - self.delta:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.apply_early_stop = True
        else:
            self.best_validation_loss = val_loss
            save_parameters(model, path)
            self.epochs_without_improvement = 0
