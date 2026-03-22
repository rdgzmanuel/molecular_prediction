# src/molecular_prediction/data/transforms.py

import torch
from torch_geometric.data import Data


class NormaliseTargets:
    """Normalise molecular property targets using precomputed statistics.
    Applied on-the-fly when accessing dataset elements.

    Args:
        stats: Dictionary mapping target index to {"mean": ..., "std": ...}.
        target_indices: List of target indices to normalise.
    """

    def __init__(
        self, stats: dict[int, dict[str, torch.Tensor]], target_indices: list[int]
    ) -> None:
        self.stats: dict[int, dict[str, torch.Tensor]] = stats
        self.target_indices: list[int] = target_indices

    def __call__(self, data: Data) -> Data:
        """Normalise the targets of a single molecule.

        Args:
            data: PyG Data object for a single molecule.

        Returns:
            Data object with normalised targets.
        """
        for idx in self.target_indices:
            data.y[0, idx] = (data.y[0, idx] - self.stats[idx]["mean"]) / (
                self.stats[idx]["std"] + 1e-6
            )

        return data


class AddGaussianNoise:
    """Add isotropic Gaussian noise to 3D atomic coordinates.
    Used for the noise robustness ablation experiment.

    Args:
        sigma: Standard deviation of the Gaussian noise.
    """

    def __init__(self, sigma: float) -> None:
        self.sigma: float = sigma

    def __call__(self, data: Data) -> Data:
        """Add noise to the 3D coordinates of a single molecule.

        Args:
            data: PyG Data object for a single molecule.

        Returns:
            Data object with perturbed coordinates.
        """
        noise: torch.Tensor = torch.rand_like(data.pos) * self.sigma
        data.pos = data.pos + noise

        return data
