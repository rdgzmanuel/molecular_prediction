"""
Configuration of the project.
"""

from dataclasses import dataclass

import torch


@dataclass
class PathsConfig:
    """
    Class to define the configuration of the paths where the artifacts are saved.
    """

    path_data: str = "./data/raw"
    path_weights: str = "models"
    output_dir: str = "runs"


@dataclass
class DataConfig:
    """
    Class to define the configuration of the data.
    """

    target_indices: tuple[int] = (0, 1, 4, 11)
    split: tuple[int] = (110000, 10000)  # train/val, rest is testing
    seed: int = 42


@dataclass
class ModelConfig:
    """
    Class to define the configuration of the model.
    """

    hidden_dim: int = 128
    num_layers: int = 4
    edge_attr_dim: int = 4
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """
    Class to define the configuration of the training.
    """

    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 1
    patience: int = 20
    delta: float = 1e-4
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = (
        torch.optim.lr_scheduler.CosineAnnealingLR
    )
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ExperimentConfig:
    """
    Class to define the configuration of the training.
    """

    name: str = "main_comparison"


@dataclass
class Config:
    """
    Main configuration class.
    """

    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    experiment: ExperimentConfig = ExperimentConfig()


config: Config = Config()
