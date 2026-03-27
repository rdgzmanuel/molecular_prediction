# src/molecular_prediction/models/base.py

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool


class BaseGNN(ABC, nn.Module):
    """Abstract base class for all GNN models in the project.
    Defines the shared structure: node embedding, message passing layers,
    coordinate update (optional), global readout, and prediction head.

    Args:
        hidden_dim: Dimensionality of node embeddings.
        num_layers: Number of message passing layers.
        num_targets: Number of target properties to predict.
    """

    def __init__(self, hidden_dim: int, num_layers: int, num_targets: int) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.num_targets: int = num_targets

        self.node_embedding: nn.Module = None  # to be defined in subclass __init__
        self.layers: nn.ModuleList = nn.ModuleList()
        self.prediction_head: nn.Module = None  # to be defined in subclass __init__

    @abstractmethod
    def message_pass(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one message passing step and return updated node features.

        Args:
            h: Node feature matrix of shape [num_nodes, hidden_dim].
            edge_index: Graph connectivity of shape [2, num_edges].
            edge_attr: Edge feature matrix of shape [num_edges, edge_dim], optional.

        Returns:
            Updated node features of shape [num_nodes, hidden_dim].
        """
        pass

    def update_coords(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        messages: torch.Tensor,
    ) -> torch.Tensor:
        """Update 3D coordinates using aggregated messages.
        Default implementation is a no-op — only overridden by EGNN.

        Args:
            pos: Atomic coordinates of shape [num_nodes, 3].
            edge_index: Graph connectivity of shape [2, num_edges].
            messages: Message tensor of shape [num_edges, hidden_dim].

        Returns:
            Updated coordinates of shape [num_nodes, 3].
        """
        return pos

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass shared by the two simple models models.

        Args:
            data: PyG Data object containing node features, edge index,
                  edge attributes and optionally 3D coordinates.

        Returns:
            Predicted targets of shape [num_graphs, num_targets].
        """
        h: torch.Tensor = self.node_embedding(data.x)
        edge_index: torch.Tensor = data.edge_index
        edge_attr: Optional[torch.Tensor] = data.edge_attr

        h = self.message_pass(h, edge_index, edge_attr)
        graph_repr = global_add_pool(h, data.batch)
        out = self.prediction_head(graph_repr)
        return out
