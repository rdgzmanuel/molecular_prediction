# src/molecular_prediction/models/gin.py

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv

from src.molecular_prediction.models.base import BaseGNN

NODE_FEATURE_DIM: int = 11


class GIN(BaseGNN):
    """Topological GNN baseline using Graph Isomorphism Network.
    Uses only atomic features and bond connectivity — no 3D geometry.

    Args:
        hidden_dim: Dimensionality of node embeddings.
        num_layers: Number of GIN message passing layers.
        num_targets: Number of target properties to predict.
    """

    def __init__(self, hidden_dim: int, num_layers: int, num_targets: int) -> None:
        super().__init__(hidden_dim, num_layers, num_targets)

        self.node_embedding: nn.Module = nn.Linear(NODE_FEATURE_DIM, hidden_dim)

        for _ in range(num_layers):
            mlp: nn.Module = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.layers.append(GINConv(mlp, train_eps=False))

        self.prediction_head: nn.Module = nn.Linear(hidden_dim, num_targets)

    def message_pass(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one GIN message passing step.

        Args:
            h: Node feature matrix of shape [num_nodes, hidden_dim].
            edge_index: Graph connectivity of shape [2, num_edges].
            edge_attr: Ignored in GIN — topology only.

        Returns:
            Updated node features of shape [num_nodes, hidden_dim].
        """
        for layer in self.layers:
            h = layer(h, edge_index)

        return h
