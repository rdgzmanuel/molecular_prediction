# src/molecular_prediction/models/gin_dist.py

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv

from molecular_prediction.models.base import BaseGNN

NODE_FEATURE_DIM: int = 11


class GINDist(BaseGNN):
    """GNN baseline with interatomic distances as edge features.
    Uses GINEConv to incorporate edge features including ||x_i - x_j||^2.
    Invariant to E(3) but loses angular information.

    Args:
        hidden_dim: Dimensionality of node embeddings.
        num_layers: Number of message passing layers.
        num_targets: Number of target properties to predict.
        edge_attr_dim: Dimensionality of the original QM9 edge features.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_targets: int,
        edge_attr_dim: int,
    ) -> None:
        super().__init__(hidden_dim, num_layers, num_targets)

        self.node_embedding: nn.Module = nn.Linear(NODE_FEATURE_DIM, hidden_dim)
        self.edge_embedding: nn.Module = nn.Linear(edge_attr_dim + 1, hidden_dim)

        for _ in range(num_layers):
            mlp: nn.Module = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.layers.append(GINEConv(mlp, train_eps=False))

        self.prediction_head: nn.Module = nn.Linear(hidden_dim, num_targets)

    def _compute_distances(
        self, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute squared interatomic distances for each edge.

        Args:
            pos: Atomic coordinates of shape [num_nodes, 3].
            edge_index: Graph connectivity of shape [2, num_edges].

        Returns:
            Squared distances of shape [num_edges, 1].
        """
        sources: torch.Tensor = pos[edge_index[0]]
        destinations: torch.Tensor = pos[edge_index[1]]

        distances: torch.Tensor = ((sources - destinations) ** 2).sum(
            dim=1, keepdim=True
        )

        return distances

    def message_pass(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run all GINEConv message passing layers.

        Args:
            h: Node feature matrix of shape [num_nodes, hidden_dim].
            edge_index: Graph connectivity of shape [2, num_edges].
            edge_attr: Enriched edge features of shape [num_edges, hidden_dim].

        Returns:
            Updated node features of shape [num_nodes, hidden_dim].
        """
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        return h

    def forward(self, data: Data) -> torch.Tensor:
        """Enrich edge_attr with interatomic distances then run forward pass.

        Args:
            data: PyG Data object with node features, edge index,
                  edge attributes and 3D coordinates.

        Returns:
            Predicted targets of shape [num_graphs, num_targets].
        """
        distances: torch.Tensor = self._compute_distances(data.pos, data.edge_index)
        enriched_edge_attr: torch.Tensor = torch.cat([data.edge_attr, distances], dim=1)

        data.edge_attr = self.edge_embedding(enriched_edge_attr)

        return super().forward(data)
