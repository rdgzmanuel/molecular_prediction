# src/molecular_prediction/models/egnn.py

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import scatter

from src.molecular_prediction.models.base import BaseGNN

NODE_FEATURE_DIM: int = 11


class EGNNConv(MessagePassing):
    """Single EGNN message passing layer.
    Updates both node features h and coordinates x equivariantly.

    Args:
        hidden_dim: Dimensionality of node embeddings.
        edge_attr_dim: Dimensionality of edge features.
    """

    def __init__(self, hidden_dim: int, edge_attr_dim: int) -> None:
        super().__init__(aggr="sum")

        self.phi_e: nn.Module = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.phi_x: nn.Linear = nn.Linear(hidden_dim, 1)
        self.phi_h: nn.Module = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one EGNN layer updating both node features and coordinates.

        Args:
            h: Node features of shape [num_nodes, hidden_dim].
            pos: Atomic coordinates of shape [num_nodes, 3].
            edge_index: Graph connectivity of shape [2, num_edges].
            edge_attr: Edge features of shape [num_edges, edge_attr_dim].

        Returns:
            Tuple of (updated h, updated pos), both with same shape as input.
        """
        h = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)

        direction_vectors: torch.Tensor = pos[edge_index[0]] - pos[edge_index[1]]
        scaled: torch.Tensor = direction_vectors * self.phi_x(self._messages)
        pos = pos + scatter(
            scaled, edge_index[0], dim=0, dim_size=pos.size(0), reduce="sum"
        )

        return h, pos

    def message(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute messages m_ij using phi_e.

        Args:
            h_i: Features of target nodes of shape [num_edges, hidden_dim].
            h_j: Features of source nodes of shape [num_edges, hidden_dim].
            pos_i: Coordinates of target nodes of shape [num_edges, 3].
            pos_j: Coordinates of source nodes of shape [num_edges, 3].
            edge_attr: Edge features of shape [num_edges, edge_attr_dim].

        Returns:
            Messages of shape [num_edges, hidden_dim].
        """
        dist_sq: torch.Tensor = ((pos_i - pos_j) ** 2).sum(dim=1, keepdim=True)
        inputs: torch.Tensor = torch.cat([h_i, h_j, dist_sq, edge_attr], dim=1)
        self._messages: torch.Tensor = self.phi_e(inputs)
        return self._messages

    def update(self, aggr_out: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Update node features using phi_h.

        Args:
            aggr_out: Aggregated messages of shape [num_nodes, hidden_dim].
            h: Current node features of shape [num_nodes, hidden_dim].

        Returns:
            Updated node features of shape [num_nodes, hidden_dim].
        """
        return self.phi_h(torch.cat([h, aggr_out], dim=1))


class EGNN(BaseGNN):
    """E(3)-equivariant GNN using EGNN layers (Satorras et al., ICML 2021).
    Updates both node features and 3D coordinates equivariantly.

    Args:
        hidden_dim: Dimensionality of node embeddings.
        num_layers: Number of EGNN layers.
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

        for _ in range(num_layers):
            self.layers.append(EGNNConv(hidden_dim, edge_attr_dim))

        self.prediction_head: nn.Module = nn.Linear(hidden_dim, num_targets)

    def message_pass(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Not used — EGNN overrides forward directly."""
        raise NotImplementedError("EGNN overrides forward, message_pass is not called.")

    def forward(self, data: Data) -> torch.Tensor:
        """Full EGNN forward pass with coordinate updates.

        Args:
            data: PyG Data object with node features, edge index,
                  edge attributes and 3D coordinates.

        Returns:
            Predicted targets of shape [num_graphs, num_targets].
        """
        h: torch.Tensor = self.node_embedding(data.x)
        pos: torch.Tensor = data.pos
        edge_index: torch.Tensor = data.edge_index
        edge_attr: Optional[torch.Tensor] = data.edge_attr

        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)

        graph_repr: torch.Tensor = global_add_pool(h, data.batch)
        return self.prediction_head(graph_repr)
