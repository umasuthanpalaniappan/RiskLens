"""LSTM baseline and dynamic-graph GNN volatility models.

``LSTMVolModel`` is a single-stock baseline: weights are shared across
tickers, but each ticker is scored independently from its own return/vol
window with no cross-stock information.

``GNNVolModel`` shares the same LSTM encoder architecture to embed each
stock's window, then mixes those embeddings across stocks using the
dynamic correlation graph (graph.py) via spectral graph-convolution layers
built on the graph Laplacian. A learnable, per-stock gate ``alpha`` (in
[0, 1], via sigmoid) blends the graph-refined prediction with the plain
LSTM prediction, so the model can fall back to the univariate signal for
stocks where the graph isn't informative.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src import config


class LSTMVolModel(nn.Module):
    """Single-stock LSTM regressor predicting the future/trailing vol ratio."""

    def __init__(
        self,
        input_size: int = len(config.FEATURE_COLUMNS),
        hidden_size: int = config.LSTM_HIDDEN,
        num_layers: int = config.LSTM_LAYERS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (batch, window, features) -> (batch,) predicted vol ratio."""
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


class GraphConvLayer(nn.Module):
    """One spectral graph-convolution step: propagate then linearly transform.

    Propagation uses ``(I - L)``, i.e. the normalised adjacency implied by
    the symmetric Laplacian from graph.graph_laplacian, so each node's
    representation is smoothed by its (weighted) neighbours before the
    per-node linear transform.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        propagation = torch.eye(n, device=h.device, dtype=h.dtype) - laplacian
        return self.linear(propagation @ h)


class GNNVolModel(nn.Module):
    """Dynamic-graph GNN volatility model with an LSTM encoder and a per-node gate."""

    def __init__(
        self,
        num_nodes: int,
        input_size: int = len(config.FEATURE_COLUMNS),
        lstm_hidden: int = config.LSTM_HIDDEN,
        lstm_layers: int = config.LSTM_LAYERS,
        gnn_hidden: int = config.GNN_HIDDEN,
        gnn_layers: int = config.GNN_LAYERS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_head = nn.Linear(lstm_hidden, 1)

        dims = [lstm_hidden] + [gnn_hidden] * gnn_layers
        self.convs = nn.ModuleList(
            GraphConvLayer(dims[i], dims[i + 1]) for i in range(gnn_layers)
        )
        self.gnn_head = nn.Linear(gnn_hidden, 1)
        self.dropout = nn.Dropout(dropout)

        # Per-node gate, one scalar per stock in the fixed universe order.
        self.raw_alpha = nn.Parameter(torch.zeros(num_nodes))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (num_nodes, window, features) -> (num_nodes, lstm_hidden)."""
        _, (h_n, _) = self.encoder(x)
        return h_n[-1]

    def forward(
        self, x: torch.Tensor, laplacian: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (blended_pred, lstm_only_pred, gnn_only_pred, alpha), each (num_nodes,)."""
        h = self.encode(x)
        lstm_pred = self.lstm_head(h).squeeze(-1)

        g = h
        for conv in self.convs:
            g = self.dropout(torch.relu(conv(g, laplacian)))
        gnn_pred = self.gnn_head(g).squeeze(-1)

        alpha = torch.sigmoid(self.raw_alpha)
        blended = alpha * gnn_pred + (1 - alpha) * lstm_pred
        return blended, lstm_pred, gnn_pred, alpha
