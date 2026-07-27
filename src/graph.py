"""Dynamic correlation-graph construction for the GNN and the portfolio optimiser.

The graph over stocks is re-estimated on a rolling basis: at any date ``t``
the adjacency is the absolute pairwise correlation of daily returns over the
trailing ``GRAPH_WINDOW`` days ending at ``t`` (no look-ahead). Both the GNN
training loop (dataset.py) and the live portfolio script use these same
functions so the graph is defined identically everywhere.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src import config


def correlation_matrix(window_returns: pd.DataFrame) -> np.ndarray:
    """Signed pairwise-correlation matrix (unit diagonal) for one window of returns.

    Distinct from ``correlation_adjacency``: this keeps the sign and the
    diagonal, because portfolio.py needs a proper correlation matrix to
    build the predicted covariance Sigma, whereas the GNN's message-passing
    graph wants an unsigned, zero-diagonal, thresholded adjacency.
    """
    return window_returns.corr().fillna(0.0).values


def correlation_adjacency(
    window_returns: pd.DataFrame,
    threshold: float = config.GRAPH_CORR_THRESHOLD,
) -> np.ndarray:
    """Absolute pairwise-correlation adjacency matrix for one window of returns.

    ``window_returns`` is (days x tickers). The diagonal is zeroed (no
    self-loops; models.py adds self-loops explicitly when needed) and edges
    with |correlation| below ``threshold`` are zeroed out.
    """
    corr = window_returns.corr().abs().fillna(0.0).values.copy()
    np.fill_diagonal(corr, 0.0)
    if threshold > 0:
        corr[corr < threshold] = 0.0
    return corr


def rolling_adjacency_series(
    returns: pd.DataFrame,
    window: int = config.GRAPH_WINDOW,
    threshold: float = config.GRAPH_CORR_THRESHOLD,
) -> List[np.ndarray]:
    """Adjacency matrix ending at each date of ``returns`` once enough history exists.

    Returns a list aligned to ``returns.index[window - 1:]`` (one matrix per
    valid date), used by dataset.py to build the dynamic-graph training set.
    """
    adjacencies = []
    values = returns.values
    for end in range(window, len(returns) + 1):
        w = returns.iloc[end - window:end]
        adjacencies.append(correlation_adjacency(w, threshold))
    return adjacencies


def graph_laplacian(adjacency: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Graph Laplacian of ``adjacency``.

    ``normalized=True`` returns the symmetric normalised Laplacian
    ``L = I - D^-1/2 A D^-1/2``; otherwise the combinatorial ``L = D - A``.
    Isolated nodes (degree 0) are handled by leaving their row/column at 0
    in the normalised case to avoid division by zero.
    """
    degree = adjacency.sum(axis=1)
    if not normalized:
        return np.diag(degree) - adjacency

    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    d_inv_sqrt_mat = np.diag(d_inv_sqrt)
    n = adjacency.shape[0]
    return np.eye(n) - d_inv_sqrt_mat @ adjacency @ d_inv_sqrt_mat
