"""Dynamic correlation-graph construction for the GNN and the portfolio optimiser.

The graph over stocks is re-estimated on a rolling basis: at any date ``t``
the adjacency is the absolute pairwise correlation over the trailing
``GRAPH_WINDOW`` days ending at ``t`` (no look-ahead), of whatever signal the
caller passes in. portfolio.py passes *returns* (it needs a real covariance
Sigma). dataset.py's GNN dataset passes *realised volatility* instead --
vol-clustering/contagion structure is what's actually relevant to a model
predicting where volatility goes next, not return correlation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def correlation_matrix(window_data: pd.DataFrame) -> np.ndarray:
    """Signed pairwise-correlation matrix (unit diagonal) for one window of data.

    Distinct from ``correlation_adjacency``: this keeps the sign and the
    diagonal, because portfolio.py needs a proper correlation matrix to
    build the predicted covariance Sigma, whereas the GNN's message-passing
    graph wants an unsigned, zero-diagonal, thresholded adjacency.
    """
    return window_data.corr().fillna(0.0).values


def correlation_adjacency(
    window_data: pd.DataFrame,
    threshold: float = config.GRAPH_CORR_THRESHOLD,
) -> np.ndarray:
    """Absolute pairwise-correlation adjacency matrix for one window of data.

    ``window_data`` is (days x tickers). The diagonal is zeroed (no
    self-loops; models.py adds self-loops explicitly when needed) and edges
    with |correlation| below ``threshold`` are zeroed out.
    """
    corr = window_data.corr().abs().fillna(0.0).values.copy()
    np.fill_diagonal(corr, 0.0)
    if threshold > 0:
        corr[corr < threshold] = 0.0
    return corr


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
