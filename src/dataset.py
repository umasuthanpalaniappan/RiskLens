"""Windowing, normalisation and train/val/test splitting.

Two dataset shapes are produced from the same underlying per-ticker feature
panel (see features.py):

* ``build_lstm_dataset`` pools ``(ticker, date)`` samples across the whole
  universe into flat tensors for the single-stock LSTM baseline.
* ``build_gnn_dataset`` keeps one sample *per date* containing every
  ticker's window plus that date's graph Laplacian (graph.py), which is
  what the dynamic GNN needs.

Normalisation statistics (mu/sd) are always fit on the train split only and
persisted to ``results/norm_stats.json`` so that a later, separate process
(the live portfolio script, or anyone doing inference) can reproduce the
exact same feature scaling without re-touching the training data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src import config, features, graph

NORM_COLUMNS = config.FEATURE_COLUMNS + ["target_vol_ratio"]


@dataclass
class Split:
    train_dates: pd.Index
    val_dates: pd.Index
    test_dates: pd.Index


def chronological_split(dates: pd.Index) -> Split:
    """Split a sorted date index into train/val/test with no shuffling."""
    n = len(dates)
    n_train = int(n * config.TRAIN_FRAC)
    n_val = int(n * config.VAL_FRAC)
    return Split(
        train_dates=dates[:n_train],
        val_dates=dates[n_train:n_train + n_val],
        test_dates=dates[n_train + n_val:],
    )


def compute_norm_stats(panel: Dict[str, pd.DataFrame], train_dates: pd.Index) -> Dict[str, Dict[str, float]]:
    """Fit mean/std for each column in ``NORM_COLUMNS`` on the train split only."""
    stats: Dict[str, Dict[str, float]] = {}
    for col in NORM_COLUMNS:
        values = np.concatenate(
            [frame.loc[frame.index.intersection(train_dates), col].dropna().values for frame in panel.values()]
        )
        stats[col] = {"mu": float(values.mean()), "sd": float(values.std() + 1e-8)}
    return stats


def save_norm_stats(stats: Dict[str, Dict[str, float]], path: str = config.NORM_STATS_PATH) -> None:
    config.ensure_dirs()
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_norm_stats(path: str = config.NORM_STATS_PATH) -> Dict[str, Dict[str, float]]:
    with open(path) as f:
        return json.load(f)


def normalize(value: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return (value - stats["mu"]) / stats["sd"]


def denormalize(value: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return value * stats["sd"] + stats["mu"]


def _normalized_feature_matrix(frame: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    cols = [normalize(frame[c].values, stats[c]) for c in config.FEATURE_COLUMNS]
    return np.stack(cols, axis=1)  # (T, F)


# --------------------------------------------------------------------------
# LSTM (pooled, single-stock) dataset
# --------------------------------------------------------------------------

def build_lstm_dataset(
    panel: Dict[str, pd.DataFrame],
    stats: Dict[str, Dict[str, float]],
    window: int = config.RETURN_WINDOW,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Pool sliding windows across all tickers into train/val/test tensors.

    Returns a dict with keys "train"/"val"/"test", each mapping to
    ``(X, y)`` where X is (N, window, F) and y is (N,) normalised target.
    """
    any_frame = next(iter(panel.values()))
    split = chronological_split(any_frame.index)
    buckets = {"train": ([], []), "val": ([], []), "test": ([], [])}

    for frame in panel.values():
        feat_mat = _normalized_feature_matrix(frame, stats)
        target = normalize(frame["target_vol_ratio"].values, stats["target_vol_ratio"])
        dates = frame.index
        for i in range(window, len(frame)):
            if np.isnan(feat_mat[i - window:i]).any() or np.isnan(target[i - 1]):
                continue
            end_date = dates[i - 1]
            if end_date in split.train_dates:
                bucket = "train"
            elif end_date in split.val_dates:
                bucket = "val"
            else:
                bucket = "test"
            buckets[bucket][0].append(feat_mat[i - window:i])
            buckets[bucket][1].append(target[i - 1])

    out = {}
    for name, (xs, ys) in buckets.items():
        X = torch.tensor(np.stack(xs), dtype=torch.float32) if xs else torch.empty(0, window, len(config.FEATURE_COLUMNS))
        y = torch.tensor(np.array(ys), dtype=torch.float32) if ys else torch.empty(0)
        out[name] = (X, y)
    return out


# --------------------------------------------------------------------------
# GNN (per-date, all-tickers) dataset
# --------------------------------------------------------------------------

@dataclass
class GraphSample:
    date: pd.Timestamp
    X: torch.Tensor       # (num_nodes, window, F)
    laplacian: torch.Tensor  # (num_nodes, num_nodes)
    y: torch.Tensor        # (num_nodes,) normalised target


def build_gnn_dataset(
    panel: Dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    stats: Dict[str, Dict[str, float]],
    window: int = config.RETURN_WINDOW,
    graph_window: int = config.GRAPH_WINDOW,
) -> Dict[str, List[GraphSample]]:
    """Build one graph sample per date, split chronologically train/val/test."""
    tickers = list(panel.keys())
    any_frame = panel[tickers[0]]
    dates = any_frame.index
    split = chronological_split(dates)

    feat_mats = {t: _normalized_feature_matrix(panel[t], stats) for t in tickers}
    targets = {t: normalize(panel[t]["target_vol_ratio"].values, stats["target_vol_ratio"]) for t in tickers}

    min_hist = max(window, graph_window)
    buckets: Dict[str, List[GraphSample]] = {"train": [], "val": [], "test": []}

    for i in range(min_hist, len(dates)):
        end_date = dates[i - 1]
        window_ok = True
        node_feats, node_targets = [], []
        for t in tickers:
            fm = feat_mats[t][i - window:i]
            tgt = targets[t][i - 1]
            if np.isnan(fm).any() or np.isnan(tgt):
                window_ok = False
                break
            node_feats.append(fm)
            node_targets.append(tgt)
        if not window_ok:
            continue

        graph_returns_window = returns.loc[dates[i - graph_window:i]]
        if graph_returns_window.isna().any().any():
            continue
        adjacency = graph.correlation_adjacency(graph_returns_window)
        laplacian = graph.graph_laplacian(adjacency)

        if end_date in split.train_dates:
            bucket = "train"
        elif end_date in split.val_dates:
            bucket = "val"
        else:
            bucket = "test"

        buckets[bucket].append(
            GraphSample(
                date=end_date,
                X=torch.tensor(np.stack(node_feats), dtype=torch.float32),
                laplacian=torch.tensor(laplacian, dtype=torch.float32),
                y=torch.tensor(node_targets, dtype=torch.float32),
            )
        )
    return buckets


def latest_graph_sample(
    panel: Dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    stats: Dict[str, Dict[str, float]],
    window: int = config.RETURN_WINDOW,
    graph_window: int = config.GRAPH_WINDOW,
) -> GraphSample:
    """Build a single graph sample from the most recent available window (no target)."""
    tickers = list(panel.keys())
    feat_mats = {t: _normalized_feature_matrix(panel[t], stats) for t in tickers}
    dates = next(iter(panel.values())).index

    node_feats = [feat_mats[t][-window:] for t in tickers]
    adjacency = graph.correlation_adjacency(returns.iloc[-graph_window:])
    laplacian = graph.graph_laplacian(adjacency)

    return GraphSample(
        date=dates[-1],
        X=torch.tensor(np.stack(node_feats), dtype=torch.float32),
        laplacian=torch.tensor(laplacian, dtype=torch.float32),
        y=torch.empty(len(tickers)),
    )
