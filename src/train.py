"""Training loops for the LSTM baseline and the dynamic-graph GNN.

Both ``train_lstm`` and ``train_gnn`` share the same data-prep step
(``prepare_universe``), the same normalisation stats (persisted once to
``results/norm_stats.json`` so live inference in run_portfolio.py can reuse
them exactly), and the same evaluation metrics (evaluate.py). This keeps
scripts/run_lstm.py and scripts/run_gnn.py thin wrappers with no duplicated
modelling logic.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import config, data, dataset, evaluate, features, models


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _save_json(obj: dict, path: str) -> None:
    config.ensure_dirs()
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def prepare_universe(tickers=None) -> Tuple[Dict[str, "pd.DataFrame"], "pd.DataFrame"]:
    """Download/cache prices and build the aligned per-ticker feature panel.

    Returns ``(panel, returns)`` where ``panel`` is aligned across tickers
    (see features.align_panel) and ``returns`` is the *unaligned* full daily
    log-return matrix (needed so the GNN dataset can look further back for
    the rolling correlation graph than the feature panel's own start date).
    """
    tickers = tickers or config.TICKERS
    prices, volumes = data.fetch_price_volume(tickers)
    returns = features.log_returns(prices)
    panel = features.build_panel(prices, volumes)
    panel = features.align_panel(panel)
    return panel, returns


def _get_or_build_norm_stats(panel, train_dates):
    stats = dataset.compute_norm_stats(panel, train_dates)
    dataset.save_norm_stats(stats)
    return stats


# --------------------------------------------------------------------------
# LSTM baseline
# --------------------------------------------------------------------------

def train_lstm(tickers=None, epochs: int = config.EPOCHS) -> dict:
    """Train the pooled single-stock LSTM baseline and save weights + results."""
    panel, _returns = prepare_universe(tickers)
    any_dates = next(iter(panel.values())).index
    split = dataset.chronological_split(any_dates)
    stats = _get_or_build_norm_stats(panel, split.train_dates)

    data_splits = dataset.build_lstm_dataset(panel, stats)
    train_X, train_y = data_splits["train"]
    val_X, val_y = data_splits["val"]
    test_X, test_y = data_splits["test"]

    model = models.LSTMVolModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()

    loader = DataLoader(TensorDataset(train_X, train_y), batch_size=config.BATCH_SIZE, shuffle=True)

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X) if len(val_X) else torch.empty(0)
            val_loss = loss_fn(val_pred, val_y).item() if len(val_X) else float("nan")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred_norm = model(test_X).numpy() if len(test_X) else np.array([])
    test_true_norm = test_y.numpy() if len(test_y) else np.array([])

    tstats = stats["target_vol_ratio"]
    test_pred = dataset.denormalize(test_pred_norm, tstats)
    test_true = dataset.denormalize(test_true_norm, tstats)

    config.ensure_dirs()
    torch.save(model.state_dict(), config.LSTM_WEIGHTS_PATH)

    results = {
        "timestamp": _now_iso(),
        "epochs": epochs,
        "tickers": list(panel.keys()),
        "n_train": int(len(train_y)),
        "n_val": int(len(val_y)),
        "n_test": int(len(test_y)),
        "test_mse": evaluate.mse(test_pred, test_true) if len(test_true) else None,
        "test_mae": evaluate.mae(test_pred, test_true) if len(test_true) else None,
        "test_direction_accuracy": evaluate.direction_accuracy(test_pred, test_true) if len(test_true) else None,
    }
    _save_json(results, config.LSTM_RESULTS_PATH)
    return results


# --------------------------------------------------------------------------
# Dynamic GNN
# --------------------------------------------------------------------------

def train_gnn(tickers=None, epochs: int = config.EPOCHS) -> dict:
    """Train the dynamic-graph GNN and save weights + results (incl. per-node alpha)."""
    tickers = tickers or config.TICKERS
    panel, _returns = prepare_universe(tickers)
    any_dates = next(iter(panel.values())).index
    split = dataset.chronological_split(any_dates)
    stats = _get_or_build_norm_stats(panel, split.train_dates)

    graph_splits = dataset.build_gnn_dataset(panel, stats)
    train_samples = graph_splits["train"]
    val_samples = graph_splits["val"]
    test_samples = graph_splits["test"]

    model = models.GNNVolModel(num_nodes=len(tickers))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()

    rng = np.random.default_rng(config.SEED)
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_samples))
        for idx in order:
            s = train_samples[idx]
            optimizer.zero_grad()
            pred, _, _, _ = model(s.X, s.laplacian)
            loss = loss_fn(pred, s.y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for s in val_samples:
                pred, _, _, _ = model(s.X, s.laplacian)
                val_losses.append(loss_fn(pred, s.y).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    tstats = stats["target_vol_ratio"]
    blended_all, lstm_all, gnn_all, true_all = [], [], [], []
    with torch.no_grad():
        for s in test_samples:
            pred, lstm_pred, gnn_pred, alpha = model(s.X, s.laplacian)
            blended_all.append(pred.numpy())
            lstm_all.append(lstm_pred.numpy())
            gnn_all.append(gnn_pred.numpy())
            true_all.append(s.y.numpy())
        final_alpha = torch.sigmoid(model.raw_alpha).numpy()

    config.ensure_dirs()
    torch.save(model.state_dict(), config.GNN_WEIGHTS_PATH)

    if true_all:
        blended = dataset.denormalize(np.concatenate(blended_all), tstats)
        lstm_only = dataset.denormalize(np.concatenate(lstm_all), tstats)
        gnn_only = dataset.denormalize(np.concatenate(gnn_all), tstats)
        true = dataset.denormalize(np.concatenate(true_all), tstats)
        metrics = {
            "test_mse_blended": evaluate.mse(blended, true),
            "test_mae_blended": evaluate.mae(blended, true),
            "test_direction_accuracy_blended": evaluate.direction_accuracy(blended, true),
            "test_mse_lstm_only": evaluate.mse(lstm_only, true),
            "test_mae_lstm_only": evaluate.mae(lstm_only, true),
            "test_direction_accuracy_lstm_only": evaluate.direction_accuracy(lstm_only, true),
            "test_mse_gnn_only": evaluate.mse(gnn_only, true),
            "test_mae_gnn_only": evaluate.mae(gnn_only, true),
            "test_direction_accuracy_gnn_only": evaluate.direction_accuracy(gnn_only, true),
        }
    else:
        metrics = {k: None for k in (
            "test_mse_blended", "test_mae_blended", "test_direction_accuracy_blended",
            "test_mse_lstm_only", "test_mae_lstm_only", "test_direction_accuracy_lstm_only",
            "test_mse_gnn_only", "test_mae_gnn_only", "test_direction_accuracy_gnn_only",
        )}

    results = {
        "timestamp": _now_iso(),
        "epochs": epochs,
        "tickers": tickers,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "alpha": {t: float(a) for t, a in zip(tickers, final_alpha)},
        **metrics,
    }
    _save_json(results, config.GNN_RESULTS_PATH)
    return results
