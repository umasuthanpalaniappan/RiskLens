#!/usr/bin/env python3
"""Live daily RiskLens portfolio construction.

Loads the trained GNN model and saved normalisation stats, pulls the latest
available prices, predicts each stock's volatility ratio for the next
horizon, builds the predicted covariance matrix and dynamic graph from the
most recent GRAPH_WINDOW-day window, classifies today's volatility regime,
solves the long-only minimum-variance portfolio, and writes the result to
results/portfolio_weights.json. Intended to be run daily (see
.github/workflows/daily_update.yml).

Usage: python scripts/run_portfolio.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src import config, data, dataset, features, graph, models, portfolio, regime


def main() -> None:
    config.ensure_dirs()
    tickers = config.TICKERS

    stats = dataset.load_norm_stats()

    model = models.GNNVolModel(num_nodes=len(tickers))
    model.load_state_dict(torch.load(config.GNN_WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    lookback = max(config.RETURN_WINDOW, config.GRAPH_WINDOW, config.REGIME_LOOKBACK) + config.VOL_WINDOW + config.HORIZON + 10
    prices, volumes = data.fetch_latest_window(tickers, lookback_days=lookback)
    returns = features.log_returns(prices)
    panel = features.build_panel(prices, volumes)
    panel = features.align_panel(panel, require_target=False)

    sample = dataset.latest_graph_sample(panel, returns, stats)
    with torch.no_grad():
        pred_norm, _, _, alpha = model(sample.X, sample.laplacian)
    predicted_ratio = dataset.denormalize(pred_norm.numpy(), stats["target_vol_ratio"])

    current_vol = np.array([panel[t]["realized_vol"].iloc[-1] for t in tickers])
    predicted_vol = current_vol * predicted_ratio

    recent_window = returns.iloc[-config.GRAPH_WINDOW:][tickers]
    corr_matrix = graph.correlation_matrix(recent_window)
    laplacian = graph.graph_laplacian(graph.correlation_adjacency(recent_window))

    market_vol = regime.market_volatility(returns[tickers])
    today_regime = regime.current_regime(market_vol)

    sigma = portfolio.predicted_covariance(predicted_vol, corr_matrix)
    weights = portfolio.optimize(sigma, regime=today_regime)

    result = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "as_of_date": str(panel[tickers[0]].index[-1].date()),
        "regime": today_regime,
        "tickers": tickers,
        "weights": {t: float(w) for t, w in zip(tickers, weights)},
        "predicted_volatility": {t: float(v) for t, v in zip(tickers, predicted_vol)},
        "predicted_vol_ratio": {t: float(v) for t, v in zip(tickers, predicted_ratio)},
        "gnn_alpha": {t: float(a) for t, a in zip(tickers, alpha.numpy())},
    }

    with open(config.PORTFOLIO_RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Regime: {today_regime}")
    for t, w in result["weights"].items():
        print(f"  {t}: weight={w:.4f}  predicted_vol={result['predicted_volatility'][t]:.4f}")
    print(f"Saved to {config.PORTFOLIO_RESULTS_PATH}")


if __name__ == "__main__":
    main()
