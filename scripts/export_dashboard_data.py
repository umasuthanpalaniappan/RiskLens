#!/usr/bin/env python3
"""Export aggregated real market data for the Vercel-deployable web dashboard
(web/index.html), which fetches this file directly -- no inline embedding,
no log-scraping.

Writes web/data.json: per-ticker full daily price/volume/return/realised-vol
history, a 60-day correlation matrix, and a historical regime timeline --
plus the three results JSONs (portfolio/gnn/lstm) and the sector map, all in
one payload so the frontend needs exactly one fetch.

Committed by daily_update.yml (and train_models.yml) so that a Vercel
project with Git integration pointed at this repo redeploys automatically,
with fresh real data, every time the daily job runs.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config, data, features, graph, regime

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "web" / "data.json"


def build_market_data() -> dict:
    tickers = config.TICKERS
    prices, volumes = data.fetch_price_volume(tickers)
    returns = features.log_returns(prices)
    vol = pd.DataFrame({t: features.realized_vol(returns[t]) for t in tickers})

    series = {}
    for t in tickers:
        series[t] = {
            "price": [round(float(v), 2) if pd.notna(v) else None for v in prices[t]],
            "volume": [int(v) if pd.notna(v) else None for v in volumes[t]],
            "return": [round(float(v), 5) if pd.notna(v) else None for v in returns[t]],
            "realized_vol": [round(float(v), 4) if pd.notna(v) else None for v in vol[t]],
        }

    recent_window = returns.iloc[-config.GRAPH_WINDOW:][tickers]
    corr = graph.correlation_matrix(recent_window)
    correlation = {"tickers": tickers, "matrix": [[round(float(x), 4) for x in row] for row in corr]}

    market_vol = regime.market_volatility(returns[tickers]).dropna()
    step = max(1, len(market_vol) // 200)
    regime_timeline = [
        [market_vol.index[i].strftime("%Y-%m-%d"), regime.current_regime(market_vol.iloc[: i + 1]), round(float(market_vol.iloc[i]), 4)]
        for i in range(0, len(market_vol), step)
    ]

    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "tickers": tickers,
        "dates": [d.strftime("%Y-%m-%d") for d in prices.index],
        "series": series,
        "correlation": correlation,
        "regime_timeline": regime_timeline,
    }


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    payload = {
        "market": build_market_data(),
        "portfolio": load_json(root / "results" / "portfolio_weights.json"),
        "gnn": load_json(root / "results" / "gnn_results.json"),
        "lstm": load_json(root / "results" / "lstm_results.json"),
        "sector_map": config.SECTOR_MAP,
        "max_weight": config.PORTFOLIO_MAX_WEIGHT,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
