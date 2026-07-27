#!/usr/bin/env python3
"""One-off export of aggregated real market data for the standalone showcase
dashboard artifact published outside this repo (not used by app/dashboard.py
or any pipeline step -- app/dashboard.py reads data/raw/ directly instead).

Prints a single compact JSON line to stdout, delimited by marker lines, so it
can be captured from an Actions job's logs without ever committing raw
market data to the repo. Date arrays are stored once at the top level (every
ticker shares the same trading calendar) rather than repeated per ticker.

Exports the FULL daily history for every ticker (not a downsampled subset)
so the showcase page can offer real lookback ranges (1Y/3Y/5Y/10Y/Max) and
a complete raw data table per stock, not just a recent window.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config, data, features, graph, regime


def main() -> None:
    tickers = config.TICKERS
    prices, volumes = data.fetch_price_volume(tickers)
    returns = features.log_returns(prices)
    vol = pd.DataFrame({t: features.realized_vol(returns[t]) for t in tickers})

    per_ticker = {}
    for t in tickers:
        per_ticker[t] = {
            "price": [round(float(v), 2) if pd.notna(v) else None for v in prices[t]],
            "volume": [int(v) if pd.notna(v) else None for v in volumes[t]],
            "return": [round(float(v), 5) if pd.notna(v) else None for v in returns[t]],
            "realized_vol": [round(float(v), 4) if pd.notna(v) else None for v in vol[t]],
        }

    recent_window = returns.iloc[-config.GRAPH_WINDOW:][tickers]
    corr = graph.correlation_matrix(recent_window)
    correlation = {"tickers": tickers, "matrix": [[round(float(x), 4) for x in row] for row in corr]}

    market_vol = regime.market_volatility(returns[tickers]).dropna()
    step = max(1, len(market_vol) // 200)  # ~200 points across full history
    regime_timeline = [
        [market_vol.index[i].strftime("%Y-%m-%d"), regime.current_regime(market_vol.iloc[: i + 1]), round(float(market_vol.iloc[i]), 4)]
        for i in range(0, len(market_vol), step)
    ]

    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "tickers": tickers,
        "dates": [d.strftime("%Y-%m-%d") for d in prices.index],
        "series": per_ticker,
        "correlation": correlation,
        "regime_timeline": regime_timeline,
    }
    blob = json.dumps(payload, separators=(",", ":"))
    print(f"payload size: {len(blob)} bytes", file=sys.stderr)

    # Printed in fixed-size chunks (not one giant line) since some CI log
    # backends cap individual line length; reassemble by concatenating
    # chunk lines in order between the markers.
    chunk_size = 50_000
    chunks = [blob[i:i + chunk_size] for i in range(0, len(blob), chunk_size)]
    print("===DASHBOARD_DATA_START===")
    print(f"CHUNKS={len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"CHUNK{i:05d}:{c}")
    print("===DASHBOARD_DATA_END===")


if __name__ == "__main__":
    main()
