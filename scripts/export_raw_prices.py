#!/usr/bin/env python3
"""Export raw (unadjusted) daily close prices for the RiskLens 18-stock
universe -- i.e. the actual traded close price, matching what Google
Finance / NSE show, as opposed to dataset/prices.csv which is
dividend/split-adjusted close (auto_adjust=True) and is what the models
train on.

This is a one-off export for cross-checking against external sources, not
part of the modelling pipeline -- the models deliberately use adjusted
close to avoid fake volatility spikes on dividend/split dates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf

from src import config

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "dataset" / "prices_raw_unadjusted.csv"


def main() -> None:
    tickers = config.TICKERS
    raw = yf.download(
        tickers,
        start=config.START_DATE,
        end=config.END_DATE,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers})
    prices = prices.sort_index()
    prices.index.name = "date"
    prices.to_csv(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH} ({len(prices)} rows, {len(tickers)} tickers)")


if __name__ == "__main__":
    main()
