"""Price data acquisition and local caching.

Downloads daily adjusted-close prices for the RiskLens universe via
yfinance and caches them to ``data/raw/prices.csv`` so that repeated runs
(and offline dashboard reloads) don't re-hit the network. Callers that only
need the freshest window (e.g. the live portfolio script) can force a
re-download with ``refresh=True``.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import List, Optional

import pandas as pd

from src import config


def _download(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    """Download adjusted close prices for ``tickers`` via yfinance.

    Returns a DataFrame indexed by date with one column per ticker.
    """
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    if isinstance(raw.columns, pd.MultiIndex):
        closes = {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)}
        prices = pd.DataFrame(closes)
    else:
        # Single-ticker download collapses the MultiIndex.
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.sort_index()
    prices.index.name = "date"
    return prices


def fetch_prices(
    tickers: Optional[List[str]] = None,
    start: str = config.START_DATE,
    end: Optional[str] = config.END_DATE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return adjusted close prices for ``tickers``, using the on-disk cache.

    If a cache file already exists and ``refresh`` is False, it is loaded
    from disk. Otherwise a fresh download is performed via yfinance and the
    cache is (over)written.
    """
    tickers = tickers or config.TICKERS
    config.ensure_dirs()

    if not refresh and os.path.exists(config.RAW_DATA_PATH):
        prices = pd.read_csv(config.RAW_DATA_PATH, index_col="date", parse_dates=True)
        missing = [t for t in tickers if t not in prices.columns]
        if not missing:
            return prices[tickers]

    end = end or dt.date.today().isoformat()
    prices = _download(tickers, start, end)
    prices = prices.dropna(how="all")
    prices.to_csv(config.RAW_DATA_PATH)
    return prices[tickers]


def fetch_latest_window(tickers: Optional[List[str]] = None, lookback_days: int = 400) -> pd.DataFrame:
    """Refresh the cache and return the most recent ``lookback_days`` rows.

    Used by the live portfolio script, which always wants today's prices
    plus enough trailing history to compute rolling volatility/correlations.
    """
    tickers = tickers or config.TICKERS
    start = (dt.date.today() - dt.timedelta(days=int(lookback_days * 1.6))).isoformat()
    prices = fetch_prices(tickers, start=start, refresh=True)
    return prices.tail(lookback_days)
