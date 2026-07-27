"""Price/volume data acquisition and local caching.

Downloads daily adjusted-close prices *and* traded volume for the RiskLens
universe via yfinance in a single request, and caches each to its own CSV
under ``data/raw/`` so that repeated runs (and offline dashboard reloads)
don't re-hit the network. Callers that only need the freshest window (e.g.
the live portfolio script) can force a re-download with ``refresh=True``.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import List, Optional, Tuple

import pandas as pd

from src import config


def _download(tickers: List[str], start: str, end: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close prices and volume for ``tickers`` via yfinance.

    Returns ``(prices, volumes)``, each indexed by date with one column per ticker.
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
        present = [t for t in tickers if t in raw.columns.get_level_values(0)]
        prices = pd.DataFrame({t: raw[t]["Close"] for t in present})
        volumes = pd.DataFrame({t: raw[t]["Volume"] for t in present})
    else:
        # Single-ticker download collapses the MultiIndex.
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volumes = raw[["Volume"]].rename(columns={"Volume": tickers[0]})

    prices = prices.sort_index()
    volumes = volumes.sort_index()
    prices.index.name = "date"
    volumes.index.name = "date"

    empty = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if empty:
        raise ValueError(
            f"yfinance returned no data for: {empty}. A ticker may have been delisted, "
            f"renamed, or suspended (e.g. after a merger/demerger) -- replace it in "
            f"config.TICKERS (and config.SECTOR_MAP) and retrain."
        )
    return prices, volumes


def fetch_price_volume(
    tickers: Optional[List[str]] = None,
    start: str = config.START_DATE,
    end: Optional[str] = config.END_DATE,
    refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(prices, volumes)`` for ``tickers``, using the on-disk cache when possible."""
    tickers = tickers or config.TICKERS
    config.ensure_dirs()

    if not refresh and os.path.exists(config.RAW_DATA_PATH) and os.path.exists(config.RAW_VOLUME_PATH):
        prices = pd.read_csv(config.RAW_DATA_PATH, index_col="date", parse_dates=True)
        volumes = pd.read_csv(config.RAW_VOLUME_PATH, index_col="date", parse_dates=True)
        if all(t in prices.columns for t in tickers) and all(t in volumes.columns for t in tickers):
            return prices[tickers], volumes[tickers]

    end = end or dt.date.today().isoformat()
    prices, volumes = _download(tickers, start, end)
    prices = prices.dropna(how="all")
    volumes = volumes.reindex(prices.index)
    prices.to_csv(config.RAW_DATA_PATH)
    volumes.to_csv(config.RAW_VOLUME_PATH)
    return prices[tickers], volumes[tickers]


def fetch_latest_window(
    tickers: Optional[List[str]] = None, lookback_days: int = 400
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Refresh the cache and return the most recent ``lookback_days`` rows of ``(prices, volumes)``.

    Used by the live portfolio script, which always wants today's prices
    plus enough trailing history to compute rolling volatility/correlations.

    The refresh always re-downloads the *full* ``config.START_DATE``-to-today
    history (not just ``lookback_days``) so that the on-disk cache -- which
    the dashboard's Market Overview tab also reads -- keeps its full
    multi-year span rather than being truncated down to a short live window
    every time this runs.
    """
    tickers = tickers or config.TICKERS
    prices, volumes = fetch_price_volume(tickers, start=config.START_DATE, refresh=True)
    return prices.tail(lookback_days), volumes.tail(lookback_days)
