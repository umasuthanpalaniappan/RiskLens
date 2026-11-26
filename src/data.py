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
import time
from typing import List, Optional, Tuple

import pandas as pd

from src import config

_DOWNLOAD_RETRIES = 3
_DOWNLOAD_RETRY_DELAY_SECONDS = 5

_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
_MARKET_CLOSE_IST = dt.time(16, 0)  # NSE closes 15:30 IST; the buffer lets the final bar settle


def _default_end_date(now: Optional[dt.datetime] = None) -> str:
    """Exclusive ``end`` date for yfinance, so the newest *finished* session is included.

    yfinance treats ``end`` as exclusive, so ``end=today`` can never contain
    today's bar: a run on the evening of a trading day used to publish data
    that was a full trading day behind. Include today only once NSE has
    closed (a manual run mid-session must not store a half-finished bar).
    ``now`` is injectable (any timezone-aware datetime) so this is unit-testable.
    """
    now = (now or dt.datetime.now(_IST)).astimezone(_IST)
    include_today = now.time() >= _MARKET_CLOSE_IST
    return (now.date() + dt.timedelta(days=1 if include_today else 0)).isoformat()


def _download(tickers: List[str], start: str, end: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close prices and volume for ``tickers`` via yfinance.

    Returns ``(prices, volumes)``, each indexed by date with one column per ticker.

    ``threads=False`` forces yfinance to fetch tickers sequentially rather
    than from a thread pool -- with threading on, concurrent tickers can
    write to yfinance's internal SQLite response cache at the same time and
    intermittently fail with "database is locked", which has nothing to do
    with any ticker actually being delisted.
    """
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
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
            f"yfinance returned no data for: {empty}. This can be a transient "
            f"fetch failure (retry), or a ticker that was actually delisted, "
            f"renamed, or suspended (e.g. after a merger/demerger) -- if it's "
            f"the latter, replace it in config.TICKERS (and config.SECTOR_MAP) "
            f"and retrain."
        )
    return prices, volumes


def _download_with_retries(tickers: List[str], start: str, end: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retry ``_download`` a few times before giving up.

    yfinance downloads occasionally fail transiently on CI runners (rate
    limiting, its internal SQLite cache momentarily locked, a flaky
    connection) with no real data-availability problem behind them.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            return _download(tickers, start, end)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: retry any transient failure
            last_error = exc
            if attempt < _DOWNLOAD_RETRIES:
                time.sleep(_DOWNLOAD_RETRY_DELAY_SECONDS * attempt)
    assert last_error is not None
    raise last_error


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

    end = end or _default_end_date()
    prices, volumes = _download_with_retries(tickers, start, end)
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
