"""Feature and target engineering shared by the LSTM and GNN pipelines.

The prediction target is a *volatility ratio*: how much realised volatility
over the next ``HORIZON`` trading days is expected to change relative to the
trailing ``VOL_WINDOW``-day realised volatility. A ratio > 1 means "vol is
about to rise", < 1 means "vol is about to fall". This is what both models
predict, and what portfolio.py consumes to build Sigma.

Beyond raw returns and realised vol, two extra signals are fed to the
models: trailing ``MOMENTUM_WINDOW``-day cumulative return (has the stock
been trending?) and a rolling z-score of traded volume (is today unusually
busy relative to its own recent history?) -- both cheap, well-known
volatility-regime signals.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src import config

TRADING_DAYS_PER_YEAR = 252


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns for each column (ticker) in ``prices``."""
    return np.log(prices / prices.shift(1))


def realized_vol(returns: pd.Series, window: int = config.VOL_WINDOW) -> pd.Series:
    """Trailing annualised realised volatility, one value per day."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def forward_realized_vol(returns: pd.Series, horizon: int = config.HORIZON) -> pd.Series:
    """Realised volatility over the ``horizon`` days *following* each date."""
    reversed_roll = returns[::-1].rolling(horizon).std()[::-1] * np.sqrt(TRADING_DAYS_PER_YEAR)
    # rolling(horizon) on the reversed series at position t covers
    # [t, t-horizon+1] in reversed order == [t, t+horizon-1] in original order.
    # Shift by -1 so the label at date t excludes today and covers t+1..t+horizon.
    return reversed_roll.shift(-1)


def momentum(returns: pd.Series, window: int = config.MOMENTUM_WINDOW) -> pd.Series:
    """Trailing cumulative log return over ``window`` days -- a simple trend/momentum signal."""
    return returns.rolling(window).sum()


def volume_zscore(volume: pd.Series, window: int = config.VOLUME_ZSCORE_WINDOW) -> pd.Series:
    """Rolling z-score of traded volume: how unusual is today's activity?"""
    rolling_mean = volume.rolling(window).mean()
    rolling_std = volume.rolling(window).std()
    return (volume - rolling_mean) / rolling_std.replace(0, np.nan)


def build_stock_features(price_series: pd.Series, volume_series: pd.Series) -> pd.DataFrame:
    """Build the per-ticker feature+target frame from that ticker's price and volume series."""
    ret = log_returns(price_series.to_frame()).iloc[:, 0]
    vol = realized_vol(ret)
    fwd_vol = forward_realized_vol(ret)
    target = fwd_vol / vol.replace(0, np.nan)

    frame = pd.DataFrame(
        {
            "log_return": ret,
            "realized_vol": vol,
            "momentum": momentum(ret),
            "volume_zscore": volume_zscore(volume_series),
            "target_vol_ratio": target,
        }
    )
    return frame


def build_panel(prices: pd.DataFrame, volumes: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build per-ticker feature frames for every column of ``prices``/``volumes``."""
    return {
        ticker: build_stock_features(prices[ticker], volumes[ticker])
        for ticker in prices.columns
    }


def align_panel(panel: Dict[str, pd.DataFrame], require_target: bool = True) -> Dict[str, pd.DataFrame]:
    """Restrict every ticker's frame to dates where all tickers have data.

    ``require_target`` drops trailing rows without a target label; set to
    False for live inference where the target is unknown/in the future.
    """
    cols_needed = config.FEATURE_COLUMNS + (["target_vol_ratio"] if require_target else [])
    valid_index = None
    for frame in panel.values():
        mask = frame[cols_needed].notna().all(axis=1)
        idx = frame.index[mask]
        valid_index = idx if valid_index is None else valid_index.intersection(idx)
    return {ticker: frame.loc[valid_index] for ticker, frame in panel.items()}
