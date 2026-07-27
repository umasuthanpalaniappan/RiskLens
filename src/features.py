"""Feature and target engineering shared by the LSTM and GNN pipelines.

The prediction target is a *volatility ratio*: how much realised volatility
over the next ``HORIZON`` trading days is expected to change relative to the
trailing ``VOL_WINDOW``-day realised volatility. A ratio > 1 means "vol is
about to rise", < 1 means "vol is about to fall". This is what both models
predict, and what portfolio.py consumes to build Sigma.
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


def build_stock_features(price_series: pd.Series) -> pd.DataFrame:
    """Build the (log_return, realized_vol, target) frame for a single ticker."""
    ret = log_returns(price_series.to_frame()).iloc[:, 0]
    vol = realized_vol(ret)
    fwd_vol = forward_realized_vol(ret)
    target = fwd_vol / vol.replace(0, np.nan)

    frame = pd.DataFrame(
        {
            "log_return": ret,
            "realized_vol": vol,
            "target_vol_ratio": target,
        }
    )
    return frame


def build_panel(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build per-ticker feature frames for every column of ``prices``."""
    return {ticker: build_stock_features(prices[ticker]) for ticker in prices.columns}


def align_panel(panel: Dict[str, pd.DataFrame], require_target: bool = True) -> Dict[str, pd.DataFrame]:
    """Restrict every ticker's frame to dates where all tickers have data.

    ``require_target`` drops trailing rows without a target label; set to
    False for live inference where the target is unknown/in the future.
    """
    cols_needed = ["log_return", "realized_vol"] + (["target_vol_ratio"] if require_target else [])
    valid_index = None
    for frame in panel.values():
        mask = frame[cols_needed].notna().all(axis=1)
        idx = frame.index[mask]
        valid_index = idx if valid_index is None else valid_index.intersection(idx)
    return {ticker: frame.loc[valid_index] for ticker, frame in panel.items()}
