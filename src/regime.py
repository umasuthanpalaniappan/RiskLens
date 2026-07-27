"""Market volatility regime classification.

The regime ("calm" / "normal" / "crisis") is a simple, transparent
volatility-quantile bucket computed from trailing *equal-weight average*
realised volatility across the universe -- not a hidden-Markov or other
latent-state model. It is used by portfolio.py as a light risk-overlay on
top of the minimum-variance solution (see optimize()'s docstring for why).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, features


def market_volatility(returns: pd.DataFrame, window: int = config.VOL_WINDOW) -> pd.Series:
    """Equal-weight average trailing realised volatility across the universe."""
    per_stock_vol = returns.apply(lambda col: features.realized_vol(col, window))
    return per_stock_vol.mean(axis=1)


def current_regime(
    vol_series: pd.Series,
    lookback: int = config.REGIME_LOOKBACK,
    calm_quantile: float = config.REGIME_CALM_QUANTILE,
    crisis_quantile: float = config.REGIME_CRISIS_QUANTILE,
) -> str:
    """Classify the most recent value of ``vol_series`` against its trailing quantiles.

    ``vol_series`` should be a market-level realised-volatility series (e.g.
    from ``market_volatility``). Returns one of "calm", "normal", "crisis".
    """
    clean = vol_series.dropna()
    if clean.empty:
        return "normal"
    history = clean.iloc[-lookback:]
    calm_threshold = history.quantile(calm_quantile)
    crisis_threshold = history.quantile(crisis_quantile)
    latest = clean.iloc[-1]

    if latest >= crisis_threshold:
        return "crisis"
    if latest <= calm_threshold:
        return "calm"
    return "normal"
