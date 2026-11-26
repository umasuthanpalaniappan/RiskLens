"""Unit tests for the calm / normal / crisis market-regime classifier."""
import numpy as np
import pandas as pd
import pytest

from src import features, regime


def _series(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))


def test_latest_value_far_above_history_is_crisis():
    assert regime.current_regime(_series(list(np.linspace(0.10, 0.30, 200)) + [0.9])) == "crisis"


def test_latest_value_far_below_history_is_calm():
    assert regime.current_regime(_series(list(np.linspace(0.10, 0.30, 200)) + [0.01])) == "calm"


def test_latest_value_in_the_middle_is_normal():
    history = list(np.linspace(0.10, 0.30, 200))
    assert regime.current_regime(_series(history + [0.20])) == "normal"


@pytest.mark.parametrize("values", [[], [np.nan, np.nan]])
def test_no_usable_data_defaults_to_normal(values):
    assert regime.current_regime(_series(values)) == "normal"


def test_only_the_trailing_lookback_window_counts():
    # Old history was extremely volatile; the recent window is quiet and the
    # latest value is its highest, so today is a crisis *relative to recent history*.
    old = [5.0] * 100
    recent = list(np.linspace(0.10, 0.20, 10))
    assert regime.current_regime(_series(old + recent), lookback=10) == "crisis"


def test_market_volatility_is_the_cross_sectional_mean_of_realised_vol():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(40, 3)), columns=list("ABC"))
    got = regime.market_volatility(returns, window=5)
    per_stock = pd.concat([features.realized_vol(returns[c], 5) for c in returns], axis=1)
    assert np.allclose(got.dropna(), per_stock.mean(axis=1).dropna())
