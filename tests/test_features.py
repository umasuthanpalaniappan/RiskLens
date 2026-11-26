"""Unit tests for feature and target engineering (volatility ratio target, indicators)."""
import numpy as np
import pandas as pd
import pytest

from src import config, features

ANN = np.sqrt(features.TRADING_DAYS_PER_YEAR)


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.Series(p, index=pd.bdate_range("2024-01-01", periods=n))


def _volume(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(1_000, 5_000, n).astype(float), index=pd.bdate_range("2024-01-01", periods=n))


def test_log_returns_match_hand_calculation():
    r = features.log_returns(pd.DataFrame({"A": [100.0, 110.0, 99.0]}))["A"]
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(1.1))
    assert r.iloc[2] == pytest.approx(np.log(0.9))


def test_realized_vol_is_annualised_sample_std_of_the_window():
    r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
    assert features.realized_vol(r, window=5).iloc[-1] == pytest.approx(r.std(ddof=1) * ANN)
    assert features.realized_vol(r, window=5).iloc[:4].isna().all()


def test_forward_vol_at_t_covers_the_next_horizon_days_only():
    r = features.log_returns(_prices().to_frame()).iloc[:, 0]
    h = config.HORIZON
    fwd = features.forward_realized_vol(r, h)
    for t in (10, 100, 200):
        expected = r.iloc[t + 1 : t + 1 + h].std(ddof=1) * ANN
        assert fwd.iloc[t] == pytest.approx(expected)
    assert fwd.iloc[-h:].isna().all()  # the future is unknown at the end of the data


def test_target_is_future_vol_divided_by_current_vol():
    p, v = _prices(), _volume()
    frame = features.build_stock_features(p, v)
    r = features.log_returns(p.to_frame()).iloc[:, 0]
    t = 120
    expected = features.forward_realized_vol(r).iloc[t] / features.realized_vol(r).iloc[t]
    assert frame["target_vol_ratio"].iloc[t] == pytest.approx(expected)


def test_features_never_look_ahead():
    """Changing or removing the future must not change any feature computed for the past."""
    p, v = _prices(), _volume()
    full = features.build_stock_features(p, v)
    cut = 200
    truncated = features.build_stock_features(p.iloc[:cut], v.iloc[:cut])
    cols = config.FEATURE_COLUMNS
    pd.testing.assert_frame_equal(full[cols].iloc[:cut], truncated[cols], check_exact=False, rtol=1e-9, atol=1e-12)


def test_momentum_is_the_trailing_sum_of_log_returns():
    r = features.log_returns(_prices().to_frame()).iloc[:, 0]
    w = config.MOMENTUM_WINDOW
    assert features.momentum(r).iloc[100] == pytest.approx(r.iloc[100 - w + 1 : 101].sum())


def test_rsi_is_rescaled_to_minus_one_one():
    rsi = features.rsi(_prices()).dropna()
    assert len(rsi) > 0
    assert rsi.between(-1, 1).all()


def test_volume_zscore_has_no_infinities_even_for_flat_volume():
    flat = pd.Series([1_000.0] * 60)
    z = features.volume_zscore(flat)
    assert not np.isinf(z).any()


def test_macd_hist_is_price_normalised():
    p = _prices()
    scaled = features.macd_hist(p * 1000)
    assert np.allclose(scaled.dropna(), features.macd_hist(p).dropna())


def test_panel_has_every_feature_and_the_target():
    frame = features.build_stock_features(_prices(), _volume())
    assert set(config.FEATURE_COLUMNS) | {"target_vol_ratio"} <= set(frame.columns)


def test_align_panel_drops_unlabelled_tail_only_when_a_target_is_required():
    prices = pd.DataFrame({"A": _prices(seed=0), "B": _prices(seed=5)})
    volumes = pd.DataFrame({"A": _volume(seed=1), "B": _volume(seed=2)})
    panel = features.build_panel(prices, volumes)
    with_target = features.align_panel(panel, require_target=True)
    live = features.align_panel(panel, require_target=False)
    assert with_target["A"].index.equals(with_target["B"].index)
    assert live["A"].index[-1] == prices.index[-1]  # live inference keeps the latest day
    assert with_target["A"].index[-1] < live["A"].index[-1]  # training drops days with no label
    assert with_target["A"]["target_vol_ratio"].notna().all()
