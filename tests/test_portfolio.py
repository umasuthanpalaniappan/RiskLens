"""Unit tests for the minimum-variance portfolio and its regime overlay."""
from types import SimpleNamespace

import numpy as np
import pytest

from src import config, portfolio


def _sigma(vols, corr=None):
    vols = np.asarray(vols, dtype=float)
    corr = np.eye(len(vols)) if corr is None else corr
    return portfolio.predicted_covariance(vols, corr)


def _random_sigma(n, seed=0):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.15, 0.40, n)
    a = rng.normal(size=(n, n))
    cov = a @ a.T + n * np.eye(n)
    d = np.sqrt(np.diag(cov))
    return portfolio.predicted_covariance(vols, cov / np.outer(d, d))


def test_covariance_is_symmetric_and_positive_definite():
    sigma = _random_sigma(18)
    assert np.allclose(sigma, sigma.T)
    assert np.linalg.eigvalsh(sigma).min() > 0


def test_covariance_diagonal_is_vol_squared_plus_ridge():
    vols = np.array([0.2, 0.3, 0.25])
    sigma = _sigma(vols)
    assert np.allclose(np.diag(sigma), vols**2 + config.PORTFOLIO_RIDGE)


def test_weights_form_a_valid_long_only_capped_portfolio():
    w = portfolio.optimize(_random_sigma(18), regime=None)
    assert w.sum() == pytest.approx(1.0)
    assert w.min() >= -1e-9
    assert w.max() <= config.PORTFOLIO_MAX_WEIGHT + 1e-9


def test_cap_binds_when_one_stock_is_much_safer():
    w = portfolio.optimize(_sigma([0.05, 0.5, 0.5, 0.5, 0.5, 0.5]), regime=None)
    assert w[0] == pytest.approx(config.PORTFOLIO_MAX_WEIGHT, abs=1e-6)


def test_two_uncorrelated_stocks_match_the_closed_form_solution():
    # Textbook result: w_A = var_B / (var_A + var_B) = 0.09 / 0.13 = 0.692
    w = portfolio.optimize(_sigma([0.20, 0.30]), regime=None, max_weight=1.0)
    assert w[0] == pytest.approx(0.09 / 0.13, abs=1e-3)
    # ... and the mix is calmer than either stock on its own.
    sigma = _sigma([0.20, 0.30])
    assert np.sqrt(w @ sigma @ w) < 0.20


def test_identical_stocks_get_equal_weights():
    w = portfolio.optimize(_sigma([0.25] * 5), regime=None)
    assert np.allclose(w, 0.2, atol=1e-3)


def test_calmer_stock_gets_more_weight():
    w = portfolio.optimize(_sigma([0.10, 0.40]), regime=None, max_weight=1.0)
    assert w[0] > w[1]


@pytest.mark.parametrize("regime, blend", [("calm", 0.0), ("normal", 0.15), ("crisis", 0.40)])
def test_regime_overlay_blends_toward_equal_weight(regime, blend):
    sigma = _random_sigma(18, seed=1)
    base = portfolio.optimize(sigma, regime=None)
    got = portfolio.optimize(sigma, regime=regime)
    expected = (1 - blend) * base + blend / len(base)
    assert np.allclose(got, expected, atol=1e-9)
    assert got.sum() == pytest.approx(1.0)


def test_crisis_overlay_puts_a_floor_under_every_weight():
    n = 18
    w = portfolio.optimize(_random_sigma(n, seed=2), regime="crisis")
    assert w.min() >= 0.40 / n - 1e-9  # why dashboard weights never fall below ~2.2%


def test_unknown_regime_is_ignored():
    sigma = _random_sigma(6, seed=3)
    assert np.allclose(portfolio.optimize(sigma, regime="nonsense"), portfolio.optimize(sigma, regime=None))


def test_solver_failure_falls_back_to_equal_weight(monkeypatch):
    monkeypatch.setattr(portfolio, "minimize", lambda *a, **k: SimpleNamespace(success=False, x=None))
    w = portfolio.optimize(_random_sigma(4), regime=None)
    assert np.allclose(w, 0.25)
