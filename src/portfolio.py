"""Minimum-variance portfolio construction from predicted volatilities.

RiskLens's models predict *volatility*, not expected returns, so there is no
honest way to run a full Markowitz mean-variance optimisation here -- doing
so would require fabricating a return forecast the models were never
trained to produce. Instead this module builds a minimum-variance portfolio:
it takes the predicted covariance matrix Sigma (built from predicted vols
and the recent correlation structure) and finds the long-only weights that
minimise portfolio variance w'.Sigma.w, subject to full investment and a
per-name cap. See README.md for the fuller rationale.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from src import config

# How much weight to blend toward the equal-weight portfolio depending on
# regime, as a simple, transparent risk overlay on top of the min-variance
# solution -- *not* a second optimisation. In "crisis" regimes single-name
# concentration risk is penalised more heavily by leaning further toward
# equal weight.
REGIME_EQUAL_WEIGHT_BLEND = {
    "calm": 0.0,
    "normal": 0.15,
    "crisis": 0.40,
}


def predicted_covariance(
    predicted_vols: np.ndarray,
    corr_matrix: np.ndarray,
    ridge: float = config.PORTFOLIO_RIDGE,
) -> np.ndarray:
    """Build Sigma = D @ Corr @ D from per-stock predicted (annualised) vol and a correlation matrix.

    ``predicted_vols`` are absolute volatilities (e.g. trailing realised vol
    scaled by the model's predicted vol ratio), not the raw ratios
    themselves. A small ridge term is added to the diagonal for numerical
    stability before optimisation.
    """
    d = np.diag(predicted_vols)
    sigma = d @ corr_matrix @ d
    sigma = sigma + ridge * np.eye(len(predicted_vols))
    return sigma


def _min_variance_weights(sigma: np.ndarray, long_only: bool, max_weight: float) -> np.ndarray:
    n = sigma.shape[0]
    x0 = np.full(n, 1.0 / n)

    def objective(w):
        return w @ sigma @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight) for _ in range(n)] if long_only else [(-max_weight, max_weight) for _ in range(n)]

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        # Fall back to equal weight rather than surfacing an unusable solution.
        return x0
    weights = np.clip(result.x, 0.0 if long_only else -max_weight, max_weight)
    return weights / weights.sum()


def optimize(
    sigma: np.ndarray,
    regime: Optional[str] = None,
    long_only: bool = config.PORTFOLIO_LONG_ONLY,
    max_weight: float = config.PORTFOLIO_MAX_WEIGHT,
) -> np.ndarray:
    """Solve the long-only minimum-variance portfolio, with a regime risk overlay.

    Returns a weight vector that sums to 1. If ``regime`` is one of
    "calm"/"normal"/"crisis", the min-variance solution is blended with the
    equal-weight portfolio per ``REGIME_EQUAL_WEIGHT_BLEND`` before being
    renormalised.
    """
    n = sigma.shape[0]
    weights = _min_variance_weights(sigma, long_only, max_weight)

    blend = REGIME_EQUAL_WEIGHT_BLEND.get(regime, 0.0)
    if blend > 0:
        equal = np.full(n, 1.0 / n)
        weights = (1 - blend) * weights + blend * equal
        weights = weights / weights.sum()
    return weights
