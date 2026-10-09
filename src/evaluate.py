"""Shared evaluation metrics for the LSTM and GNN training loops.

Kept separate from train.py so both scripts compute metrics identically and
neither duplicates the logic.
"""
from __future__ import annotations

import numpy as np


def mse(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean squared error between predicted and actual vol ratios."""
    return float(np.mean((pred - actual) ** 2))


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error between predicted and actual vol ratios.

    Reported alongside MSE since MAE is in the same units as the target
    (vol ratio) and isn't dominated by the occasional large miss the way
    MSE is -- useful context given vol ratios are fat-tailed (crisis spikes).
    """
    return float(np.mean(np.abs(pred - actual)))


def direction_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of samples where predicted and actual agree on vol direction.

    A vol ratio > 1 means volatility is expected to rise over the horizon;
    < 1 means it's expected to fall. This measures whether the model gets
    that up/down call right, independent of the predicted magnitude.
    """
    pred_dir = np.sign(pred - 1.0)
    actual_dir = np.sign(actual - 1.0)
    agree = pred_dir == actual_dir
    return float(np.mean(agree))
