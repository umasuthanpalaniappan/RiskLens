"""Unit tests for the shared evaluation metrics."""
import numpy as np
import pytest

from src import evaluate


def test_mse_and_mae():
    pred, actual = np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 2.0])
    assert evaluate.mse(pred, actual) == pytest.approx((0 + 1 + 4) / 3)
    assert evaluate.mae(pred, actual) == pytest.approx((0 + 1 + 2) / 3)


def test_direction_accuracy_counts_agreement_around_one():
    pred = np.array([1.2, 0.8, 1.5, 0.5])
    actual = np.array([1.1, 0.9, 0.7, 0.4])
    assert evaluate.direction_accuracy(pred, actual) == pytest.approx(0.75)


def test_direction_accuracy_extremes():
    r = np.array([0.5, 1.5, 0.9])
    assert evaluate.direction_accuracy(r, r) == 1.0
    assert evaluate.direction_accuracy(r, 2 - r) == 0.0  # mirrored around 1 flips every call


def test_a_constant_call_of_rising_scores_the_share_of_rising_days():
    actual = np.array([1.3, 0.8, 1.1, 0.9, 1.2])  # 3 of 5 rise
    assert evaluate.direction_accuracy(np.full(5, 1.5), actual) == pytest.approx(0.6)
