"""Tests for aggregation strategies."""

from __future__ import annotations

import pytest

from flp.aggregation import aggregate, fedavg, mean_update_norm, trimmed_mean


def test_fedavg_weighted():
    deltas = [[2.0, 0.0], [0.0, 2.0]]
    assert fedavg(deltas, [3.0, 1.0]) == pytest.approx([1.5, 0.5])


def test_fedavg_equal_weights_is_mean():
    deltas = [[1.0, 2.0], [3.0, 4.0]]
    assert fedavg(deltas, [1.0, 1.0]) == pytest.approx([2.0, 3.0])


def test_fedavg_errors():
    with pytest.raises(ValueError):
        fedavg([], [])
    with pytest.raises(ValueError):
        fedavg([[1.0]], [0.0])
    with pytest.raises(ValueError):
        fedavg([[1.0], [1.0, 2.0]], [1.0, 1.0])


def test_trimmed_mean_removes_outlier():
    # One client sends a huge outlier; trimmed mean should ignore it.
    deltas = [[1.0], [1.1], [0.9], [1.05], [100.0]]
    assert trimmed_mean(deltas, 0.2) == pytest.approx([1.0125], abs=0.05)
    assert trimmed_mean(deltas, 0.0) == pytest.approx([20.81], abs=0.01)


def test_trimmed_mean_invalid_fraction():
    with pytest.raises(ValueError):
        trimmed_mean([[1.0]], 0.5)


def test_aggregate_dispatch():
    assert aggregate("fedprox", [[1.0], [3.0]], [1.0, 1.0]) == pytest.approx([2.0])
    assert aggregate("trimmed_mean", [[1.0], [3.0]], [1.0, 1.0], 0.0) == pytest.approx([2.0])
    with pytest.raises(ValueError):
        aggregate("bogus", [[1.0]], [1.0])


def test_mean_update_norm():
    assert mean_update_norm([[3.0, 4.0]]) == pytest.approx(5.0)
    assert mean_update_norm([]) == 0.0
