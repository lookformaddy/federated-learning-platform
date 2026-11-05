"""Tests for the from-scratch logistic model."""

from __future__ import annotations

import math

import pytest

from flp.model import LogisticModel, clip_vector, vector_norm


def test_sigmoid_extremes():
    from flp.model import sigmoid

    assert sigmoid(100.0) == pytest.approx(1.0)
    assert sigmoid(-100.0) == pytest.approx(0.0)
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_zero_model_loss_is_log2(test_data):
    X, y = test_data
    model = LogisticModel(4)
    assert model.loss(X, y) == pytest.approx(math.log(2), rel=1e-6)


def test_training_reduces_loss(test_data):
    X, y = test_data
    model = LogisticModel(4)
    before = model.loss(X, y)
    for _ in range(30):
        gw, gb = model.gradient(X, y)
        model.step(gw, gb, lr=0.5)
    assert model.loss(X, y) < before
    assert model.accuracy(X, y) > 0.9


def test_gradient_empty_data(zero_model):
    gw, gb = zero_model.gradient([], [])
    assert gw == [0.0] * 4 and gb == 0.0


def test_delta_roundtrip(zero_model):
    other = LogisticModel(4, weights=[0.1, -0.2, 0.3, 0.0], bias=0.5)
    delta = other.delta_from(zero_model)
    restored = zero_model.copy()
    restored.apply_delta(delta)
    assert restored.weights == pytest.approx(other.weights)
    assert restored.bias == pytest.approx(0.5)


def test_apply_delta_wrong_length(zero_model):
    with pytest.raises(ValueError):
        zero_model.apply_delta([1.0, 2.0])


def test_serialization_roundtrip():
    model = LogisticModel(3, weights=[0.5, -0.5, 0.25], bias=-0.1)
    assert LogisticModel.from_dict(model.to_dict()).weights == pytest.approx(model.weights)
    assert model.nbytes() == 8 * 4


def test_clip_vector():
    v = [3.0, 4.0]  # norm 5
    clipped, norm = clip_vector(v, 2.5)
    assert norm == pytest.approx(5.0)
    assert vector_norm(clipped) == pytest.approx(2.5)
    small, _ = clip_vector([0.3, 0.4], 2.5)
    assert small == [0.3, 0.4]
