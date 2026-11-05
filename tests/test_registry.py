"""Tests for the model registry: promotion gates and rollback."""

from __future__ import annotations

import pytest

from flp.model import LogisticModel
from flp.registry import ModelRegistry


def _model() -> LogisticModel:
    return LogisticModel(4, weights=[0.1, 0.2, 0.3, 0.4], bias=0.0)


def test_first_version_promotes():
    reg = ModelRegistry()
    v = reg.register(_model(), accuracy=0.8, loss=0.5, round_no=1,
                     dp_epsilon_spent=1.0, dp_budget_intact=True)
    assert v.promoted
    assert reg.promoted_version == 1
    assert reg.best_accuracy == 0.8


def test_worse_version_does_not_promote():
    reg = ModelRegistry()
    reg.register(_model(), accuracy=0.8, loss=0.5, round_no=1,
                 dp_epsilon_spent=1.0, dp_budget_intact=True)
    v2 = reg.register(_model(), accuracy=0.7, loss=0.6, round_no=2,
                      dp_epsilon_spent=2.0, dp_budget_intact=True)
    assert not v2.promoted
    assert reg.promoted_version == 1


def test_dp_budget_failure_blocks_promotion():
    reg = ModelRegistry()
    reg.register(_model(), accuracy=0.8, loss=0.5, round_no=1,
                 dp_epsilon_spent=1.0, dp_budget_intact=True)
    v2 = reg.register(_model(), accuracy=0.9, loss=0.4, round_no=2,
                      dp_epsilon_spent=9.0, dp_budget_intact=False)
    assert not v2.promoted
    assert reg.promoted_version == 1


def test_rollback_returns_last_promoted():
    reg = ModelRegistry()
    reg.register(_model(), accuracy=0.8, loss=0.5, round_no=1,
                 dp_epsilon_spent=1.0, dp_budget_intact=True)
    reg.register(_model(), accuracy=0.7, loss=0.6, round_no=2,
                 dp_epsilon_spent=2.0, dp_budget_intact=True)
    rolled = reg.rollback()
    assert rolled is not None
    assert rolled.version == 1
    assert rolled.model().weights == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_rollback_empty_registry():
    assert ModelRegistry().rollback() is None


def test_registry_serialization():
    reg = ModelRegistry()
    reg.register(_model(), accuracy=0.8, loss=0.5, round_no=1,
                 dp_epsilon_spent=1.0, dp_budget_intact=True)
    d = reg.to_dict()
    assert d["promoted_version"] == 1
    assert len(d["versions"]) == 1
