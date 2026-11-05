"""Tests for differential privacy: clipping, noise, budget fail-closed."""

from __future__ import annotations

import random

import pytest

from flp.privacy import (
    PrivacyAccountant,
    PrivacyBudgetExhausted,
    add_gaussian_noise,
    privatize_delta,
)


def test_per_round_epsilon_positive():
    acc = PrivacyAccountant(noise_multiplier=1.0, delta=1e-5, epsilon_budget=8.0)
    eps = acc.per_round_epsilon()
    assert eps > 0
    # Larger noise multiplier -> smaller per-round epsilon.
    acc2 = PrivacyAccountant(noise_multiplier=2.0, delta=1e-5, epsilon_budget=8.0)
    assert acc2.per_round_epsilon() < eps


def test_budget_fail_closed():
    acc = PrivacyAccountant(noise_multiplier=1.0, delta=1e-5, epsilon_budget=1.0)
    with pytest.raises(PrivacyBudgetExhausted):
        # One round costs ~4.8 eps at multiplier 1.0, over the 1.0 budget.
        acc.charge_round()
    assert acc.rounds_charged == 0


def test_budget_charges_and_tracks():
    acc = PrivacyAccountant(noise_multiplier=1.0, delta=1e-5, epsilon_budget=100.0)
    spent = acc.charge_round()
    assert spent == pytest.approx(acc.per_round_epsilon())
    assert acc.can_afford_round()
    assert acc.remaining < 100.0


def test_privatize_clips():
    delta, norm = privatize_delta([3.0, 4.0], clip_norm=1.0)
    assert norm == pytest.approx(5.0)
    assert delta == pytest.approx([0.6, 0.8])


def test_noise_changes_vector_and_is_deterministic_with_seed():
    rng1 = random.Random(0)
    rng2 = random.Random(0)
    v = [0.0, 0.0, 0.0]
    assert add_gaussian_noise(v, 0.5, 1.0, rng1) == add_gaussian_noise(v, 0.5, 1.0, rng2)
    rng3 = random.Random(1)
    assert add_gaussian_noise(v, 0.5, 1.0, rng3) != v


def test_zero_noise_multiplier_is_identity():
    assert add_gaussian_noise([1.0, 2.0], 0.0, 1.0, random.Random(0)) == [1.0, 2.0]
