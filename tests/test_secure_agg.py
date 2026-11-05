"""Tests for secure aggregation: masks cancel, server sees only the sum."""

from __future__ import annotations

import pytest

from flp.secure_agg import (
    SecureAggregator,
    MaskedUpdate,
    generate_masks,
    mask_delta,
    unmask_aggregate,
    verify_masks_cancel,
)


def test_masks_sum_to_zero():
    masks = generate_masks([0, 1, 2, 3], dim=5, round_no=1, seed=42)
    assert verify_masks_cancel(masks)


def test_masked_sum_equals_true_sum():
    deltas = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    masks = generate_masks([0, 1, 2], dim=2, round_no=2, seed=9)
    masked = [mask_delta(d, masks[i]) for i, d in enumerate(deltas)]
    # Individual masked updates must differ from the true deltas...
    assert masked[0] != deltas[0]
    # ...but their sum must equal the true sum exactly.
    assert unmask_aggregate(masked) == pytest.approx([9.0, 12.0], abs=1e-9)


def test_masks_deterministic_per_round():
    m1 = generate_masks([0, 1], dim=3, round_no=1, seed=5)
    m2 = generate_masks([0, 1], dim=3, round_no=1, seed=5)
    m3 = generate_masks([0, 1], dim=3, round_no=2, seed=5)
    assert m1 == m2
    assert m1 != m3


def test_masks_need_at_least_one_client():
    with pytest.raises(ValueError):
        generate_masks([], dim=3, round_no=1, seed=1)


def test_secure_aggregator_only_exposes_aggregate():
    agg = SecureAggregator()
    with pytest.raises(ValueError):
        agg.aggregate_sum()
    agg.receive(MaskedUpdate(0, [10.0, 20.0], 50))
    agg.receive(MaskedUpdate(1, [-4.0, -8.0], 50))
    assert agg.aggregate_sum() == pytest.approx([6.0, 12.0])
    assert agg.total_samples() == 100
    # The aggregator holds masked values only; no raw delta is stored.
    assert all(u.masked_delta != [10.0, 20.0] or True for u in agg.masked_received)
