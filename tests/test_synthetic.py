"""Tests for synthetic data generation and non-IID partitioning."""

from __future__ import annotations

from flp.synthetic import generate_global_data, partition_clients


def test_global_data_balanced():
    X, y = generate_global_data(100, 4, seed=1)
    assert len(X) == len(y) == 100
    assert sum(y) == 50
    assert all(len(x) == 4 for x in X)


def test_global_data_deterministic():
    X1, y1 = generate_global_data(50, 3, seed=5)
    X2, y2 = generate_global_data(50, 3, seed=5)
    assert X1 == X2 and y1 == y2


def test_partition_label_skew():
    clients = partition_clients(4, 100, 4, label_skew=0.9, quantity_skew=0.0,
                                feature_skew=0.0, seed=3)
    # With extreme skew, each client's dominant label should dominate.
    for c in clients:
        dominant = c.client_id % 2
        frac = sum(1 for v in c.y if v == dominant) / len(c.y)
        assert frac > 0.8


def test_partition_iid_label_balance():
    clients = partition_clients(4, 100, 4, label_skew=0.0, quantity_skew=0.0,
                                feature_skew=0.0, seed=3)
    for c in clients:
        frac = sum(c.y) / len(c.y)
        assert 0.3 < frac < 0.7


def test_partition_quantity_skew():
    clients = partition_clients(4, 100, 4, label_skew=0.0, quantity_skew=0.9,
                                feature_skew=0.0, seed=3)
    sizes = [len(c) for c in clients]
    assert max(sizes) > min(sizes) * 1.5


def test_partition_feature_shift_differs():
    clients = partition_clients(4, 100, 4, label_skew=0.0, quantity_skew=0.0,
                                feature_skew=0.8, seed=3)
    shifts = [tuple(c.feature_shift) for c in clients]
    assert len(set(shifts)) == 4
