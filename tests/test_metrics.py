"""Tests for evaluation metrics and communication accounting."""

from __future__ import annotations

import pytest

from flp.metrics import CommAccounting, evaluate_global, fairness_report
from flp.model import LogisticModel
from flp.orchestrator import build_clients
from flp.config import FLConfig


def test_evaluate_global(test_data):
    X, y = test_data
    model = LogisticModel(4)
    result = evaluate_global(model, X, y)
    assert result["accuracy"] == pytest.approx(0.5, abs=0.1)
    assert result["loss"] > 0


def test_fairness_report_shape():
    config = FLConfig(n_clients=4, seed=7)
    clients = build_clients(config)
    model = LogisticModel(4)
    report = fairness_report(model, clients)
    assert len(report.per_client_accuracy) == 4
    assert report.min_accuracy <= report.mean_accuracy <= report.max_accuracy
    assert report.spread == pytest.approx(report.max_accuracy - report.min_accuracy)
    assert report.std_accuracy >= 0


def test_comm_accounting_totals():
    comm = CommAccounting()
    comm.record_round(bytes_up=100, bytes_down=200)
    comm.record_round(bytes_up=100, bytes_down=200)
    assert comm.total_up == 200
    assert comm.total_down == 400
    assert comm.total == 600
    assert comm.to_dict()["total"] == 600
