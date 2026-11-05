"""End-to-end orchestrator tests: loss falls, DP halts, stragglers drop."""

from __future__ import annotations

from flp.config import FLConfig
from flp.orchestrator import run_training


def test_training_reduces_loss_no_dp(small_config):
    result = run_training(small_config)
    assert len(result.rounds) == small_config.n_rounds
    assert result.final_loss < result.initial_loss
    assert result.final_accuracy > 0.85
    assert result.halted_reason == ""
    assert result.registry.promoted_version >= 1


def test_training_with_dp_completes_within_budget(dp_config):
    result = run_training(dp_config)
    assert len(result.rounds) == dp_config.n_rounds
    assert result.final_accuracy > 0.7  # noise hurts, but the task is learnable
    assert result.halted_reason == ""
    total_eps = result.rounds[-1].dp_epsilon_spent
    assert 0 < total_eps <= dp_config.dp_epsilon_budget


def test_dp_budget_exhaustion_halts_training():
    config = FLConfig(
        n_clients=4, n_rounds=10, participation_fraction=1.0,
        dp_enabled=True, dp_noise_multiplier=1.0, dp_epsilon_budget=1.0, seed=7,
    )
    result = run_training(config)
    assert "DP budget exhausted" in result.halted_reason
    assert len(result.rounds) == 0


def test_straggler_is_dropped_when_sampled():
    config = FLConfig(
        n_clients=6, n_rounds=8, participation_fraction=1.0,
        dp_enabled=False, straggler_client_ids=(5,), seed=42,
    )
    result = run_training(config)
    assert len(result.rounds) == 8
    # Client 5 straggles every round it is sampled, so it never completes.
    for r in result.rounds:
        assert 5 not in r.participants
        assert 5 in r.dropped
    # Training still converges despite the permanent straggler.
    assert result.final_loss < result.initial_loss
    assert result.final_accuracy > 0.85


def test_trimmed_mean_strategy_runs():
    config = FLConfig(
        n_clients=6, n_rounds=3, participation_fraction=1.0,
        strategy="trimmed_mean", dp_enabled=False, seed=7,
    )
    result = run_training(config)
    assert len(result.rounds) == 3
    assert result.final_loss < result.initial_loss


def test_fedprox_strategy_runs():
    config = FLConfig(
        n_clients=4, n_rounds=3, participation_fraction=1.0,
        strategy="fedprox", fedprox_mu=0.5, dp_enabled=False, seed=7,
    )
    result = run_training(config)
    assert len(result.rounds) == 3
    assert result.final_loss < result.initial_loss


def test_comm_accounting_matches_rounds(small_config):
    result = run_training(small_config)
    assert result.comm.total_up == sum(r.bytes_up for r in result.rounds)
    assert result.comm.total_down == sum(r.bytes_down for r in result.rounds)
    assert result.comm.total > 0


def test_fairness_report_present(small_config):
    result = run_training(small_config)
    assert result.fairness is not None
    assert len(result.fairness.per_client_accuracy) == small_config.n_clients
    assert result.fairness.spread >= 0


def test_result_serializes_to_dict(small_config):
    result = run_training(small_config)
    d = result.to_dict()
    assert d["final_accuracy"] == result.final_accuracy
    assert len(d["rounds"]) == small_config.n_rounds
    assert d["registry"]["promoted_version"] >= 1


def test_deterministic_same_seed(small_config):
    r1 = run_training(small_config)
    r2 = run_training(small_config)
    assert r1.final_loss == r2.final_loss
    assert r1.final_accuracy == r2.final_accuracy
