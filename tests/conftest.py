"""Shared fixtures for the federated learning platform tests."""

from __future__ import annotations

import pytest

from flp.config import FLConfig
from flp.model import LogisticModel
from flp.synthetic import generate_global_data, partition_clients


@pytest.fixture
def small_config() -> FLConfig:
    return FLConfig(
        n_clients=4,
        n_rounds=3,
        participation_fraction=1.0,
        dp_enabled=False,
        seed=7,
    )


@pytest.fixture
def dp_config() -> FLConfig:
    return FLConfig(
        n_clients=4,
        n_rounds=3,
        participation_fraction=1.0,
        dp_enabled=True,
        dp_noise_multiplier=0.5,
        dp_epsilon_budget=100.0,
        seed=7,
    )


@pytest.fixture
def test_data():
    return generate_global_data(200, 4, seed=99)


@pytest.fixture
def clients():
    return partition_clients(
        n_clients=4,
        samples_per_client=60,
        n_features=4,
        label_skew=0.5,
        quantity_skew=0.3,
        feature_skew=0.2,
        seed=7,
    )


@pytest.fixture
def zero_model() -> LogisticModel:
    return LogisticModel(4)
