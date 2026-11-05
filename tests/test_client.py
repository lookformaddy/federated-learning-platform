"""Tests for the simulated client: local training, FedProx, deadlines."""

from __future__ import annotations

import random

from flp.client import FederatedClient, straggler_aware_weights
from flp.model import LogisticModel
from flp.orchestrator import build_clients


def test_local_training_improves_local_loss(clients):
    model = LogisticModel(4)
    client = FederatedClient(client_id=clients[0].client_id, data=clients[0])
    before = model.loss(clients[0].X, clients[0].y)
    update = client.train(model, local_epochs=3, lr=0.5, rng=random.Random(1))
    assert update.completed
    after_model = model.copy()
    after_model.apply_delta(update.delta)
    assert after_model.loss(clients[0].X, clients[0].y) < before


def test_straggler_exceeds_deadline(clients):
    model = LogisticModel(4)
    client = FederatedClient(client_id=clients[0].client_id, data=clients[0])
    update = client.train(model, local_epochs=1, lr=0.5, deadline_ms=500,
                          straggler=True, rng=random.Random(1))
    assert not update.completed
    assert "deadline exceeded" in update.dropped_reason
    assert update.latency_ms > 500


def test_non_straggler_meets_deadline(clients):
    model = LogisticModel(4)
    client = FederatedClient(client_id=clients[0].client_id, data=clients[0])
    update = client.train(model, local_epochs=1, lr=0.5, deadline_ms=500,
                          straggler=False, rng=random.Random(1))
    assert update.completed


def test_clipping_bounds_delta_norm(clients):
    model = LogisticModel(4)
    client = FederatedClient(client_id=clients[0].client_id, data=clients[0])
    update = client.train(model, local_epochs=5, lr=2.0, clip_norm=0.5,
                          rng=random.Random(1))
    from flp.model import vector_norm

    assert vector_norm(update.delta) <= 0.5 + 1e-9


def test_fedprox_delta_smaller_than_fedavg(clients):
    # A proximal term pulls local updates back toward the global model.
    # mu must satisfy lr*mu < 1 for a stable contraction (here 0.5*0.5).
    model = LogisticModel(4)
    c1 = FederatedClient(client_id=clients[0].client_id, data=clients[0])
    c2 = FederatedClient(client_id=clients[0].client_id, data=clients[0])
    u_plain = c1.train(model, local_epochs=3, lr=0.5, rng=random.Random(1))
    u_prox = c2.train(model, local_epochs=3, lr=0.5, fedprox_mu=0.5, rng=random.Random(1))
    from flp.model import vector_norm

    assert vector_norm(u_prox.delta) < vector_norm(u_plain.delta)


def test_straggler_aware_weights_penalize_slow_clients():
    from flp.config import FLConfig

    clients = build_clients(FLConfig(n_clients=4, seed=1))
    slow = clients[0]
    slow.latency_history = [900, 950, 1000]
    fast = clients[1]
    fast.latency_history = [100, 110, 105]
    weights = straggler_aware_weights(clients)
    assert abs(sum(weights) - 1.0) < 1e-9
    assert weights[1] > weights[0]
