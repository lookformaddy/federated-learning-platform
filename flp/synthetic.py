"""Deterministic synthetic data with controllable non-IID partitions.

The underlying task is a generic binary classification (e.g. "flagged vs
not flagged") over a small feature vector. Data is generated from seeded
Gaussian class-conditional distributions, so every run is reproducible.

Non-IID axes, each controlled by a skew parameter in [0, 1]:
- label skew: each client gets a dominant label; higher skew means the
  client's local label mix is further from the global 50/50 balance.
- quantity skew: clients hold unequal numbers of samples.
- feature skew: each client gets a fixed per-feature shift, simulating
  different sensors / populations / collection conditions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class ClientData:
    client_id: int
    X: list[list[float]]
    y: list[int]
    feature_shift: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)


def _gauss(rng: random.Random, mean: float, std: float = 1.0) -> float:
    return rng.gauss(mean, std)


def generate_global_data(
    n_samples: int, n_features: int, seed: int
) -> tuple[list[list[float]], list[int]]:
    """Balanced, IID, unshifted data — used for the held-out test set."""
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[int] = []
    for i in range(n_samples):
        label = i % 2  # exact balance, deterministic order shuffled below
        mean = 1.0 if label == 1 else -1.0
        X.append([_gauss(rng, mean) for _ in range(n_features)])
        y.append(label)
    order = list(range(n_samples))
    rng.shuffle(order)
    return [X[i] for i in order], [y[i] for i in order]


def partition_clients(
    n_clients: int,
    samples_per_client: int,
    n_features: int,
    label_skew: float,
    quantity_skew: float,
    feature_skew: float,
    seed: int,
) -> list[ClientData]:
    rng = random.Random(seed + 1000)
    clients: list[ClientData] = []
    # Quantity skew: deterministic weights in [1 - q, 1 + q], normalized.
    raw_weights = [
        1.0 + quantity_skew * rng.uniform(-1.0, 1.0) for _ in range(n_clients)
    ]
    total_w = sum(raw_weights)
    for cid in range(n_clients):
        n = max(10, int(round(samples_per_client * n_clients * raw_weights[cid] / total_w)))
        # Label skew: dominant label for this client.
        dominant = cid % 2
        p_dominant = 0.5 + 0.5 * label_skew
        shift = [rng.uniform(-feature_skew, feature_skew) for _ in range(n_features)]
        X: list[list[float]] = []
        y: list[int] = []
        for _ in range(n):
            label = dominant if rng.random() < p_dominant else 1 - dominant
            mean = 1.0 if label == 1 else -1.0
            X.append([_gauss(rng, mean + shift[j]) for j in range(n_features)])
            y.append(label)
        clients.append(ClientData(client_id=cid, X=X, y=y, feature_shift=shift))
    return clients
