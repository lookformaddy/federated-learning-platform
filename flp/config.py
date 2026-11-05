"""Configuration for the federated learning platform."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FLConfig:
    """All knobs for one federated training run."""

    n_clients: int = 6
    n_rounds: int = 8
    participation_fraction: float = 0.8
    local_epochs: int = 2
    learning_rate: float = 0.5
    n_features: int = 4
    samples_per_client: int = 120
    test_samples: int = 400
    # Non-IID skew knobs (0 = IID, 1 = extreme).
    label_skew: float = 0.7
    quantity_skew: float = 0.5
    feature_skew: float = 0.4
    # Aggregation: "fedavg" | "trimmed_mean".
    strategy: str = "fedavg"
    trim_fraction: float = 0.1
    # FedProx proximal term (0 disables it; only used when strategy is fedprox).
    fedprox_mu: float = 0.0
    # Differential privacy.
    dp_enabled: bool = True
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 0.8
    dp_delta: float = 1e-5
    dp_epsilon_budget: float = 50.0
    # Stragglers / deadlines (simulated latency in ms).
    round_deadline_ms: int = 500
    min_participants: int = 2
    straggler_client_ids: tuple[int, ...] = ()
    straggler_latency_ms: int = 900
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 < self.participation_fraction <= 1.0:
            raise ValueError("participation_fraction must be in (0, 1]")
        if self.strategy not in {"fedavg", "fedprox", "trimmed_mean"}:
            raise ValueError(f"unknown strategy: {self.strategy}")
        if self.strategy == "fedprox" and self.fedprox_mu <= 0:
            # FedProx with mu=0 degenerates to FedAvg; make it explicit.
            self.fedprox_mu = 0.1
        if self.n_clients < 1 or self.n_rounds < 1:
            raise ValueError("n_clients and n_rounds must be >= 1")
        if self.min_participants < 1:
            raise ValueError("min_participants must be >= 1")
        if self.dp_epsilon_budget <= 0:
            raise ValueError("dp_epsilon_budget must be positive")

    @property
    def participants_per_round(self) -> int:
        return max(1, int(round(self.n_clients * self.participation_fraction)))
