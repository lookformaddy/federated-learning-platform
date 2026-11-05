"""Differential privacy: per-update clipping + Gaussian noise + accountant.

Mechanism: each client update is clipped to `clip_norm` (bounding its
sensitivity), then the server adds Gaussian noise to the aggregate. The
privacy accountant tracks epsilon with the standard Gaussian-mechanism
bound, eps_round = sqrt(2 * ln(1.25 / delta)) / noise_multiplier, composed
linearly across rounds (a conservative, auditable composition). The
budget is fail-closed: once the next round would exceed the budget,
training halts instead of silently spending privacy it does not have.

Note: this is a simulation-grade accountant for a teaching/platform
demo, not a certified DP implementation. Real deployments should use a
moments/RDP accountant (e.g. Opacus) and subsampling amplification.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from flp.model import clip_vector


@dataclass
class PrivacyAccountant:
    noise_multiplier: float
    delta: float
    epsilon_budget: float
    epsilon_spent: float = 0.0
    rounds_charged: int = 0

    def per_round_epsilon(self) -> float:
        if self.noise_multiplier <= 0:
            return math.inf
        return math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.noise_multiplier

    def can_afford_round(self) -> bool:
        return self.epsilon_spent + self.per_round_epsilon() <= self.epsilon_budget

    def charge_round(self) -> float:
        """Charge one round; raises if the budget would be exceeded."""
        eps = self.per_round_epsilon()
        if self.epsilon_spent + eps > self.epsilon_budget:
            raise PrivacyBudgetExhausted(
                f"DP budget exhausted: spent={self.epsilon_spent:.3f}, "
                f"next round needs {eps:.3f}, budget={self.epsilon_budget:.3f}"
            )
        self.epsilon_spent += eps
        self.rounds_charged += 1
        return self.epsilon_spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.epsilon_budget - self.epsilon_spent)


class PrivacyBudgetExhausted(RuntimeError):
    """Raised when a round would exceed the DP epsilon budget (fail-closed)."""


def add_gaussian_noise(
    vector: list[float],
    noise_multiplier: float,
    clip_norm: float,
    rng: random.Random,
    n_participants: int = 1,
) -> list[float]:
    """Add Gaussian noise with std = noise_multiplier * clip_norm / n.

    The sensitivity of the *mean* of n clipped updates is clip_norm / n,
    so the noise scale shrinks with the cohort size — the standard
    DP-FedAvg calibration. The accountant's per-round epsilon depends
    only on the noise multiplier, so the two stay consistent.
    """
    if noise_multiplier <= 0:
        return list(vector)
    sigma = noise_multiplier * clip_norm / max(1, n_participants)
    return [x + rng.gauss(0.0, sigma) for x in vector]


def privatize_delta(
    delta: list[float], clip_norm: float
) -> tuple[list[float], float]:
    """Clip a client delta to the DP sensitivity bound."""
    return clip_vector(delta, clip_norm)
