"""From-scratch binary logistic regression (pure stdlib).

No sklearn/torch: the math is small, auditable, and deterministic.
Weights are plain float lists so they can be serialized, masked for
secure aggregation, clipped for DP, and averaged by the aggregator
without any framework dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class LogisticModel:
    n_features: int
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = [0.0] * self.n_features
        if len(self.weights) != self.n_features:
            raise ValueError("weights length must equal n_features")

    def copy(self) -> "LogisticModel":
        return LogisticModel(self.n_features, list(self.weights), self.bias)

    def predict_proba(self, x: list[float]) -> float:
        z = self.bias + sum(w * xi for w, xi in zip(self.weights, x))
        return sigmoid(z)

    def predict(self, x: list[float]) -> int:
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def loss(self, X: list[list[float]], y: list[int]) -> float:
        if not X:
            return 0.0
        total = 0.0
        for xi, yi in zip(X, y):
            p = min(max(self.predict_proba(xi), 1e-12), 1 - 1e-12)
            total += -(yi * math.log(p) + (1 - yi) * math.log(1 - p))
        return total / len(X)

    def accuracy(self, X: list[list[float]], y: list[int]) -> float:
        if not X:
            return 0.0
        return sum(1 for xi, yi in zip(X, y) if self.predict(xi) == yi) / len(X)

    def gradient(
        self, X: list[list[float]], y: list[int]
    ) -> tuple[list[float], float]:
        """Full-batch gradient of the mean binary cross-entropy loss."""
        n = len(X)
        if n == 0:
            return [0.0] * self.n_features, 0.0
        gw = [0.0] * self.n_features
        gb = 0.0
        for xi, yi in zip(X, y):
            err = self.predict_proba(xi) - yi
            for j in range(self.n_features):
                gw[j] += err * xi[j]
            gb += err
        return [g / n for g in gw], gb / n

    def step(self, grad_w: list[float], grad_b: float, lr: float) -> None:
        self.weights = [w - lr * g for w, g in zip(self.weights, grad_w)]
        self.bias -= lr * grad_b

    def delta_from(self, other: "LogisticModel") -> list[float]:
        """Parameter delta (self - other), bias appended last."""
        return [a - b for a, b in zip(self.weights, other.weights)] + [
            self.bias - other.bias
        ]

    def apply_delta(self, delta: list[float]) -> None:
        if len(delta) != self.n_features + 1:
            raise ValueError("delta must have n_features + 1 entries")
        self.weights = [w + d for w, d in zip(self.weights, delta[: self.n_features])]
        self.bias += delta[self.n_features]

    def to_dict(self) -> dict:
        return {"n_features": self.n_features, "weights": list(self.weights), "bias": self.bias}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticModel":
        return cls(n_features=int(d["n_features"]), weights=list(d["weights"]), bias=float(d["bias"]))

    def nbytes(self) -> int:
        """Wire size of the parameter vector (8 bytes per float64)."""
        return 8 * (self.n_features + 1)


def vector_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def clip_vector(v: list[float], max_norm: float) -> tuple[list[float], float]:
    """Scale v down to max_norm if needed; returns (clipped, original_norm)."""
    n = vector_norm(v)
    if n <= max_norm or n == 0.0:
        return list(v), n
    scale = max_norm / n
    return [x * scale for x in v], n
