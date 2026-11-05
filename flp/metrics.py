"""Evaluation metrics: global accuracy, per-client fairness, comm costs."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from flp.model import LogisticModel


@dataclass
class FairnessReport:
    per_client_accuracy: dict[int, float]
    min_accuracy: float
    max_accuracy: float
    spread: float  # max - min
    std_accuracy: float
    mean_accuracy: float

    def to_dict(self) -> dict:
        return {
            "per_client_accuracy": self.per_client_accuracy,
            "min_accuracy": self.min_accuracy,
            "max_accuracy": self.max_accuracy,
            "spread": self.spread,
            "std_accuracy": self.std_accuracy,
            "mean_accuracy": self.mean_accuracy,
        }


def evaluate_global(
    model: LogisticModel, X: list[list[float]], y: list[int]
) -> dict[str, float]:
    return {"loss": model.loss(X, y), "accuracy": model.accuracy(X, y)}


def fairness_report(
    model: LogisticModel, clients: list
) -> FairnessReport:
    """Per-client accuracy on each client's own (non-IID) data."""
    per_client = {
        c.client_id: model.accuracy(c.data.X, c.data.y) for c in clients
    }
    vals = list(per_client.values())
    return FairnessReport(
        per_client_accuracy=per_client,
        min_accuracy=min(vals) if vals else 0.0,
        max_accuracy=max(vals) if vals else 0.0,
        spread=(max(vals) - min(vals)) if vals else 0.0,
        std_accuracy=statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        mean_accuracy=sum(vals) / len(vals) if vals else 0.0,
    )


@dataclass
class CommAccounting:
    """Bytes moved per round and cumulatively (model-size based)."""

    bytes_up_per_round: list[int]
    bytes_down_per_round: list[int]

    def __init__(self) -> None:
        self.bytes_up_per_round = []
        self.bytes_down_per_round = []

    def record_round(self, bytes_up: int, bytes_down: int) -> None:
        self.bytes_up_per_round.append(bytes_up)
        self.bytes_down_per_round.append(bytes_down)

    @property
    def total_up(self) -> int:
        return sum(self.bytes_up_per_round)

    @property
    def total_down(self) -> int:
        return sum(self.bytes_down_per_round)

    @property
    def total(self) -> int:
        return self.total_up + self.total_down

    def to_dict(self) -> dict:
        return {
            "bytes_up_per_round": self.bytes_up_per_round,
            "bytes_down_per_round": self.bytes_down_per_round,
            "total_up": self.total_up,
            "total_down": self.total_down,
            "total": self.total,
        }
