"""Versioned model registry with promotion gates and rollback.

A global model version promotes only when it (a) beats the previous
global accuracy on the held-out set and (b) was produced within the DP
budget. Anything else is recorded but stays unpromoted. Rollback
restores the last promoted version's parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flp.model import LogisticModel


@dataclass
class ModelVersion:
    version: int
    weights: list[float]
    bias: float
    n_features: int
    accuracy: float
    loss: float
    round_no: int
    dp_epsilon_spent: float
    promoted: bool = False
    promotion_reason: str = ""

    def model(self) -> LogisticModel:
        return LogisticModel(self.n_features, list(self.weights), self.bias)


@dataclass
class ModelRegistry:
    versions: list[ModelVersion] = field(default_factory=list)
    promoted_version: int = 0

    @property
    def best_accuracy(self) -> float:
        promoted = [v for v in self.versions if v.promoted]
        return max((v.accuracy for v in promoted), default=0.0)

    def register(
        self,
        model: LogisticModel,
        accuracy: float,
        loss: float,
        round_no: int,
        dp_epsilon_spent: float,
        dp_budget_intact: bool,
    ) -> ModelVersion:
        version = len(self.versions) + 1
        promotes = dp_budget_intact and accuracy > self.best_accuracy
        reason = (
            "promoted: beat previous global accuracy with DP budget intact"
            if promotes
            else (
                "not promoted: DP budget exhausted"
                if not dp_budget_intact
                else f"not promoted: accuracy {accuracy:.4f} did not beat best {self.best_accuracy:.4f}"
            )
        )
        entry = ModelVersion(
            version=version,
            weights=list(model.weights),
            bias=model.bias,
            n_features=model.n_features,
            accuracy=accuracy,
            loss=loss,
            round_no=round_no,
            dp_epsilon_spent=dp_epsilon_spent,
            promoted=promotes,
            promotion_reason=reason,
        )
        self.versions.append(entry)
        if promotes:
            self.promoted_version = version
        return entry

    def rollback(self) -> ModelVersion | None:
        """Return the last promoted version (or None if nothing promoted)."""
        promoted = [v for v in self.versions if v.promoted]
        if not promoted:
            return None
        latest = promoted[-1]
        self.promoted_version = latest.version
        return latest

    def to_dict(self) -> dict:
        return {
            "promoted_version": self.promoted_version,
            "best_accuracy": self.best_accuracy,
            "versions": [
                {
                    "version": v.version,
                    "accuracy": v.accuracy,
                    "loss": v.loss,
                    "round": v.round_no,
                    "dp_epsilon_spent": v.dp_epsilon_spent,
                    "promoted": v.promoted,
                    "reason": v.promotion_reason,
                }
                for v in self.versions
            ],
        }
