"""Simulated federated client: local training with straggler/dropout behaviour."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from flp.model import LogisticModel, clip_vector
from flp.synthetic import ClientData


@dataclass
class ClientUpdate:
    client_id: int
    delta: list[float]  # parameter delta vs the global model, bias last
    n_samples: int
    latency_ms: int
    clipped_norm: float
    completed: bool = True
    dropped_reason: str = ""


@dataclass
class FederatedClient:
    """One simulated device/organization holding private local data."""

    client_id: int
    data: ClientData
    base_latency_ms: int = 120
    latency_history: list[int] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latency_history) / len(self.latency_history) if self.latency_history else float(self.base_latency_ms)

    def simulate_latency(self, rng: random.Random, straggler: bool) -> int:
        jitter = rng.randint(0, 80)
        latency = self.base_latency_ms + jitter
        if straggler:
            latency += 800  # simulated slow device / bad link
        return latency

    def train(
        self,
        global_model: LogisticModel,
        local_epochs: int,
        lr: float,
        fedprox_mu: float = 0.0,
        clip_norm: float | None = None,
        deadline_ms: int = 500,
        straggler: bool = False,
        rng: random.Random | None = None,
    ) -> ClientUpdate:
        """Train locally from the global model.

        Returns a (possibly dropped) update: if the simulated latency
        exceeds the round deadline, the client is a straggler for this
        round and its update is not delivered (partial participation).
        """
        rng = rng or random.Random(self.client_id)
        latency = self.simulate_latency(rng, straggler)
        self.latency_history.append(latency)
        if latency > deadline_ms:
            return ClientUpdate(
                client_id=self.client_id,
                delta=[0.0] * (global_model.n_features + 1),
                n_samples=len(self.data),
                latency_ms=latency,
                clipped_norm=0.0,
                completed=False,
                dropped_reason=f"deadline exceeded: {latency}ms > {deadline_ms}ms",
            )
        local = global_model.copy()
        for _ in range(local_epochs):
            gw, gb = local.gradient(self.data.X, self.data.y)
            if fedprox_mu > 0:
                # FedProx proximal term: mu * (w_local - w_global).
                gw = [g + fedprox_mu * (w - wg) for g, w, wg in zip(gw, local.weights, global_model.weights)]
                gb = gb + fedprox_mu * (local.bias - global_model.bias)
            local.step(gw, gb, lr)
        delta = local.delta_from(global_model)
        norm = 0.0
        if clip_norm is not None:
            delta, norm = clip_vector(delta, clip_norm)
        return ClientUpdate(
            client_id=self.client_id,
            delta=delta,
            n_samples=len(self.data),
            latency_ms=latency,
            clipped_norm=norm,
            completed=True,
        )


def straggler_aware_weights(clients: list[FederatedClient]) -> list[float]:
    """Selection weights inversely proportional to observed latency.

    Slow clients are still eligible (fairness) but sampled less often,
    so a chronic straggler does not dominate round deadlines.
    """
    weights = [1.0 / (1.0 + c.avg_latency_ms / 200.0) for c in clients]
    total = sum(weights)
    return [w / total for w in weights]
