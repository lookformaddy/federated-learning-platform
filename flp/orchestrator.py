"""Federated training orchestrator: sampling, rounds, DP, secure agg, registry."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from flp.aggregation import aggregate
from flp.client import ClientUpdate, FederatedClient, straggler_aware_weights
from flp.config import FLConfig
from flp.metrics import CommAccounting, FairnessReport, evaluate_global, fairness_report
from flp.model import LogisticModel
from flp.privacy import PrivacyAccountant, PrivacyBudgetExhausted, add_gaussian_noise
from flp.registry import ModelRegistry
from flp.secure_agg import SecureAggregator, generate_masks, mask_delta
from flp.synthetic import generate_global_data, partition_clients


@dataclass
class RoundResult:
    round_no: int
    participants: list[int]
    dropped: list[int]
    loss: float
    accuracy: float
    fairness_spread: float
    bytes_up: int
    bytes_down: int
    dp_epsilon_spent: float
    promoted_version: int


@dataclass
class TrainingResult:
    rounds: list[RoundResult] = field(default_factory=list)
    final_loss: float = 0.0
    final_accuracy: float = 0.0
    initial_loss: float = 0.0
    halted_reason: str = ""
    comm: CommAccounting = field(default_factory=CommAccounting)
    registry: ModelRegistry = field(default_factory=ModelRegistry)
    fairness: FairnessReport | None = None

    def to_dict(self) -> dict:
        return {
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "final_accuracy": self.final_accuracy,
            "halted_reason": self.halted_reason,
            "rounds": [
                {
                    "round": r.round_no,
                    "participants": r.participants,
                    "dropped": r.dropped,
                    "loss": r.loss,
                    "accuracy": r.accuracy,
                    "fairness_spread": r.fairness_spread,
                    "bytes_up": r.bytes_up,
                    "bytes_down": r.bytes_down,
                    "dp_epsilon_spent": r.dp_epsilon_spent,
                    "promoted_version": r.promoted_version,
                }
                for r in self.rounds
            ],
            "comm": self.comm.to_dict(),
            "registry": self.registry.to_dict(),
            "fairness": self.fairness.to_dict() if self.fairness else None,
        }


def build_clients(config: FLConfig) -> list[FederatedClient]:
    partitions = partition_clients(
        config.n_clients,
        config.samples_per_client,
        config.n_features,
        config.label_skew,
        config.quantity_skew,
        config.feature_skew,
        config.seed,
    )
    clients = []
    for i, part in enumerate(partitions):
        base = 100 + (i * 37) % 150  # deterministic per-client base latency
        clients.append(FederatedClient(client_id=part.client_id, data=part, base_latency_ms=base))
    return clients


def sample_clients(
    clients: list[FederatedClient], weights: list[float], k: int, rng: random.Random
) -> list[FederatedClient]:
    """Weighted sampling *without* replacement, so the cohort size is
    exactly min(k, len(clients)) — sampling with replacement plus
    dedup would silently shrink rounds."""
    pool = list(clients)
    pool_weights = list(weights)
    chosen: list[FederatedClient] = []
    for _ in range(min(k, len(pool))):
        pick = rng.choices(pool, weights=pool_weights, k=1)[0]
        idx = pool.index(pick)
        chosen.append(pick)
        pool.pop(idx)
        pool_weights.pop(idx)
    return chosen


def run_training(config: FLConfig) -> TrainingResult:
    rng = random.Random(config.seed)
    clients = build_clients(config)
    test_X, test_y = generate_global_data(config.test_samples, config.n_features, config.seed + 7)
    model = LogisticModel(config.n_features)
    result = TrainingResult()
    result.initial_loss = model.loss(test_X, test_y)

    accountant = PrivacyAccountant(
        noise_multiplier=config.dp_noise_multiplier if config.dp_enabled else 0.0,
        delta=config.dp_delta,
        epsilon_budget=config.dp_epsilon_budget,
    ) if config.dp_enabled else None

    for round_no in range(1, config.n_rounds + 1):
        # --- DP budget gate (fail-closed): halt before spending past budget.
        if accountant is not None and not accountant.can_afford_round():
            result.halted_reason = (
                f"DP budget exhausted at round {round_no}: "
                f"spent={accountant.epsilon_spent:.3f}, budget={config.dp_epsilon_budget:.3f}"
            )
            break
        # --- Straggler-aware client sampling (without replacement).
        k = config.participants_per_round
        weights = straggler_aware_weights(clients)
        participants = sample_clients(clients, weights, k, rng)

        # --- Local training (secure-aggregation masked exchange).
        updates: dict[int, ClientUpdate] = {}
        completed: list[FederatedClient] = []
        dropped: list[int] = []
        for c in participants:
            is_straggler = c.client_id in config.straggler_client_ids
            update = c.train(
                model,
                local_epochs=config.local_epochs,
                lr=config.learning_rate,
                fedprox_mu=config.fedprox_mu if config.strategy == "fedprox" else 0.0,
                clip_norm=config.dp_clip_norm if config.dp_enabled else None,
                deadline_ms=config.round_deadline_ms,
                straggler=is_straggler,
                rng=rng,
            )
            if not update.completed:
                dropped.append(c.client_id)
                continue
            completed.append(c)
            updates[c.client_id] = update

        if len(completed) < config.min_participants:
            result.halted_reason = (
                f"round {round_no}: only {len(completed)} participants "
                f"(min {config.min_participants}) after deadline drops"
            )
            break

        # Generate zero-sum masks over the completed set and mask updates.
        # The server only ever receives the masked deltas.
        from flp.secure_agg import MaskedUpdate

        aggregator = SecureAggregator()
        dim = config.n_features + 1
        masks = generate_masks([c.client_id for c in completed], dim, round_no, config.seed)
        deltas: list[list[float]] = []
        sample_counts: list[int] = []
        for c in completed:
            update = updates[c.client_id]
            masked = mask_delta(update.delta, masks[c.client_id])
            aggregator.receive(MaskedUpdate(c.client_id, masked, update.n_samples))
            deltas.append(update.delta)
            sample_counts.append(update.n_samples)

        # Server sees only the aggregate sum (masks cancel exactly):
        # the masked sum must equal the plain sum of client deltas.
        agg_sum = aggregator.aggregate_sum()
        for j in range(dim):
            assert abs(agg_sum[j] - sum(d[j] for d in deltas)) < 1e-6, "secure-agg masks did not cancel"
        # Aggregate through the chosen strategy (FedAvg / FedProx /
        # trimmed mean) using sample counts as weights.
        agg_delta = aggregate(config.strategy, deltas, sample_counts, config.trim_fraction)

        # --- Differential privacy: noise on the aggregate, budget charged.
        eps_spent = accountant.epsilon_spent if accountant else 0.0
        if accountant is not None:
            try:
                eps_spent = accountant.charge_round()
            except PrivacyBudgetExhausted as exc:
                result.halted_reason = str(exc)
                break
            agg_delta = add_gaussian_noise(
                agg_delta, config.dp_noise_multiplier, config.dp_clip_norm, rng,
                n_participants=len(completed),
            )

        model.apply_delta(agg_delta)

        # --- Evaluation, fairness, comm accounting.
        evals = evaluate_global(model, test_X, test_y)
        fair = fairness_report(model, clients)
        model_bytes = model.nbytes()
        bytes_up = len(completed) * model_bytes
        bytes_down = len(participants) * model_bytes
        result.comm.record_round(bytes_up, bytes_down)
        entry = result.registry.register(
            model,
            accuracy=evals["accuracy"],
            loss=evals["loss"],
            round_no=round_no,
            dp_epsilon_spent=eps_spent,
            dp_budget_intact=(accountant is None or accountant.epsilon_spent <= config.dp_epsilon_budget),
        )
        result.rounds.append(
            RoundResult(
                round_no=round_no,
                participants=[c.client_id for c in completed],
                dropped=dropped,
                loss=evals["loss"],
                accuracy=evals["accuracy"],
                fairness_spread=fair.spread,
                bytes_up=bytes_up,
                bytes_down=bytes_down,
                dp_epsilon_spent=eps_spent,
                promoted_version=entry.version if entry.promoted else result.registry.promoted_version,
            )
        )

    result.final_loss = model.loss(test_X, test_y)
    result.final_accuracy = model.accuracy(test_X, test_y)
    result.fairness = fairness_report(model, clients)
    return result
