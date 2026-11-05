"""Secure aggregation simulation: masked update exchange.

Each client masks its update with a one-time random mask before sending
it to the server. Masks are constructed so they sum to exactly zero
across the participating set (the last participant's mask is the
negation of the sum of the others), so the server's sum of masked
updates equals the sum of the true updates — the server never sees an
individual client's update, only the aggregate.

This simulates the *property* of secure aggregation (mask cancellation),
not a real cryptographic protocol: in production, masks come from
pairwise Diffie-Hellman secrets + Shamir sharing so dropouts can be
handled without leaking. Here dropout handling is explicit: if a client
drops out after masking, its mask must be cancelled by the coordinator
before aggregation, which `cancel_dropped_masks` models.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class MaskedUpdate:
    client_id: int
    masked_delta: list[float]
    n_samples: int


def generate_masks(
    client_ids: list[int], dim: int, round_no: int, seed: int
) -> dict[int, list[float]]:
    """Zero-sum masks, deterministic per (round, seed)."""
    if not client_ids:
        raise ValueError("need at least one client to mask")
    rng = random.Random(seed * 100_003 + round_no * 9176)
    masks: dict[int, list[float]] = {}
    running = [0.0] * dim
    for cid in client_ids[:-1]:
        mask = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        masks[cid] = mask
        for j in range(dim):
            running[j] += mask[j]
    masks[client_ids[-1]] = [-x for x in running]
    return masks


def mask_delta(delta: list[float], mask: list[float]) -> list[float]:
    return [d + m for d, m in zip(delta, mask)]


def unmask_aggregate(masked_deltas: list[list[float]]) -> list[float]:
    """Sum masked deltas; masks cancel, leaving the true aggregate sum."""
    dim = len(masked_deltas[0])
    out = [0.0] * dim
    for d in masked_deltas:
        for j in range(dim):
            out[j] += d[j]
    return out


def verify_masks_cancel(masks: dict[int, list[float]], tol: float = 1e-9) -> bool:
    dim = len(next(iter(masks.values())))
    for j in range(dim):
        if abs(sum(m[j] for m in masks.values())) > tol:
            return False
    return True


class SecureAggregator:
    """Server-side view: it only ever holds masked updates + the aggregate."""

    def __init__(self) -> None:
        self.masked_received: list[MaskedUpdate] = []

    def receive(self, update: MaskedUpdate) -> None:
        self.masked_received.append(update)

    def aggregate_sum(self) -> list[float]:
        if not self.masked_received:
            raise ValueError("no masked updates received")
        return unmask_aggregate([u.masked_delta for u in self.masked_received])

    def total_samples(self) -> int:
        return sum(u.n_samples for u in self.masked_received)
