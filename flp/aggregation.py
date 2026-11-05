"""Aggregation strategies: FedAvg, FedProx (client-side), trimmed mean.

FedProx's proximal term lives in the client trainer (flp/client.py); the
server-side aggregation for FedProx is the same size-weighted average as
FedAvg. Trimmed mean is the robust option: per coordinate, drop the top
and bottom `trim_fraction` of client updates before averaging, which
bounds the influence of any single (possibly corrupted) client.
"""

from __future__ import annotations

import math


def fedavg(deltas: list[list[float]], weights: list[float]) -> list[float]:
    """Size-weighted average of client deltas."""
    if not deltas:
        raise ValueError("no deltas to aggregate")
    if len(deltas) != len(weights):
        raise ValueError("deltas and weights must have the same length")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    dim = len(deltas[0])
    out = [0.0] * dim
    for delta, w in zip(deltas, weights):
        if len(delta) != dim:
            raise ValueError("all deltas must have the same dimension")
        for j in range(dim):
            out[j] += (w / total) * delta[j]
    return out


def trimmed_mean(deltas: list[list[float]], trim_fraction: float = 0.1) -> list[float]:
    """Per-coordinate trimmed mean; falls back to plain mean when trimming
    would leave too few values."""
    if not deltas:
        raise ValueError("no deltas to aggregate")
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    dim = len(deltas[0])
    k = int(len(deltas) * trim_fraction)
    out: list[float] = []
    for j in range(dim):
        vals = sorted(d[j] for d in deltas)
        kept = vals[k : len(vals) - k] if len(vals) - 2 * k >= 1 else vals
        out.append(sum(kept) / len(kept))
    return out


def aggregate(
    strategy: str,
    deltas: list[list[float]],
    weights: list[float],
    trim_fraction: float = 0.1,
) -> list[float]:
    if strategy in {"fedavg", "fedprox"}:
        return fedavg(deltas, weights)
    if strategy == "trimmed_mean":
        return trimmed_mean(deltas, trim_fraction)
    raise ValueError(f"unknown aggregation strategy: {strategy}")


def mean_update_norm(deltas: list[list[float]]) -> float:
    if not deltas:
        return 0.0
    return sum(math.sqrt(sum(x * x for x in d)) for d in deltas) / len(deltas)
