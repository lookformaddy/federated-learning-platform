# Federated Learning Platform

A simulation-first federated learning platform: train a shared model
across distributed clients **without moving their data**. Pure-stdlib
core (from-scratch logistic regression), with FedAvg / FedProx /
trimmed-mean aggregation, secure-aggregation simulation, differential
privacy with a fail-closed epsilon budget, straggler handling, fairness
and communication accounting, and a versioned model registry with
promotion gates and rollback.

## The enterprise problem

Regulated data (health, finance, on-device) cannot be centralized, yet
teams still need one good shared model. Federated learning solves the
data-movement problem but creates new ones: clients are non-IID,
unreliable, and slow; individual updates can leak private data; and a
bad round can silently poison the global model. This project implements
each of those concerns as small, testable, pure-Python components — the
same seams a production platform (Flower / FATE / TensorFlow Federated)
exposes — without any device fleet, network, or cloud dependency.

## Architecture overview

```
  Synthetic clients ── non-IID partitions (label / quantity / feature skew)
        │  local training (from-scratch logistic regression)
        │  FedProx proximal term (optional) · gradient clipping (DP)
        ▼
  Secure aggregation ── zero-sum masks; server only ever sees the sum
        │
        ▼
  Aggregation ── FedAvg (size-weighted) · FedProx · trimmed mean (robust)
        │  + Gaussian noise on the aggregate (DP)
        ▼
  Privacy accountant ── epsilon tracked per round; budget cap halts training
        │
        ▼
  Global model ── held-out eval each round · per-client fairness metrics
        │  · communication-cost accounting (bytes up/down)
        ▼
  Model registry ── versioned; promotes only if accuracy beats the
                    previous global model AND the DP budget is intact;
                    rollback to the last promoted version
        │
        ▼
  FastAPI service (optional, guarded import) ── POST /train, GET /rounds,
        /registry, /fairness
```

See [docs/architecture.md](docs/architecture.md) for the full design,
including the threat model and the decisions below.

## How to run it

```bash
pip install -r requirements.txt   # stdlib only for core; pytest + FastAPI for tests/service
python demo.py                    # end-to-end demo: 8 rounds, 1 straggler, DP on
python -m flp.cli train --clients 6 --rounds 8 --strategy fedavg
python -m flp.cli train --strategy trimmed_mean --no-dp --json
python -m pytest tests/ -q        # full suite; API tests auto-skip w/o FastAPI
```

With FastAPI installed (`pip install fastapi uvicorn`), start the service:

```bash
uvicorn flp.api:app --reload
curl -X POST localhost:8000/train -H 'Content-Type: application/json' \
  -d '{"n_clients": 6, "n_rounds": 8, "straggler_client_ids": [5]}'
curl localhost:8000/registry
```

## Key design decisions and trade-offs

- **Simulation-first, no real devices.** Clients are deterministic
  synthetic partitions, so every run is reproducible and the whole
  platform is testable in CI. The seams (client trainer, aggregator,
  registry) are the ones a real deployment would swap for gRPC/device
  code — the orchestration logic does not change.
- **Non-IID by construction.** Label skew, quantity skew, and per-client
  feature shifts are first-class knobs, because IID federated learning
  is a toy: real fleets are skewed, and skew is what makes FedProx and
  fairness metrics matter.
- **Secure aggregation as mask cancellation.** Masks sum to exactly zero
  across the participating set, so the server's sum of masked updates
  equals the sum of true updates. This simulates the *property* (server
  never sees an individual update); a production system would derive
  masks from pairwise secrets + Shamir sharing so dropouts can be
  handled cryptographically.
- **DP: clip, then noise the aggregate, then charge the budget.**
  Per-update clipping bounds sensitivity; Gaussian noise is calibrated
  to the *mean* (sensitivity `clip/n_participants`); the accountant
  charges linear composition per round and halts training when the
  budget would be exceeded — fail-closed, never silent overspend. The
  accountant is simulation-grade (linear composition, no subsampling
  amplification); production should use an RDP/moments accountant.
  The default budget (ε=50) is sized for an 8-round demo under this
  conservative accountant — it is a demo parameter, not a privacy claim.
- **Stragglers are dropped, not waited on.** Each round has a deadline;
  clients whose simulated latency exceeds it are dropped for that round
  (partial participation). Straggler-aware sampling down-weights
  chronically slow clients while keeping them eligible, so one slow
  device cannot stall every round.
- **Trimmed mean as the robust option.** Per-coordinate trimming bounds
  any single client's influence, which is the cheap defense against a
  corrupted or poisoned update. It costs a little accuracy on clean
  data and is only used when selected.
- **Promotion gate, not auto-promotion.** A global version promotes only
  if it beats the previous global accuracy *and* the DP budget is
  intact. Everything else is recorded but stays unpromoted, and
  rollback restores the last promoted version.
- **Pure stdlib core.** The trainer uses no sklearn/torch, so the math
  is auditable and the suite runs anywhere; FastAPI is an optional layer.

## What it demonstrates

Federated orchestration (sampling, deadlines, partial participation),
non-IID data handling, FedAvg/FedProx/robust aggregation, secure
aggregation properties, differential privacy with budget accounting,
straggler-aware scheduling, fairness and communication-cost evaluation,
and gated model promotion — the full loop a Director-level ML platform
interview asks about, in one runnable codebase.
