"""FastAPI service for the federated learning platform.

Guarded import: if FastAPI is not installed, `app` is None and the
core package still works (tests skip the API suite in that case).
"""

from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - exercised by skipping API tests
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]

from flp.config import FLConfig
from flp.orchestrator import TrainingResult, run_training

if FastAPI is not None:
    app = FastAPI(title="Federated Learning Platform", version="0.1.0")
    _last_result: TrainingResult | None = None

    class TrainRequest(BaseModel):
        n_clients: int = 6
        n_rounds: int = 8
        participation_fraction: float = 0.8
        strategy: str = "fedavg"
        dp_enabled: bool = True
        dp_epsilon_budget: float = 50.0
        straggler_client_ids: list[int] = []
        seed: int = 42

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "federated-learning-platform"}

    @app.post("/train")
    def train(req: TrainRequest) -> dict:
        global _last_result
        try:
            config = FLConfig(
                n_clients=req.n_clients,
                n_rounds=req.n_rounds,
                participation_fraction=req.participation_fraction,
                strategy=req.strategy,
                dp_enabled=req.dp_enabled,
                dp_epsilon_budget=req.dp_epsilon_budget,
                straggler_client_ids=tuple(req.straggler_client_ids),
                seed=req.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _last_result = run_training(config)
        return _last_result.to_dict()

    @app.get("/rounds")
    def rounds() -> dict:
        if _last_result is None:
            raise HTTPException(status_code=404, detail="no training run yet")
        return {"rounds": _last_result.to_dict()["rounds"]}

    @app.get("/registry")
    def registry() -> dict:
        if _last_result is None:
            raise HTTPException(status_code=404, detail="no training run yet")
        return _last_result.registry.to_dict()

    @app.get("/fairness")
    def fairness() -> dict:
        if _last_result is None:
            raise HTTPException(status_code=404, detail="no training run yet")
        return _last_result.fairness.to_dict() if _last_result.fairness else {}
else:  # pragma: no cover
    app = None
