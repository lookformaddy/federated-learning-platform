"""API tests — auto-skip when FastAPI is not installed."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from flp.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_registry_before_train_is_404(client):
    assert client.get("/registry").status_code == 404


def test_train_endpoint(client):
    r = client.post("/train", json={
        "n_clients": 4, "n_rounds": 2, "participation_fraction": 1.0,
        "dp_enabled": False, "seed": 7,
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["rounds"]) == 2
    assert data["final_loss"] < data["initial_loss"]
    assert data["registry"]["promoted_version"] >= 1


def test_rounds_and_registry_after_train(client):
    client.post("/train", json={"n_clients": 4, "n_rounds": 2,
                                "participation_fraction": 1.0, "dp_enabled": False, "seed": 7})
    assert len(client.get("/rounds").json()["rounds"]) == 2
    assert client.get("/registry").json()["promoted_version"] >= 1
    fairness = client.get("/fairness").json()
    assert len(fairness["per_client_accuracy"]) == 4


def test_train_rejects_bad_strategy(client):
    r = client.post("/train", json={"n_clients": 4, "n_rounds": 2, "strategy": "bogus"})
    assert r.status_code == 422
