from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _client() -> TestClient:
    from oransim.api_routers.zevup import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_zevup_compare_endpoint_mock_mode(monkeypatch):
    monkeypatch.setenv("ZEVUP_LLM_MODE", "mock")
    with _client() as client:
        response = client.post(
            "/api/zevup/scotland/segments/compare",
            json={"year": 2025, "segment_limit": 10, "use_llm": True},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["market"] == "scotland"
    assert body["llm_status"] == "mock"
    assert body["segment_results"]
    assert set(body["wp_report"]) == {"WP1", "WP5", "WP6"}


def test_zevup_compare_endpoint_rejects_unknown_intervention(monkeypatch):
    monkeypatch.setenv("ZEVUP_LLM_MODE", "mock")
    with _client() as client:
        response = client.post(
            "/api/zevup/scotland/segments/compare",
            json={"interventions": ["unknown"]},
        )

    assert response.status_code == 400
    assert "Unknown intervention" in response.json()["detail"]


def test_zevup_compare_endpoint_maps_internal_value_error(monkeypatch):
    from oransim import api_routers

    monkeypatch.setenv("ZEVUP_LLM_MODE", "mock")
    monkeypatch.setattr(
        api_routers.zevup,
        "compare_scotland_segments",
        lambda req: (_ for _ in ()).throw(
            ValueError("feasible_h values must be in the range [0, 1].")
        ),
    )
    with _client() as client:
        response = client.post(
            "/api/zevup/scotland/segments/compare",
            json={"year": 2025, "segment_limit": 10, "use_llm": True},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "ZEV-UP comparison failed."


def test_zevup_compare_endpoint_maps_file_not_found(monkeypatch):
    from oransim import api_routers

    monkeypatch.setenv("ZEVUP_LLM_MODE", "mock")
    monkeypatch.setattr(
        api_routers.zevup,
        "compare_scotland_segments",
        lambda req: (_ for _ in ()).throw(FileNotFoundError("/tmp/missing.csv")),
    )
    with _client() as client:
        response = client.post(
            "/api/zevup/scotland/segments/compare",
            json={"year": 2025, "segment_limit": 10, "use_llm": True},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Required ZEV-UP source data is missing."
