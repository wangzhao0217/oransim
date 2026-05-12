"""Regression tests for the frugal EV opinion simulator MVP."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _load_personas_and_concepts() -> tuple[list[dict], dict[str, dict]]:
    root = Path(__file__).parent.parent
    personas = json.loads((root / "data/personas/frugal_ev_personas.json").read_text())
    concepts = {
        concept["concept_id"]: concept
        for concept in json.loads((root / "data/concepts/frugal_ev_concepts.json").read_text())
    }
    return personas, concepts


def test_frugal_ev_simulate_returns_default_segment_report(monkeypatch):
    from oransim.api_routers import frugal_ev
    from oransim.schemas.frugal_ev_opinion import FrugalEVSimulateRequest

    scores = {
        "urban_commuter": 0.72,
        "first_time_car_buyer": 0.68,
        "delivery_driver": 0.58,
        "rural_budget_family": 0.38,
        "ev_skeptic": 0.31,
    }

    async def fake_reaction(persona: dict, concept: dict) -> dict:
        score = scores[persona["segment"]]
        return {
            "purchase_intent_score": score,
            "trust_score": 0.5,
            "price_acceptance_score": 0.6,
            "range_confidence_score": 0.7,
            "charging_confidence_score": 0.4,
            "top_positive_reason": f"{persona['segment']} hook",
            "top_objection": f"{persona['segment']} objection",
            "what_would_change_my_mind": "More proof.",
            "likely_social_comment": f"{concept['name']} comment.",
        }

    monkeypatch.setattr(frugal_ev, "simulate_persona_reaction", fake_reaction)

    report = asyncio.run(
        frugal_ev.simulate(
            FrugalEVSimulateRequest(concept_id="frugal_ev_city_9999", target_market="UK")
        )
    )

    assert report.concept_name == "\u00a39,999 City EV"
    assert report.target_market == "UK"
    assert report.overall_acceptance_score == 0.534
    assert [opinion.segment for opinion in report.segment_opinions] == list(scores)
    assert report.strongest_segments == ["urban_commuter", "first_time_car_buyer"]
    assert report.weakest_segments == ["rural_budget_family", "ev_skeptic"]
    assert len(report.top_objection_themes) == 5
    assert "Promising for urban_commuter" in report.verdict
    assert "Weak for rural_budget_family, ev_skeptic" in report.verdict


def test_frugal_ev_mock_agent_returns_bounded_schema_scores(monkeypatch):
    from oransim.agents.frugal_ev_persona_prompt import simulate_persona_reaction

    monkeypatch.setenv("LLM_MODE", "mock")
    personas, concepts = _load_personas_and_concepts()

    reaction = asyncio.run(simulate_persona_reaction(personas[0], concepts["frugal_ev_city_9999"]))

    for key in (
        "purchase_intent_score",
        "trust_score",
        "price_acceptance_score",
        "range_confidence_score",
        "charging_confidence_score",
    ):
        assert 0.0 <= reaction[key] <= 1.0
    assert reaction["top_positive_reason"]
    assert reaction["top_objection"]
    assert reaction["what_would_change_my_mind"]
    assert reaction["likely_social_comment"]


def test_frugal_ev_live_llm_call_uses_reasoning_model_budget(monkeypatch):
    from oransim.agents import frugal_ev_persona_prompt as agent

    calls = {}

    def fake_llm_available() -> bool:
        return True

    def fake_call_llm_json_with_retry(body: dict, **kwargs):
        calls["body"] = body
        calls["kwargs"] = kwargs
        return (
            {
                "purchase_intent_score": 0.8,
                "trust_score": 0.7,
                "price_acceptance_score": 0.9,
                "range_confidence_score": 0.6,
                "charging_confidence_score": 0.5,
                "top_positive_reason": "low running costs",
                "top_objection": "charging access",
                "what_would_change_my_mind": "home charging",
                "likely_social_comment": "worth a look",
            },
            {},
        )

    monkeypatch.setattr(agent, "llm_available", fake_llm_available)
    monkeypatch.setattr(agent, "call_llm_json_with_retry", fake_call_llm_json_with_retry)
    personas, concepts = _load_personas_and_concepts()

    reaction = asyncio.run(
        agent.simulate_persona_reaction(personas[0], concepts["frugal_ev_city_9999"])
    )

    assert reaction["purchase_intent_score"] == 0.8
    assert calls["body"]["max_tokens"] >= 1500
    assert calls["kwargs"]["use_stream"] is False
    assert calls["kwargs"]["timeout"] >= 90


def test_frugal_ev_simulate_rejects_unknown_concept():
    from oransim.api_routers.frugal_ev import simulate
    from oransim.schemas.frugal_ev_opinion import FrugalEVSimulateRequest

    with pytest.raises(HTTPException) as exc:
        asyncio.run(simulate(FrugalEVSimulateRequest(concept_id="missing")))

    assert exc.value.status_code == 404
