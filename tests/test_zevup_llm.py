from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _segments() -> list[dict[str, str | float]]:
    return [
        {
            "segment_id": "scot_seg_0001",
            "age_group": "35_to_49",
            "sex": "female",
            "education_group": "degree_or_above",
            "socioeconomic_group": "managerial_professional",
            "household_type": "couple_with_dependent_children",
            "dependent_children": "youngest_5_to_11",
            "accommodation": "whole_house",
            "household_size": "three_plus",
            "car_ownership": "one",
            "tenure": "owned",
            "charging_readiness_group": "moderate",
            "household_weight": 100.0,
            "person_weight": 300.0,
            "baseline_feasible_share": 0.2,
        }
    ]


class _FakeDeepSeekResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeDeepSeekResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _deepseek_payload(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def _valid_content(effect: float = 0.025) -> str:
    return json.dumps(
        {
            "effects": [
                {
                    "segment_id": "scot_seg_0001",
                    "intervention_id": "awareness_campaign",
                    "effect": effect,
                    "reasoning": "Reasoned from segment attributes.",
                    "message_recommendation": "Use clear local mobility messaging.",
                }
            ],
            "wp1_summary": "Affordability summary.",
            "wp5_summary": "Market uptake summary.",
            "wp6_summary": "Messaging summary.",
            "caveats": ["Synthetic test response."],
        }
    )


def test_default_model_is_deepseek_v4_flash(monkeypatch):
    from oransim.agents.zevup_llm import llm_config

    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    assert llm_config()["model"] == "deepseek-v4-flash"


def test_mock_mode_returns_effects(monkeypatch):
    from oransim.agents.zevup_llm import assess_segments

    monkeypatch.setenv("ZEVUP_LLM_MODE", "mock")
    result = assess_segments(
        segments=_segments(),
        vehicle_concept={
            "concept_id": "zevup_l7e_passenger_2_seat",
            "name": "Vehicle",
        },
        interventions=["awareness_campaign", "price_subsidy", "charging_support"],
        year=2025,
        baseline_scenario="neutral_lr_central",
    )

    assert result.status == "mock"
    assert result.response.effects
    assert {effect.intervention_id for effect in result.response.effects} == {
        "awareness_campaign",
        "price_subsidy",
        "charging_support",
    }


def test_off_mode_uses_configured_defaults(monkeypatch):
    from oransim.data.zevup_static import get_intervention
    from oransim.agents.zevup_llm import assess_segments

    monkeypatch.setenv("ZEVUP_LLM_MODE", "off")
    result = assess_segments(
        segments=_segments(),
        vehicle_concept={
            "concept_id": "zevup_l7e_passenger_2_seat",
            "name": "Vehicle",
        },
        interventions=["awareness_campaign"],
        year=2025,
        baseline_scenario="neutral_lr_central",
    )

    assert result.status == "off"
    assert (
        result.response.effects[0].effect
        == get_intervention("awareness_campaign").default_effect
    )


def test_force_off_overrides_mock_mode(monkeypatch):
    from oransim.agents.zevup_llm import assess_segments

    monkeypatch.setenv("ZEVUP_LLM_MODE", "mock")
    result = assess_segments(
        segments=_segments(),
        vehicle_concept={
            "concept_id": "zevup_l7e_passenger_2_seat",
            "name": "Vehicle",
        },
        interventions=["price_subsidy"],
        year=2025,
        baseline_scenario="neutral_lr_central",
        force_off=True,
    )

    assert result.status == "off"


def test_deepseek_mode_without_api_key_returns_fallback(monkeypatch):
    from oransim.agents.zevup_llm import assess_segments

    monkeypatch.setenv("ZEVUP_LLM_MODE", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = assess_segments(
        segments=_segments(),
        vehicle_concept={
            "concept_id": "zevup_l7e_passenger_2_seat",
            "name": "Vehicle",
        },
        interventions=["charging_support"],
        year=2025,
        baseline_scenario="neutral_lr_central",
    )

    assert result.status == "fallback"


def test_deepseek_mode_uses_live_call_payload(monkeypatch):
    from oransim.agents.zevup_llm import assess_segments

    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeDeepSeekResponse(_deepseek_payload(_valid_content()))

    monkeypatch.setenv("ZEVUP_LLM_MODE", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.setattr("oransim.agents.zevup_llm.urllib.request.urlopen", fake_urlopen)

    result = assess_segments(
        segments=_segments(),
        vehicle_concept={
            "concept_id": "zevup_l7e_passenger_2_seat",
            "name": "Vehicle",
        },
        interventions=["awareness_campaign"],
        year=2025,
        baseline_scenario="neutral_lr_central",
    )

    assert result.status == "used"
    request = captured["request"]
    assert request.full_url.endswith("/chat/completions")
    assert request.headers["Authorization"] == "Bearer test-key"
    assert captured["timeout"] == 30.0
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "deepseek-v4-flash"
    assert body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    messages = body["messages"]
    assert "strict JSON" in messages[0]["content"]
    assert "required_json_shape" in messages[1]["content"]
    assert "intervention_metadata" in messages[1]["content"]
    assert "effect_rules" in messages[1]["content"]


def test_prompt_includes_effect_rules_and_intervention_bounds():
    from oransim.agents.zevup_llm import _prompt

    prompt = _prompt(
        segments=_segments(),
        vehicle_concept={
            "concept_id": "zevup_l7e_passenger_2_seat",
            "name": "Vehicle",
        },
        interventions=["awareness_campaign", "price_subsidy", "charging_support"],
        year=2025,
        baseline_scenario="neutral_lr_central",
    )
    payload = json.loads(prompt)

    assert payload["effect_rules"]
    metadata = {
        item["intervention_id"]: item for item in payload["intervention_metadata"]
    }
    assert metadata["charging_support"]["default_effect"] == 0.075
    assert metadata["charging_support"]["max_effect"] == 0.20
    assert any("poor charging readiness" in rule for rule in payload["effect_rules"])
    assert any("routine_manual" in rule for rule in payload["effect_rules"])


def test_unreasonable_deepseek_effect_returns_fallback(monkeypatch):
    from oransim.agents.zevup_llm import assess_segments

    content = json.dumps(
        {
            "effects": [
                {
                    "segment_id": "scot_seg_0001",
                    "intervention_id": "charging_support",
                    "effect": 0.19,
                    "reasoning": "Unsupported charging lift.",
                    "message_recommendation": "Use charging support.",
                }
            ],
            "wp1_summary": "Affordability summary.",
            "wp5_summary": "Market uptake summary.",
            "wp6_summary": "Messaging summary.",
            "caveats": ["Synthetic test response."],
        }
    )

    def fake_urlopen(request, timeout):
        return _FakeDeepSeekResponse(_deepseek_payload(content))

    monkeypatch.setenv("ZEVUP_LLM_MODE", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("oransim.agents.zevup_llm.urllib.request.urlopen", fake_urlopen)

    result = assess_segments(
        segments=_segments(),
        vehicle_concept={
            "concept_id": "zevup_l7e_passenger_2_seat",
            "name": "Vehicle",
        },
        interventions=["charging_support"],
        year=2025,
        baseline_scenario="neutral_lr_central",
    )

    assert result.status == "fallback"


def test_invalid_json_content_returns_fallback(monkeypatch):
    from oransim.agents.zevup_llm import assess_segments

    def fake_urlopen(request, timeout):
        return _FakeDeepSeekResponse(_deepseek_payload("not json"))

    monkeypatch.setenv("ZEVUP_LLM_MODE", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("oransim.agents.zevup_llm.urllib.request.urlopen", fake_urlopen)

    result = assess_segments(
        segments=_segments(),
        vehicle_concept={
            "concept_id": "zevup_l7e_passenger_2_seat",
            "name": "Vehicle",
        },
        interventions=["awareness_campaign"],
        year=2025,
        baseline_scenario="neutral_lr_central",
    )

    assert result.status == "fallback"


def test_pydantic_invalid_content_returns_fallback(monkeypatch):
    from oransim.agents.zevup_llm import assess_segments

    def fake_urlopen(request, timeout):
        return _FakeDeepSeekResponse(_deepseek_payload(_valid_content(effect=1.5)))

    monkeypatch.setenv("ZEVUP_LLM_MODE", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("oransim.agents.zevup_llm.urllib.request.urlopen", fake_urlopen)

    result = assess_segments(
        segments=_segments(),
        vehicle_concept={
            "concept_id": "zevup_l7e_passenger_2_seat",
            "name": "Vehicle",
        },
        interventions=["awareness_campaign"],
        year=2025,
        baseline_scenario="neutral_lr_central",
    )

    assert result.status == "fallback"


def test_malformed_provider_shape_returns_fallback(monkeypatch):
    from oransim.agents.zevup_llm import assess_segments

    def fake_urlopen(request, timeout):
        return _FakeDeepSeekResponse({"choices": []})

    monkeypatch.setenv("ZEVUP_LLM_MODE", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("oransim.agents.zevup_llm.urllib.request.urlopen", fake_urlopen)

    result = assess_segments(
        segments=_segments(),
        vehicle_concept={
            "concept_id": "zevup_l7e_passenger_2_seat",
            "name": "Vehicle",
        },
        interventions=["awareness_campaign"],
        year=2025,
        baseline_scenario="neutral_lr_central",
    )

    assert result.status == "fallback"


def test_invalid_timeout_env_returns_fallback_without_crashing(monkeypatch):
    from oransim.agents.zevup_llm import assess_segments

    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        raise OSError("network unavailable")

    monkeypatch.setenv("ZEVUP_LLM_MODE", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("ZEVUP_LLM_TIMEOUT", "abc")
    monkeypatch.setattr("oransim.agents.zevup_llm.urllib.request.urlopen", fake_urlopen)

    result = assess_segments(
        segments=_segments(),
        vehicle_concept={
            "concept_id": "zevup_l7e_passenger_2_seat",
            "name": "Vehicle",
        },
        interventions=["awareness_campaign"],
        year=2025,
        baseline_scenario="neutral_lr_central",
    )

    assert result.status == "fallback"
    assert captured["timeout"] == 30.0


def test_mock_and_off_modes_do_not_call_deepseek(monkeypatch):
    from oransim.agents.zevup_llm import assess_segments

    def fail_urlopen(request, timeout):
        raise AssertionError("urlopen should not be called")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("oransim.agents.zevup_llm.urllib.request.urlopen", fail_urlopen)

    for mode in ("mock", "off"):
        monkeypatch.setenv("ZEVUP_LLM_MODE", mode)
        result = assess_segments(
            segments=_segments(),
            vehicle_concept={
                "concept_id": "zevup_l7e_passenger_2_seat",
                "name": "Vehicle",
            },
            interventions=["awareness_campaign"],
            year=2025,
            baseline_scenario="neutral_lr_central",
        )

        assert result.status == mode


def test_unknown_mode_returns_fallback_without_live_call(monkeypatch):
    from oransim.agents.zevup_llm import assess_segments

    def fail_urlopen(request, timeout):
        raise AssertionError("urlopen should not be called")

    monkeypatch.setenv("ZEVUP_LLM_MODE", "surprise")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("oransim.agents.zevup_llm.urllib.request.urlopen", fail_urlopen)

    result = assess_segments(
        segments=_segments(),
        vehicle_concept={
            "concept_id": "zevup_l7e_passenger_2_seat",
            "name": "Vehicle",
        },
        interventions=["awareness_campaign"],
        year=2025,
        baseline_scenario="neutral_lr_central",
    )

    assert result.status == "fallback"
