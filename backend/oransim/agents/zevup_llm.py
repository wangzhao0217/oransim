"""DeepSeek-compatible LLM helper for ZEV-UP segment reasoning."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from ..data.zevup_static import get_intervention
from ..schemas.zevup import ZEVUPLLMEffect, ZEVUPLLMResponse

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
ZEVUPLLMStatus = Literal["off", "mock", "fallback", "used"]
LOW_SOCIOECONOMIC_GROUPS = {"routine_manual", "other"}
YOUNG_AGE_GROUPS = {"16_to_24", "25_to_34"}


@dataclass
class ZEVUPLLMResult:
    status: ZEVUPLLMStatus
    response: ZEVUPLLMResponse
    raw: dict[str, Any] | None = None


def llm_config() -> dict[str, str | float | bool]:
    return {
        "mode": os.environ.get("ZEVUP_LLM_MODE", "mock"),
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        "model": os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
        "api_key_set": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "timeout": _timeout_seconds(),
    }


def _timeout_seconds() -> float:
    try:
        return float(os.environ.get("ZEVUP_LLM_TIMEOUT", "30"))
    except ValueError:
        return 30.0


def assess_segments(
    *,
    segments: list[dict[str, Any]],
    vehicle_concept: dict[str, Any],
    interventions: list[str],
    year: int,
    baseline_scenario: str,
    force_off: bool = False,
) -> ZEVUPLLMResult:
    """Return structured intervention reasoning from DeepSeek, mock, or defaults."""
    if force_off:
        return ZEVUPLLMResult("off", _default_response(segments, interventions))

    cfg = llm_config()
    mode = str(cfg["mode"])
    if mode == "off":
        return ZEVUPLLMResult("off", _default_response(segments, interventions))
    if mode == "mock":
        return ZEVUPLLMResult("mock", _mock_response(segments, interventions))
    if mode != "deepseek":
        return ZEVUPLLMResult("fallback", _default_response(segments, interventions))
    if not cfg["api_key_set"]:
        return ZEVUPLLMResult("fallback", _default_response(segments, interventions))

    raw: dict[str, Any] | None = None
    try:
        raw = _call_deepseek(
            segments=segments,
            vehicle_concept=vehicle_concept,
            interventions=interventions,
            year=year,
            baseline_scenario=baseline_scenario,
            config=cfg,
        )
        parsed = ZEVUPLLMResponse(**raw)
        validate_effects(parsed, segments=segments, interventions=interventions)
        return ZEVUPLLMResult("used", parsed, raw)
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValidationError,
        TimeoutError,
        ValueError,
    ):
        return ZEVUPLLMResult("fallback", _default_response(segments, interventions), raw)


def _call_deepseek(
    *,
    segments: list[dict[str, Any]],
    vehicle_concept: dict[str, Any],
    interventions: list[str],
    year: int,
    baseline_scenario: str,
    config: dict[str, str | float | bool],
) -> dict[str, Any]:
    prompt = _prompt(segments, vehicle_concept, interventions, year, baseline_scenario)
    body = {
        "model": config["model"],
        "temperature": 0.0,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a ZEV-UP transport acceptance analyst. "
                    "Return strict JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        f"{config['base_url']}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(config["timeout"])) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    return json.loads(content)


def _prompt(
    segments: list[dict[str, Any]],
    vehicle_concept: dict[str, Any],
    interventions: list[str],
    year: int,
    baseline_scenario: str,
) -> str:
    metadata = []
    for intervention_id in interventions:
        intervention = get_intervention(intervention_id)
        metadata.append(
            {
                "intervention_id": intervention.intervention_id,
                "name": intervention.name,
                "description": intervention.description,
                "default_effect": intervention.default_effect,
                "min_effect": intervention.min_effect,
                "max_effect": intervention.max_effect,
                "effect_targets": intervention.effect_targets,
            }
        )
    return json.dumps(
        {
            "task": (
                "Estimate bounded intervention effects for Scotland ZEV-UP "
                "weighted segments."
            ),
            "year": year,
            "baseline_scenario": baseline_scenario,
            "vehicle_concept": vehicle_concept,
            "interventions": interventions,
            "intervention_metadata": metadata,
            "effect_rules": [
                (
                    "Use the default_effect unless a named segment attribute gives a "
                    "clear reason to move up or down within min_effect and max_effect."
                ),
                (
                    "poor charging readiness should only increase charging_support "
                    "above default; it should not by itself increase awareness_campaign "
                    "or price_subsidy."
                ),
                (
                    "Segments with socioeconomic_group routine_manual or other may "
                    "receive a higher price_subsidy effect when affordability is the "
                    "main barrier."
                ),
                (
                    "Age groups 16_to_24 or 25_to_34 may receive a slightly higher "
                    "awareness_campaign effect when awareness or willingness is the "
                    "main barrier."
                ),
                (
                    "Reasoning must cite the segment fields that justify any effect "
                    "above default."
                ),
            ],
            "segments": segments,
            "required_json_shape": {
                "effects": [
                    {
                        "segment_id": "string",
                        "intervention_id": "string",
                        "effect": "float 0-1",
                        "reasoning": "string",
                        "message_recommendation": "string",
                    }
                ],
                "wp1_summary": "string",
                "wp5_summary": "string",
                "wp6_summary": "string",
                "caveats": ["string"],
            },
        },
        ensure_ascii=False,
    )


def validate_effects(
    response: ZEVUPLLMResponse,
    *,
    segments: list[dict[str, Any]],
    interventions: list[str],
) -> None:
    """Reject unsupported live-LLM effects before they reach uplift math."""
    segment_lookup = {str(segment["segment_id"]): segment for segment in segments}
    selected_interventions = set(interventions)

    for effect in response.effects:
        segment = segment_lookup.get(effect.segment_id)
        if segment is None:
            raise ValueError(f"Unknown segment_id in LLM effect: {effect.segment_id}")
        if effect.intervention_id not in selected_interventions:
            raise ValueError(
                f"Unexpected intervention_id in LLM effect: {effect.intervention_id}"
            )

        intervention = get_intervention(effect.intervention_id)
        value = float(effect.effect)
        if not intervention.min_effect <= value <= intervention.max_effect:
            raise ValueError(
                f"{effect.intervention_id} effect {value} outside configured bounds "
                f"{intervention.min_effect}-{intervention.max_effect}"
            )
        if value <= intervention.default_effect:
            continue

        if (
            effect.intervention_id == "charging_support"
            and segment.get("charging_readiness_group") != "poor"
        ):
            raise ValueError(
                "charging_support above default requires poor charging_readiness_group"
            )
        if (
            effect.intervention_id == "price_subsidy"
            and segment.get("socioeconomic_group") not in LOW_SOCIOECONOMIC_GROUPS
        ):
            raise ValueError(
                "price_subsidy above default requires a lower socioeconomic proxy group"
            )
        if (
            effect.intervention_id == "awareness_campaign"
            and segment.get("age_group") not in YOUNG_AGE_GROUPS
        ):
            raise ValueError(
                "awareness_campaign above default requires a younger age group"
            )


def _default_response(
    segments: list[dict[str, Any]], interventions: list[str]
) -> ZEVUPLLMResponse:
    effects = []
    for segment in segments:
        for intervention_id in interventions:
            intervention = get_intervention(intervention_id)
            effects.append(
                ZEVUPLLMEffect(
                    segment_id=str(segment["segment_id"]),
                    intervention_id=intervention_id,
                    effect=intervention.default_effect,
                    reasoning=f"Default configured effect for {intervention.name}.",
                    message_recommendation=(
                        f"Use {intervention.name.lower()} messaging for this segment."
                    ),
                )
            )
    return ZEVUPLLMResponse(
        effects=effects,
        wp1_summary=(
            "Default assumptions applied to affordability, awareness, and charging "
            "readiness."
        ),
        wp5_summary="Segment effects are aggregated from configured defaults.",
        wp6_summary="Prioritise segments with high weighted uplift.",
        caveats=["LLM disabled or unavailable; configured default assumptions used."],
    )


def _mock_response(
    segments: list[dict[str, Any]], interventions: list[str]
) -> ZEVUPLLMResponse:
    effects = []
    for segment in segments:
        for intervention_id in interventions:
            intervention = get_intervention(intervention_id)
            effect = intervention.default_effect
            if intervention_id == "price_subsidy" and segment.get(
                "socioeconomic_group"
            ) in {"routine_manual", "other"}:
                effect += 0.035
            if (
                intervention_id == "charging_support"
                and segment.get("charging_readiness_group") == "poor"
            ):
                effect += 0.045
            if intervention_id == "awareness_campaign" and segment.get(
                "age_group"
            ) in {"16_to_24", "25_to_34"}:
                effect += 0.01
            effects.append(
                ZEVUPLLMEffect(
                    segment_id=str(segment["segment_id"]),
                    intervention_id=intervention_id,
                    effect=min(effect, intervention.max_effect),
                    reasoning=f"Mock segment adjustment for {intervention_id}.",
                    message_recommendation=(
                        f"Tailor {intervention_id.replace('_', ' ')} to this segment."
                    ),
                )
            )
    return ZEVUPLLMResponse(
        effects=effects,
        wp1_summary=(
            "Mock mode estimates affordability and charging effects by weighted "
            "segment."
        ),
        wp5_summary="Mock mode aggregates segment effects into market readiness uplift.",
        wp6_summary="Mock mode recommends targeting high-uplift segments first.",
        caveats=["Mock LLM mode used; no external DeepSeek call was made."],
    )
