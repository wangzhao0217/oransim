"""Frugal EV persona reaction prompt and LLM/mock execution."""

from __future__ import annotations

import asyncio
from typing import Any

from .soul_llm import MODEL, TIMEOUT, call_llm_json_with_retry, llm_available

FRUGAL_EV_PERSONA_PROMPT = """\
You are simulating a real consumer reacting to a new frugal electric vehicle.

Consumer profile:
- Segment: {segment}
- Income level: {income_level}
- Main need: {main_need}
- Concerns: {concerns}
- Decision drivers: {decision_drivers}

EV concept:
- Name: {concept_name}
- Price: GBP {price:,}
- Range: {range_km} km
- Battery warranty: {battery_warranty_years} years
- Positioning: {positioning}
- Message: {message}
- Features: {features}

Return ONLY valid JSON with these exact keys:
{{
  "purchase_intent_score": <float 0-1>,
  "trust_score": <float 0-1>,
  "price_acceptance_score": <float 0-1>,
  "range_confidence_score": <float 0-1>,
  "charging_confidence_score": <float 0-1>,
  "top_positive_reason": "<string>",
  "top_objection": "<string>",
  "what_would_change_my_mind": "<string>",
  "likely_social_comment": "<string>"
}}
"""

_SYSTEM = (
    "You are a market-research persona simulator. Respond as the named consumer "
    "segment, grounded in the supplied profile and EV concept. Return strict JSON only."
)

_SCORE_KEYS = (
    "purchase_intent_score",
    "trust_score",
    "price_acceptance_score",
    "range_confidence_score",
    "charging_confidence_score",
)
_TEXT_KEYS = (
    "top_positive_reason",
    "top_objection",
    "what_would_change_my_mind",
    "likely_social_comment",
)
_MAX_TOKENS = 1500
_TIMEOUT = max(TIMEOUT, 90.0)


async def simulate_persona_reaction(persona: dict[str, Any], concept: dict[str, Any]) -> dict:
    """Call the configured LLM for one persona/concept reaction, or use a local mock."""
    prompt = FRUGAL_EV_PERSONA_PROMPT.format(
        segment=persona["segment"],
        income_level=persona["income_level"],
        main_need=persona["main_need"],
        concerns=", ".join(persona["concerns"]),
        decision_drivers=", ".join(persona["decision_drivers"]),
        concept_name=concept["name"],
        price=concept["price"],
        range_km=concept["range_km"],
        battery_warranty_years=concept["battery_warranty_years"],
        positioning=concept["positioning"],
        message=concept["message"],
        features=", ".join(concept["features"]),
    )
    if llm_available():
        body = {
            "model": MODEL,
            "temperature": 0.4,
            "max_tokens": _MAX_TOKENS,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            parsed, _usage = await asyncio.to_thread(
                call_llm_json_with_retry,
                body,
                max_retries=1,
                use_stream=False,
                timeout=_TIMEOUT,
            )
            return _normalize_response(parsed, persona, concept)
        except Exception:
            pass
    return _mock_persona_reaction(persona, concept)


def _normalize_response(
    raw: dict[str, Any],
    persona: dict[str, Any],
    concept: dict[str, Any],
) -> dict:
    fallback = _mock_persona_reaction(persona, concept)
    normalized = dict(fallback)
    for key in _SCORE_KEYS:
        normalized[key] = _score(raw.get(key), fallback[key])
    for key in _TEXT_KEYS:
        value = raw.get(key)
        normalized[key] = str(value).strip() if value else fallback[key]
    return normalized


def _score(value: Any, fallback: float) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 3)
    except (TypeError, ValueError):
        return fallback


def _mock_persona_reaction(persona: dict[str, Any], concept: dict[str, Any]) -> dict:
    segment = persona["segment"]
    price = float(concept["price"])
    range_km = float(concept["range_km"])
    warranty_years = float(concept["battery_warranty_years"])
    features = " ".join(concept.get("features", [])).lower()

    price_score = _clamp(1.05 - (price - 7_999) / 14_000)
    range_score = _clamp((range_km - 120) / 220)
    warranty_score = _clamp(warranty_years / 8)
    charging_score = 0.55 + (0.1 if "home charging" in features else 0.0)
    trust_score = _clamp(0.45 + warranty_score * 0.35)

    segment_adjustments = {
        "urban_commuter": (0.12, 0.05, "compact daily trips and low running cost"),
        "first_time_car_buyer": (0.08, -0.02, "a monthly payment that competes with used cars"),
        "delivery_driver": (0.02, -0.06, "lower cost per kilometre for work"),
        "rural_budget_family": (-0.08, -0.15, "warranty and household running-cost savings"),
        "ev_skeptic": (-0.12, -0.08, "clear proof that battery risk is covered"),
    }
    intent_delta, charging_delta, positive_reason = segment_adjustments.get(
        segment, (0.0, 0.0, "low ownership cost")
    )
    charging_score = _clamp(charging_score + charging_delta)
    purchase_intent = _clamp(
        price_score * 0.35
        + range_score * 0.25
        + trust_score * 0.2
        + charging_score * 0.2
        + intent_delta
    )

    objection = _top_objection(segment, price, range_km, warranty_years)
    return {
        "purchase_intent_score": round(purchase_intent, 3),
        "trust_score": round(trust_score, 3),
        "price_acceptance_score": round(price_score, 3),
        "range_confidence_score": round(range_score, 3),
        "charging_confidence_score": round(charging_score, 3),
        "top_positive_reason": positive_reason,
        "top_objection": objection,
        "what_would_change_my_mind": _mind_changer(segment),
        "likely_social_comment": _social_comment(segment, concept),
    }


def _top_objection(segment: str, price: float, range_km: float, warranty_years: float) -> str:
    if segment == "rural_budget_family" and range_km < 250:
        return "Range and public charging may not cover rural family use."
    if segment == "delivery_driver" and range_km < 240:
        return "Charging downtime could interrupt paid work."
    if segment == "first_time_car_buyer" and price > 12_000:
        return "Monthly payments may still feel high for a first car."
    if segment == "ev_skeptic" and warranty_years < 8:
        return "Battery durability proof is not strong enough yet."
    if segment == "urban_commuter":
        return "Apartment charging and parking access are uncertain."
    return "Resale value and real-world battery life remain uncertain."


def _mind_changer(segment: str) -> str:
    by_segment = {
        "urban_commuter": "Reliable kerbside or workplace charging near home.",
        "first_time_car_buyer": "A low monthly payment with insurance and battery warranty included.",
        "delivery_driver": "Evidence that daily routes can be completed with fast charging backup.",
        "rural_budget_family": "A longer real-world range test and accessible local service network.",
        "ev_skeptic": "Independent owner reviews and a transferable long battery warranty.",
    }
    return by_segment.get(segment, "More proof from real owners.")


def _social_comment(segment: str, concept: dict[str, Any]) -> str:
    comments = {
        "urban_commuter": f"{concept['name']} sounds useful if I can charge it near my flat.",
        "first_time_car_buyer": "If the finance deal is cheap enough, I would compare it with used petrol cars.",
        "delivery_driver": "I care less about the badge and more about daily cost and downtime.",
        "rural_budget_family": "The idea is good, but range and charging outside the city would decide it.",
        "ev_skeptic": "Cheap is attractive, but I would wait for real battery reliability proof.",
    }
    return comments.get(
        segment, f"I would consider {concept['name']} if ownership costs are clear."
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
