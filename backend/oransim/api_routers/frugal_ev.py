"""Frugal EV market-entry opinion simulator router."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from fastapi import APIRouter, HTTPException

from ..agents.frugal_ev_persona_prompt import simulate_persona_reaction
from ..schemas.frugal_ev_opinion import (
    FrugalEVOpinionReport,
    FrugalEVSimulateRequest,
    SegmentOpinion,
)

router = APIRouter(prefix="/api/frugal-ev", tags=["frugal-ev"])

_DATA = Path(__file__).resolve().parents[3] / "data"
_PERSONAS = json.loads((_DATA / "personas/frugal_ev_personas.json").read_text(encoding="utf-8"))
_CONCEPTS = {
    concept["concept_id"]: concept
    for concept in json.loads(
        (_DATA / "concepts/frugal_ev_concepts.json").read_text(encoding="utf-8")
    )
}


@router.post("/simulate", response_model=FrugalEVOpinionReport)
async def simulate(req: FrugalEVSimulateRequest):
    concept = _CONCEPTS.get(req.concept_id)
    if not concept:
        raise HTTPException(status_code=404, detail=f"concept_id {req.concept_id!r} not found")

    target_segments = req.segments or [persona["segment"] for persona in _PERSONAS]
    unknown = sorted(set(target_segments) - {persona["segment"] for persona in _PERSONAS})
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown segment(s): {', '.join(unknown)}")

    personas = [persona for persona in _PERSONAS if persona["segment"] in target_segments]
    if not personas:
        raise HTTPException(status_code=400, detail="at least one segment is required")

    opinions = []
    for persona in personas:
        raw = await simulate_persona_reaction(persona, concept)
        opinions.append(SegmentOpinion(segment=persona["segment"], **raw))

    sorted_by_intent = sorted(
        opinions, key=lambda opinion: opinion.purchase_intent_score, reverse=True
    )
    overall = mean(opinion.purchase_intent_score for opinion in opinions)

    return FrugalEVOpinionReport(
        concept_name=concept["name"],
        target_market=req.target_market,
        overall_acceptance_score=round(overall, 3),
        segment_opinions=opinions,
        strongest_segments=[opinion.segment for opinion in sorted_by_intent[:2]],
        weakest_segments=[opinion.segment for opinion in sorted_by_intent[-2:]],
        top_objection_themes=_unique([opinion.top_objection for opinion in opinions])[:5],
        top_message_hooks=[opinion.top_positive_reason for opinion in sorted_by_intent[:3]],
        verdict=_verdict(overall, sorted_by_intent),
    )


def _unique(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _verdict(overall: float, opinions: list[SegmentOpinion]) -> str:
    strong = [opinion.segment for opinion in opinions if opinion.purchase_intent_score >= 0.6]
    weak = [opinion.segment for opinion in opinions if opinion.purchase_intent_score < 0.4]
    parts = []
    if strong:
        parts.append(f"Promising for {', '.join(strong)}.")
    if weak:
        parts.append(f"Weak for {', '.join(weak)}; address {opinions[-1].top_objection}.")
    if not parts:
        parts.append("Mixed reception across segments.")
    parts.append(f"Overall acceptance score is {overall:.3f}.")
    return " ".join(parts)
