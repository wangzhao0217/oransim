"""Schemas for the frugal EV opinion simulator."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SegmentOpinion(BaseModel):
    segment: str
    purchase_intent_score: float = Field(ge=0.0, le=1.0)
    trust_score: float = Field(ge=0.0, le=1.0)
    price_acceptance_score: float = Field(ge=0.0, le=1.0)
    range_confidence_score: float = Field(ge=0.0, le=1.0)
    charging_confidence_score: float = Field(ge=0.0, le=1.0)
    top_positive_reason: str
    top_objection: str
    what_would_change_my_mind: str
    likely_social_comment: str


class FrugalEVSimulateRequest(BaseModel):
    concept_id: str
    target_market: str = "UK"
    segments: list[str] | None = None


class FrugalEVOpinionReport(BaseModel):
    concept_name: str
    target_market: str
    overall_acceptance_score: float = Field(ge=0.0, le=1.0)
    segment_opinions: list[SegmentOpinion]
    strongest_segments: list[str]
    weakest_segments: list[str]
    top_objection_themes: list[str]
    top_message_hooks: list[str]
    verdict: str
