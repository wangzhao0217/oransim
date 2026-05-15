"""Schemas for the ZEV-UP Scotland segment simulator."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ZEVUPMarket(BaseModel):
    market_id: str
    display_name: str
    country: str
    first_supported_year: int
    last_supported_year: int
    default_scenario: str
    census_dir: str
    household_feasibility_path: str
    study3_long_path: str
    t54_penetration_path: str
    t55_entry_year_path: str
    caveats: list[str] = Field(default_factory=list)


class ZEVUPVehicleConcept(BaseModel):
    concept_id: str
    name: str
    vehicle_class: str
    seats: int
    primary_use_case: str
    range_class: str
    charging_assumption: str
    battery_assumption: str
    business_model_compatibility: list[str]


class ZEVUPIntervention(BaseModel):
    intervention_id: str
    name: str
    description: str
    default_effect: float = Field(ge=0.0, le=1.0)
    min_effect: float = Field(ge=0.0, le=1.0)
    max_effect: float = Field(ge=0.0, le=1.0)
    primary_wp: str
    effect_targets: list[str]


class ZEVUPCompareRequest(BaseModel):
    year: int = 2025
    baseline_scenario: str = "neutral_lr_central"
    vehicle_concept_id: str = "zevup_l7e_passenger_2_seat"
    interventions: list[str] = Field(
        default_factory=lambda: [
            "awareness_campaign",
            "price_subsidy",
            "charging_support",
        ],
        min_length=1,
    )
    segment_limit: int = Field(default=80, ge=1, le=500)
    use_llm: bool = True


class ZEVUPSegmentSummary(BaseModel):
    segment_id: str
    age_group: str
    sex: str
    education_group: str
    socioeconomic_group: str
    household_type: str
    dependent_children: str
    accommodation: str
    household_size: str
    car_ownership: str
    tenure: str
    charging_readiness_group: str
    household_weight: float = Field(ge=0.0)
    person_weight: float = Field(ge=0.0)
    baseline_feasible_share: float = Field(ge=0.0, le=1.0)


class ZEVUPLLMEffect(BaseModel):
    segment_id: str
    intervention_id: str
    effect: float = Field(ge=0.0, le=1.0)
    reasoning: str
    message_recommendation: str


class ZEVUPLLMResponse(BaseModel):
    effects: list[ZEVUPLLMEffect]
    wp1_summary: str
    wp5_summary: str
    wp6_summary: str
    caveats: list[str] = Field(default_factory=list)


class ZEVUPSegmentResult(ZEVUPSegmentSummary):
    post_feasible_share: float = Field(ge=0.0, le=1.0)
    uplift: float
    weighted_household_uplift: float
    top_intervention: str
    message_recommendation: str


class ZEVUPCompareResponse(BaseModel):
    market: str
    year: int
    baseline_scenario: str
    vehicle_concept: ZEVUPVehicleConcept
    interventions: list[ZEVUPIntervention]
    baseline_readiness: float = Field(ge=0.0, le=1.0)
    post_intervention_readiness: float = Field(ge=0.0, le=1.0)
    overall_uplift: float
    segment_results: list[ZEVUPSegmentResult]
    demographic_rankings: dict[str, list[dict[str, float | str]]]
    subsidy_impact_summary: dict[str, float | str]
    wp_report: dict[str, str]
    llm_status: str
    caveats: list[str]
    provenance: dict[str, str]
