"""Comparison engine for Scotland ZEV-UP segment interventions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..data.zevup_segments import build_scotland_segments
from ..data.zevup_static import (
    get_intervention,
    get_market,
    get_vehicle_concept,
    interventions,
)
from ..schemas.zevup import (
    ZEVUPCompareRequest,
    ZEVUPCompareResponse,
    ZEVUPIntervention,
    ZEVUPSegmentResult,
)
from .zevup_llm import assess_segments

_RETURNED_COLUMNS = [
    "segment_id",
    "age_group",
    "sex",
    "education_group",
    "socioeconomic_group",
    "household_type",
    "dependent_children",
    "accommodation",
    "household_size",
    "car_ownership",
    "tenure",
    "charging_readiness_group",
    "household_weight",
    "person_weight",
    "baseline_feasible_share",
]

_DIMENSIONS = [
    "age_group",
    "sex",
    "education_group",
    "socioeconomic_group",
    "household_type",
    "car_ownership",
    "charging_readiness_group",
]

_LOWER_SOCIOECONOMIC_GROUPS = {"routine_manual", "other"}


def compare_scotland_segments(req: ZEVUPCompareRequest) -> ZEVUPCompareResponse:
    """Compare selected interventions across Scotland ZEV-UP segments."""
    _validate_interventions(req.interventions)
    market = get_market("scotland")
    concept = get_vehicle_concept(req.vehicle_concept_id)
    selected_interventions = [get_intervention(item) for item in req.interventions]

    all_segments = build_scotland_segments(
        year=req.year,
        scenario=req.baseline_scenario,
        max_segments=None,
    )
    llm_segments = all_segments.head(req.segment_limit).copy()
    segment_records = _segment_records(llm_segments)
    llm = assess_segments(
        segments=segment_records,
        vehicle_concept=concept.model_dump(),
        interventions=req.interventions,
        year=req.year,
        baseline_scenario=req.baseline_scenario,
        force_off=not req.use_llm,
    )

    effect_lookup = _effect_lookup(llm.response.effects, selected_interventions)
    results = _apply_effects(all_segments, selected_interventions, effect_lookup)
    returned_results = (
        results.sort_values("weighted_household_uplift", ascending=False)
        .head(req.segment_limit)
        .copy()
    )

    baseline = _weighted_mean(results, "baseline_feasible_share")
    post = _weighted_mean(results, "post_feasible_share")
    return ZEVUPCompareResponse(
        market="scotland",
        year=req.year,
        baseline_scenario=req.baseline_scenario,
        vehicle_concept=concept,
        interventions=selected_interventions,
        baseline_readiness=round(baseline, 6),
        post_intervention_readiness=round(post, 6),
        overall_uplift=round(post - baseline, 6),
        segment_results=[
            ZEVUPSegmentResult(**record)
            for record in returned_results[_RETURNED_COLUMNS + [
                "post_feasible_share",
                "uplift",
                "weighted_household_uplift",
                "top_intervention",
                "message_recommendation",
            ]].to_dict("records")
        ],
        demographic_rankings=_rankings(results),
        subsidy_impact_summary=_subsidy_summary(results, selected_interventions),
        wp_report={
            "WP1": llm.response.wp1_summary,
            "WP5": llm.response.wp5_summary,
            "WP6": llm.response.wp6_summary,
        },
        llm_status=llm.status,
        caveats=_merge_caveats(market.caveats, llm.response.caveats),
        provenance={
            "household_feasibility_path": market.household_feasibility_path,
            "census_dir": market.census_dir,
            "segment_method": "census-constrained weighted household allocation",
        },
    )


def _validate_interventions(ids: list[str]) -> None:
    known = set(interventions())
    unknown = sorted(set(ids) - known)
    if unknown:
        raise ValueError(f"Unknown intervention(s): {', '.join(unknown)}")


def _segment_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df[_RETURNED_COLUMNS].to_dict("records")


def _effect_lookup(
    effects: Iterable[Any],
    selected: list[ZEVUPIntervention],
) -> dict[tuple[str, str], dict[str, Any]]:
    selected_by_id = {item.intervention_id: item for item in selected}
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for effect in effects:
        intervention = selected_by_id.get(effect.intervention_id)
        if intervention is None:
            continue
        bounded = max(
            intervention.min_effect,
            min(intervention.max_effect, float(effect.effect)),
        )
        lookup[(str(effect.segment_id), effect.intervention_id)] = {
            "effect": bounded,
            "message_recommendation": str(effect.message_recommendation),
        }
    return lookup


def _apply_effects(
    segments: pd.DataFrame,
    selected: list[ZEVUPIntervention],
    effect_lookup: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in segments[_RETURNED_COLUMNS].to_dict("records"):
        current = float(record["baseline_feasible_share"])
        household_weight = float(record["household_weight"])
        uplift_by_intervention: dict[str, float] = {}
        message_by_intervention: dict[str, str] = {}
        intervention_lifts: dict[str, float] = {}
        for intervention in selected:
            lookup = effect_lookup.get((record["segment_id"], intervention.intervention_id))
            bounded_effect = (
                float(lookup["effect"]) if lookup is not None else intervention.default_effect
            )
            lift = (1.0 - current) * bounded_effect
            current = min(1.0, max(0.0, current + lift))
            uplift_by_intervention[intervention.intervention_id] = lift
            intervention_lifts[f"lift_{intervention.intervention_id}"] = round(lift, 6)
            intervention_lifts[f"weighted_lift_{intervention.intervention_id}"] = round(
                lift * household_weight,
                6,
            )
            message_by_intervention[intervention.intervention_id] = (
                str(lookup["message_recommendation"])
                if lookup is not None
                else intervention.description
            )

        # This is the largest marginal lift within the selected intervention sequence.
        top_intervention = max(
            uplift_by_intervention,
            key=uplift_by_intervention.get,
        )
        post_share = min(1.0, max(0.0, current))
        baseline_share = float(record["baseline_feasible_share"])
        uplift = post_share - baseline_share
        rows.append(
            {
                **record,
                "post_feasible_share": round(post_share, 6),
                "uplift": round(uplift, 6),
                "weighted_household_uplift": round(
                    uplift * household_weight,
                    6,
                ),
                "top_intervention": top_intervention,
                "message_recommendation": message_by_intervention[top_intervention],
                **intervention_lifts,
            }
        )
    return pd.DataFrame(rows)


def _weighted_mean(df: pd.DataFrame, value_col: str) -> float:
    weights = df["household_weight"].astype(float)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return 0.0
    return float((df[value_col].astype(float) * weights).sum() / total_weight)


def _rankings(df: pd.DataFrame) -> dict[str, list[dict[str, float | str]]]:
    rankings: dict[str, list[dict[str, float | str]]] = {}
    for dimension in _DIMENSIONS:
        grouped = (
            df.groupby(dimension, as_index=False, observed=False)
            .agg(
                weighted_household_uplift=("weighted_household_uplift", "sum"),
                household_weight=("household_weight", "sum"),
            )
            .sort_values(
                ["weighted_household_uplift", "household_weight", dimension],
                ascending=[False, False, True],
                kind="mergesort",
            )
            .head(5)
        )
        rankings[dimension] = [
            {
                "value": str(row[dimension]),
                "weighted_household_uplift": round(
                    float(row["weighted_household_uplift"]),
                    6,
                ),
                "household_weight": round(float(row["household_weight"]), 6),
            }
            for row in grouped.to_dict("records")
        ]
    return rankings


def _subsidy_summary(
    df: pd.DataFrame,
    selected: list[ZEVUPIntervention],
) -> dict[str, float | str]:
    group_definition = "routine_manual or other NS-SeC proxy groups"
    if "price_subsidy" not in {item.intervention_id for item in selected}:
        return {
            "group_definition": group_definition,
            "weighted_households_made_feasible_proxy": 0.0,
            "note": (
                "price_subsidy was not selected; no subsidy-specific uplift "
                "is reported."
            ),
        }

    lower = df[df["socioeconomic_group"].isin(_LOWER_SOCIOECONOMIC_GROUPS)]
    return {
        "group_definition": group_definition,
        "weighted_households_made_feasible_proxy": round(
            float(lower["weighted_lift_price_subsidy"].sum()),
            6,
        ),
        "note": (
            "This is the marginal price-subsidy lift among lower socioeconomic "
            "proxy groups; direct income is not observed."
        ),
    }


def _merge_caveats(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged
