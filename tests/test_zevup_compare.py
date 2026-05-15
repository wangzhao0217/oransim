from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def test_compare_scotland_segments_returns_wp_sections(monkeypatch):
    from oransim.agents.zevup_compare import compare_scotland_segments
    from oransim.schemas.zevup import ZEVUPCompareRequest

    monkeypatch.setenv("ZEVUP_LLM_MODE", "mock")
    report = compare_scotland_segments(
        ZEVUPCompareRequest(year=2025, segment_limit=20, use_llm=True)
    )

    assert report.market == "scotland"
    assert 0 <= report.baseline_readiness <= 1
    assert report.post_intervention_readiness >= report.baseline_readiness
    assert all(0 <= item.post_feasible_share <= 1 for item in report.segment_results)
    assert report.segment_results
    assert len(report.segment_results) <= 20
    assert {"WP1", "WP5", "WP6"}.issubset(report.wp_report)
    assert report.demographic_rankings["age_group"]
    assert report.llm_status == "mock"


def test_use_llm_false_forces_off_status(monkeypatch):
    from oransim.agents.zevup_compare import compare_scotland_segments
    from oransim.schemas.zevup import ZEVUPCompareRequest

    monkeypatch.setenv("ZEVUP_LLM_MODE", "mock")
    report = compare_scotland_segments(
        ZEVUPCompareRequest(year=2025, segment_limit=10, use_llm=False)
    )

    assert report.llm_status == "off"


def test_unknown_intervention_is_rejected():
    import pytest

    from oransim.agents.zevup_compare import compare_scotland_segments
    from oransim.schemas.zevup import ZEVUPCompareRequest

    with pytest.raises(ValueError, match="Unknown intervention"):
        compare_scotland_segments(
            ZEVUPCompareRequest(interventions=["unknown"], use_llm=False)
        )


def test_segment_limit_only_limits_returned_rows(monkeypatch):
    from oransim.agents import zevup_compare
    from oransim.agents.zevup_compare import compare_scotland_segments
    from oransim.schemas.zevup import (
        ZEVUPCompareRequest,
        ZEVUPIntervention,
        ZEVUPLLMEffect,
        ZEVUPLLMResponse,
    )

    segments = pd.DataFrame(
        [
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
                "baseline_feasible_share": 0.1,
            },
            {
                "segment_id": "scot_seg_0002",
                "age_group": "50_to_64",
                "sex": "male",
                "education_group": "upper_school",
                "socioeconomic_group": "routine_manual",
                "household_type": "one_person_66_plus",
                "dependent_children": "none",
                "accommodation": "flat",
                "household_size": "one_person",
                "car_ownership": "zero",
                "tenure": "social_rented",
                "charging_readiness_group": "poor",
                "household_weight": 200.0,
                "person_weight": 200.0,
                "baseline_feasible_share": 0.2,
            },
            {
                "segment_id": "scot_seg_0003",
                "age_group": "65_plus",
                "sex": "female",
                "education_group": "lower_school",
                "socioeconomic_group": "other",
                "household_type": "other",
                "dependent_children": "none",
                "accommodation": "flat",
                "household_size": "two_person",
                "car_ownership": "one",
                "tenure": "owned",
                "charging_readiness_group": "strong",
                "household_weight": 300.0,
                "person_weight": 600.0,
                "baseline_feasible_share": 0.3,
            },
        ]
    )

    def fake_build_scotland_segments(**kwargs):
        assert kwargs["max_segments"] is None
        return segments.copy()

    def fake_assess_segments(**kwargs):
        assert kwargs["force_off"] is False
        return SimpleNamespace(
            status="mock",
            response=ZEVUPLLMResponse(
                effects=[
                    ZEVUPLLMEffect(
                        segment_id="scot_seg_0001",
                        intervention_id="price_subsidy",
                        effect=0.2,
                        reasoning="Targeted subsidy effect.",
                        message_recommendation="Lead with affordability.",
                    )
                ],
                wp1_summary="WP1",
                wp5_summary="WP5",
                wp6_summary="WP6",
                caveats=["Synthetic partial coverage."],
            ),
        )

    monkeypatch.setenv("ZEVUP_LLM_MODE", "mock")
    monkeypatch.setattr(zevup_compare, "build_scotland_segments", fake_build_scotland_segments)
    monkeypatch.setattr(zevup_compare, "assess_segments", fake_assess_segments)

    report = compare_scotland_segments(
        ZEVUPCompareRequest(segment_limit=1, interventions=["price_subsidy"], use_llm=True)
    )

    assert len(report.segment_results) == 1
    assert report.baseline_readiness == 0.233333
    assert report.post_intervention_readiness == 0.29125
    assert report.overall_uplift == 0.057917
    assert report.segment_results[0].segment_id == "scot_seg_0001"


def test_price_subsidy_summary_reports_proxy_uplift(monkeypatch):
    from oransim.agents.zevup_compare import compare_scotland_segments
    from oransim.schemas.zevup import ZEVUPCompareRequest

    monkeypatch.setenv("ZEVUP_LLM_MODE", "mock")
    report = compare_scotland_segments(
        ZEVUPCompareRequest(
            year=2025,
            segment_limit=10,
            interventions=["price_subsidy"],
            use_llm=True,
        )
    )

    assert report.subsidy_impact_summary["weighted_households_made_feasible_proxy"] >= 0
    assert "proxy" in report.subsidy_impact_summary["note"]


def test_subsidy_summary_is_zero_when_price_subsidy_not_selected(monkeypatch):
    from oransim.agents import zevup_compare
    from oransim.agents.zevup_compare import compare_scotland_segments
    from oransim.schemas.zevup import (
        ZEVUPCompareRequest,
        ZEVUPLLMEffect,
        ZEVUPLLMResponse,
    )

    segments = pd.DataFrame(
        [
            {
                "segment_id": "scot_seg_0001",
                "age_group": "25_to_34",
                "sex": "female",
                "education_group": "upper_school",
                "socioeconomic_group": "routine_manual",
                "household_type": "one_person_under_66",
                "dependent_children": "none",
                "accommodation": "flat",
                "household_size": "one_person",
                "car_ownership": "zero",
                "tenure": "private_rented",
                "charging_readiness_group": "poor",
                "household_weight": 100.0,
                "person_weight": 100.0,
                "baseline_feasible_share": 0.2,
            }
        ]
    )

    def fake_build_scotland_segments(**kwargs):
        assert kwargs["max_segments"] is None
        return segments.copy()

    def fake_assess_segments(**kwargs):
        return SimpleNamespace(
            status="mock",
            response=ZEVUPLLMResponse(
                effects=[
                    ZEVUPLLMEffect(
                        segment_id="scot_seg_0001",
                        intervention_id="awareness_campaign",
                        effect=0.08,
                        reasoning="Awareness can still lift lower proxy groups.",
                        message_recommendation="Lead with practical use cases.",
                    )
                ],
                wp1_summary="WP1",
                wp5_summary="WP5",
                wp6_summary="WP6",
                caveats=[],
            ),
        )

    monkeypatch.setattr(zevup_compare, "build_scotland_segments", fake_build_scotland_segments)
    monkeypatch.setattr(zevup_compare, "assess_segments", fake_assess_segments)

    report = compare_scotland_segments(
        ZEVUPCompareRequest(interventions=["awareness_campaign"], use_llm=True)
    )

    assert report.overall_uplift > 0
    assert (
        report.subsidy_impact_summary["weighted_households_made_feasible_proxy"]
        == 0.0
    )
    assert "price_subsidy was not selected" in report.subsidy_impact_summary["note"]


def test_subsidy_summary_uses_price_subsidy_marginal_lift(monkeypatch):
    from oransim.agents import zevup_compare
    from oransim.agents.zevup_compare import compare_scotland_segments
    from oransim.schemas.zevup import (
        ZEVUPCompareRequest,
        ZEVUPLLMEffect,
        ZEVUPLLMResponse,
    )

    segments = pd.DataFrame(
        [
            {
                "segment_id": "scot_seg_0001",
                "age_group": "25_to_34",
                "sex": "female",
                "education_group": "upper_school",
                "socioeconomic_group": "routine_manual",
                "household_type": "one_person_under_66",
                "dependent_children": "none",
                "accommodation": "flat",
                "household_size": "one_person",
                "car_ownership": "zero",
                "tenure": "private_rented",
                "charging_readiness_group": "poor",
                "household_weight": 100.0,
                "person_weight": 100.0,
                "baseline_feasible_share": 0.2,
            }
        ]
    )

    def fake_build_scotland_segments(**kwargs):
        assert kwargs["max_segments"] is None
        return segments.copy()

    def fake_assess_segments(**kwargs):
        return SimpleNamespace(
            status="mock",
            response=ZEVUPLLMResponse(
                effects=[
                    ZEVUPLLMEffect(
                        segment_id="scot_seg_0001",
                        intervention_id="awareness_campaign",
                        effect=0.08,
                        reasoning="Awareness lifts this segment first.",
                        message_recommendation="Lead with practical use cases.",
                    ),
                    ZEVUPLLMEffect(
                        segment_id="scot_seg_0001",
                        intervention_id="price_subsidy",
                        effect=0.10,
                        reasoning="Subsidy adds a marginal affordability lift.",
                        message_recommendation="Lead with affordability.",
                    ),
                ],
                wp1_summary="WP1",
                wp5_summary="WP5",
                wp6_summary="WP6",
                caveats=[],
            ),
        )

    monkeypatch.setattr(
        zevup_compare,
        "build_scotland_segments",
        fake_build_scotland_segments,
    )
    monkeypatch.setattr(zevup_compare, "assess_segments", fake_assess_segments)

    report = compare_scotland_segments(
        ZEVUPCompareRequest(
            interventions=["awareness_campaign", "price_subsidy"],
            use_llm=True,
        )
    )

    assert report.overall_uplift == 0.1376
    assert (
        report.subsidy_impact_summary["weighted_households_made_feasible_proxy"]
        == 7.36
    )
    assert "marginal price-subsidy lift" in report.subsidy_impact_summary["note"]
