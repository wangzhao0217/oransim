from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _household_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "oa_code": ["OA1", "OA1"],
            "household_id": ["OA1-1", "OA1-2"],
            "nssec_group": ["routine_manual", "managerial_professional"],
            "accommodation": ["flat", "whole_house"],
            "household_size": ["one_person", "two_person"],
            "car_ownership": ["zero", "one"],
            "tenure": ["social_rented", "owned"],
            "trip_suit_weight": [0.4, 0.7],
            "household_weight": [2.0, 8.0],
            "feasible_share": [0.2, 0.2],
            "AP_i": [0.5, 0.8],
            "G_it": [0.3, 0.6],
            "A_it": [0.4, 0.9],
            "feasible_h": [0.08, 0.25],
            "scenario": ["test_scenario", "test_scenario"],
            "year": [2025, 2025],
            "adopts": [0, 1],
            "trip_suit_i": [0.55, 0.55],
        }
    )


def _manifest(tmp_path: Path, source_path: Path) -> Path:
    path = tmp_path / "scenario_slices.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "supported_slices": [
                    {
                        "scenario": "test_scenario",
                        "year": 2025,
                        "household_feasibility_path": str(source_path),
                        "status": "available",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_requires_generation_slice_fails_with_upstream_command(tmp_path: Path):
    from oransim.data.zevup_personas import resolve_household_feasibility_path

    missing = tmp_path / "missing.parquet"
    manifest = tmp_path / "scenario_slices.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "supported_slices": [
                    {
                        "scenario": "future",
                        "year": 2040,
                        "household_feasibility_path": str(missing),
                        "status": "requires_generation",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="generate_household_feasibility_slice"):
        resolve_household_feasibility_path("future", 2040, manifest)


def test_build_personas_preserves_lineage_and_imputes_demographics(tmp_path: Path):
    from oransim.data.zevup_personas import build_personas

    source_path = tmp_path / "household.parquet"
    _household_source().to_parquet(source_path, index=False)
    manifest = _manifest(tmp_path, source_path)
    demo = pd.DataFrame(
        {
            "oa_code": ["OA1", "OA1"],
            "age_group": ["25_to_34", "65_plus"],
            "sex": ["female", "male"],
            "education_group": ["degree_or_above", "lower_school"],
            "share": [0.7, 0.3],
        }
    )
    household_types = pd.DataFrame(
        {
            "oa_code": ["OA1", "OA1"],
            "household_type": ["one_person_66_plus", "couple_no_children"],
            "dependent_children": ["none", "none"],
            "share": [0.2, 0.8],
        }
    )
    acceptance_config = {
        "base": 0.65,
        "bounds": [0.4, 0.95],
        "age_group_adjustments": {"65_plus": -0.01},
        "education_group_adjustments": {"lower_school": -0.01},
        "socioeconomic_group_adjustments": {"routine_manual": -0.01},
        "household_adjustments": {"accommodation": {}, "car_ownership": {}, "tenure": {}},
    }

    personas = build_personas(
        scenario="test_scenario",
        year=2025,
        manifest_path=manifest,
        demographic_distribution=demo,
        household_type_distribution=household_types,
        acceptance_config=acceptance_config,
    )

    assert len(personas) == 2
    assert personas["persona_id"].str.startswith("scotland__test_scenario__2025__").all()
    assert set(personas["oa_feasible_share"]) == {0.2}
    assert set(personas["oa_trip_suit_i"]) == {0.55}
    assert personas["acceptance_gate_h"].between(0.4, 0.95).all()
    one_person = personas[personas["household_size"].eq("one_person")].iloc[0]
    assert one_person["household_type"] == "one_person_66_plus"
    assert one_person["age_group"] == "65_plus"

