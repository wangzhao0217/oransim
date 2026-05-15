from __future__ import annotations

from functools import lru_cache
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

ROOT = Path(__file__).parent.parent


@lru_cache(maxsize=1)
def _full_segments() -> pd.DataFrame:
    from oransim.data.zevup_segments import build_scotland_segments

    return build_scotland_segments(
        year=2025,
        scenario="neutral_lr_central",
        max_segments=None,
    )


def test_build_scotland_segments_returns_weighted_rows():
    from oransim.data.zevup_segments import build_scotland_segments

    df = build_scotland_segments(
        year=2025,
        scenario="neutral_lr_central",
        max_segments=120,
    )

    required = {
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
    }
    assert required.issubset(df.columns)
    assert len(df) > 0
    assert len(df) <= 120
    assert df["household_weight"].sum() > 0
    assert df["person_weight"].sum() > 0
    assert df["baseline_feasible_share"].between(0, 1).all()


def test_segment_builder_rejects_missing_scenario():
    from oransim.data.zevup_segments import build_scotland_segments

    with pytest.raises(ValueError, match="No household feasibility rows"):
        build_scotland_segments(
            year=2025,
            scenario="missing_scenario",
            max_segments=10,
        )


def test_required_source_manifest_includes_spec_files():
    from oransim.data.zevup_segments import required_scotland_source_paths

    names = {path.name for path in required_scotland_source_paths()}
    expected = {
        "UV102b - Age (20) by sex.csv",
        "UV501b - Highest level of qualification by age (5).csv",
        "MV104 - Household composition - households (10) by dependent children in household (4).csv",
        "MV407 - Number of cars or vans by accomodation type - households (3) by household size (4).csv",
        "MV609 - National Statistics Socio-economic Classification (NS-SeC) (9) by household tenure - People.csv",
        "UV405 - Car or van availability.csv",
        "outputarea2022_usualresidentpopulation.csv",
        "household_feasibility.parquet",
        "feasible_market_long.parquet",
        "zevup_t54_penetration.csv",
        "zevup_t55_entry_year.csv",
    }
    assert expected.issubset(names)


def test_unlimited_segments_include_metadata_and_more_rows():
    from oransim.data.zevup_segments import build_scotland_segments

    limited = build_scotland_segments(
        year=2025,
        scenario="neutral_lr_central",
        max_segments=10,
    )
    full = _full_segments()

    assert len(full) > len(limited)
    assert {"baseline_adopted_weight", "market", "scenario", "year"}.issubset(
        full.columns
    )
    assert set(full["market"]) == {"scotland"}
    assert set(full["scenario"]) == {"neutral_lr_central"}
    assert set(full["year"]) == {2025}


def test_unlimited_segments_conserve_household_weight():
    full = _full_segments()
    source = pd.read_parquet(
        ROOT.parent.parent
        / "output"
        / "microsimulation"
        / "household_feasibility.parquet",
        columns=["year", "scenario", "household_weight"],
    )
    source = source[
        (source["year"].astype(int) == 2025)
        & (source["scenario"].astype(str) == "neutral_lr_central")
    ]

    assert full["household_weight"].sum() == pytest.approx(
        source["household_weight"].sum(),
        rel=1e-9,
        abs=1e-6,
    )


def test_segments_do_not_include_impossible_household_combinations():
    full = _full_segments()

    one_person = full["household_type"].str.startswith("one_person")
    assert (full.loc[one_person, "household_size"] == "one_person").all()

    dependent_couple_or_other = full["household_type"].isin(
        ["couple_with_dependent_children", "other_with_dependent_children"]
    )
    assert (
        full.loc[dependent_couple_or_other, "household_size"] == "three_plus"
    ).all()

    lone_parent_dependent = (
        full["household_type"] == "lone_parent_with_dependent_children"
    )
    assert full.loc[lone_parent_dependent, "household_size"].isin(
        ["two_person", "three_plus"]
    ).all()

    other_non_one_person = (
        ~one_person
        & ~dependent_couple_or_other
        & ~lone_parent_dependent
    )
    assert full.loc[other_non_one_person, "household_size"].isin(
        ["two_person", "three_plus"]
    ).all()


def test_base_household_segments_rejects_out_of_range_feasible_h(monkeypatch):
    from oransim.data import zevup_segments

    source = pd.DataFrame(
        {
            "year": [2025],
            "scenario": ["neutral_lr_central"],
            "nssec_group": ["routine_manual"],
            "accommodation": ["flat"],
            "household_size": ["one_person"],
            "car_ownership": ["zero"],
            "tenure": ["social_rented"],
            "household_weight": [1.0],
            "feasible_h": [1.2],
        }
    )
    monkeypatch.setattr(zevup_segments.pd, "read_parquet", lambda *_args, **_kw: source)

    with pytest.raises(ValueError, match="feasible_h"):
        zevup_segments._base_household_segments(
            year=2025,
            scenario="neutral_lr_central",
        )


def test_census_column_validation_names_source_and_missing_columns(monkeypatch):
    from oransim.data import zevup_segments

    monkeypatch.setattr(
        zevup_segments,
        "_read_census",
        lambda _name: pd.DataFrame({"code": ["S00135307"]}),
    )

    with pytest.raises(
        ValueError,
        match=r"Required columns missing from UV102b - Age \(20\) by sex\.csv",
    ):
        zevup_segments._sex_age_counts()


def test_build_segments_rejects_unknown_base_household_size_before_filtering(
    monkeypatch,
):
    from oransim.data import zevup_segments

    base = pd.DataFrame(
        {
            "socioeconomic_group": ["routine_manual"],
            "accommodation": ["flat"],
            "household_size": ["four_plus"],
            "car_ownership": ["zero"],
            "tenure": ["social_rented"],
            "charging_readiness_group": ["strong"],
            "household_weight": [1.0],
            "weighted_feasible": [0.5],
            "baseline_feasible_share": [0.5],
        }
    )
    monkeypatch.setattr(zevup_segments, "_ensure_required_sources", lambda: None)
    monkeypatch.setattr(zevup_segments, "_base_household_segments", lambda *_args: base)

    with pytest.raises(ValueError, match="Unknown household_size value\\(s\\): four_plus"):
        zevup_segments.build_scotland_segments(
            year=2025,
            scenario="neutral_lr_central",
            max_segments=None,
        )
