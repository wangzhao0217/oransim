"""Build Scotland ZEV-UP weighted segment cells."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .zevup_static import get_market

REPO_ROOT = Path(__file__).resolve().parents[5]

REQUIRED_CENSUS_FILES = [
    "UV102b - Age (20) by sex.csv",
    "UV501b - Highest level of qualification by age (5).csv",
    "MV104 - Household composition - households (10) by dependent children in household (4).csv",
    "MV407 - Number of cars or vans by accomodation type - households (3) by household size (4).csv",
    "MV609 - National Statistics Socio-economic Classification (NS-SeC) (9) by household tenure - People.csv",
    "UV405 - Car or van availability.csv",
    "outputarea2022_usualresidentpopulation.csv",
]

REQUIRED_MODEL_PATH_FIELDS = [
    "household_feasibility_path",
    "study3_long_path",
    "t54_penetration_path",
    "t55_entry_year_path",
]

ALLOWED_HOUSEHOLD_SIZES = {"one_person", "two_person", "three_plus"}


def required_scotland_source_paths() -> list[Path]:
    """Return every Scotland source path required by the first ZEV-UP slice."""
    market = get_market("scotland")
    census_dir = REPO_ROOT / market.census_dir
    paths = [census_dir / name for name in REQUIRED_CENSUS_FILES]
    paths.extend(
        REPO_ROOT / str(getattr(market, field))
        for field in REQUIRED_MODEL_PATH_FIELDS
    )
    return paths


def _ensure_required_sources() -> None:
    missing = [path for path in required_scotland_source_paths() if not path.exists()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Required Scotland ZEV-UP source file(s) missing. "
            "Regenerate the Scotland census cleanup and Study 3 "
            "microsimulation outputs:\n"
            f"{formatted}"
        )


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert Scotland census counts to numeric values."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({"-": "0", "": "0"})
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _read_census(name: str) -> pd.DataFrame:
    """Read a required Scotland census table and coerce count columns."""
    market = get_market("scotland")
    path = REPO_ROOT / market.census_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required Scotland census file missing: {path}")

    df = pd.read_csv(path, low_memory=False)
    for column in df.columns:
        if column != "code":
            df[column] = _to_numeric(df[column])
    return df


def _require_columns(
    df: pd.DataFrame,
    columns: list[str],
    source_name: str,
) -> None:
    """Raise if a census source does not include expected columns."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        formatted = "\n".join(missing)
        raise ValueError(
            f"Required columns missing from {source_name}:\n{formatted}"
        )


def _sex_age_counts() -> pd.DataFrame:
    source_name = "UV102b - Age (20) by sex.csv"
    df = _read_census(source_name)
    age_map = {
        "16_to_24": ["16 - 17", "18 - 19", "20 - 24"],
        "25_to_34": ["25 - 29", "30 - 34"],
        "35_to_49": ["35 - 39", "40 - 44", "45 - 49"],
        "50_to_64": ["50 - 54", "55 - 59", "60 - 64"],
        "65_plus": ["65 - 69", "70 - 74", "75 - 79", "80 - 84", "85 and over"],
    }
    required_columns = [
        f"{sex}__{label}"
        for labels in age_map.values()
        for sex in ["Female", "Male"]
        for label in labels
    ]
    _require_columns(df, required_columns, source_name)

    rows: list[dict[str, float | str]] = []
    for age_group, labels in age_map.items():
        for sex in ["Female", "Male"]:
            columns = [f"{sex}__{label}" for label in labels]
            rows.append(
                {
                    "age_group": age_group,
                    "sex": sex.lower(),
                    "count": float(df[columns].sum(axis=1).sum()),
                }
            )

    out = pd.DataFrame(rows)
    out["age_total"] = out.groupby("age_group")["count"].transform("sum")
    out["sex_share_in_age"] = (
        out["count"] / out["age_total"].replace(0, pd.NA)
    ).fillna(0.5)
    return out[["age_group", "sex", "sex_share_in_age"]]


def _education_key(label: str) -> str:
    return {
        "No qualifications": "no_qualifications",
        "Lower school qualifications": "lower_school",
        "Upper school qualifications": "upper_school",
        "Apprenticeship qualifications": "apprenticeship",
        "Further Education and sub-degree Higher Education qualifications incl. HNC/HNDs": "further_or_subdegree",
        "Degree level qualifications or above": "degree_or_above",
    }[label]


def _education_age_shares() -> pd.DataFrame:
    source_name = "UV501b - Highest level of qualification by age (5).csv"
    df = _read_census(source_name)
    education_labels = [
        "No qualifications",
        "Lower school qualifications",
        "Upper school qualifications",
        "Apprenticeship qualifications",
        "Further Education and sub-degree Higher Education qualifications incl. HNC/HNDs",
        "Degree level qualifications or above",
    ]
    age_labels = {
        "16_to_24": "16 to 24",
        "25_to_34": "25 to 34",
        "35_to_49": "35 to 49",
        "50_to_64": "50 to 64",
        "65_plus": "65 and over",
    }
    required_columns = [
        f"{education}__{age_label}"
        for age_label in age_labels.values()
        for education in education_labels
    ]
    _require_columns(df, required_columns, source_name)

    rows: list[dict[str, float | str]] = []
    for age_group, age_label in age_labels.items():
        for education in education_labels:
            column = f"{education}__{age_label}"
            rows.append(
                {
                    "age_group": age_group,
                    "education_group": _education_key(education),
                    "count": float(df[column].sum()),
                }
            )

    out = pd.DataFrame(rows)
    total = out["count"].sum()
    out["education_age_share"] = out["count"] / total if total > 0 else 0.0
    return out[["age_group", "education_group", "education_age_share"]]


def _demographic_shares() -> pd.DataFrame:
    education = _education_age_shares()
    sex = _sex_age_counts()
    demo = education.merge(sex, on="age_group", how="left")
    demo["share"] = demo["education_age_share"] * demo["sex_share_in_age"]
    demo = demo[["age_group", "sex", "education_group", "share"]]
    total = demo["share"].sum()
    demo["share"] = demo["share"] / total if total > 0 else 0.0
    return demo[demo["share"] > 0].reset_index(drop=True)


def _household_type_shares() -> pd.DataFrame:
    source_name = (
        "MV104 - Household composition - households (10) "
        "by dependent children in household (4).csv"
    )
    df = _read_census(source_name)
    wanted_households = {
        "One person household: Aged 66 and over": "one_person_66_plus",
        "One person household: Aged under 66": "one_person_under_66",
        "One family household: All aged 66 and over": "family_all_66_plus",
        "One family household: Couple family: No children": "couple_no_children",
        "One family household: Couple family: With dependent children": "couple_with_dependent_children",
        "One family household: Couple family: All children non-dependent": "couple_non_dependent_children",
        "One family household: Lone parent: With dependent children": "lone_parent_with_dependent_children",
        "One family household: Lone parent: All children non-dependent": "lone_parent_non_dependent_children",
        "Other household types: With dependent children": "other_with_dependent_children",
        "Other household types: Other (including all full-time students and all aged 65 and over)": "other",
    }
    dependent_labels = {
        "No dependent children in household": "none",
        "Dependent children in household: Youngest aged 0 to 4": "youngest_0_to_4",
        "Dependent children in household: Youngest aged 5 to 11": "youngest_5_to_11",
        "Dependent children in household: Youngest aged 12 to 18": "youngest_12_to_18",
    }
    required_columns = [
        f"{source}__{dep_source}"
        for source in wanted_households
        for dep_source in dependent_labels
    ]
    _require_columns(df, required_columns, source_name)

    rows: list[dict[str, float | str]] = []
    for source, household_type in wanted_households.items():
        for dep_source, dependent_children in dependent_labels.items():
            column = f"{source}__{dep_source}"
            rows.append(
                {
                    "household_type": household_type,
                    "dependent_children": dependent_children,
                    "count": float(df[column].sum()),
                }
            )

    out = pd.DataFrame(rows)
    total = out["count"].sum()
    out["share"] = out["count"] / total if total > 0 else 0.0
    return out[out["share"] > 0][["household_type", "dependent_children", "share"]]


def _charging_group(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    invalid = numeric[numeric.isna() | ~numeric.between(0.0, 1.0)]
    if not invalid.empty:
        raise ValueError("feasible_h values must be in the range [0, 1].")
    return pd.cut(
        numeric,
        bins=[-0.001, 0.10, 0.25, 1.0],
        labels=["poor", "moderate", "strong"],
    ).astype(str)


def _base_household_segments(year: int, scenario: str) -> pd.DataFrame:
    market = get_market("scotland")
    path = REPO_ROOT / market.household_feasibility_path
    if not path.exists():
        raise FileNotFoundError(f"Required household feasibility file missing: {path}")

    df = pd.read_parquet(path)
    df = df[
        (df["year"].astype(int) == int(year))
        & (df["scenario"].astype(str) == scenario)
    ].copy()
    if df.empty:
        raise ValueError(
            f"No household feasibility rows for scenario={scenario!r}, year={year!r}."
        )

    df["charging_readiness_group"] = _charging_group(df["feasible_h"])
    df["weighted_feasible"] = (
        df["feasible_h"].astype(float) * df["household_weight"].astype(float)
    )
    grouped = (
        df.groupby(
            [
                "nssec_group",
                "accommodation",
                "household_size",
                "car_ownership",
                "tenure",
                "charging_readiness_group",
            ],
            as_index=False,
            observed=False,
        )
        .agg(
            household_weight=("household_weight", "sum"),
            weighted_feasible=("weighted_feasible", "sum"),
        )
    )
    grouped["baseline_feasible_share"] = (
        grouped["weighted_feasible"] / grouped["household_weight"].replace(0, pd.NA)
    ).fillna(0.0)
    return grouped.rename(columns={"nssec_group": "socioeconomic_group"})


def _validate_household_size_values(size: pd.Series) -> None:
    unknown = sorted(set(size[~size.isin(ALLOWED_HOUSEHOLD_SIZES)].astype(str)))
    if unknown:
        raise ValueError(f"Unknown household_size value(s): {', '.join(unknown)}")


def _is_household_type_size_compatible(
    household_type: pd.Series,
    household_size: pd.Series,
) -> pd.Series:
    """Return whether household type and household size can coexist."""
    one_person = household_type.str.startswith("one_person")
    dependent_couple_or_other = household_type.isin(
        ["couple_with_dependent_children", "other_with_dependent_children"]
    )
    lone_parent_dependent = household_type == "lone_parent_with_dependent_children"
    other_non_one_person = (
        ~one_person & ~dependent_couple_or_other & ~lone_parent_dependent
    )

    return (
        (one_person & household_size.eq("one_person"))
        | (dependent_couple_or_other & household_size.eq("three_plus"))
        | (lone_parent_dependent & household_size.isin(["two_person", "three_plus"]))
        | (other_non_one_person & household_size.isin(["two_person", "three_plus"]))
    )


def build_scotland_segments(
    year: int = 2025,
    scenario: str = "neutral_lr_central",
    max_segments: int | None = 80,
) -> pd.DataFrame:
    """Build census-constrained weighted Scotland segment rows."""
    _ensure_required_sources()
    base = _base_household_segments(year, scenario)
    _validate_household_size_values(base["household_size"])
    demo = _demographic_shares()
    household = _household_type_shares()

    cross = base.merge(demo, how="cross").rename(columns={"share": "demo_share"})
    cross = cross.merge(household, how="cross").rename(
        columns={"share": "household_type_share"}
    )
    cross = cross[
        _is_household_type_size_compatible(
            cross["household_type"],
            cross["household_size"],
        )
    ].copy()
    cross["household_type_share"] = cross["household_type_share"] / cross.groupby(
        [
            "socioeconomic_group",
            "accommodation",
            "household_size",
            "car_ownership",
            "tenure",
            "charging_readiness_group",
            "age_group",
            "sex",
            "education_group",
        ],
        observed=False,
    )["household_type_share"].transform("sum")
    cross["allocation_share"] = cross["demo_share"] * cross["household_type_share"]
    cross["household_weight"] = cross["household_weight"] * cross["allocation_share"]
    cross["person_weight"] = (
        cross["household_weight"] * _household_size_factor(cross["household_size"])
    )
    cross = cross[cross["household_weight"] > 0].copy()
    cross = (
        cross.groupby(
            [
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
            ],
            as_index=False,
        )
        .agg(
            household_weight=("household_weight", "sum"),
            person_weight=("person_weight", "sum"),
            baseline_feasible_share=("baseline_feasible_share", "first"),
        )
    )
    cross = cross.sort_values("household_weight", ascending=False)
    if max_segments is not None:
        cross = cross.head(max_segments).copy()
    else:
        cross = cross.copy()

    cross["segment_id"] = [f"scot_seg_{i:04d}" for i in range(1, len(cross) + 1)]
    cross["market"] = "scotland"
    cross["scenario"] = scenario
    cross["year"] = int(year)
    cross["baseline_adopted_weight"] = (
        cross["household_weight"] * cross["baseline_feasible_share"]
    )
    return cross.reset_index(drop=True)


def _household_size_factor(size: pd.Series) -> pd.Series:
    mapping = {"one_person": 1.0, "two_person": 2.0, "three_plus": 3.2}
    factors = size.map(mapping)
    if factors.isna().any():
        _validate_household_size_values(size)
    return factors
