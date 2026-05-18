"""Build OA-preserving ZEV-UP household decision-maker personas."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .zevup_static import get_market

REPO_ROOT = Path(__file__).resolve().parents[5]
ORANSIM_ROOT = Path(__file__).resolve().parents[3]
ZEVUP_DATA = ORANSIM_ROOT / "data" / "zevup"

SCENARIO_SLICES_PATH = ZEVUP_DATA / "scenario_slices.json"
ACCEPTANCE_CONFIG_PATH = ZEVUP_DATA / "persona_acceptance_config.json"

AGE_SEX_FILE = "UV102b - Age (20) by sex.csv"
EDUCATION_FILE = "UV501b - Highest level of qualification by age (5).csv"
HOUSEHOLD_COMPOSITION_FILE = (
    "MV104 - Household composition - households (10) "
    "by dependent children in household (4).csv"
)
PANEL_PATH = REPO_ROOT / "output" / "study3" / "feasible_market_long.parquet"

REQUIRED_HOUSEHOLD_COLUMNS = [
    "oa_code",
    "household_id",
    "nssec_group",
    "accommodation",
    "household_size",
    "car_ownership",
    "tenure",
    "trip_suit_weight",
    "household_weight",
    "feasible_share",
    "AP_i",
    "G_it",
    "A_it",
    "feasible_h",
    "scenario",
    "year",
    "adopts",
]

PERSONA_COLUMNS = [
    "persona_id",
    "scenario",
    "year",
    "oa_code",
    "household_id",
    "household_weight",
    "nssec_group",
    "accommodation",
    "household_size",
    "car_ownership",
    "tenure",
    "household_type",
    "dependent_children",
    "trip_suit_weight",
    "oa_trip_suit_i",
    "oa_feasible_share",
    "AP_i",
    "G_it",
    "A_it",
    "feasible_h",
    "adopts",
    "age_group",
    "sex",
    "education_group",
    "acceptance_gate_h",
]

AGE_MAP = {
    "16_to_24": ["16 - 17", "18 - 19", "20 - 24"],
    "25_to_34": ["25 - 29", "30 - 34"],
    "35_to_49": ["35 - 39", "40 - 44", "45 - 49"],
    "50_to_64": ["50 - 54", "55 - 59", "60 - 64"],
    "65_plus": ["65 - 69", "70 - 74", "75 - 79", "80 - 84", "85 and over"],
}

EDUCATION_LABELS = {
    "No qualifications": "no_qualifications",
    "Lower school qualifications": "lower_school",
    "Upper school qualifications": "upper_school",
    "Apprenticeship qualifications": "apprenticeship",
    "Further Education and sub-degree Higher Education qualifications incl. HNC/HNDs": "further_or_subdegree",
    "Degree level qualifications or above": "degree_or_above",
}

EDUCATION_AGE_LABELS = {
    "16_to_24": "16 to 24",
    "25_to_34": "25 to 34",
    "35_to_49": "35 to 49",
    "50_to_64": "50 to 64",
    "65_plus": "65 and over",
}

HOUSEHOLD_TYPES = {
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

DEPENDENT_CHILDREN = {
    "No dependent children in household": "none",
    "Dependent children in household: Youngest aged 0 to 4": "youngest_0_to_4",
    "Dependent children in household: Youngest aged 5 to 11": "youngest_5_to_11",
    "Dependent children in household: Youngest aged 12 to 18": "youngest_12_to_18",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({"-": "0", "": "0"})
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _read_census(name: str) -> pd.DataFrame:
    market = get_market("scotland")
    path = REPO_ROOT / market.census_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required Scotland census file missing: {path}")
    df = pd.read_csv(path, low_memory=False)
    for column in df.columns:
        if column != "code":
            df[column] = _to_numeric(df[column])
    return df


def load_scenario_slice(
    scenario: str,
    year: int,
    manifest_path: Path = SCENARIO_SLICES_PATH,
) -> dict[str, Any]:
    """Return one scenario-slice manifest entry."""
    payload = _read_json(manifest_path)
    for item in payload.get("supported_slices", []):
        if item.get("scenario") == scenario and int(item.get("year")) == int(year):
            return dict(item)
    raise ValueError(f"Unsupported ZEV-UP persona slice: {scenario} {year}.")


def upstream_generation_command(scenario: str, year: int, output_path: Path) -> str:
    return (
        "python -m studies.microsimulation.scripts.generate_household_feasibility_slice "
        f"--scenario {scenario} --year {year} --output {output_path}"
    )


def resolve_household_feasibility_path(
    scenario: str,
    year: int,
    manifest_path: Path = SCENARIO_SLICES_PATH,
) -> Path:
    """Resolve and validate the household-feasibility source path."""
    entry = load_scenario_slice(scenario, year, manifest_path)
    path = REPO_ROOT / str(entry["household_feasibility_path"])
    if path.exists():
        return path

    command = upstream_generation_command(scenario, year, path)
    if entry.get("status") == "requires_generation":
        raise FileNotFoundError(
            "Household feasibility slice is not available:\n"
            f"{path}\n\nGenerate it first with:\n{command}"
        )
    raise FileNotFoundError(f"Household feasibility parquet missing: {path}")


def load_household_feasibility_slice(
    scenario: str,
    year: int,
    *,
    manifest_path: Path = SCENARIO_SLICES_PATH,
    household_feasibility_path: Path | None = None,
) -> pd.DataFrame:
    """Read the requested Paper 2 household-feasibility slice."""
    path = household_feasibility_path or resolve_household_feasibility_path(
        scenario,
        year,
        manifest_path,
    )
    df = pd.read_parquet(path)
    missing = [column for column in REQUIRED_HOUSEHOLD_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Household feasibility source is missing columns: {missing}")

    selected = df[
        (df["scenario"].astype(str) == str(scenario))
        & (df["year"].astype(int) == int(year))
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No household feasibility rows for scenario={scenario!r}, year={year!r}."
        )
    return selected


def _normalise_distribution(
    df: pd.DataFrame,
    group_col: str,
    weight_col: str = "count",
) -> pd.DataFrame:
    out = df.copy()
    totals = out.groupby(group_col)[weight_col].transform("sum")
    out["share"] = np.where(totals > 0, out[weight_col] / totals, 0.0)
    return out[out["share"] > 0].copy()


@lru_cache(maxsize=1)
def oa_demographic_distribution() -> pd.DataFrame:
    """Build OA-level P(age, sex, education) from Census 2022 marginals."""
    age_df = _read_census(AGE_SEX_FILE)
    age_rows: list[pd.DataFrame] = []
    for age_group, labels in AGE_MAP.items():
        for sex in ["Female", "Male"]:
            columns = [f"{sex}__{label}" for label in labels]
            missing = [column for column in columns if column not in age_df.columns]
            if missing:
                raise ValueError(f"Missing age/sex columns in {AGE_SEX_FILE}: {missing}")
            age_rows.append(
                pd.DataFrame(
                    {
                        "oa_code": age_df["code"].astype(str),
                        "age_group": age_group,
                        "sex": sex.lower(),
                        "age_sex_count": age_df[columns].sum(axis=1).astype(float),
                    }
                )
            )
    age_sex = pd.concat(age_rows, ignore_index=True)

    edu_df = _read_census(EDUCATION_FILE)
    edu_rows: list[pd.DataFrame] = []
    for age_group, age_label in EDUCATION_AGE_LABELS.items():
        for source, key in EDUCATION_LABELS.items():
            column = f"{source}__{age_label}"
            if column not in edu_df.columns:
                raise ValueError(f"Missing education column in {EDUCATION_FILE}: {column}")
            edu_rows.append(
                pd.DataFrame(
                    {
                        "oa_code": edu_df["code"].astype(str),
                        "age_group": age_group,
                        "education_group": key,
                        "education_count": edu_df[column].astype(float),
                    }
                )
            )
    education = pd.concat(edu_rows, ignore_index=True)
    education["age_total"] = education.groupby(["oa_code", "age_group"])[
        "education_count"
    ].transform("sum")
    education["education_share_in_age"] = np.where(
        education["age_total"] > 0,
        education["education_count"] / education["age_total"],
        1.0 / len(EDUCATION_LABELS),
    )

    demo = age_sex.merge(
        education[["oa_code", "age_group", "education_group", "education_share_in_age"]],
        on=["oa_code", "age_group"],
        how="left",
    )
    demo["education_share_in_age"] = demo["education_share_in_age"].fillna(
        1.0 / len(EDUCATION_LABELS)
    )
    demo["count"] = demo["age_sex_count"] * demo["education_share_in_age"]
    demo = demo[["oa_code", "age_group", "sex", "education_group", "count"]]
    return _normalise_distribution(demo, "oa_code")


@lru_cache(maxsize=1)
def oa_household_type_distribution() -> pd.DataFrame:
    """Build OA-level household-type distribution from Census 2022 MV104."""
    df = _read_census(HOUSEHOLD_COMPOSITION_FILE)
    rows: list[pd.DataFrame] = []
    for source, household_type in HOUSEHOLD_TYPES.items():
        for dep_source, dependent_children in DEPENDENT_CHILDREN.items():
            column = f"{source}__{dep_source}"
            if column not in df.columns:
                raise ValueError(
                    f"Missing household-composition column in "
                    f"{HOUSEHOLD_COMPOSITION_FILE}: {column}"
                )
            rows.append(
                pd.DataFrame(
                    {
                        "oa_code": df["code"].astype(str),
                        "household_type": household_type,
                        "dependent_children": dependent_children,
                        "count": df[column].astype(float),
                    }
                )
            )
    households = pd.concat(rows, ignore_index=True)
    return _normalise_distribution(households, "oa_code")


def _is_household_type_size_compatible(
    household_type: pd.Series,
    household_size: pd.Series,
) -> pd.Series:
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


def _assign_distribution(
    group: pd.DataFrame,
    distribution: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    """Assign distribution categories to weighted rows by cumulative midpoints."""
    out = group.sort_values(["household_id"]).copy()
    if distribution.empty:
        for column in value_cols:
            out[column] = "unknown"
        return out

    dist = distribution.sort_values(value_cols).copy()
    weight_col = "share" if "share" in dist.columns else "count"
    weights = dist[weight_col].astype(float).clip(lower=0)
    total = float(weights.sum())
    if total <= 0:
        for column in value_cols:
            out[column] = str(dist.iloc[0][column])
        return out

    cumulative = (weights / total).cumsum().to_numpy()
    row_weights = out["household_weight"].astype(float).clip(lower=0)
    row_total = float(row_weights.sum())
    if row_total <= 0:
        positions = np.linspace(0, 1, num=len(out), endpoint=False) + 0.5 / max(len(out), 1)
    else:
        positions = ((row_weights.cumsum() - row_weights / 2) / row_total).to_numpy()
    indexes = np.searchsorted(cumulative, positions, side="left").clip(
        0,
        len(dist) - 1,
    )
    assigned = dist.iloc[indexes].reset_index(drop=True)
    for column in value_cols:
        out[column] = assigned[column].astype(str).to_numpy()
    return out


def _dominant_distribution(
    distribution: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    """Return the dominant category for each group."""
    if distribution.empty:
        return pd.DataFrame(columns=group_cols + value_cols)
    weight_col = "share" if "share" in distribution.columns else "count"
    return (
        distribution.sort_values(
            [*group_cols, weight_col, *value_cols],
            ascending=[*[True] * len(group_cols), False, *[True] * len(value_cols)],
            kind="mergesort",
        )
        .groupby(group_cols, as_index=False)
        .head(1)[group_cols + value_cols]
        .reset_index(drop=True)
    )


def assign_household_types(
    households: pd.DataFrame,
    distribution: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach deterministic OA-level household type labels to weighted rows."""
    distribution = distribution if distribution is not None else oa_household_type_distribution()
    mappings: list[pd.DataFrame] = []
    for household_size in ["one_person", "two_person", "three_plus"]:
        dist = distribution.copy()
        dist["household_size"] = household_size
        compatible = _is_household_type_size_compatible(
            dist["household_type"],
            dist["household_size"],
        )
        mappings.append(
            _dominant_distribution(
                dist[compatible],
                ["oa_code", "household_size"],
                ["household_type", "dependent_children"],
            )
        )
    mapping = pd.concat(mappings, ignore_index=True)
    out = households.merge(mapping, on=["oa_code", "household_size"], how="left")
    out["household_type"] = out["household_type"].fillna("unknown")
    out["dependent_children"] = out["dependent_children"].fillna("unknown")
    return out


def _age_rule(personas: pd.DataFrame) -> pd.Series:
    rule = pd.Series("none", index=personas.index)
    rule = rule.mask(personas["household_type"].eq("one_person_66_plus"), "force_65")
    rule = rule.mask(personas["household_type"].eq("one_person_under_66"), "exclude_65")
    rule = rule.mask(
        personas["dependent_children"].ne("none") & rule.eq("none"),
        "downweight_65",
    )
    return rule


def _adjust_demo_distribution(dist: pd.DataFrame, rule: str) -> pd.DataFrame:
    if dist.empty or rule == "none":
        return dist
    adjusted = dist.copy()
    if rule == "force_65":
        filtered = adjusted[adjusted["age_group"].eq("65_plus")].copy()
        return filtered if not filtered.empty else adjusted
    if rule == "exclude_65":
        filtered = adjusted[~adjusted["age_group"].eq("65_plus")].copy()
        return filtered if not filtered.empty else adjusted
    if rule == "downweight_65":
        adjusted["share"] = np.where(
            adjusted["age_group"].eq("65_plus"),
            adjusted["share"] * 0.25,
            adjusted["share"],
        )
        return adjusted[adjusted["share"] > 0].copy()
    return adjusted


def assign_demographics(
    personas: pd.DataFrame,
    distribution: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach deterministic OA-level age/sex/education labels."""
    distribution = distribution if distribution is not None else oa_demographic_distribution()
    source = personas.copy()
    source["_age_rule"] = _age_rule(source)

    mappings: list[pd.DataFrame] = []
    for rule in ["none", "force_65", "exclude_65", "downweight_65"]:
        dist = _adjust_demo_distribution(distribution, rule)
        dist = dist.copy()
        dist["_age_rule"] = rule
        mappings.append(
            _dominant_distribution(
                dist,
                ["oa_code", "_age_rule"],
                ["age_group", "sex", "education_group"],
            )
        )
    mapping = pd.concat(mappings, ignore_index=True)
    out = source.merge(mapping, on=["oa_code", "_age_rule"], how="left")

    missing = out["age_group"].isna()
    if missing.any():
        fallback = _dominant_distribution(
            distribution,
            ["oa_code"],
            ["age_group", "sex", "education_group"],
        ).rename(
            columns={
                "age_group": "_fallback_age_group",
                "sex": "_fallback_sex",
                "education_group": "_fallback_education_group",
            }
        )
        out = out.merge(fallback, on="oa_code", how="left")
        out["age_group"] = out["age_group"].fillna(out["_fallback_age_group"])
        out["sex"] = out["sex"].fillna(out["_fallback_sex"])
        out["education_group"] = out["education_group"].fillna(
            out["_fallback_education_group"]
        )
        out = out.drop(
            columns=[
                "_fallback_age_group",
                "_fallback_sex",
                "_fallback_education_group",
            ]
        )

    out = out.drop(columns=["_age_rule"])
    out["age_group"] = out["age_group"].fillna("unknown")
    out["sex"] = out["sex"].fillna("unknown")
    out["education_group"] = out["education_group"].fillna("unknown")
    return out


def load_acceptance_config(path: Path = ACCEPTANCE_CONFIG_PATH) -> dict[str, Any]:
    return _read_json(path)


def compute_acceptance_gate(
    personas: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.Series:
    """Compute the Oransim-only baseline acceptance gate."""
    config = config if config is not None else load_acceptance_config()
    score = pd.Series(float(config.get("base", 0.65)), index=personas.index)
    for column, config_key in [
        ("age_group", "age_group_adjustments"),
        ("education_group", "education_group_adjustments"),
        ("nssec_group", "socioeconomic_group_adjustments"),
    ]:
        score = score + personas[column].map(config.get(config_key, {})).fillna(0.0)

    household_adjustments = config.get("household_adjustments", {})
    for column in ["accommodation", "car_ownership", "tenure"]:
        score = score + personas[column].map(
            household_adjustments.get(column, {})
        ).fillna(0.0)

    lower, upper = config.get("bounds", [0.4, 0.95])
    return score.clip(float(lower), float(upper))


def _attach_oa_trip_suitability(
    households: pd.DataFrame,
    scenario: str,
    year: int,
    panel_path: Path = PANEL_PATH,
) -> pd.DataFrame:
    out = households.copy()
    if "trip_suit_i" in out.columns:
        return out.rename(columns={"trip_suit_i": "oa_trip_suit_i"})
    out["oa_trip_suit_i"] = pd.NA
    if not panel_path.exists():
        return out
    panel = pd.read_parquet(
        panel_path,
        columns=["scenario", "geo_code", "year", "trip_suit_i"],
    )
    panel = panel[
        (panel["scenario"].astype(str) == str(scenario))
        & (panel["year"].astype(int) == int(year))
    ][["geo_code", "trip_suit_i"]].drop_duplicates("geo_code")
    if panel.empty:
        return out
    out = out.drop(columns=["oa_trip_suit_i"]).merge(
        panel.rename(columns={"geo_code": "oa_code", "trip_suit_i": "oa_trip_suit_i"}),
        on="oa_code",
        how="left",
    )
    return out


def build_personas(
    *,
    scenario: str,
    year: int,
    manifest_path: Path = SCENARIO_SLICES_PATH,
    household_feasibility_path: Path | None = None,
    demographic_distribution: pd.DataFrame | None = None,
    household_type_distribution: pd.DataFrame | None = None,
    acceptance_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build weighted household decision-maker personas for one scenario/year."""
    households = load_household_feasibility_slice(
        scenario,
        year,
        manifest_path=manifest_path,
        household_feasibility_path=household_feasibility_path,
    )
    households = _attach_oa_trip_suitability(households, scenario, year)
    households = households.rename(columns={"feasible_share": "oa_feasible_share"})
    households["persona_id"] = (
        "scotland__"
        + households["scenario"].astype(str)
        + "__"
        + households["year"].astype(str)
        + "__"
        + households["oa_code"].astype(str)
        + "__"
        + households["household_id"].astype(str)
    )

    personas = assign_household_types(households, household_type_distribution)
    personas = assign_demographics(personas, demographic_distribution)
    personas["acceptance_gate_h"] = compute_acceptance_gate(
        personas,
        acceptance_config,
    )

    for column in PERSONA_COLUMNS:
        if column not in personas.columns:
            personas[column] = pd.NA
    return personas[PERSONA_COLUMNS].reset_index(drop=True)
