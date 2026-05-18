"""Gate-aware ZEV-UP reactions for household decision-maker personas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.zevup_personas import ORANSIM_ROOT
from ..data.zevup_static import get_intervention, list_intervention_ids

EFFECT_MODIFIERS_PATH = (
    ORANSIM_ROOT / "data" / "zevup" / "persona_effect_modifiers.json"
)

THETA = 0.10
NEAR_THRESHOLD_MARGIN = 0.05
LOW_GATE_THRESHOLD = 0.30
MULTI_BARRIER_MARGIN = 0.05
EPSILON_GATE = 1e-6

REACTION_PRIORITY = {
    "already_ready": 0,
    "moved_by_policy": 1,
    "low_relevance": 2,
    "near_threshold_not_moved": 3,
    "still_blocked": 4,
}

GATE_COLUMNS = {
    "readiness": "readiness_gate",
    "charging": "charging_gate",
    "affordability": "affordability_gate",
    "trip": "trip_gate",
    "acceptance": "acceptance_gate",
}


def load_effect_modifiers(path: Path = EFFECT_MODIFIERS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _modifier_from_mapping(
    personas: pd.DataFrame,
    column: str,
    mapping: dict[str, float],
) -> pd.Series:
    return personas[column].astype(str).map(mapping).fillna(1.0).astype(float)


def intervention_effects(
    personas: pd.DataFrame,
    intervention_id: str,
    modifiers: dict[str, Any] | None = None,
) -> pd.Series:
    """Return bounded persona-specific effect values for one intervention."""
    modifiers = modifiers if modifiers is not None else load_effect_modifiers()
    intervention = get_intervention(intervention_id)
    config = modifiers.get("interventions", {}).get(intervention_id, {})
    multiplier = pd.Series(1.0, index=personas.index, dtype=float)

    if intervention_id == "awareness_campaign":
        if "age_group" in config:
            multiplier *= _modifier_from_mapping(
                personas,
                "age_group",
                config["age_group"],
            )
        low = config.get("low_acceptance_gate")
        if low:
            multiplier *= np.where(
                personas["acceptance_gate"].astype(float) < float(low["threshold"]),
                float(low["multiplier"]),
                1.0,
            )

    if intervention_id == "price_subsidy":
        if "socioeconomic_group" in config:
            multiplier *= _modifier_from_mapping(
                personas,
                "nssec_group",
                config["socioeconomic_group"],
            )
        low = config.get("low_affordability_gate")
        if low:
            multiplier *= np.where(
                personas["affordability_gate"].astype(float) < float(low["threshold"]),
                float(low["multiplier"]),
                1.0,
            )

    if intervention_id == "charging_support":
        if "accommodation" in config:
            multiplier *= _modifier_from_mapping(
                personas,
                "accommodation",
                config["accommodation"],
            )
        if "tenure" in config:
            multiplier *= _modifier_from_mapping(personas, "tenure", config["tenure"])
        low = config.get("low_charging_gate")
        if low:
            multiplier *= np.where(
                personas["charging_gate"].astype(float) < float(low["threshold"]),
                float(low["multiplier"]),
                1.0,
            )

    effects = float(intervention.default_effect) * multiplier
    return effects.clip(float(intervention.min_effect), float(intervention.max_effect))


def _new_gate(old: pd.Series, effect: pd.Series) -> pd.Series:
    old = old.astype(float).clip(0.0, 1.0)
    return (old + (1.0 - old) * effect.astype(float)).clip(0.0, 1.0)


def _gate_ratio(old: pd.Series, new: pd.Series) -> pd.Series:
    return new.astype(float) / old.astype(float).clip(lower=EPSILON_GATE)


def _classify_reactions(df: pd.DataFrame) -> pd.Series:
    baseline = df["feasible_h"].astype(float)
    post = df["post_zevup_feasible_h"].astype(float)
    already = (df["adopts"].astype(int) == 1) | (baseline >= THETA)
    moved = (~already) & (post >= THETA)
    low_gate_count = (
        df[list(GATE_COLUMNS.values())].astype(float).lt(LOW_GATE_THRESHOLD).sum(axis=1)
    )
    low_relevance = (~already) & (~moved) & (low_gate_count >= 4)
    near_not_moved = (
        (~already)
        & (~moved)
        & (~low_relevance)
        & baseline.ge(THETA - NEAR_THRESHOLD_MARGIN)
        & baseline.lt(THETA)
    )
    return pd.Series(
        np.select(
            [already, moved, low_relevance, near_not_moved],
            [
                "already_ready",
                "moved_by_policy",
                "low_relevance",
                "near_threshold_not_moved",
            ],
            default="still_blocked",
        ),
        index=df.index,
    )


def _barriers(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    gates = df[list(GATE_COLUMNS.values())].astype(float)
    labels = list(GATE_COLUMNS)
    values = gates.to_numpy()
    min_idx = values.argmin(axis=1)
    min_values = values[np.arange(len(values)), min_idx]
    lowest = pd.Series([labels[i] for i in min_idx], index=df.index)
    close_counts = (values <= (min_values[:, None] + MULTI_BARRIER_MARGIN)).sum(axis=1)
    main = lowest.mask(close_counts > 1, "multi_constraint")
    main = main.mask(df["reaction_class"].eq("low_relevance"), "multi_constraint")
    main = main.mask(df["reaction_class"].eq("already_ready"), "none_or_low")
    return main, lowest


def _best_intervention(df: pd.DataFrame, interventions: list[str]) -> tuple[pd.Series, pd.Series]:
    lift_cols = [f"lift_{intervention_id}" for intervention_id in interventions]
    if not lift_cols:
        best = pd.Series("none", index=df.index)
    else:
        lift_values = df[lift_cols].astype(float)
        best = lift_values.idxmax(axis=1).str.removeprefix("lift_")
    actionable = df["reaction_class"].isin(
        ["moved_by_policy", "near_threshold_not_moved"]
    )
    status = pd.Series(
        np.where(actionable, "actionable", "diagnostic_only"),
        index=df.index,
    )
    return best, status


def score_persona_reactions(
    personas: pd.DataFrame,
    *,
    interventions: list[str] | None = None,
    modifiers: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply deterministic gate-aware ZEV-UP interventions to personas."""
    selected = interventions or list_intervention_ids()
    unknown = sorted(set(selected) - set(list_intervention_ids()))
    if unknown:
        raise ValueError(f"Unknown intervention(s): {', '.join(unknown)}")
    modifiers = modifiers if modifiers is not None else load_effect_modifiers()

    results = personas.copy()
    results["readiness_gate"] = results["AP_i"].astype(float).clip(0.0, 1.0)
    results["charging_gate"] = results["G_it"].astype(float).clip(0.0, 1.0)
    results["affordability_gate"] = results["A_it"].astype(float).clip(0.0, 1.0)
    results["trip_gate"] = results["trip_suit_weight"].astype(float).clip(0.0, 1.0)
    results["acceptance_gate"] = results["acceptance_gate_h"].astype(float).clip(0.0, 1.0)

    current = results["feasible_h"].astype(float).clip(0.0, 1.0)
    current_charging = results["charging_gate"].copy()
    current_affordability = results["affordability_gate"].copy()
    current_acceptance = results["acceptance_gate"].copy()

    for intervention_id in list_intervention_ids():
        results[f"effect_{intervention_id}"] = 0.0
        results[f"lift_{intervention_id}"] = 0.0

    for intervention_id in selected:
        effect = intervention_effects(results, intervention_id, modifiers)
        before = current.copy()
        if intervention_id == "awareness_campaign":
            new_gate = _new_gate(current_acceptance, effect)
            current = (current * _gate_ratio(current_acceptance, new_gate)).clip(0.0, 1.0)
            current_acceptance = new_gate
        elif intervention_id == "price_subsidy":
            new_gate = _new_gate(current_affordability, effect)
            current = (current * _gate_ratio(current_affordability, new_gate)).clip(0.0, 1.0)
            current_affordability = new_gate
        elif intervention_id == "charging_support":
            new_gate = _new_gate(current_charging, effect)
            current = (current * _gate_ratio(current_charging, new_gate)).clip(0.0, 1.0)
            current_charging = new_gate
        results[f"effect_{intervention_id}"] = effect.round(6)
        results[f"lift_{intervention_id}"] = (current - before).clip(lower=0).round(6)

    results["post_charging_gate"] = current_charging.round(6)
    results["post_affordability_gate"] = current_affordability.round(6)
    results["post_acceptance_gate"] = current_acceptance.round(6)
    results["post_zevup_feasible_h"] = current.round(6)
    results["post_zevup_adopts"] = (results["post_zevup_feasible_h"] >= THETA).astype(int)
    results["zevup_uplift"] = (
        results["post_zevup_feasible_h"].astype(float)
        - results["feasible_h"].astype(float)
    ).round(6)
    results["weighted_uplift"] = (
        results["zevup_uplift"].astype(float)
        * results["household_weight"].astype(float)
    ).round(6)
    results["reaction_class"] = _classify_reactions(results)
    results["main_barrier"], results["lowest_gate"] = _barriers(results)
    results["best_intervention"], results["best_intervention_status"] = _best_intervention(
        results,
        selected,
    )
    return results.reset_index(drop=True)


def select_representative_sample(
    reactions: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    """Select a deterministic balanced sample from scored reactions."""
    if sample_size <= 0 or len(reactions) <= sample_size:
        return reactions.sort_values("persona_id").reset_index(drop=True)

    source = reactions.copy()
    source["_priority"] = source["reaction_class"].map(REACTION_PRIORITY).fillna(99)
    source = source.sort_values(
        ["_priority", "main_barrier", "weighted_uplift", "persona_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    strata = list(source.groupby(["reaction_class", "main_barrier"], sort=False))
    quota = max(1, sample_size // max(len(strata), 1))
    selected_indexes: list[int] = []
    for _, group in strata:
        selected_indexes.extend(group.head(quota).index.tolist())

    selected = source.loc[~source.index.duplicated() & source.index.isin(selected_indexes)]
    if len(selected) < sample_size:
        remainder = source.drop(index=selected.index, errors="ignore").head(
            sample_size - len(selected)
        )
        selected = pd.concat([selected, remainder], ignore_index=False)

    selected = selected.head(sample_size).drop(columns=["_priority"])
    return selected.sort_values("persona_id").reset_index(drop=True)


def reaction_summary(reactions: pd.DataFrame) -> dict[str, Any]:
    weights = reactions["household_weight"].astype(float)
    total_weight = float(weights.sum())
    baseline = (
        float((reactions["feasible_h"].astype(float) * weights).sum() / total_weight)
        if total_weight > 0
        else 0.0
    )
    post = (
        float(
            (reactions["post_zevup_feasible_h"].astype(float) * weights).sum()
            / total_weight
        )
        if total_weight > 0
        else 0.0
    )
    moved = reactions[reactions["reaction_class"].eq("moved_by_policy")]
    return {
        "rows": int(len(reactions)),
        "weighted_households": round(total_weight, 6),
        "baseline_readiness": round(baseline, 6),
        "post_zevup_readiness": round(post, 6),
        "overall_uplift": round(post - baseline, 6),
        "weighted_households_moved": round(float(moved["household_weight"].sum()), 6),
        "reaction_class_counts": {
            str(k): int(v) for k, v in reactions["reaction_class"].value_counts().items()
        },
        "main_barrier_counts": {
            str(k): int(v) for k, v in reactions["main_barrier"].value_counts().items()
        },
    }

