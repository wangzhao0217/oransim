from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _personas() -> pd.DataFrame:
    base = {
        "scenario": "neutral_lr_central",
        "year": 2025,
        "oa_code": "OA1",
        "household_weight": 10.0,
        "nssec_group": "routine_manual",
        "accommodation": "flat",
        "household_size": "one_person",
        "car_ownership": "one",
        "tenure": "social_rented",
        "household_type": "one_person_under_66",
        "dependent_children": "none",
        "trip_suit_weight": 0.7,
        "oa_trip_suit_i": 0.6,
        "oa_feasible_share": 0.1,
        "age_group": "35_to_49",
        "sex": "female",
        "education_group": "upper_school",
    }
    return pd.DataFrame(
        [
            {
                **base,
                "persona_id": "p1",
                "household_id": "h1",
                "AP_i": 0.8,
                "G_it": 0.6,
                "A_it": 0.6,
                "feasible_h": 0.095,
                "adopts": 0,
                "acceptance_gate_h": 0.65,
            },
            {
                **base,
                "persona_id": "p2",
                "household_id": "h2",
                "AP_i": 0.2,
                "G_it": 0.2,
                "A_it": 0.2,
                "trip_suit_weight": 0.2,
                "feasible_h": 0.02,
                "adopts": 0,
                "acceptance_gate_h": 0.2,
            },
        ]
    )


def test_score_persona_reactions_applies_gate_aware_uplift():
    from oransim.agents.zevup_persona_reactions import score_persona_reactions

    scored = score_persona_reactions(
        _personas().head(1),
        interventions=["price_subsidy", "charging_support"],
        modifiers={"interventions": {}},
    )

    row = scored.iloc[0]
    assert row["post_affordability_gate"] > row["affordability_gate"]
    assert row["post_charging_gate"] > row["charging_gate"]
    assert row["post_zevup_feasible_h"] > row["feasible_h"]
    assert row["best_intervention"] in {"price_subsidy", "charging_support"}


def test_low_relevance_requires_four_low_gates():
    from oransim.agents.zevup_persona_reactions import score_persona_reactions

    scored = score_persona_reactions(
        _personas(),
        interventions=["awareness_campaign"],
        modifiers={"interventions": {}},
    )

    low = scored[scored["persona_id"].eq("p2")].iloc[0]
    assert low["reaction_class"] == "low_relevance"
    assert low["main_barrier"] == "multi_constraint"
    assert low["lowest_gate"] in {"readiness", "charging", "affordability", "trip", "acceptance"}


def test_representative_sample_is_deterministic():
    from oransim.agents.zevup_persona_reactions import (
        score_persona_reactions,
        select_representative_sample,
    )

    scored = score_persona_reactions(_personas(), modifiers={"interventions": {}})

    first = select_representative_sample(scored, 1)
    second = select_representative_sample(scored.sample(frac=1.0, random_state=1), 1)

    assert first["persona_id"].tolist() == second["persona_id"].tolist()

