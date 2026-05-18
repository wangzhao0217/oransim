from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _reactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "persona_id": ["p1", "p2"],
            "oa_code": ["OA1", "OA2"],
            "nssec_group": ["routine_manual", "managerial_professional"],
            "tenure": ["social_rented", "owned"],
            "household_size": ["one_person", "two_person"],
            "main_barrier": ["charging", "none_or_low"],
            "best_intervention": ["charging_support", "awareness_campaign"],
            "reaction_class": ["moved_by_policy", "already_ready"],
            "weighted_uplift": [2.0, 0.1],
        }
    )


def test_template_narrative_returns_json_shape():
    from oransim.agents.zevup_persona_narratives import (
        NARRATIVE_FIELDS,
        template_narrative,
    )

    narrative = template_narrative(_reactions().iloc[0].to_dict())

    assert set(narrative) == set(NARRATIVE_FIELDS)
    assert "synthetic weighted persona" in narrative["policy_note"]


def test_attach_narratives_limits_selected_rows(tmp_path: Path):
    from oransim.agents.zevup_persona_narratives import attach_narratives

    output, rows = attach_narratives(
        _reactions(),
        narrative_limit=1,
        interventions=["charging_support"],
        cache_dir=tmp_path,
    )

    assert len(rows) == 1
    assert output["narrative_status"].value_counts()["template"] == 1
    assert output["narrative_status"].value_counts()["not_selected"] == 1

