from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

ROOT = Path(__file__).parent.parent
ZEVUP_DATA = ROOT / "data" / "zevup"


def _load(name: str) -> dict:
    return json.loads((ZEVUP_DATA / name).read_text(encoding="utf-8"))


def test_static_zevup_json_files_parse():
    assert (_load("markets.json"))["markets"][0]["market_id"] == "scotland"
    assert len(_load("vehicle_concepts.json")["vehicle_concepts"]) == 3
    assert len(_load("interventions.json")["interventions"]) == 3
    assert set(_load("wp_output_map.json")["work_packages"]) == {
        "WP1",
        "WP5",
        "WP6",
    }


def test_only_first_slice_interventions_are_present():
    interventions = _load("interventions.json")["interventions"]
    assert [item["intervention_id"] for item in interventions] == [
        "awareness_campaign",
        "price_subsidy",
        "charging_support",
    ]
    for item in interventions:
        assert 0.0 <= item["default_effect"] <= item["max_effect"] <= 0.25
        assert item["min_effect"] >= 0.0


def test_static_loader_returns_market_vehicle_and_intervention():
    from oransim.data.zevup_static import (
        get_intervention,
        get_market,
        get_vehicle_concept,
        list_intervention_ids,
    )

    assert get_market("scotland").display_name == "Scotland"
    assert get_vehicle_concept("zevup_l7e_passenger_2_seat").seats == 2
    subsidy = get_intervention("price_subsidy")
    assert subsidy.default_effect > 0
    assert list_intervention_ids() == [
        "awareness_campaign",
        "price_subsidy",
        "charging_support",
    ]
