"""Static data loading for the ZEV-UP simulator."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ..schemas.zevup import ZEVUPIntervention, ZEVUPMarket, ZEVUPVehicleConcept

_DATA = Path(__file__).resolve().parents[3] / "data" / "zevup"


def _read_json(name: str) -> dict:
    return json.loads((_DATA / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def markets() -> dict[str, ZEVUPMarket]:
    payload = _read_json("markets.json")
    return {item["market_id"]: ZEVUPMarket(**item) for item in payload["markets"]}


@lru_cache(maxsize=1)
def vehicle_concepts() -> dict[str, ZEVUPVehicleConcept]:
    payload = _read_json("vehicle_concepts.json")
    return {
        item["concept_id"]: ZEVUPVehicleConcept(**item)
        for item in payload["vehicle_concepts"]
    }


@lru_cache(maxsize=1)
def interventions() -> dict[str, ZEVUPIntervention]:
    payload = _read_json("interventions.json")
    return {
        item["intervention_id"]: ZEVUPIntervention(**item)
        for item in payload["interventions"]
    }


def get_market(market_id: str) -> ZEVUPMarket:
    return markets()[market_id]


def get_vehicle_concept(concept_id: str) -> ZEVUPVehicleConcept:
    return vehicle_concepts()[concept_id]


def get_intervention(intervention_id: str) -> ZEVUPIntervention:
    return interventions()[intervention_id]


def list_intervention_ids() -> list[str]:
    return list(interventions().keys())
