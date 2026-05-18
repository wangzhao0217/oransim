"""Narrative-only persona explanations for ZEV-UP reactions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .llm_providers import get_provider
from .zevup_persona_reactions import REACTION_PRIORITY

ORANSIM_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = ORANSIM_ROOT / ".cache" / "zevup_persona_narratives"
PROMPT_VERSION = "zevup-persona-narrative-v1"
DEFAULT_NARRATIVE_MAX_TOKENS = int(os.environ.get("LLM_NARRATIVE_MAX_TOKENS", "700"))

NARRATIVE_FIELDS = [
    "persona_reaction_text",
    "hesitation_explanation",
    "message_hook",
    "policy_note",
]

PROMPT_RECORD_FIELDS = [
    "persona_id",
    "scenario",
    "year",
    "oa_code",
    "household_weight",
    "nssec_group",
    "accommodation",
    "household_size",
    "car_ownership",
    "tenure",
    "household_type",
    "dependent_children",
    "age_group",
    "sex",
    "education_group",
    "feasible_h",
    "post_zevup_feasible_h",
    "zevup_uplift",
    "reaction_class",
    "main_barrier",
    "lowest_gate",
    "best_intervention",
    "best_intervention_status",
    "readiness_gate",
    "charging_gate",
    "affordability_gate",
    "trip_gate",
    "acceptance_gate",
]


def template_narrative(record: dict[str, Any]) -> dict[str, str]:
    """Return deterministic analyst-voice narrative fields."""
    household = (
        f"{record.get('nssec_group', 'unknown')} {record.get('tenure', 'unknown')} "
        f"{record.get('household_size', 'unknown')} household in {record.get('oa_code', 'unknown')}"
    )
    barrier = str(record.get("main_barrier", "multi_constraint")).replace("_", " ")
    intervention = str(record.get("best_intervention", "the selected intervention")).replace(
        "_",
        " ",
    )
    reaction = str(record.get("reaction_class", "still_blocked")).replace("_", " ")
    return {
        "persona_reaction_text": (
            f"This weighted household decision-maker persona represents a {household}. "
            f"The deterministic scenario classifies the reaction as {reaction}."
        ),
        "hesitation_explanation": (
            f"The main modelled barrier is {barrier}, based on the inherited "
            "feasibility gates and the Oransim acceptance gate."
        ),
        "message_hook": (
            f"If treated as actionable, lead with {intervention} and keep the "
            "message tied to the identified barrier."
        ),
        "policy_note": (
            "This is a synthetic weighted persona interpretation, not observed "
            "survey evidence or a causal effect estimate."
        ),
    }


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def cache_key(
    record: dict[str, Any],
    *,
    interventions: list[str],
    model: str,
    prompt_version: str = PROMPT_VERSION,
    config_hash: str = "",
) -> str:
    payload = {
        "persona_id": record["persona_id"],
        "interventions": interventions,
        "model": model,
        "prompt_version": prompt_version,
        "config_hash": config_hash,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _prompt(record: dict[str, Any], interventions: list[str]) -> tuple[str, str]:
    system = (
        "/no_think You are a transport policy analyst. Return strict JSON only "
        "in the assistant message content. Use analyst voice, not first person. "
        "Interpret ZEV-UP as a frugal electric vehicle policy scenario. Do not "
        "mention hydrogen, fuel cells, or fuel-cell vehicles. Do not change "
        "numeric values. Do not round decimal model values; avoid quoting decimal "
        "model values unless copying them exactly. Do not include reasoning, "
        "markdown, or extra fields."
    )
    compact_record = {
        field: record.get(field)
        for field in PROMPT_RECORD_FIELDS
        if field in record and record.get(field) is not None
    }
    user = json.dumps(
        {
            "task": "Write a concise narrative interpretation for one synthetic weighted household decision-maker persona.",
            "persona_record": compact_record,
            "selected_interventions": interventions,
            "required_json_shape": {
                "persona_reaction_text": "string",
                "hesitation_explanation": "string",
                "message_hook": "string",
                "policy_note": "string",
            },
        },
        ensure_ascii=False,
        default=str,
    )
    return system, user


def _parse_narrative(content: str) -> dict[str, str]:
    parsed = json.loads(content)
    return {field: str(parsed[field]) for field in NARRATIVE_FIELDS}


def _call_ollama(record: dict[str, Any], interventions: list[str]) -> dict[str, str]:
    provider = get_provider()
    model = os.environ.get("LLM_MODEL", "llama3.1:8b")
    system, user = _prompt(record, interventions)
    result = provider.generate(
        system,
        user,
        model=model,
        temperature=0.2,
        max_tokens=DEFAULT_NARRATIVE_MAX_TOKENS,
        stream=False,
    )
    return _parse_narrative(result.content)


def select_narrative_rows(reactions: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Select a balanced deterministic set of rows to narrate."""
    if limit <= 0 or reactions.empty:
        return reactions.head(0).copy()
    source = reactions.copy()
    source["_priority"] = source["reaction_class"].map(REACTION_PRIORITY).fillna(99)
    source = source.sort_values(
        ["_priority", "weighted_uplift", "persona_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    strata = list(source.groupby("reaction_class", sort=False))
    quota = max(1, limit // max(len(strata), 1))
    selected_indexes: list[int] = []
    for _, group in strata:
        selected_indexes.extend(group.head(quota).index.tolist())
    selected = source.loc[source.index.isin(selected_indexes)].copy()
    if len(selected) < limit:
        selected = pd.concat(
            [
                selected,
                source.drop(index=selected.index, errors="ignore").head(limit - len(selected)),
            ],
            ignore_index=False,
        )
    return selected.head(limit).drop(columns=["_priority"]).reset_index(drop=True)


def attach_narratives(
    reactions: pd.DataFrame,
    *,
    narrative_limit: int,
    interventions: list[str],
    use_ollama: bool = False,
    allow_template_fallback: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    config_hash: str = "",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Attach narrative fields to selected rows and return JSONL records."""
    output = reactions.copy()
    output["narrative_cache_key"] = ""
    output["narrative_status"] = "not_selected"
    for field in NARRATIVE_FIELDS:
        output[field] = ""

    selected = select_narrative_rows(output, narrative_limit)
    if selected.empty:
        return output, []

    cache_dir.mkdir(parents=True, exist_ok=True)
    model = os.environ.get("LLM_MODEL", "template")
    narrative_records: list[dict[str, Any]] = []

    for record in selected.to_dict("records"):
        key = cache_key(
            record,
            interventions=interventions,
            model=model,
            config_hash=config_hash,
        )
        cache_path = cache_dir / f"{key}.json"
        status = "template"
        try:
            if use_ollama and cache_path.exists():
                narrative = json.loads(cache_path.read_text(encoding="utf-8"))["narrative"]
                status = "ollama_cached"
            elif use_ollama:
                narrative = _call_ollama(record, interventions)
                cache_path.write_text(
                    json.dumps(
                        {
                            "cache_key": key,
                            "model": model,
                            "prompt_version": PROMPT_VERSION,
                            "persona_id": record["persona_id"],
                            "narrative": narrative,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                status = "ollama_generated"
            else:
                narrative = template_narrative(record)
        except Exception as exc:
            if use_ollama and not allow_template_fallback:
                raise RuntimeError(
                    "Ollama narrative generation failed. Check local setup:\n"
                    "  ollama list\n"
                    "  curl http://localhost:11434/api/tags\n"
                    "  export LLM_PROVIDER=ollama\n"
                    "  export LLM_BASE_URL=http://localhost:11434\n"
                    "  export LLM_MODEL=<your local model>\n"
                    "For a host Ollama server from a devcontainer, use:\n"
                    "  export LLM_BASE_URL=http://host.docker.internal:11434\n"
                    f"Current LLM_PROVIDER={os.environ.get('LLM_PROVIDER', '')!r}, "
                    f"LLM_BASE_URL={os.environ.get('LLM_BASE_URL', '')!r}, "
                    f"LLM_MODEL={os.environ.get('LLM_MODEL', '')!r}.\n"
                    f"Underlying error: {type(exc).__name__}: {exc}"
                ) from exc
            narrative = template_narrative(record)
            status = "fallback_parse_error" if use_ollama else "template"

        mask = output["persona_id"].eq(record["persona_id"])
        output.loc[mask, "narrative_cache_key"] = key
        output.loc[mask, "narrative_status"] = status
        for field in NARRATIVE_FIELDS:
            output.loc[mask, field] = narrative[field]
        narrative_records.append(
            {
                "cache_key": key,
                "status": status,
                "persona_id": record["persona_id"],
                "narrative": narrative,
            }
        )

    return output, narrative_records
