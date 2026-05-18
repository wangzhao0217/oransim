"""Audit ZEV-UP segment LLM effects across off, mock, and DeepSeek modes."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from oransim.agents.zevup_llm import ZEVUPLLMResult, assess_segments, llm_config
from oransim.data.zevup_segments import build_scotland_segments
from oransim.data.zevup_static import get_vehicle_concept, interventions

RETURNED_COLUMNS = [
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
]


@contextmanager
def _temporary_mode(mode: str) -> Iterator[None]:
    old_mode = os.environ.get("ZEVUP_LLM_MODE")
    os.environ["ZEVUP_LLM_MODE"] = mode
    try:
        yield
    finally:
        if old_mode is None:
            os.environ.pop("ZEVUP_LLM_MODE", None)
        else:
            os.environ["ZEVUP_LLM_MODE"] = old_mode


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _effect_rows(
    *,
    result: ZEVUPLLMResult,
    mode: str,
    run: int,
) -> list[dict[str, Any]]:
    rows = []
    for effect in result.response.effects:
        rows.append(
            {
                "mode": mode,
                "run": run,
                "status": result.status,
                "segment_id": effect.segment_id,
                "intervention_id": effect.intervention_id,
                "effect": float(effect.effect),
                "reasoning": effect.reasoning,
                "message_recommendation": effect.message_recommendation,
            }
        )
    return rows


def _variation_table(effects: pd.DataFrame) -> pd.DataFrame:
    deepseek = effects[
        (effects["mode"] == "deepseek") & (effects["status"] == "used")
    ].copy()
    if deepseek.empty:
        return pd.DataFrame(
            columns=[
                "segment_id",
                "intervention_id",
                "runs",
                "mean_effect",
                "std_effect",
                "min_effect",
                "max_effect",
                "range_effect",
                "unique_effects",
            ]
        )
    grouped = deepseek.groupby(["segment_id", "intervention_id"], as_index=False)
    variation = grouped.agg(
        runs=("effect", "count"),
        mean_effect=("effect", "mean"),
        std_effect=("effect", "std"),
        min_effect=("effect", "min"),
        max_effect=("effect", "max"),
        unique_effects=("effect", "nunique"),
    )
    variation["std_effect"] = variation["std_effect"].fillna(0.0)
    variation["range_effect"] = variation["max_effect"] - variation["min_effect"]
    return variation.round(6)


def _comparison_table(effects: pd.DataFrame) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame()
    means = (
        effects.groupby(["segment_id", "intervention_id", "mode"], as_index=False)
        .agg(effect=("effect", "mean"))
        .pivot(index=["segment_id", "intervention_id"], columns="mode", values="effect")
        .reset_index()
    )
    means.columns.name = None
    for column in ["off", "mock", "deepseek"]:
        if column not in means.columns:
            means[column] = pd.NA
    means["mock_minus_off"] = means["mock"] - means["off"]
    means["deepseek_minus_off"] = means["deepseek"] - means["off"]
    means["deepseek_minus_mock"] = means["deepseek"] - means["mock"]
    return means.round(6)


def run_audit(
    *,
    year: int,
    scenario: str,
    segment_limit: int,
    runs: int,
    output_dir: Path,
    selected_interventions: list[str],
    live_mode: str = "deepseek",
) -> Path:
    """Run the effect audit and return the created run directory."""
    if live_mode != "deepseek":
        raise ValueError("run_audit currently supports live_mode='deepseek' only.")

    run_dir = output_dir / _run_id()
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)

    segments = build_scotland_segments(
        year=year,
        scenario=scenario,
        max_segments=segment_limit,
    )
    segment_records = segments[RETURNED_COLUMNS].to_dict("records")
    vehicle_concept = get_vehicle_concept("zevup_l7e_passenger_2_seat").model_dump()

    all_rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []

    print(
        f"Running deterministic comparison modes for {len(segment_records)} segment(s): "
        "off, mock",
        flush=True,
    )
    for mode in ["off", "mock"]:
        with _temporary_mode(mode):
            result = assess_segments(
                segments=segment_records,
                vehicle_concept=vehicle_concept,
                interventions=selected_interventions,
                year=year,
                baseline_scenario=scenario,
            )
        all_rows.extend(_effect_rows(result=result, mode=mode, run=1))
        statuses.append({"mode": mode, "run": 1, "status": result.status})
        print(f"Completed mode {mode}: status={result.status}", flush=True)

    deepseek_key_available = bool(os.environ.get("DEEPSEEK_API_KEY"))
    live_prompts_planned = runs if deepseek_key_available else 0
    expected_live_effect_rows = (
        live_prompts_planned * len(segment_records) * len(selected_interventions)
    )
    print(
        f"DeepSeek live prompt plan: {live_prompts_planned} prompt(s), "
        f"{len(segment_records)} segment(s) per prompt, "
        f"{len(selected_interventions)} intervention(s), "
        f"about {expected_live_effect_rows} live effect row(s).",
        flush=True,
    )
    if not deepseek_key_available:
        print("DeepSeek key not available; skipping live prompts.", flush=True)
    if deepseek_key_available:
        for run in range(1, runs + 1):
            print(
                f"Sending DeepSeek prompt {run}/{runs} "
                f"for {len(segment_records)} segment(s)...",
                flush=True,
            )
            with _temporary_mode(live_mode):
                result = assess_segments(
                    segments=segment_records,
                    vehicle_concept=vehicle_concept,
                    interventions=selected_interventions,
                    year=year,
                    baseline_scenario=scenario,
                )
            all_rows.extend(_effect_rows(result=result, mode=live_mode, run=run))
            statuses.append({"mode": live_mode, "run": run, "status": result.status})
            if result.raw is not None:
                _write_json(raw_dir / f"{live_mode}_run_{run:02d}.json", result.raw)
            print(
                f"Completed DeepSeek prompt {run}/{runs}: status={result.status}",
                flush=True,
            )

    effects = pd.DataFrame(all_rows)
    comparison = _comparison_table(effects)
    variation = _variation_table(effects)

    effects.to_csv(run_dir / "effects_long.csv", index=False)
    comparison.to_csv(run_dir / "mode_comparison.csv", index=False)
    variation.to_csv(run_dir / "effect_variation.csv", index=False)
    _write_json(run_dir / "segment_inputs.json", segment_records)

    deepseek_used = sum(
        1 for item in statuses if item["mode"] == "deepseek" and item["status"] == "used"
    )
    summary = {
        "run_dir": str(run_dir),
        "year": year,
        "scenario": scenario,
        "segment_limit": segment_limit,
        "segments_returned": len(segment_records),
        "runs_requested": runs,
        "deepseek_key_available": deepseek_key_available,
        "deepseek_runs_used": deepseek_used,
        "deepseek_runs_skipped": 0 if deepseek_key_available else runs,
        "live_prompts_planned": live_prompts_planned,
        "expected_live_effect_rows": expected_live_effect_rows,
        "live_mode": live_mode,
        "comparison_modes": ["off", "mock", live_mode],
        "interventions": selected_interventions,
        "llm_config": llm_config(),
        "statuses": statuses,
        "artifacts": [
            "audit_summary.json",
            "effects_long.csv",
            "mode_comparison.csv",
            "effect_variation.csv",
            "segment_inputs.json",
            f"raw/{live_mode}_run_XX.json",
        ],
        "publication_hierarchy": {
            "main_result": "ZEVUP_LLM_MODE=off",
            "robustness_check": "ZEVUP_LLM_MODE=mock",
            "exploratory_scenario": "ZEVUP_LLM_MODE=deepseek",
        },
    }
    if not deepseek_key_available:
        summary["note"] = (
            "DeepSeek repetitions were skipped because DEEPSEEK_API_KEY is not set. "
            "Set the key and rerun this script to measure live effect variation."
        )
    _write_json(run_dir / "audit_summary.json", summary)
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit ZEV-UP LLM segment effects across off, mock, and DeepSeek.",
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--scenario", default="neutral_lr_central")
    parser.add_argument("--segment-limit", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "zevup_llm_audits",
    )
    parser.add_argument("--interventions", nargs="*", default=interventions())
    parser.add_argument(
        "--live-mode",
        choices=["deepseek"],
        default="deepseek",
        help="Live LLM mode to audit. Currently only deepseek is supported.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dir = run_audit(
        year=args.year,
        scenario=args.scenario,
        segment_limit=args.segment_limit,
        runs=args.runs,
        output_dir=args.output_dir,
        selected_interventions=list(args.interventions),
        live_mode=args.live_mode,
    )
    print(f"Wrote ZEV-UP LLM effect audit to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
