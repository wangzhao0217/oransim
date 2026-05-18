"""Build a deterministic ZEV-UP household decision-maker persona sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oransim.agents.zevup_persona_narratives import attach_narratives
from oransim.agents.zevup_persona_reactions import (
    reaction_summary,
    score_persona_reactions,
    select_representative_sample,
)
from oransim.data.zevup_personas import (
    ACCEPTANCE_CONFIG_PATH,
    SCENARIO_SLICES_PATH,
    build_personas,
)
from oransim.data.zevup_static import list_intervention_ids

ORANSIM_ROOT = Path(__file__).resolve().parents[2]
ZEVUP_DATA = ORANSIM_ROOT / "data" / "zevup"
OUTPUT_ROOT = ORANSIM_ROOT / "output" / "zevup_persona_runs"
CONFIG_SNAPSHOT_NAMES = [
    "scenario_slices.json",
    "interventions.json",
    "persona_acceptance_config.json",
    "persona_effect_modifiers.json",
]


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b""


def config_hash() -> str:
    hasher = hashlib.sha256()
    for name in CONFIG_SNAPSHOT_NAMES:
        hasher.update(name.encode("utf-8"))
        hasher.update(_read_bytes(ZEVUP_DATA / name))
    return hasher.hexdigest()


def create_run_dir(run_id: str | None = None, output_root: Path = OUTPUT_ROOT) -> Path:
    run_id = run_id or utc_run_id()
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def snapshot_configs(run_dir: Path) -> None:
    snapshot_dir = run_dir / "config_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_SNAPSHOT_NAMES:
        source = ZEVUP_DATA / name
        if source.exists():
            shutil.copy2(source, snapshot_dir / name)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_sample(
    *,
    scenario: str,
    year: int,
    sample_size: int,
    narrative_limit: int,
    interventions: list[str] | None = None,
    use_ollama: bool = False,
    allow_template_fallback: bool = False,
    run_id: str | None = None,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    selected_interventions = interventions or list_intervention_ids()
    cfg_hash = config_hash()
    run_config = {
        "scenario": scenario,
        "year": year,
        "sample_size": sample_size,
        "narrative_limit": narrative_limit,
        "interventions": selected_interventions,
        "use_ollama": use_ollama,
        "allow_template_fallback": allow_template_fallback,
        "scenario_manifest": str(SCENARIO_SLICES_PATH),
        "acceptance_config": str(ACCEPTANCE_CONFIG_PATH),
        "config_hash": cfg_hash,
    }

    personas = build_personas(scenario=scenario, year=year)
    reactions = score_persona_reactions(
        personas,
        interventions=selected_interventions,
    )
    sample_reactions = select_representative_sample(reactions, sample_size)
    sample_personas = personas[
        personas["persona_id"].isin(set(sample_reactions["persona_id"]))
    ].copy()
    sample_personas = sample_personas.sort_values("persona_id").reset_index(drop=True)

    sample_reactions, narrative_rows = attach_narratives(
        sample_reactions,
        narrative_limit=narrative_limit,
        interventions=selected_interventions,
        use_ollama=use_ollama,
        allow_template_fallback=allow_template_fallback,
        config_hash=cfg_hash,
    )

    run_dir = create_run_dir(run_id=run_id, output_root=output_root)
    write_json(run_dir / "run_config.json", run_config)
    snapshot_configs(run_dir)
    sample_personas.to_parquet(run_dir / "personas_sample.parquet", index=False)
    sample_reactions.to_parquet(run_dir / "reactions_sample.parquet", index=False)
    write_jsonl(run_dir / "narratives_sample.jsonl", narrative_rows)
    summary = reaction_summary(sample_reactions)
    summary["run_dir"] = str(run_dir)
    summary["narratives_written"] = len(narrative_rows)
    write_json(run_dir / "summary.json", summary)
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a ZEV-UP household decision-maker persona sample.",
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--narrative-limit", type=int, default=100)
    parser.add_argument("--interventions", nargs="*")
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--allow-template-fallback", action="store_true")
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_dir = run_sample(
            scenario=args.scenario,
            year=args.year,
            sample_size=args.sample_size,
            narrative_limit=args.narrative_limit,
            interventions=args.interventions,
            use_ollama=args.use_ollama,
            allow_template_fallback=args.allow_template_fallback,
            run_id=args.run_id,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote ZEV-UP persona sample run to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
