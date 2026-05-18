from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _load_script():
    path = BACKEND / "scripts" / "audit_zevup_llm_effects.py"
    spec = importlib.util.spec_from_file_location("audit_zevup_llm_effects", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_script_writes_off_mock_artifacts_without_deepseek_key(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    result = module.main(
        [
            "--segment-limit",
            "2",
            "--runs",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    runs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "audit_summary.json").exists()
    assert (run_dir / "effects_long.csv").exists()
    assert (run_dir / "mode_comparison.csv").exists()
    assert (run_dir / "effect_variation.csv").exists()
    assert (run_dir / "segment_inputs.json").exists()

    effects = pd.read_csv(run_dir / "effects_long.csv")
    assert set(effects["mode"]) == {"off", "mock"}
    assert set(effects["status"]) == {"off", "mock"}

    comparison = pd.read_csv(run_dir / "mode_comparison.csv")
    assert {"off", "mock"}.issubset(comparison.columns)
