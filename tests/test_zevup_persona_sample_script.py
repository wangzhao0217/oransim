from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

SCRIPT = Path(__file__).parent.parent / "backend" / "scripts" / "run_zevup_persona_sample.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_zevup_persona_sample", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_sample_writes_timestamped_outputs(monkeypatch, tmp_path: Path):
    module = _load_script()
    personas = pd.DataFrame(
        {
            "persona_id": ["p1"],
            "scenario": ["neutral_lr_central"],
            "year": [2025],
            "oa_code": ["OA1"],
            "household_id": ["h1"],
            "household_weight": [10.0],
            "nssec_group": ["routine_manual"],
            "accommodation": ["flat"],
            "household_size": ["one_person"],
            "car_ownership": ["one"],
            "tenure": ["social_rented"],
            "household_type": ["one_person_under_66"],
            "dependent_children": ["none"],
            "trip_suit_weight": [0.7],
            "oa_trip_suit_i": [0.6],
            "oa_feasible_share": [0.1],
            "AP_i": [0.8],
            "G_it": [0.6],
            "A_it": [0.6],
            "feasible_h": [0.095],
            "adopts": [0],
            "age_group": ["35_to_49"],
            "sex": ["female"],
            "education_group": ["upper_school"],
            "acceptance_gate_h": [0.65],
        }
    )

    monkeypatch.setattr(module, "build_personas", lambda **_kwargs: personas)

    run_dir = module.run_sample(
        scenario="neutral_lr_central",
        year=2025,
        sample_size=1,
        narrative_limit=1,
        run_id="test_run",
        output_root=tmp_path,
    )

    assert run_dir == tmp_path / "test_run"
    assert (run_dir / "personas_sample.parquet").exists()
    assert (run_dir / "reactions_sample.parquet").exists()
    assert (run_dir / "narratives_sample.jsonl").exists()
    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "summary.json").exists()

