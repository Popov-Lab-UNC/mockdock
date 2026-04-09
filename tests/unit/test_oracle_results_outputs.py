from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from mockdock.oracle import MDOracle


def test_update_results_df_writes_live_outputs(tmp_path: Path):
    oracle = MDOracle.__new__(MDOracle)
    oracle.results_df = pl.DataFrame()
    oracle._yaml_results = []
    oracle._generation_round = 1
    oracle._run_dir = tmp_path
    oracle.benchmark_name = "CHK1"
    oracle.max_budget = 10
    oracle.budget_used = 3

    new_results = [
        {
            "smiles": "CC",
            "original_smiles": "CC",
            "docking_score": -10.0,
            "normalized_score": 0.5,
            "valid_pose_found": True,
            "dlg_path": str(tmp_path / "x.dlg"),
            "pose_index": 0,
            "best_any_score": -10.0,
            "skip_reason": None,
            "n_conformers": 1,
        }
    ]
    oracle._update_results_df(new_results)

    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "results.yaml").exists()
    status_path = tmp_path / "status.json"
    assert status_path.exists()
    status = json.loads(status_path.read_text())
    assert status["benchmark"] == "CHK1"
    assert status["n_molecules_total"] == 1
