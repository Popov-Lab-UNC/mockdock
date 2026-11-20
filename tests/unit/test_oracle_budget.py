from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from mockdock.oracle import MDOracle


def test_oracle_init_overwrite_guard(tmp_path: Path):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    (run_dir / "status.json").write_text("{}")

    # Should raise FileExistsError if overwrite is False (default)
    with pytest.raises(FileExistsError, match="already contains status.json or results.csv"):
        MDOracle("CHK1", run_dir=run_dir, overwrite=False)

    # With overwrite=True, stale files are cleared and status.json is re-written
    (run_dir / "results.csv").write_text("dummy")
    (run_dir / "eval_metrics.json").write_text("dummy")
    oracle = MDOracle("CHK1", budget=1000, run_dir=run_dir, overwrite=True)

    assert oracle.run_dir == run_dir
    assert (run_dir / "status.json").exists()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["budget_total"] == 1000
    assert status["budget_used"] == 0
    assert not (run_dir / "eval_metrics.json").exists()


def test_oracle_score_batch_submission_order_and_budget_exhaustion(tmp_path: Path, monkeypatch):
    oracle = MDOracle("CHK1", budget=3, run_dir=tmp_path / "run_order", overwrite=True)

    # Mock components to isolate scoring flow
    oracle._ensure_components = lambda: None
    oracle._docking_analyzer = type("Analyzer", (), {"fragment_mol": None})()

    class MockPreparer:
        @staticmethod
        def prepare_batch(smiles, tmp_dir):
            return [{"smiles": s} for s in smiles]

    class MockDocking:
        @staticmethod
        def dock_batch(tasks, chunk_idx):
            return [{"dlg_path": "fake.dlg"}] * len(tasks)

    oracle._ligand_preparer = MockPreparer()
    oracle._docking_oracle = MockDocking()

    # Mock filter_smiles: CCO valid, CC invalid, CCN valid, CCC out of budget, CCO out of budget
    # We will test a batch of 5 smiles with budget=3
    # First 3 smiles:
    # slot 0: CCO (valid, gets docked)
    # slot 1: CC (invalid)
    # slot 2: CCN (valid, gets docked)
    # Remaining 2 smiles (slots 3, 4):
    # CCC (budget_exhausted)
    # C#C (budget_exhausted)
    smiles_input = ["CCO", "CC", "CCN", "CCC", "C#C"]

    # We mock standardize_smiles
    def mock_standardize(s):
        if s == "CC":
            return None  # simulates invalid SMILES
        return s

    monkeypatch.setattr("mockdock.oracle.standardize_smiles", mock_standardize)
    oracle._loader.require_fragment_match = False
    oracle._loader.filter_during_optimization = False
    oracle._loader.low_score = -5.0
    oracle._loader.high_score = -10.0

    # Mock filter_poses to return finite valid scores
    oracle._filter_poses_for_molecule = lambda smi, states: (-8.0, True, -8.0, "fake.dlg", 0)

    scores = oracle.score(smiles_input)

    assert len(scores) == 5
    assert oracle.budget_used == 3
    assert oracle.budget_remaining == 0

    results_df = oracle.results_df
    assert len(results_df) == 5

    # Check exact submission order:
    # Row 0: CCO, oracle_call = 1, skip_reason = None
    # Row 1: CC,  oracle_call = 2, skip_reason = "invalid_molecule"
    # Row 2: CCN, oracle_call = 3, skip_reason = None
    # Row 3: CCC, oracle_call = None, skip_reason = "budget_exhausted"
    # Row 4: C#C, oracle_call = None, skip_reason = "budget_exhausted"
    assert results_df["original_smiles"].to_list() == ["CCO", "CC", "CCN", "CCC", "C#C"]
    assert results_df["oracle_call"].to_list() == [1, 2, 3, None, None]
    assert results_df["skip_reason"].to_list() == [
        None,
        "invalid_molecule",
        None,
        "budget_exhausted",
        "budget_exhausted",
    ]


def test_oracle_evaluate_method(tmp_path: Path):
    run_dir = tmp_path / "run_eval"
    oracle = MDOracle("CHK1", budget=10, run_dir=run_dir, overwrite=True)

    # Populate dummy results
    df = pl.DataFrame(
        {
            "smiles": ["c1ccccc1", "c1ccccc1O"],
            "original_smiles": ["c1ccccc1", "c1ccccc1O"],
            "docking_score": [-8.0, -9.0],
            "norm_score": [0.5, 0.7],
            "reward_score": [0.5, 0.7],
            "valid_pose_found": [True, True],
            "dlg_path": ["a.dlg", "b.dlg"],
            "pose_index": [0, 0],
            "best_any_score": [-8.0, -9.0],
            "skip_reason": [None, None],
            "n_conformers": [1, 1],
            "generation_round": [1, 1],
            "generation_index": [1, 2],
            "oracle_call": [1, 2],
        }
    )
    oracle._update_results_df(df.to_dicts())

    metrics = oracle.evaluate()
    assert isinstance(metrics, dict)
    assert metrics["benchmark"] == "CHK1"
    assert metrics["budget_total"] == 10
    assert (run_dir / "eval_metrics.json").exists()
