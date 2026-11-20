from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from mockdock.evaluator import MDEvaluator


def test_evaluator_requires_status_json(tmp_path: Path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="does not contain 'status.json'"):
        MDEvaluator(empty_dir)


def test_evaluator_forgiving_path_resolution(tmp_path: Path):
    run_dir = tmp_path / "run_res"
    run_dir.mkdir()
    status = {"benchmark": "CHK1", "budget_total": 1000, "budget_used": 1000}
    (run_dir / "status.json").write_text(json.dumps(status))
    (run_dir / "results.csv").write_text("smiles,reward_score\nCCO,0.5\n")

    # Pass results.csv path instead of directory
    ev = MDEvaluator(run_dir / "results.csv")
    assert ev.run_dir == run_dir
    assert ev.benchmark_name == "CHK1"
    assert ev.budget_total == 1000


def test_evaluator_lazy_results_csv_check(tmp_path: Path):
    run_dir = tmp_path / "pre_score_dir"
    run_dir.mkdir()
    status = {"benchmark": "CHK1", "budget_total": 1000, "budget_used": 0}
    (run_dir / "status.json").write_text(json.dumps(status))

    # __init__ succeeds because status.json is present
    ev = MDEvaluator(run_dir)
    assert ev.benchmark_name == "CHK1"

    # load_results raises informative error because results.csv is not written yet
    with pytest.raises(FileNotFoundError, match="Has oracle.score\\(\\) been run yet\\?"):
        ev.load_results()


def test_evaluator_subbudget_autosave_protection(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run_subbudget"
    run_dir.mkdir()
    status = {
        "benchmark": "CHK1",
        "budget_total": 1000,
        "budget_used": 1000,
        "fragment_smiles": "c1ccccc1",
    }
    (run_dir / "status.json").write_text(json.dumps(status))

    # Mock BenchmarkLoader so it doesn't need external assets
    class _FakeLoader:
        def __init__(self, benchmark_name, scratch_dir=None):
            self.fragment_smiles = "c1ccccc1"

        @staticmethod
        def get_initial_compounds():
            return pl.DataFrame({"canonical_smiles": ["CC"]})

    monkeypatch.setattr("mockdock.evaluator.BenchmarkLoader", _FakeLoader)

    # 1000 rows
    rows = [{"smiles": f"CCO{i}", "reward_score": 0.5, "oracle_call": i + 1} for i in range(1000)]
    pl.DataFrame(rows).write_csv(run_dir / "results.csv")

    ev = MDEvaluator(run_dir)

    # Evaluating at sub-budget 500: save defaults to False, eval_metrics.json must NOT be written
    metrics_500 = ev.compute_metrics(budget=500)
    assert metrics_500["evaluated_budget"] == 500
    assert metrics_500["n_evaluated_rows"] == 500
    assert not (run_dir / "eval_metrics.json").exists()

    # Evaluating at full budget 1000: save defaults to True, eval_metrics.json MUST be written
    metrics_1000 = ev.compute_metrics()
    assert metrics_1000["evaluated_budget"] is None
    assert metrics_1000["n_evaluated_rows"] == 1000
    assert (run_dir / "eval_metrics.json").exists()


def test_avg_top_k_none_guard_and_zero_padding():
    # External CSV with budget=5: avg_top_1 and avg_top_10/100 behavior
    df = pl.DataFrame(
        {
            "smiles": ["c1ccccc1", "c1ccccc1O"],
            "reward_score": [0.8, 0.6],
            "skip_reason": [None, None],
            "valid_pose_found": [True, True],
        }
    )

    ev_small = MDEvaluator.from_csv(
        results_csv=None, benchmark="CHK1", budget=5, fragment_smiles="c1ccccc1"
    )
    ev_small.load_results = lambda budget=None: df
    out = ev_small.compute_metrics(save=False)

    # With budget=5:
    # avg_top_1 is defined: 0.8
    # avg_top_10 is undefined for budget=5: must be None
    # avg_top_100 is undefined for budget=5: must be None
    assert out["avg_top_1"] == 0.8
    assert out["avg_top_10"] is None
    assert out["avg_top_100"] is None

    # With budget=10:
    # avg_top_10 is defined: zero-padded to 10 -> (0.8 + 0.6 + 8 * 0.0) / 10 = 0.14
    # avg_top_100 is undefined for budget=10: None
    ev_10 = MDEvaluator.from_csv(
        results_csv=None, benchmark="CHK1", budget=10, fragment_smiles="c1ccccc1"
    )
    ev_10.load_results = lambda budget=None: df
    out_10 = ev_10.compute_metrics(save=False)
    assert round(out_10["avg_top_10"], 4) == 0.14
    assert out_10["avg_top_100"] is None


def test_valid_pose_rate_denominator():
    # 4 molecules docked: 2 passed RMSD, 2 failed RMSD (skip_reason="failed_rmsd")
    df = pl.DataFrame(
        {
            "smiles": ["c1ccccc1", "c1ccccc1O", "c1ccccc1N", "c1ccccc1F"],
            "reward_score": [0.8, 0.6, 0.0, 0.0],
            "skip_reason": [None, None, "failed_rmsd", "failed_rmsd"],
            "valid_pose_found": [True, True, False, False],
        }
    )

    ev = MDEvaluator.from_csv(
        results_csv=None, benchmark="CHK1", budget=10, fragment_smiles="c1ccccc1"
    )
    ev.load_results = lambda budget=None: df
    out = ev.compute_metrics(save=False)

    # valid_pose_rate must be 2 / 4 = 0.5 (NOT 1.0)
    assert out["valid_pose_rate"] == 0.5


def test_auc_top_10_early_stopped_runs():
    # 2 calls completed with scores 0.5, 0.5, but budget was 4
    # Step 1: top 10 = [0.5, 0*9] = 0.05
    # Step 2: top 10 = [0.5, 0.5, 0*8] = 0.10
    # Carried forward to step 4: step 3 = 0.10, step 4 = 0.10
    # Area over 1..4: 0.5*(0.05 + 0.10) + 0.5*(0.10 + 0.10) + 0.5*(0.10 + 0.10) = 0.075 + 0.10 + 0.10 = 0.275
    # Divided by (4 - 1) = 3: 0.275 / 3 = 0.091666...
    df = pl.DataFrame({"reward_score": [0.5, 0.5], "smiles": ["CCO", "CCN"]})
    auc = MDEvaluator._auc_top_k(df, k=10, effective_budget=4)
    assert round(auc, 4) == 0.0917


def test_compute_reference_metrics():
    # CHK1 reference compounds
    ref_df = MDEvaluator.compute_reference_metrics("CHK1")
    assert isinstance(ref_df, pl.DataFrame)
    assert not ref_df.is_empty()
    assert "smiles" in ref_df.columns
    assert "qed" in ref_df.columns
    assert "sa" in ref_df.columns
    assert ref_df["qed"].null_count() == 0


def test_internal_diversity_deterministic():
    smiles = [f"c1ccccc1{i * 'C'}" for i in range(50)]
    div1 = MDEvaluator._tanimoto_diversity(smiles, max_samples=10, seed=42)
    div2 = MDEvaluator._tanimoto_diversity(smiles, max_samples=10, seed=42)
    div3 = MDEvaluator._tanimoto_diversity(smiles, max_samples=10, seed=99)
    assert div1 == div2
    assert div1 != div3
