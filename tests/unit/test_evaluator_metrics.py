from __future__ import annotations

from pathlib import Path

import polars as pl

from mockdock import evaluator as evaluator_mod


def test_compute_metrics_generates_expected_keys(tmp_path: Path, monkeypatch):
    class _FakeLoader:
        def __init__(self, benchmark_name, scratch_dir=None):
            del benchmark_name, scratch_dir
            self.fragment_smiles = "c1ccccc1"

        @staticmethod
        def get_initial_compounds():
            return pl.DataFrame({"canonical_smiles": ["CC"]})

    class _PassAllFilters:
        @staticmethod
        def evaluate(mol):
            return {"pass": True, "rules_hit": [], "mw": 200, "logp": 2.0}

    monkeypatch.setattr(evaluator_mod, "BenchmarkLoader", _FakeLoader)
    monkeypatch.setattr(evaluator_mod, "MDFilters", lambda **kwargs: _PassAllFilters())

    df = pl.DataFrame(
        {
            "smiles": ["CCO", "CCN"],
            "original_smiles": ["CC", "CCN"],
            "norm_score": [0.3, 0.9],
            "reward_score": [0.3, 0.9],
            "skip_reason": [None, None],
            "valid_pose_found": [True, False],
        }
    )
    csv_path = tmp_path / "results.csv"
    df.write_csv(csv_path)

    ev = evaluator_mod.MDEvaluator("CHK1")
    out = ev.compute_metrics(csv_path, output_path=tmp_path / "metrics.json")
    assert "validity" in out
    assert "avg_top_1" in out
    assert "valid_pose_rate" in out
    assert out["avg_top_1"] >= out["avg_top_10"]

    # Check filtered metrics
    assert "avg_top_1_filtered" in out
    assert "avg_top_10_filtered" in out
    assert "auc_top_10_filtered" in out
    assert "fraction_medchem_pass" in out


def test_compute_metrics_deduplicates_unique_smiles(tmp_path: Path, monkeypatch):
    class _FakeLoader:
        def __init__(self, benchmark_name, scratch_dir=None):
            del benchmark_name, scratch_dir
            self.fragment_smiles = "c1ccccc1"

        @staticmethod
        def get_initial_compounds():
            return pl.DataFrame({"canonical_smiles": ["CC"]})

    class _PassAllFilters:
        @staticmethod
        def evaluate(mol):
            return {"pass": True, "rules_hit": [], "mw": 200, "logp": 2.0}

    monkeypatch.setattr(evaluator_mod, "BenchmarkLoader", _FakeLoader)
    monkeypatch.setattr(evaluator_mod, "MDFilters", lambda **kwargs: _PassAllFilters())

    # CCO is repeated 3 times with 0.9, CCN is 0.1
    df = pl.DataFrame(
        {
            "smiles": ["CCO", "CCO", "CCO", "CCN"],
            "original_smiles": ["CCO", "CCO", "CCO", "CCN"],
            "norm_score": [0.9, 0.9, 0.9, 0.1],
            "reward_score": [0.9, 0.9, 0.9, 0.1],
            "skip_reason": [None, None, None, None],
            "valid_pose_found": [True, True, True, True],
        }
    )
    csv_path = tmp_path / "results.csv"
    df.write_csv(csv_path)

    ev = evaluator_mod.MDEvaluator("CHK1")
    out = ev.compute_metrics(csv_path, output_path=tmp_path / "metrics.json")

    # With deduplication: top 2 unique are CCO (0.9) and CCN (0.1) -> avg_top_10 = (0.9 + 0.1) / 2 = 0.5
    assert out["avg_top_1"] == 0.9
    assert round(out["avg_top_10"], 4) == 0.5
    assert round(out["avg_top_10_norm"], 4) == 0.5
    assert round(out["avg_top_10_filtered"], 4) == 0.5
