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

    class _NoPains:
        @staticmethod
        def HasMatch(_):
            return False

    monkeypatch.setattr(evaluator_mod, "BenchmarkLoader", _FakeLoader)
    monkeypatch.setattr(
        evaluator_mod.MDEvaluator, "_build_pains_catalog", staticmethod(lambda: _NoPains())
    )

    df = pl.DataFrame(
        {
            "smiles": ["CCO", "CCN"],
            "original_smiles": ["CC", "CCN"],
            "normalized_score": [0.3, 0.9],
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
