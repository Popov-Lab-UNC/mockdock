from __future__ import annotations

from pathlib import Path

import polars as pl

from mockdock.loader import BenchmarkLoader


def test_loader_initial_and_validation_split_quartile(monkeypatch, tmp_path: Path):
    df = pl.DataFrame(
        {
            "canonical_smiles": ["A", "B", "C", "D"],
            "pchembl_value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    loader = BenchmarkLoader("CHK1", scratch_dir=tmp_path)
    monkeypatch.setattr("mockdock.loader.fetch_chembl_data", lambda *_: df)
    loader._pkg_bioactivity_dir = loader._bioactivity_data_dir / "nonexistent_pkg"

    full_df, threshold, act_col = loader.get_full_data_and_threshold()
    assert len(full_df) == 4
    assert act_col == "pchembl_value"
    assert threshold == 1.75

    initial = loader.get_initial_compounds()
    validation = loader.get_validation_compounds()
    assert initial["canonical_smiles"].to_list() == ["A"]
    assert validation["canonical_smiles"].to_list() == ["B", "C", "D"]


def test_loader_uses_memory_cache_after_first_load(monkeypatch, tmp_path: Path):
    calls = {"n": 0}

    def _fetch(*_):
        calls["n"] += 1
        return pl.DataFrame({"canonical_smiles": ["C"], "pchembl_value": [5.0]})

    loader = BenchmarkLoader("CHK1", scratch_dir=tmp_path)
    monkeypatch.setattr("mockdock.loader.fetch_chembl_data", _fetch)
    loader._pkg_bioactivity_dir = loader._bioactivity_data_dir / "nonexistent_pkg"

    loader.get_full_data_and_threshold()
    loader.get_full_data_and_threshold()
    assert calls["n"] == 1
