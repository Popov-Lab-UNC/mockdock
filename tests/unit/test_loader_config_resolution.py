from __future__ import annotations

from pathlib import Path

import pytest

from mockdock.loader import BenchmarkLoader


def test_find_config_prefers_exact_then_case_variants(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "CHK1.toml").write_text("x = 1\n")

    out = BenchmarkLoader._find_config("chk1", config_dir)
    assert out.name == "CHK1.toml"


def test_find_config_raises_when_missing(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        BenchmarkLoader._find_config("missing", config_dir)
