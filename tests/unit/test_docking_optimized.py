from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mockdock.docking import AutoDockGPUOracle


@pytest.mark.unit
@patch("shutil.which")
@patch("subprocess.run")
def test_dock_batch_symlinks_only_allowed_files(mock_run, mock_which, tmp_path: Path):
    del mock_run
    mock_which.return_value = "/usr/bin/true"

    receptor_dir = tmp_path / "receptor"
    receptor_dir.mkdir()
    receptor_file = receptor_dir / "rec.maps.fld"
    receptor_file.touch()
    (receptor_dir / "rec.pdbqt").touch()
    (receptor_dir / "rec.A.map").touch()
    (receptor_dir / "junk.log").touch()

    oracle = AutoDockGPUOracle(receptor_file=receptor_file)
    adgpu_tmp = tmp_path / "adgpu_run_tmp"
    adgpu_tmp.mkdir()
    lig_pdbqt = tmp_path / "ligand.pdbqt"
    lig_pdbqt.touch()

    with patch("tempfile.TemporaryDirectory") as mock_temp_dir:
        mock_temp_dir.return_value.__enter__.return_value = str(adgpu_tmp)
        oracle.dock_batch([{"smiles": "C", "pdbqt_paths": [lig_pdbqt]}], chunk_idx=0)

    files = os.listdir(adgpu_tmp)
    assert "rec.maps.fld" in files
    assert "rec.pdbqt" in files
    assert "rec.A.map" in files
    assert "junk.log" not in files
