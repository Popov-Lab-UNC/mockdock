import os
import shutil
from pathlib import Path
from unittest.mock import patch
import pytest

from mockdock.docking import AutoDockGPUOracle
from mockdock.oracle import MDOracle


@pytest.mark.unit
@patch("shutil.which")
def test_adgpu_executable_direct_instantiation(mock_which, tmp_path):
    mock_which.return_value = "/mock/custom_adgpu"

    receptor_file = tmp_path / "rec.pdbqt"
    receptor_file.touch()

    # 1. Test direct passing to AutoDockGPUOracle
    oracle = AutoDockGPUOracle(receptor_file=receptor_file, adgpu_executable="/mock/custom_adgpu")
    assert oracle.adgpu_executable == "/mock/custom_adgpu"


@pytest.mark.unit
@patch("shutil.which")
def test_adgpu_executable_env_var(mock_which, tmp_path, monkeypatch):
    mock_which.return_value = "/mock/env_adgpu"

    receptor_file = tmp_path / "rec.pdbqt"
    receptor_file.touch()

    monkeypatch.setenv("ADGPU_EXECUTABLE", "/mock/env_adgpu")

    # 2. Test environment variable resolution
    oracle = AutoDockGPUOracle(receptor_file=receptor_file)
    assert oracle.adgpu_executable == "/mock/env_adgpu"


@pytest.mark.unit
@patch("shutil.which")
def test_adgpu_executable_import_class_default(mock_which, tmp_path, monkeypatch):
    mock_which.return_value = "/mock/class_default_adgpu"

    receptor_file = tmp_path / "rec.pdbqt"
    receptor_file.touch()

    # 3. Test changing class-level default attribute
    monkeypatch.setattr(AutoDockGPUOracle, "DEFAULT_ADGPU_EXECUTABLE", "/mock/class_default_adgpu")
    oracle = AutoDockGPUOracle(receptor_file=receptor_file)
    assert oracle.adgpu_executable == "/mock/class_default_adgpu"


@pytest.mark.unit
@patch("shutil.which")
@patch("mockdock.oracle.BenchmarkLoader")
def test_mdoracle_adgpu_executable_config(mock_loader, mock_which, tmp_path, monkeypatch):
    mock_which.return_value = "/mock/oracle_adgpu"

    # Set up mock loader return values so MDOracle init succeeds without file access errors
    mock_loader_instance = mock_loader.return_value
    mock_loader_instance.pdb_id = "2R0U"
    mock_loader_instance.fragment_smiles = "C"
    mock_loader_instance.rmsd_threshold = 2.0
    mock_loader_instance.require_fragment_match = True
    mock_loader_instance.require_pose_rmsd = True
    mock_loader_instance.filter_during_optimization = False
    mock_loader_instance.clip_reward_upper_bound = True
    mock_loader_instance.low_score = 0.0
    mock_loader_instance.high_score = 1.0
    mock_loader_instance.ligand_resname = "LIG"

    # Mock MDOracle._ensure_components to not raise file errors or load grids
    def dummy_ensure_components(self):
        pass
    monkeypatch.setattr(MDOracle, "_ensure_components", dummy_ensure_components)

    # 4. Test direct instantiation on MDOracle
    oracle = MDOracle("CHK1", adgpu_executable="/mock/oracle_adgpu")
    assert oracle._backend_config["adgpu_executable"] == "/mock/oracle_adgpu"

    # 5. Test class default resolution on MDOracle
    monkeypatch.setattr(AutoDockGPUOracle, "DEFAULT_ADGPU_EXECUTABLE", "/mock/class_default_oracle")
    oracle_default = MDOracle("CHK1")
    assert oracle_default._backend_config["adgpu_executable"] == "/mock/class_default_oracle"
