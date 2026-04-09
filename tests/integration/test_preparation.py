from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
from rdkit import Chem

from mockdock.analysis import DockingAnalyzer
from mockdock.ligand_prep import LigandPreparer
from mockdock.oracle import MDOracle
from mockdock.receptor import ReceptorPreparer
from mockdock.utils import check_2d_match


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_receptor_tools
def test_preparation_pipeline():
    required_execs = ["autogrid4", "mk_prepare_receptor.py", "mmtbx.reduce2"]
    missing = [exe for exe in required_execs if shutil.which(exe) is None]
    if missing:
        pytest.skip(f"Missing receptor prep executables: {missing}")

    benchmark_name = "CHK1"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        oracle = MDOracle(benchmark_name, scratch_dir=tmp_path)
        preparer = ReceptorPreparer()
        fld_path = preparer.prepare_receptor_and_grid(
            oracle.pdb_id,
            ligand_resname=oracle.ligand_resname,
            output_dir=tmp_path / "grids" / oracle.pdb_id,
            allow_bad_res=True,
        )
        assert fld_path.exists()

        ligand_preparer = LigandPreparer(n_cpus=2)
        test_smiles = [
            "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
            "Cc1cc(Nc2nc(C(=O)N3CCN[C@@H](C)C3)nc3ccc(Cl)cc23)n[nH]1",
        ]
        docking_tasks = ligand_preparer.prepare_batch(test_smiles, tmp_path / "ligands")
        assert len(docking_tasks) == len(test_smiles)

        grid_dir = tmp_path / "grids" / oracle.pdb_id
        analyzer = DockingAnalyzer(
            reference_ligand_path=grid_dir / f"{oracle.pdb_id}_ligand.pdb",
            fragment_smiles=oracle.fragment_smiles,
            rmsd_threshold=oracle.rmsd_threshold,
        )
        assert analyzer.ref_mol is not None
        assert analyzer.fragment_mol is not None
        for smi in test_smiles:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None
            assert isinstance(check_2d_match(mol, analyzer.fragment_mol), bool)
