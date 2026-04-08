import tempfile
from pathlib import Path

from rdkit import Chem

from mockdock.analysis import DockingAnalyzer
from mockdock.ligand_prep import LigandPreparer
from mockdock.oracle import MDOracle
from mockdock.receptor import ReceptorPreparer
from mockdock.utils import check_2d_match


def test_preparation_pipeline():
    """
    Test the full preparation pipeline until the docking stage.
    """
    # Use a known benchmark for configuration
    benchmark_name = "CHK1"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Working in temporary directory: {tmp_path}")

        # 1. Initialize MDOracle to get config
        print(f"\n[1] Initializing MDOracle for {benchmark_name}")
        oracle = MDOracle(benchmark_name, scratch_dir=tmp_path)

        # 2. Test Receptor Preparation
        print("\n[2] Testing ReceptorPreparer")
        # We need to make sure the executables are in the path or mock them if they are not.
        # For a real integration test, we assume they are present.
        try:
            preparer = ReceptorPreparer()
        except FileNotFoundError as e:
            print(f"Skipping real receptor prep as executables are missing: {e}")
            return

        print(f"Preparing receptor and grid for {oracle.pdb_id}...")
        fld_path = preparer.prepare_receptor_and_grid(
            oracle.pdb_id,
            ligand_resname=oracle.ligand_resname,
            output_dir=tmp_path / "grids" / oracle.pdb_id,
            allow_bad_res=True,
        )

        assert fld_path.exists(), f"Grid file {fld_path} was not created"
        print(f"Successfully created grid file: {fld_path}")

        # 3. Test Ligand Preparation
        print("\n[3] Testing LigandPreparer")
        ligand_preparer = LigandPreparer(n_cpus=2)
        test_smiles = [
            "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
            "Cc1cc(Nc2nc(C(=O)N3CCN[C@@H](C)C3)nc3ccc(Cl)cc23)n[nH]1",
        ]

        ligand_dir = tmp_path / "ligands"
        docking_tasks = ligand_preparer.prepare_batch(test_smiles, ligand_dir)
        assert len(docking_tasks) == len(test_smiles)

        for i, task in enumerate(docking_tasks):
            smi = task["smiles"]
            pdbqts = task["pdbqt_paths"]
            assert smi == test_smiles[i], f"Order mismatch or SMILES mismatch at index {i}"
            assert len(pdbqts) > 0, f"No PDBQTs generated for {smi}"
            for pdbqt_path in pdbqts:
                assert pdbqt_path.exists(), f"PDBQT file {pdbqt_path} does not exist"

        print(f"Successfully prepared {len(test_smiles)} ligands.")

        # 4. Test Docking Analysis Initialization
        print("\n[4] Testing DockingAnalyzer")
        grid_dir = tmp_path / "grids" / oracle.pdb_id
        ref_path = grid_dir / f"{oracle.pdb_id}_ligand.pdb"

        analyzer = DockingAnalyzer(
            reference_ligand_path=ref_path,
            fragment_smiles=oracle.fragment_smiles,
            rmsd_threshold=oracle.rmsd_threshold,
        )

        assert analyzer.ref_mol is not None, "Failed to load reference molecule"
        assert analyzer.fragment_mol is not None, "Failed to load fragment molecule"

        if analyzer.ref_match:
            print(f"Reference molecule matches fragment. Match: {analyzer.ref_match}")
        else:
            print(
                "Warning: Reference molecule does not match fragment (this might be expected for some PDBs without bond orders)"
            )

        # 5. Verify 2D Filtering
        print("\n[5] Verifying 2D filtering")
        for smi in test_smiles:
            mol = Chem.MolFromSmiles(smi)
            is_valid = mol is not None
            has_match = check_2d_match(mol, analyzer.fragment_mol) if is_valid else False
            print(f"SMILES: {smi} valid={is_valid} matches fragment: {has_match}")

        print("\nPreparation stage test COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    test_preparation_pipeline()
