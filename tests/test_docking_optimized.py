import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestAutoDockGPUOracleOptimized(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Patch sys.modules to avoid missing dependency errors during import
        modules_to_mock = [
            "polars",
            "rdkit",
            "rdkit.Chem",
            "rdkit.Chem.MolStandardize",
            "rdkit.RDLogger",
            "vina",
            "chembl_webresource_client",
            "meeko",
            "matplotlib",
            "matplotlib.pyplot",
            "seaborn",
            "scipy",
            "scipy.stats",
            "gemmi",
            "prody",
            "molscrub",
            "pyyaml",
            "tqdm",
            "aiohttp",
            "asttokens",
            "comm",
            "requests",
            "numpy",
            "pandas",
            "yaml",
        ]

        cls.mocked_modules = {name: MagicMock() for name in modules_to_mock}
        cls.modules_patcher = patch.dict("sys.modules", cls.mocked_modules)
        cls.modules_patcher.start()

        # Ensure 'fcgmb' is in path
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        )

        # Now we can safely import
        try:
            from fcgmb.docking import AutoDockGPUOracle

            cls.AutoDockGPUOracle = AutoDockGPUOracle
        except ImportError as e:
            print(f"Import error in test setup: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        cls.modules_patcher.stop()

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.receptor_dir = Path(self.tmp_dir.name) / "receptor"
        self.receptor_dir.mkdir()
        self.receptor_file = self.receptor_dir / "rec.maps.fld"
        self.receptor_file.touch()

        # Necessary files
        (self.receptor_dir / "rec.pdbqt").touch()
        (self.receptor_dir / "rec.A.map").touch()

        # Junk files
        (self.receptor_dir / "junk.log").touch()

    def tearDown(self):
        self.tmp_dir.cleanup()

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_dock_batch_symlinks_only_allowed_files(self, mock_run, mock_which):
        # Setup mocks for ADGPU execution
        mock_which.return_value = "/usr/bin/true"

        oracle = self.AutoDockGPUOracle(receptor_file=self.receptor_file)

        # Use a controlled temporary directory for the docking run
        adgpu_tmp = Path(self.tmp_dir.name) / "adgpu_run_tmp"
        adgpu_tmp.mkdir()

        # Ligand file
        lig_pdbqt = Path(self.tmp_dir.name) / "ligand.pdbqt"
        lig_pdbqt.touch()

        with patch("tempfile.TemporaryDirectory") as mock_temp_dir:
            mock_temp_dir.return_value.__enter__.return_value = str(adgpu_tmp)

            oracle.dock_batch({"C": [lig_pdbqt]}, chunk_idx=0)

            # Verify symlinks
            files = os.listdir(adgpu_tmp)
            self.assertIn("rec.maps.fld", files)
            self.assertIn("rec.pdbqt", files)
            self.assertIn("rec.A.map", files)
            self.assertNotIn("junk.log", files)

            # Check filelist.txt
            filelist_path = adgpu_tmp / "filelist.txt"
            self.assertTrue(filelist_path.exists())
            with open(filelist_path) as f:
                lines = f.read().splitlines()
                self.assertEqual(lines[0], "rec.maps.fld")


if __name__ == "__main__":
    unittest.main()
