import unittest
from unittest.mock import MagicMock, patch
import polars as pl
from pathlib import Path
import tempfile
import sys

# Mock imports before fcgmb.oracle is imported if needed,
# but here we can patch 'fcgmb.oracle.fetch_chembl_data'

from fcgmb.oracle import FCGMBOracle

class TestFCGMBOracleCaching(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.scratch_dir = Path(self.tmp_dir.name)
        self.benchmark_name = "CHEMBL4630_2R0U_CHEMBL1140535"

        # Ensure necessary dirs exist
        (self.scratch_dir / "data").mkdir(parents=True)
        (self.scratch_dir / "grids" / "2R0U").mkdir(parents=True)

        # Create a mock config file in the expected location if it doesn't verify existence differently
        # The Oracle loads config using package resources or relative paths.
        # Since we are using a real benchmark name, it should find the config in the package.

    def tearDown(self):
        self.tmp_dir.cleanup()

    @patch('fcgmb.oracle.fetch_chembl_data')
    @patch('polars.read_csv')
    def test_caching_logic(self, mock_read_csv, mock_fetch):
        # Setup mock data
        mock_df = pl.DataFrame({
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
            "pchembl_value": [5.0, 6.0],
            "canonical_smiles": ["C", "CC"]
        })

        oracle = FCGMBOracle(
            benchmark_name=self.benchmark_name,
            scratch_dir=self.scratch_dir
        )

        # 1. Test Fetch (file doesn't exist)
        mock_fetch.return_value = mock_df

        # Ensure file does not exist
        cache_file = self.scratch_dir / "data" / f"{self.benchmark_name}_chembl.csv"
        if cache_file.exists():
            cache_file.unlink()

        df, _, _ = oracle._get_full_data_and_threshold()

        self.assertTrue(mock_fetch.called, "Should fetch data if cache file missing")
        self.assertFalse(mock_read_csv.called, "Should not read csv if fetching")
        self.assertIsNotNone(oracle.chembl_data, "Should cache data in memory")
        self.assertTrue(cache_file.exists(), "Should write cache file")

        # Reset mocks
        mock_fetch.reset_mock()
        mock_read_csv.reset_mock()

        # 2. Test Memory Cache (call again)
        df2, _, _ = oracle._get_full_data_and_threshold()

        self.assertFalse(mock_fetch.called, "Should not fetch if in memory")
        self.assertFalse(mock_read_csv.called, "Should not read csv if in memory")
        self.assertTrue(df.equals(df2))

        # 3. Test File Cache (new oracle instance, file exists)
        oracle2 = FCGMBOracle(
            benchmark_name=self.benchmark_name,
            scratch_dir=self.scratch_dir
        )
        mock_read_csv.return_value = mock_df

        df3, _, _ = oracle2._get_full_data_and_threshold()

        self.assertTrue(mock_read_csv.called, "Should read from file if exists and not in memory")
        self.assertFalse(mock_fetch.called, "Should not fetch if file exists")
        self.assertIsNotNone(oracle2.chembl_data, "Should populate memory cache after reading file")

        # 4. Test Memory Cache on new instance (call again)
        mock_read_csv.reset_mock()
        df4, _, _ = oracle2._get_full_data_and_threshold()
        self.assertFalse(mock_read_csv.called, "Should use memory cache on second call")

if __name__ == '__main__':
    unittest.main()
