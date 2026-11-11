import csv
import pathlib
import sys
import tempfile
import polars as pl
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from exps.libinvent.run import reconcile_libinvent_run


def test_reconcile_libinvent_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        benchmark_run_dir = tmp_path / "run" / "CHK1"
        benchmark_run_dir.mkdir(parents=True)
        out_dir = tmp_path / "outputs" / "CHK1"
        out_dir.mkdir(parents=True)

        results_csv = benchmark_run_dir / "results.csv"
        # Initial mocked results.csv with docked valid molecules
        mock_docked = [
            {
                "smiles": "c1ccccc1",
                "original_smiles": "c1ccccc1",
                "docking_score": -8.5,
                "norm_score": 0.5,
                "reward_score": 0.5,
                "valid_pose_found": True,
                "dlg_path": "/fake/path/pose.dlg",
                "pose_index": 0,
                "best_any_score": -8.5,
                "skip_reason": None,
                "n_conformers": 1,
                "generation_round": 1,
                "generation_index": 1,
            },
            {
                "smiles": "Cc1ccccc1",
                "original_smiles": "Cc1ccccc1",
                "docking_score": -9.0,
                "norm_score": 0.6,
                "reward_score": 0.6,
                "valid_pose_found": True,
                "dlg_path": "/fake/path/pose2.dlg",
                "pose_index": 1,
                "best_any_score": -9.0,
                "skip_reason": None,
                "n_conformers": 1,
                "generation_round": 1,
                "generation_index": 2,
            },
        ]
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(mock_docked[0].keys()))
            writer.writeheader()
            writer.writerows(mock_docked)

        # Mocked summary CSV from REINVENT
        # Attempt 1: Valid c1ccccc1 (state 1)
        # Attempt 2: Invalid C===C (state 0)
        # Attempt 3: Valid Cc1ccccc1 (state 1)
        # Attempt 4: Duplicate c1ccccc1 (state 2)
        # Attempt 5: Over budget molecule (index >= budget=4)
        summary_csv = out_dir / "libinvent_CHK1_20261107_120000_1.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Agent",
                    "Prior",
                    "Target",
                    "Score",
                    "SMILES",
                    "SMILES_state",
                    "Input_Scaffold",
                    "R-groups",
                    "Scaffold",
                    "mockdock_score",
                    "mockdock_score (raw)",
                    "step",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "SMILES": "c1ccccc1",
                "SMILES_state": "1",
                "step": "1",
                "mockdock_score": "0.5",
            })
            writer.writerow({
                "SMILES": "C===C",
                "SMILES_state": "0",
                "step": "1",
                "mockdock_score": "0.0",
            })
            writer.writerow({
                "SMILES": "Cc1ccccc1",
                "SMILES_state": "1",
                "step": "1",
                "mockdock_score": "0.6",
            })
            writer.writerow({
                "SMILES": "c1ccccc1",
                "SMILES_state": "2",
                "step": "2",
                "mockdock_score": "0.5",
            })
            writer.writerow({
                "SMILES": "CCc1ccccc1",
                "SMILES_state": "1",
                "step": "2",
                "mockdock_score": "0.0",
            })

        reconcile_libinvent_run(
            benchmark="CHK1",
            benchmark_run_dir=benchmark_run_dir,
            summary_csv_path=summary_csv,
            budget=4,
        )

        df = pl.read_csv(results_csv)
        assert len(df) == 5
        rows = df.to_dicts()

        # Row 0: Valid c1ccccc1
        assert rows[0]["smiles"] == "c1ccccc1"
        assert rows[0]["docking_score"] == -8.5
        assert rows[0]["skip_reason"] is None

        # Row 1: Invalid C===C
        assert rows[1]["smiles"] == "C===C"
        assert rows[1]["skip_reason"] == "invalid_molecule"
        assert rows[1]["reward_score"] == 0.0

        # Row 2: Valid Cc1ccccc1
        assert rows[2]["smiles"] == "Cc1ccccc1"
        assert rows[2]["docking_score"] == -9.0
        assert rows[2]["skip_reason"] is None

        # Row 3: Duplicate c1ccccc1
        assert rows[3]["smiles"] == "c1ccccc1"
        assert rows[3]["docking_score"] == -8.5

        # Row 4: Over budget
        assert rows[4]["skip_reason"] == "budget_exhausted"


def test_reconcile_libinvent_run_with_medchem_filter_schema():
    """Verify that >100 rows with None skip_reason followed by string skip_reason doesn't fail Polars schema builder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        benchmark_run_dir = tmp_path / "run" / "PEPCK"
        benchmark_run_dir.mkdir(parents=True)
        out_dir = tmp_path / "outputs" / "PEPCK"
        out_dir.mkdir(parents=True)

        results_csv = benchmark_run_dir / "results.csv"
        # 120 valid compounds with None skip_reason, followed by 5 compounds with failed_medchem_filter
        mock_docked = []
        for i in range(120):
            mock_docked.append(
                {
                    "smiles": f"c1ccc(C{i})cc1",
                    "original_smiles": f"c1ccc(C{i})cc1",
                    "docking_score": -8.0,
                    "norm_score": 0.5,
                    "reward_score": 0.5,
                    "valid_pose_found": True,
                    "dlg_path": f"/fake/{i}.dlg",
                    "pose_index": 0,
                    "best_any_score": -8.0,
                    "skip_reason": None,
                    "n_conformers": 1,
                    "generation_round": 1,
                    "generation_index": i + 1,
                }
            )
        for i in range(120, 125):
            mock_docked.append(
                {
                    "smiles": f"c1ccc(N{i})cc1",
                    "original_smiles": f"c1ccc(N{i})cc1",
                    "docking_score": None,
                    "norm_score": 0.0,
                    "reward_score": 0.0,
                    "valid_pose_found": False,
                    "dlg_path": None,
                    "pose_index": None,
                    "best_any_score": None,
                    "skip_reason": "failed_medchem_filter",
                    "n_conformers": 0,
                    "generation_round": 1,
                    "generation_index": i + 1,
                }
            )
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(mock_docked[0].keys()))
            writer.writeheader()
            writer.writerows(mock_docked)

        summary_csv = out_dir / "libinvent_PEPCK_20261107_120000_1.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Agent",
                    "Prior",
                    "Target",
                    "Score",
                    "SMILES",
                    "SMILES_state",
                    "Input_Scaffold",
                    "R-groups",
                    "Scaffold",
                    "mockdock_score",
                    "mockdock_score (raw)",
                    "step",
                ],
            )
            writer.writeheader()
            for r in mock_docked:
                writer.writerow(
                    {
                        "SMILES": r["smiles"],
                        "SMILES_state": "1",
                        "step": "1",
                        "mockdock_score": "0.5" if r["docking_score"] is not None else "0.0",
                    }
                )

        reconcile_libinvent_run(
            benchmark="PEPCK",
            benchmark_run_dir=benchmark_run_dir,
            summary_csv_path=summary_csv,
            budget=200,
        )

        df = pl.read_csv(results_csv)
        assert len(df) == 125
        assert df["skip_reason"][122] == "failed_medchem_filter"
