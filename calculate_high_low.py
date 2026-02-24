import polars as pl
import yaml
from pathlib import Path
import numpy as np


def get_system_key(config):
    target_id = config.get("target_id")
    pdb_id = config.get("pdb_id")
    doc_id = config.get("doc_id")
    assay_id = config.get("assay_id")
    key = f"{target_id}_{pdb_id}_{doc_id}"
    if assay_id:
        key += f"_{assay_id}"
    return key


def calculate_scores():
    config_dir = Path("fcgmb/configs")
    runs_dir = Path("variance_runs")
    configs = list(config_dir.glob("*.yaml"))

    results = []

    for config_path in configs:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        system_key = get_system_key(config)
        target_pdb = f"{config.get('target_id')}_{config.get('pdb_id')}"
        doc_id = str(config.get("doc_id"))

        best_docking_scores = []  # best (minimum) docking score per run
        all_molecules = None      # collected from first valid run

        for i in range(1, 6):
            run_dir = runs_dir / f"run_{i}"
            csv_path = run_dir / target_pdb / doc_id / f"{system_key}_results.csv"

            if not csv_path.exists():
                print(f"Warning: {csv_path} not found")
                continue

            df = pl.read_csv(csv_path)

            # Keep only rows with valid docking scores
            df = df.filter(
                pl.col("docking_score").is_not_null() & (pl.col("docking_score") < 900)
            )

            if df.is_empty():
                continue

            # Best docking score for this run (minimum = most negative = best)
            best_score = df.get_column("docking_score").min()
            best_docking_scores.append(best_score)

            # Collect molecule activity data from the first valid run
            if all_molecules is None:
                all_molecules = df.select(["molecule_chembl_id", "pchembl_value", "docking_score"]).unique(
                    subset=["molecule_chembl_id"]
                )

        if not best_docking_scores:
            print(f"No valid runs for {system_key}")
            continue

        # High score: mean docking score of the best-docked molecule across runs
        high_score = float(np.mean(best_docking_scores))

        # Low score: mean docking score of molecules in the bottom 25% by activity (pchembl_value)
        if all_molecules is not None:
            activities = all_molecules.get_column("pchembl_value").to_numpy()
            docking_scores = all_molecules.get_column("docking_score").to_numpy()
            q25_threshold = np.percentile(activities, 25)
            low_mask = activities <= q25_threshold
            low_score = float(np.mean(docking_scores[low_mask]))
        else:
            low_score = 0.0

        print(f"System: {config_path.stem}")
        print(f"  High score (mean docking score of best-docked molecule): {high_score:.3f}")
        print(f"  Low score (mean docking score of bottom 25% by activity): {low_score:.3f}")

        config["high_score"] = round(high_score, 3)
        config["low_score"] = round(low_score, 3)

        with open(config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        results.append({
            "config": config_path.stem,
            "high_score": high_score,
            "low_score": low_score,
        })


if __name__ == "__main__":
    calculate_scores()
