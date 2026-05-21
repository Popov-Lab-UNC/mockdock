#!/usr/bin/env python3
# scripts/run_molskill.py
"""
Batch scores generated molecules using MolSkill.
Finds all molecule_metrics_cache.csv files, extracts SMILES, computes drug-likeness scores,
and appends a `molskill_score` column back to each cache file.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl
import numpy as np

# =====================================================================
# MolSkill Integration Setup
# =====================================================================
# TODO: To run the real MolSkill model:
# 1. Ensure you have molskill installed (`pip install molskill`)
# 2. Download or locate your model checkpoint (e.g., preference model)
# 3. Replace the placeholder function below with your actual model call:
#
#    from molskill.scorer import Scorer
#    scorer = Scorer()
#
#    def score_molecules(smiles_list: list[str]) -> list[float]:
#        # Score a batch of smiles
#        predictions = scorer.score(smiles_list)
#        return [float(p) for p in predictions]
# =====================================================================

def mock_score_molskill(smiles_list: list[str]) -> list[float]:
    """
    Placeholder scoring function for MolSkill.
    Returns simulated drug-likeness preference scores between 0.0 and 1.0.
    Replace this with real MolSkill API calls!
    """
    print(f"    [Placeholder] Scoring {len(smiles_list)} molecules with MolSkill...")
    # Just a deterministic mock score based on SMILES string characteristics
    scores = []
    for s in smiles_list:
        scores.append(float(0.2 + 0.65 * (len(s) % 17) / 17.0))
    return scores


def process_cache_file(cache_path: Path, force: bool):
    """Load molecule metrics cache, score molecules, and write back."""
    try:
        df = pl.read_csv(cache_path)
    except Exception as e:
        print(f"Error reading cache {cache_path}: {e}")
        return

    if "smiles" not in df.columns:
        print(f"Cache file {cache_path} is missing 'smiles' column. Please run analyze_experiments.py with --force once first.")
        return

    if "molskill_score" in df.columns and not force:
        print(f"  MolSkill scores already present in {cache_path.name}, skipping (use --force to re-run).")
        return

    smiles_list = df["smiles"].to_list()
    
    # Run the scoring function
    scores = mock_score_molskill(smiles_list)
    
    # Add score column and write back
    df = df.with_columns(pl.Series("molskill_score", scores))
    df.write_csv(cache_path)
    print(f"  Successfully updated {cache_path} with molskill_score!")


def main():
    parser = argparse.ArgumentParser(description="Score Mockdock generated molecules with MolSkill.")
    parser.add_argument("--exps-dir", type=Path, default=Path("exps"), help="Path to exps folder containing run directories")
    parser.add_argument("--force", action="store_true", help="Overwrite existing molskill scores")
    args = parser.parse_args()

    if not args.exps_dir.exists():
        print(f"Directory {args.exps_dir} does not exist.")
        return

    print(f"Locating cached molecule files in {args.exps_dir}...")
    cache_files = list(args.exps_dir.glob("**/molecule_metrics_cache.csv"))
    print(f"Found {len(cache_files)} cache files.")

    for i, cache_path in enumerate(cache_files, 1):
        print(f"[{i}/{len(cache_files)}] Processing {cache_path}...")
        process_cache_file(cache_path, args.force)

    print("\nMolSkill batch scoring complete!")


if __name__ == "__main__":
    main()
