#!/usr/bin/env python3
# scripts/run_aizynthfinder.py
"""
Batch scores generated molecules using AIZynthFinder.
Finds all molecule_metrics_cache.csv files, extracts SMILES, computes synthetic feasibility,
and appends an `aizynthfinder_score` column back to each cache file.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl
import numpy as np

# =====================================================================
# AIZynthFinder Integration Setup
# =====================================================================
# TODO: To run the real AIZynthFinder model:
# 1. Ensure you have aizynthfinder installed (`pip install aizynthfinder`)
# 2. Set up your policy and stock files (e.g., config.yml)
# 3. Replace the placeholder function below with your actual model call:
#
#    from aizynthfinder.api import SimpleAzAPI
#    # Load AIZynthFinder API
#    finder = SimpleAzAPI(configfile="path/to/your/config.yml")
#
#    def score_single_molecule(smiles: str) -> float:
#        finder.target_smiles = smiles
#        finder.tree_search()
#        finder.build_routes()
#        # Return feasibility (e.g., 1.0 if solved, 0.0 if not, or search tree score)
#        return 1.0 if finder.top_policy_route else 0.0
# =====================================================================

def mock_score_aizynthfinder(smiles_list: list[str]) -> list[float]:
    """
    Placeholder scoring function for AIZynthFinder.
    Returns simulated retro-synthetic scores between 0.0 and 1.0.
    Replace this with real AIZynthFinder API calls!
    """
    print(f"    [Placeholder] Scoring {len(smiles_list)} molecules with AIZynthFinder...")
    # Just a deterministic mock score based on SMILES hash
    scores = []
    for s in smiles_list:
        h = hash(s) % 100
        scores.append(float(0.1 + 0.8 * (h / 100.0)))
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

    if "aizynthfinder_score" in df.columns and not force:
        print(f"  AIZynthFinder scores already present in {cache_path.name}, skipping (use --force to re-run).")
        return

    smiles_list = df["smiles"].to_list()
    
    # Run the scoring function
    scores = mock_score_aizynthfinder(smiles_list)
    
    # Add score column and write back
    df = df.with_columns(pl.Series("aizynthfinder_score", scores))
    df.write_csv(cache_path)
    print(f"  Successfully updated {cache_path} with aizynthfinder_score!")


def main():
    parser = argparse.ArgumentParser(description="Score Mockdock generated molecules with AIZynthFinder.")
    parser.add_argument("--exps-dir", type=Path, default=Path("exps"), help="Path to exps folder containing run directories")
    parser.add_argument("--force", action="store_true", help="Overwrite existing aizynthfinder scores")
    args = parser.parse_args()

    if not args.exps-dir.exists():
        print(f"Directory {args.exps_dir} does not exist.")
        return

    print(f"Locating cached molecule files in {args.exps_dir}...")
    cache_files = list(args.exps_dir.glob("**/molecule_metrics_cache.csv"))
    print(f"Found {len(cache_files)} cache files.")

    for i, cache_path in enumerate(cache_files, 1):
        print(f"[{i}/{len(cache_files)}] Processing {cache_path}...")
        process_cache_file(cache_path, args.force)

    print("\nAIZynthFinder batch scoring complete!")


if __name__ == "__main__":
    main()
