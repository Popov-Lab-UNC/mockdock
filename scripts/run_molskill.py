#!/usr/bin/env python3
# scripts/run_molskill.py
"""
Batch scores effective-yield compounds using MolSkill.

Finds all molecule_metrics_cache.csv files, scores molecules in the
Effective Yield Rate set (novel + fragment 2D match), and writes a
`molskill_score` column back to each cache file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from molskill.scorer import MolSkillScorer


def effective_yield_filter(df: pl.DataFrame) -> pl.DataFrame:
    """Return rows in the Effective Yield Rate compound set."""
    return df.filter(
        pl.col("is_novel").cast(pl.Boolean) & pl.col("has_fragment").cast(pl.Boolean)
    )


def score_smiles_batches(
    smiles_list: list[str],
    scorer: MolSkillScorer,
    batch_size: int,
) -> list[float]:
    """Score SMILES in batches and return flat float scores."""
    scores: list[float] = []
    for start in range(0, len(smiles_list), batch_size):
        batch = smiles_list[start : start + batch_size]
        batch_scores = scorer.score(batch, batch_size=batch_size)
        scores.extend(float(score) for score in batch_scores)
    return scores


def process_cache_file(
    cache_path: Path,
    scorer: MolSkillScorer,
    force: bool,
    batch_size: int,
) -> None:
    """Load molecule metrics cache, score effective-yield compounds, and write back."""
    try:
        df = pl.read_csv(cache_path)
    except Exception as e:
        print(f"Error reading cache {cache_path}: {e}")
        return

    required_cols = {"smiles", "is_novel", "has_fragment"}
    if not required_cols.issubset(set(df.columns)):
        print(
            f"Cache file {cache_path} is missing required columns {required_cols}. "
            "Run analyze_experiments.py first."
        )
        return

    if "molskill_score" in df.columns and not force:
        print(
            f"  MolSkill scores already present in {cache_path.name}, "
            "skipping (use --force to re-run)."
        )
        return

    effective_df = effective_yield_filter(df)
    n_effective = len(effective_df)
    if n_effective == 0:
        print(f"  No effective yield compounds in {cache_path.name}, skipping.")
        return

    smiles_to_score = effective_df["smiles"].to_list()
    print(f"  Scoring {n_effective} effective yield compounds...")

    scores = score_smiles_batches(smiles_to_score, scorer, batch_size)
    scores_df = pl.DataFrame({"smiles": smiles_to_score, "molskill_score": scores})

    if "molskill_score" in df.columns:
        df = df.drop("molskill_score")

    df = df.join(scores_df, on="smiles", how="left")
    df.write_csv(cache_path)
    n_scored = df.filter(pl.col("molskill_score").is_not_null()).height
    print(f"  Updated {cache_path} with molskill_score for {n_scored} compounds.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score mockdock effective-yield compounds with MolSkill."
    )
    parser.add_argument(
        "--exps-dir",
        type=Path,
        default=Path("exps"),
        help="Path to exps folder containing run directories",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing molskill scores",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for MolSkill inference",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable MolSkill progress bars",
    )
    args = parser.parse_args()

    if not args.exps_dir.exists():
        print(f"Directory {args.exps_dir} does not exist.")
        return

    print("Loading MolSkill scorer...")
    scorer = MolSkillScorer(verbose=not args.quiet)

    print(f"Locating cached molecule files in {args.exps_dir}...")
    cache_files = sorted(args.exps_dir.glob("**/molecule_metrics_cache.csv"))
    print(f"Found {len(cache_files)} cache files.")

    for i, cache_path in enumerate(cache_files, 1):
        print(f"[{i}/{len(cache_files)}] Processing {cache_path}...")
        process_cache_file(cache_path, scorer, args.force, args.batch_size)

    print("\nMolSkill batch scoring complete!")


if __name__ == "__main__":
    main()
