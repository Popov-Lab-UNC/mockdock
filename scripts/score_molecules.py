#!/usr/bin/env python3
# scripts/score_molecules.py
"""
Unified scoring script for mockdock generated molecules.
Supports:
1. MolSkill scoring (requires molskill environment)
2. Stoplight ADMET scoring (calls Stoplight isolated python in subprocess)
3. AIZynthFinder retrosynthetic accessibility scoring (requires aizynthfinder environment)

To prevent race conditions when running in parallel via SLURM, each scorer writes 
its results to a dedicated output CSV file (e.g., scores_molskill.csv) instead of 
modifying the shared molecule_metrics_cache.csv file directly.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import subprocess
from pathlib import Path

import polars as pl


def effective_yield_filter(df: pl.DataFrame) -> pl.DataFrame:
    """Return rows in the Effective Yield Rate compound set (novel + fragment 2D match)."""
    return df.filter(
        pl.col("is_novel").cast(pl.Boolean) & pl.col("has_fragment").cast(pl.Boolean)
    )


# ─── MOLSKILL SCORER ──────────────────────────────────────────────────

def run_molskill(cache_path: Path, force: bool, batch_size: int, quiet: bool) -> None:
    """Score molecules using MolSkill and save to scores_molskill.csv."""
    try:
        from molskill.scorer import MolSkillScorer
    except ImportError as e:
        print(f"Error: Could not import MolSkillScorer. Make sure the 'molskill' conda environment is activated. ({e})")
        sys.exit(1)

    out_path = cache_path.parent / "scores_molskill.csv"
    if out_path.exists() and not force:
        print(f"  MolSkill scores already present at {out_path.name}, skipping.")
        return

    df = pl.read_csv(cache_path)
    required_cols = {"smiles", "is_novel", "has_fragment"}
    if not required_cols.issubset(set(df.columns)):
        print(f"  Error: {cache_path.name} missing {required_cols}. Run analyze_experiments.py first.")
        return

    eff_df = effective_yield_filter(df)
    if eff_df.is_empty():
        print(f"  No effective yield compounds in {cache_path.name}, skipping.")
        return

    smiles_list = eff_df["smiles"].to_list()
    print(f"  Scoring {len(smiles_list)} effective yield compounds with MolSkill...")

    scorer = MolSkillScorer(verbose=not quiet)
    scores = []
    for start in range(0, len(smiles_list), batch_size):
        batch = smiles_list[start : start + batch_size]
        batch_scores = scorer.score(batch, batch_size=batch_size)
        scores.extend(float(s) for s in batch_scores)

    out_df = pl.DataFrame({"smiles": smiles_list, "molskill_score": scores})
    out_df.write_csv(out_path)
    print(f"  Saved MolSkill scores to {out_path}")


# ─── STOPLIGHT SCORER ─────────────────────────────────────────────────

def run_stoplight(
    cache_path: Path,
    force: bool,
    stoplight_python: str,
    stoplight_script: str,
    stoplight_dir: str,
) -> None:
    """Score molecules using Stoplight ADMET model via subprocess and save to scores_stoplight.csv."""
    out_path = cache_path.parent / "scores_stoplight.csv"
    if out_path.exists() and not force:
        print(f"  Stoplight scores already present at {out_path.name}, skipping.")
        return

    df = pl.read_csv(cache_path)
    required_cols = {"smiles", "is_novel", "has_fragment"}
    if not required_cols.issubset(set(df.columns)):
        print(f"  Error: {cache_path.name} missing {required_cols}. Run analyze_experiments.py first.")
        return

    eff_df = effective_yield_filter(df)
    if eff_df.is_empty():
        print(f"  No effective yield compounds in {cache_path.name}, skipping.")
        return

    smiles_list = eff_df["smiles"].to_list()

    # Write unique smiles to a temporary TSV file for Stoplight
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as temp_in:
        temp_in.write("smiles\n")
        for s in smiles_list:
            temp_in.write(f"{s}\n")
        temp_infile = temp_in.name

    # Create temporary file for Stoplight CSV output
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_out:
        temp_outfile = temp_out.name

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{stoplight_dir}:{env.get('PYTHONPATH', '')}"

        cmd = [
            stoplight_python,
            stoplight_script,
            "--infile",
            temp_infile,
            "--smi_col",
            "smiles",
            "--outfile",
            temp_outfile,
            "--props",
            "all",
        ]

        print(f"  Running Stoplight subprocess for {len(smiles_list)} molecules...")
        subprocess.run(
            cmd,
            cwd=stoplight_dir,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        res_df = pl.read_csv(temp_outfile)
        if "SMILES" not in res_df.columns or "OverallScore" not in res_df.columns:
            raise ValueError(f"Stoplight output missing required columns. Columns: {res_df.columns}")

        res_df = res_df.with_columns(
            pl.col("OverallScore").cast(pl.Float64, strict=False).alias("overall_score")
        )

        scores_map = {
            row["SMILES"]: row["overall_score"]
            for row in res_df.select(["SMILES", "overall_score"]).iter_rows(named=True)
        }

        scores = [scores_map.get(s, None) for s in smiles_list]
        out_df = pl.DataFrame({"smiles": smiles_list, "stoplight_score": scores})
        out_df.write_csv(out_path)
        print(f"  Saved Stoplight scores to {out_path}")

    except subprocess.CalledProcessError as e:
        print(f"  [Error] Stoplight subprocess failed with exit code {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
    except Exception as e:
        print(f"  [Error] Failed to process Stoplight scoring: {e}")
    finally:
        if os.path.exists(temp_infile):
            os.remove(temp_infile)
        if os.path.exists(temp_outfile):
            os.remove(temp_outfile)


# ─── AIZYNTHFINDER SCORER ─────────────────────────────────────────────

def run_aizynthfinder(cache_path: Path, force: bool) -> None:
    """Score molecules using AIZynthFinder and save to scores_aizynthfinder.csv."""
    # Dynamically inject paths to sys.path
    BENCHMARK_VENV_PATH = "/work/users/s/h/shuhang/benchmark/.venv/lib/python3.12/site-packages"
    AIZYNTH_VENV_PATH = "/work/users/s/h/shuhang/aizynthfinder/.venv/lib/python3.12/site-packages"

    if BENCHMARK_VENV_PATH not in sys.path:
        sys.path.append(BENCHMARK_VENV_PATH)
    if AIZYNTH_VENV_PATH not in sys.path:
        sys.path.append(AIZYNTH_VENV_PATH)

    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder
    except ImportError as e:
        print(f"Error: Could not import AiZynthFinder from {AIZYNTH_VENV_PATH}: {e}")
        sys.exit(1)

    out_path = cache_path.parent / "scores_aizynthfinder.csv"
    if out_path.exists() and not force:
        print(f"  AIZynthFinder scores already present at {out_path.name}, skipping.")
        return

    df = pl.read_csv(cache_path)
    required_cols = {"smiles", "is_novel", "has_fragment"}
    if not required_cols.issubset(set(df.columns)):
        print(f"  Error: {cache_path.name} missing {required_cols}. Run analyze_experiments.py first.")
        return

    eff_df = effective_yield_filter(df)
    if eff_df.is_empty():
        print(f"  No effective yield compounds in {cache_path.name}, skipping.")
        return

    smiles_list = eff_df["smiles"].to_list()
    print(f"  Scoring {len(smiles_list)} effective yield compounds with AIZynthFinder...")

    config_path = Path(__file__).parent / "aizynthfinder_config.yml"
    if not config_path.exists():
        print(f"  Error: Config file not found at {config_path}")
        sys.exit(1)

    try:
        finder = AiZynthFinder(configfile=str(config_path))
    except Exception as e:
        print(f"  Error initializing AiZynthFinder: {e}")
        sys.exit(1)

    scores = []
    state_scores = []
    for i, s in enumerate(smiles_list, 1):
        try:
            finder.target_smiles = s
            finder.expansion_policy.select(finder.expansion_policy.items)
            finder.filter_policy.select(finder.filter_policy.items)
            finder.stock.select(finder.stock.items)

            finder.tree_search()
            finder.build_routes()
            solved = any(rt.is_solved for rt in finder.routes.reaction_trees)
            binary_score = 1.0 if solved else 0.0

            # Extract top route's state score (RouteCollection.from_analysis sorts by StateScorer)
            top_score_dict = finder.routes.scores[0] if len(finder.routes) > 0 else {}
            state_score = top_score_dict.get("state score", 0.0) if isinstance(top_score_dict, dict) else 0.0

            print(f"    [{i}/{len(smiles_list)}] {s} -> Solved: {solved} | State Score: {state_score:.3f}")
            scores.append(binary_score)
            state_scores.append(state_score)
        except Exception as e:
            print(f"    [{i}/{len(smiles_list)}] {s} -> Error: {e} (Score: 0.0)")
            scores.append(0.0)
            state_scores.append(0.0)

    out_df = pl.DataFrame({
        "smiles": smiles_list,
        "aizynthfinder_score": scores,
        "aizynthfinder_state_score": state_scores
    })
    out_df.write_csv(out_path)
    print(f"  Saved AIZynthFinder scores to {out_path}")


# ─── MAIN ORCHESTRATOR ────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified molecule scoring utility for mockdock."
    )
    parser.add_argument(
        "--scorer",
        type=str,
        required=True,
        choices=["molskill", "stoplight", "aizynthfinder"],
        help="Which model scorer to run",
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
        help="Overwrite existing scores for the selected scorer",
    )

    # Scorer-specific optional arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for MolSkill inference",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars or verbose output",
    )
    parser.add_argument(
        "--stoplight-python",
        type=str,
        default="/work/users/s/h/shuhang/stoplight/.venv/bin/python",
        help="Path to Python interpreter in Stoplight's virtual environment",
    )
    parser.add_argument(
        "--stoplight-script",
        type=str,
        default="/work/users/s/h/shuhang/stoplight/Stoplight/stoplight.py",
        help="Path to Stoplight/stoplight.py script",
    )
    parser.add_argument(
        "--stoplight-dir",
        type=str,
        default="/work/users/s/h/shuhang/stoplight",
        help="Base directory of Stoplight codebase",
    )

    args = parser.parse_args()

    if not args.exps_dir.exists():
        print(f"Directory {args.exps_dir} does not exist.")
        sys.exit(1)

    print(f"Scanning for molecule_metrics_cache.csv files in {args.exps_dir}...")
    cache_files = sorted(args.exps_dir.glob("*/*/*/molecule_metrics_cache.csv"))
    print(f"Found {len(cache_files)} cache files.")

    for i, cache_path in enumerate(cache_files, 1):
        print(f"\n[{i}/{len(cache_files)}] Processing cache: {cache_path}")
        if args.scorer == "molskill":
            run_molskill(cache_path, args.force, args.batch_size, args.quiet)
        elif args.scorer == "stoplight":
            run_stoplight(
                cache_path,
                args.force,
                args.stoplight_python,
                args.stoplight_script,
                args.stoplight_dir,
            )
        elif args.scorer == "aizynthfinder":
            run_aizynthfinder(cache_path, args.force)

    print(f"\nScoring completed successfully for: {args.scorer.upper()}!")


if __name__ == "__main__":
    main()
