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
BENCHMARK_VENV_PATH = "/work/users/s/h/shuhang/benchmark/.venv/lib/python3.12/site-packages"
if BENCHMARK_VENV_PATH not in sys.path:
    sys.path.append(BENCHMARK_VENV_PATH)

import polars as pl

REFERENCE_SET_CACHE_DIRNAME = "reference_set_scores"
REFERENCE_SET_CACHE_FILENAME = "molecule_metrics_cache.csv"
EXPENSIVE_SCORE_COLUMNS = {
    "molskill_score",
    "stoplight_score",
    "aizynthfinder_score",
    "aizynthfinder_state_score",
}


def effective_yield_filter(df: pl.DataFrame) -> pl.DataFrame:
    """Return rows in the Effective Yield Rate compound set (novel + fragment 2D match)."""
    return df.filter(
        pl.col("is_novel").cast(pl.Boolean) & pl.col("has_fragment").cast(pl.Boolean)
    )


def _reference_cache_path(reference_cache_dir: Path, target: str) -> Path:
    return reference_cache_dir / target / REFERENCE_SET_CACHE_FILENAME


def _read_metric_cache(cache_path: Path) -> pl.DataFrame:
    """Read a metric cache and remove legacy expensive-score columns if present."""
    df = pl.read_csv(cache_path)
    legacy_score_cols = [col for col in EXPENSIVE_SCORE_COLUMNS if col in df.columns]
    if legacy_score_cols:
        df = df.drop(legacy_score_cols)
        df.write_csv(cache_path)
        print(
            f"  Cleaned {cache_path}: moved expensive scores out of the shared metric cache."
        )
    return df


def _discover_reference_targets(exps_dir: Path | None) -> list[str]:
    """Use experiment targets when available; otherwise fall back to bundled benchmarks."""
    targets = set()
    if exps_dir is not None and exps_dir.exists():
        for results_csv in exps_dir.glob("*/*/*/results.csv"):
            targets.add(results_csv.parent.name)
    if targets:
        return sorted(targets)

    try:
        from mockdock.loader import BenchmarkLoader
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from mockdock.loader import BenchmarkLoader

    return BenchmarkLoader.list_benchmarks()


def build_reference_set_cache(target: str, reference_cache_dir: Path, force: bool) -> Path | None:
    """Create the shared reference-set molecule cache used by all expensive scorers."""
    cache_path = _reference_cache_path(reference_cache_dir, target)
    required_cols = {"smiles", "qed", "sa", "is_novel", "has_fragment", "passes_medchem"}
    if cache_path.exists() and not force:
        try:
            cached_df = _read_metric_cache(cache_path)
            if required_cols.issubset(set(cached_df.columns)):
                print(f"  Reference cache already present for {target}, skipping QED/SA.")
                return cache_path
            print(f"  Reference cache for {target} missing required columns, recomputing.")
        except Exception as e:
            print(f"  Error reading reference cache for {target}: {e}. Recomputing.")

    try:
        from rdkit import Chem
        from rdkit.Chem import QED
        from mockdock import MDEvaluator
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from rdkit import Chem
        from rdkit.Chem import QED
        from mockdock import MDEvaluator

    evaluator = MDEvaluator(target)
    reference_df, _, _ = evaluator._loader.get_full_data_and_threshold()
    if reference_df.is_empty():
        print(f"  No reference-set data found for {target}.")
        return None

    smiles_col = (
        "canonical_smiles"
        if "canonical_smiles" in reference_df.columns
        else "smiles"
        if "smiles" in reference_df.columns
        else None
    )
    if smiles_col is None:
        print(f"  Reference-set data for {target} has no SMILES column.")
        return None

    seen = set()
    rows = []
    for row_data in reference_df.iter_rows(named=True):
        mol = Chem.MolFromSmiles(str(row_data[smiles_col]))
        if mol is None:
            continue
        smiles = Chem.MolToSmiles(mol)
        if smiles in seen:
            continue
        seen.add(smiles)
        row = {
            "smiles": smiles,
            "qed": float(QED.qed(mol)),
            "sa": float(evaluator._sa_score(mol)),
            # Reference-set compounds are the baseline set, so all are retained by scorers.
            "is_novel": True,
            "has_fragment": True,
            "passes_medchem": True,
        }
        for optional_col in ("molecule_chembl_id", "pchembl_value"):
            if optional_col in reference_df.columns:
                row[optional_col] = row_data.get(optional_col)
        rows.append(row)

    if not rows:
        print(f"  No valid reference-set SMILES found for {target}.")
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(cache_path)
    print(f"  Saved reference cache for {target} to {cache_path}")
    return cache_path


# ─── MOLSKILL SCORER ──────────────────────────────────────────────────

def run_molskill(cache_path: Path, force: bool, batch_size: int, quiet: bool) -> None:
    """Score molecules using MolSkill and save to scores_molskill.csv."""
    try:
        from molskill.scorer import MolSkillScorer
    except ImportError as e:
        print(f"Error: Could not import MolSkillScorer. Make sure the 'molskill' conda environment is activated. ({e})")
        sys.exit(1)

    df = _read_metric_cache(cache_path)
    out_path = cache_path.parent / "scores_molskill.csv"
    if out_path.exists() and not force:
        print(f"  MolSkill scores already present at {out_path.name}, skipping.")
        return

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
    df = _read_metric_cache(cache_path)
    out_path = cache_path.parent / "scores_stoplight.csv"
    if out_path.exists() and not force:
        print(f"  Stoplight scores already present at {out_path.name}, skipping.")
        return

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

def _aizynthfinder_worker(smiles_chunk: list[str], config_path: str) -> list[tuple[str, float, float]]:
    """Worker process targeting a chunk of SMILES sequentially with a single AiZynthFinder instance."""
    import sys
    BENCHMARK_VENV_PATH = "/work/users/s/h/shuhang/benchmark/.venv/lib/python3.12/site-packages"
    AIZYNTH_VENV_PATH = "/work/users/s/h/shuhang/aizynthfinder/.venv/lib/python3.12/site-packages"

    if BENCHMARK_VENV_PATH not in sys.path:
        sys.path.append(BENCHMARK_VENV_PATH)
    if AIZYNTH_VENV_PATH not in sys.path:
        sys.path.append(AIZYNTH_VENV_PATH)

    from aizynthfinder.aizynthfinder import AiZynthFinder

    try:
        finder = AiZynthFinder(configfile=config_path)
    except Exception as e:
        print(f"  [Worker] Error initializing AiZynthFinder: {e}")
        return [(s, 0.0, 0.0) for s in smiles_chunk]

    results = []
    for s in smiles_chunk:
        try:
            finder.target_smiles = s
            finder.expansion_policy.select(finder.expansion_policy.items)
            finder.filter_policy.select(finder.filter_policy.items)
            finder.stock.select(finder.stock.items)

            finder.tree_search()
            finder.build_routes()
            solved = any(rt.is_solved for rt in finder.routes.reaction_trees)
            binary_score = 1.0 if solved else 0.0

            # Extract top route's state score
            top_score_dict = finder.routes.scores[0] if len(finder.routes) > 0 else {}
            state_score = top_score_dict.get("state score", 0.0) if isinstance(top_score_dict, dict) else 0.0
            results.append((s, binary_score, state_score))
        except Exception as e:
            results.append((s, 0.0, 0.0))
    return results


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

    df = _read_metric_cache(cache_path)
    out_path = cache_path.parent / "scores_aizynthfinder.csv"
    if out_path.exists() and not force:
        print(f"  AIZynthFinder scores already present at {out_path.name}, skipping.")
        return

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

    # Determine standard multiprocessing settings, using SLURM allocation if present
    import os
    from concurrent.futures import ProcessPoolExecutor

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    # AIZynthFinder is highly memory-intensive (approx 4-6 GB per worker).
    # To prevent OOM errors, limit the number of parallel workers to at most 3.
    num_workers = min(num_workers, 3)
    # Prevent over-allocating on very small jobs
    num_workers = min(num_workers, len(smiles_list))
    
    print(f"  Running AIZynthFinder with {num_workers} parallel workers...")

    # Partition SMILES list into chunks
    chunk_size = (len(smiles_list) + num_workers - 1) // num_workers
    chunks = [smiles_list[i : i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]

    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_aizynthfinder_worker, chunk, str(config_path)) for chunk in chunks]
        for idx, future in enumerate(futures, 1):
            try:
                chunk_res = future.result()
                all_results.extend(chunk_res)
                print(f"    Completed batch {idx}/{len(futures)} ({len(chunk_res)} SMILES)")
            except Exception as e:
                print(f"    [Error] Worker batch {idx} failed: {e}")

    # Map parallel results back to original order
    results_map = {res[0]: (res[1], res[2]) for res in all_results}
    scores = [results_map.get(s, (0.0, 0.0))[0] for s in smiles_list]
    state_scores = [results_map.get(s, (0.0, 0.0))[1] for s in smiles_list]

    out_df = pl.DataFrame({
        "smiles": smiles_list,
        "aizynthfinder_score": scores,
        "aizynthfinder_state_score": state_scores
    })
    out_df.write_csv(out_path)
    print(f"  Saved AIZynthFinder scores to {out_path}")


def _stoplight_worker(task_args):
    cache_p, force, s_python, s_script, s_dir = task_args
    try:
        run_stoplight(cache_p, force, s_python, s_script, s_dir)
        return True
    except Exception as e:
        print(f"Error running Stoplight for {cache_p}: {e}")
        return False


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
    parser.add_argument(
        "--reference-cache-dir",
        type=Path,
        default=None,
        help="Shared cache directory for reference-set molecule metrics and scores",
    )
    parser.add_argument(
        "--reference-targets",
        nargs="*",
        default=None,
        help="Optional benchmark targets to score for the reference set",
    )
    parser.add_argument(
        "--skip-reference-set",
        action="store_true",
        help="Only score generated-molecule caches under --exps-dir",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Only build/score reference-set caches",
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

    if not args.reference_only and not args.exps_dir.exists():
        print(f"Directory {args.exps_dir} does not exist.")
        sys.exit(1)

    cache_files = []
    if not args.reference_only:
        print(f"Scanning for molecule_metrics_cache.csv files in {args.exps_dir}...")
        cache_files = sorted(args.exps_dir.glob("*/*/*/molecule_metrics_cache.csv"))
        print(f"Found {len(cache_files)} generated-molecule cache files.")

    if not args.skip_reference_set:
        reference_cache_dir = args.reference_cache_dir or (
            args.exps_dir.parent / REFERENCE_SET_CACHE_DIRNAME
        )
        reference_targets = args.reference_targets or _discover_reference_targets(args.exps_dir)
        print(
            f"Preparing {len(reference_targets)} reference-set caches in {reference_cache_dir}..."
        )
        for target in reference_targets:
            cache_path = build_reference_set_cache(target, reference_cache_dir, force=args.force)
            if cache_path is not None:
                cache_files.append(cache_path)

    print(f"Scoring {len(cache_files)} cache files.")

    if args.scorer == "stoplight":
        import os
        from concurrent.futures import ProcessPoolExecutor
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
        print(f"\nRunning Stoplight ADMET scorer in parallel with {num_workers} workers...")

        tasks = [
            (
                cache_path,
                args.force,
                args.stoplight_python,
                args.stoplight_script,
                args.stoplight_dir,
            )
            for cache_path in cache_files
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(_stoplight_worker, tasks))
    else:
        for i, cache_path in enumerate(cache_files, 1):
            print(f"\n[{i}/{len(cache_files)}] Processing cache: {cache_path}")
            if args.scorer == "molskill":
                run_molskill(cache_path, args.force, args.batch_size, args.quiet)
            elif args.scorer == "aizynthfinder":
                run_aizynthfinder(cache_path, args.force)

    print(f"\nScoring completed successfully for: {args.scorer.upper()}!")


if __name__ == "__main__":
    main()
