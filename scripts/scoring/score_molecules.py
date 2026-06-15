#!/usr/bin/env python3
# scripts/scoring/score_molecules.py
"""
Unified scoring script for mockdock generated molecules.
Supports:
1. MolSkill scoring (requires molskill environment)
2. Stoplight ADMET scoring (persistent Stoplight-venv worker daemons)
3. AIZynthFinder retrosynthetic accessibility scoring (persistent worker pool)

To prevent race conditions when running in parallel via SLURM, each scorer writes
its results to a dedicated output CSV file (e.g., scores_molskill.csv) instead of
modifying the shared molecule_metrics_cache.csv file directly.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from pathlib import Path
from queue import Empty, Queue

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.experiment_utils import (
    EXPENSIVE_SCORE_COLUMNS,
    REFERENCE_SET_CACHE_DIRNAME,
    REFERENCE_SET_CACHE_FILENAME,
    effective_yield_filter,
    ensure_src_on_path,
)

BENCHMARK_VENV_PATH = "/work/users/s/h/shuhang/benchmark/.venv/lib/python3.12/site-packages"
AIZYNTH_VENV_PATH = "/work/users/s/h/shuhang/aizynthfinder/.venv/lib/python3.12/site-packages"
DEFAULT_AIZYNTH_CONFIG = Path(__file__).resolve().parent / "aizynthfinder_config.yml"
if BENCHMARK_VENV_PATH not in sys.path:
    sys.path.append(BENCHMARK_VENV_PATH)

import polars as pl

# Per-process worker state populated by pool initializers.
_aizynth_worker_ctx: dict | None = None
DEFAULT_STOPLIGHT_WORKER_SCRIPT = Path(__file__).resolve().parent / "stoplight_worker_daemon.py"


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
        ensure_src_on_path()
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
        ensure_src_on_path()
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


# ─── SHARED SCORING HELPERS ───────────────────────────────────────────

def _effective_smiles_for_cache(cache_path: Path, force: bool, out_filename: str) -> list[str] | None:
    """Return effective-yield SMILES for a cache, or None when scoring should be skipped."""
    out_path = cache_path.parent / out_filename
    if out_path.exists() and not force:
        print(f"  Scores already present at {out_path.name}, skipping {cache_path}.")
        return None

    required_cols = {"smiles", "is_novel", "has_fragment"}
    df = _read_metric_cache(cache_path)
    if not required_cols.issubset(set(df.columns)):
        print(f"  Error: {cache_path.name} missing {required_cols}. Run analyze_experiments.py first.")
        return None

    eff_df = effective_yield_filter(df)
    if eff_df.is_empty():
        print(f"  No effective yield compounds in {cache_path.name}, skipping.")
        return None

    return eff_df["smiles"].to_list()


def _append_sys_path(path: str) -> None:
    if path not in sys.path:
        sys.path.append(path)


def _stoplight_timeout_sec(
    n_molecules: int,
    timeout_startup_sec: int,
    timeout_per_mol_sec: int,
) -> int | None:
    if timeout_startup_sec <= 0 and timeout_per_mol_sec <= 0:
        return None
    return timeout_startup_sec + timeout_per_mol_sec * n_molecules


# ─── STOPLIGHT SCORER ─────────────────────────────────────────────────

def _start_stoplight_daemon(
    stoplight_python: str,
    stoplight_dir: str,
    worker_script: Path,
) -> subprocess.Popen[str]:
    """Start a long-lived Stoplight worker using the Stoplight virtualenv Python."""
    env = os.environ.copy()
    env["STOPLIGHT_DIR"] = stoplight_dir
    env["PYTHONPATH"] = f"{stoplight_dir}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.Popen(
        [stoplight_python, str(worker_script)],
        cwd=stoplight_dir,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ready = proc.stdout.readline().strip() if proc.stdout else ""
    if ready != "READY":
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"Stoplight worker failed to start: {ready!r} {stderr}")
    return proc


def _run_stoplight_task_on_daemon(
    proc: subprocess.Popen[str],
    task_args: tuple[str, bool, list[str], int, int],
    timeout_sec: int | None,
) -> tuple[str, bool, str]:
    cache_path_str, _, smiles_list, _, _ = task_args
    payload = {"cache_path": cache_path_str, "smiles_list": smiles_list}
    if proc.stdin is None or proc.stdout is None:
        return cache_path_str, False, "Stoplight worker pipes are unavailable."

    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()

    response: dict[str, str | bool] = {}
    error: dict[str, Exception] = {}

    def _read_response() -> None:
        try:
            line = proc.stdout.readline()
            response["payload"] = json.loads(line)
        except Exception as exc:
            error["exc"] = exc

    reader = threading.Thread(target=_read_response, daemon=True)
    reader.start()
    reader.join(timeout=timeout_sec)
    if reader.is_alive():
        proc.kill()
        return cache_path_str, False, f"Timed out after {timeout_sec} seconds."

    if error:
        return cache_path_str, False, f"Failed to read worker response: {error['exc']}"

    result = response.get("payload", {})
    return (
        str(result.get("cache_path", cache_path_str)),
        bool(result.get("ok", False)),
        str(result.get("message", "Unknown Stoplight worker response.")),
    )


def _stoplight_daemon_worker(
    daemon: subprocess.Popen[str],
    task_queue: Queue,
    progress: dict,
    progress_lock: threading.Lock,
    total_tasks: int,
) -> None:
    while True:
        try:
            task = task_queue.get_nowait()
        except Empty:
            return

        cache_path_str, _, smiles_list, timeout_startup_sec, timeout_per_mol_sec = task
        timeout_sec = _stoplight_timeout_sec(
            len(smiles_list), timeout_startup_sec, timeout_per_mol_sec
        )
        _, ok, message = _run_stoplight_task_on_daemon(daemon, task, timeout_sec)
        with progress_lock:
            progress["completed"] += 1
            idx = progress["completed"]
        status = "OK" if ok else "ERROR"
        print(f"  [{idx}/{total_tasks}] {status} {cache_path_str}: {message}")


def _shutdown_stoplight_daemons(daemons: list[subprocess.Popen[str]]) -> None:
    for daemon in daemons:
        if daemon.stdin is not None:
            daemon.stdin.write("STOP\n")
            daemon.stdin.flush()
        daemon.wait(timeout=30)


def _collect_stoplight_tasks(
    cache_files: list[Path],
    force: bool,
    timeout_startup_sec: int,
    timeout_per_mol_sec: int,
) -> list[tuple[str, bool, list[str], int, int]]:
    tasks = []
    for cache_path in cache_files:
        smiles_list = _effective_smiles_for_cache(cache_path, force, "scores_stoplight.csv")
        if not smiles_list:
            continue
        tasks.append(
            (
                str(cache_path.resolve()),
                force,
                smiles_list,
                timeout_startup_sec,
                timeout_per_mol_sec,
            )
        )
    return tasks


def _run_stoplight_pool(
    cache_files: list[Path],
    force: bool,
    stoplight_python: str,
    stoplight_dir: str,
    timeout_startup_sec: int,
    timeout_per_mol_sec: int,
    num_workers: int,
    worker_script: Path = DEFAULT_STOPLIGHT_WORKER_SCRIPT,
) -> None:
    tasks = _collect_stoplight_tasks(
        cache_files, force, timeout_startup_sec, timeout_per_mol_sec
    )
    if not tasks:
        print("No Stoplight scoring tasks to run.")
        return

    if not worker_script.exists():
        print(f"Error: Stoplight worker script not found at {worker_script}")
        sys.exit(1)

    daemons = [
        _start_stoplight_daemon(stoplight_python, stoplight_dir, worker_script)
        for _ in range(num_workers)
    ]
    task_queue: Queue = Queue()
    for task in tasks:
        task_queue.put(task)

    progress = {"completed": 0}
    progress_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_stoplight_daemon_worker,
            args=(daemon, task_queue, progress, progress_lock, len(tasks)),
            daemon=True,
        )
        for daemon in daemons
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        _shutdown_stoplight_daemons(daemons)


# ─── AIZYNTHFINDER SCORER ─────────────────────────────────────────────

def _init_aizynth_worker(config_path: str) -> None:
    """Load AiZynthFinder once per worker process."""
    global _aizynth_worker_ctx
    _append_sys_path(BENCHMARK_VENV_PATH)
    _append_sys_path(AIZYNTH_VENV_PATH)
    
    # Restrict TensorFlow threads per process on CPU to prevent contention/thrashing
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    
    from aizynthfinder.aizynthfinder import AiZynthFinder

    _aizynth_worker_ctx = {"finder": AiZynthFinder(configfile=config_path)}


def _score_smiles_with_finder(smiles: str) -> tuple[float, float]:
    finder = _aizynth_worker_ctx["finder"]
    finder.target_smiles = smiles
    finder.expansion_policy.select(finder.expansion_policy.items)
    finder.filter_policy.select(finder.filter_policy.items)
    finder.stock.select(finder.stock.items)
    finder.tree_search()
    finder.build_routes()
    solved = any(rt.is_solved for rt in finder.routes.reaction_trees)
    binary_score = 1.0 if solved else 0.0
    top_score_dict = finder.routes.scores[0] if len(finder.routes) > 0 else {}
    state_score = (
        top_score_dict.get("state score", 0.0)
        if isinstance(top_score_dict, dict)
        else 0.0
    )
    return binary_score, state_score


def _aizynth_worker_task(
    task_args: tuple[str, int, list[str]],
) -> tuple[str, int, list[tuple[str, float, float]]]:
    """Score one SMILES chunk in-process using a persistent AiZynthFinder instance."""
    cache_path_str, chunk_idx, smiles_chunk = task_args
    if _aizynth_worker_ctx is None:
        return cache_path_str, chunk_idx, [(s, 0.0, 0.0) for s in smiles_chunk]

    results = []
    for smiles in smiles_chunk:
        try:
            binary_score, state_score = _score_smiles_with_finder(smiles)
            results.append((smiles, binary_score, state_score))
        except Exception:
            results.append((smiles, 0.0, 0.0))
    return cache_path_str, chunk_idx, results


def _collect_aizynth_tasks(
    cache_files: list[Path],
    force: bool,
    num_workers: int,
) -> tuple[list[tuple[str, int, list[str]]], dict[str, list[str]]]:
    tasks: list[tuple[str, int, list[str]]] = []
    cache_orders: dict[str, list[str]] = {}

    for cache_path in cache_files:
        smiles_list = _effective_smiles_for_cache(cache_path, force, "scores_aizynthfinder.csv")
        if not smiles_list:
            continue

        cache_path_str = str(cache_path)
        cache_orders[cache_path_str] = smiles_list
        chunk_workers = max(1, min(num_workers, len(smiles_list)))
        chunk_size = (len(smiles_list) + chunk_workers - 1) // chunk_workers
        for chunk_idx, start in enumerate(range(0, len(smiles_list), chunk_size)):
            chunk = smiles_list[start : start + chunk_size]
            if chunk:
                tasks.append((cache_path_str, chunk_idx, chunk))

    return tasks, cache_orders


def _write_aizynth_results(
    cache_orders: dict[str, list[str]],
    chunk_results: dict[str, list[tuple[int, list[tuple[str, float, float]]]]],
) -> None:
    for cache_path_str, smiles_order in cache_orders.items():
        merged: list[tuple[str, float, float]] = []
        for _, chunk in sorted(chunk_results.get(cache_path_str, []), key=lambda item: item[0]):
            merged.extend(chunk)

        results_map = {smiles: (binary, state) for smiles, binary, state in merged}
        scores = [results_map.get(s, (0.0, 0.0))[0] for s in smiles_order]
        state_scores = [results_map.get(s, (0.0, 0.0))[1] for s in smiles_order]
        out_path = Path(cache_path_str).parent / "scores_aizynthfinder.csv"
        pl.DataFrame(
            {
                "smiles": smiles_order,
                "aizynthfinder_score": scores,
                "aizynthfinder_state_score": state_scores,
            }
        ).write_csv(out_path)
        print(f"  Saved AIZynthFinder scores to {out_path}", flush=True)


def _run_aizynth_pool(
    cache_files: list[Path],
    force: bool,
    config_path: Path,
    num_workers: int,
) -> None:
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    tasks, cache_orders = _collect_aizynth_tasks(cache_files, force, num_workers)
    if not tasks:
        print("No AIZynthFinder scoring tasks to run.")
        return

    mp_ctx = mp.get_context("spawn")
    chunk_results: dict[str, list[tuple[int, list[tuple[str, float, float]]]]] = defaultdict(list)
    completed = 0
    caches_done: set[str] = set()
    expected_chunks = defaultdict(int)
    for cache_path_str, _, _ in tasks:
        expected_chunks[cache_path_str] += 1

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp_ctx,
        initializer=_init_aizynth_worker,
        initargs=(str(config_path),),
    ) as executor:
        futures = {
            executor.submit(_aizynth_worker_task, task): task for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            cache_path_str, chunk_idx, _ = task
            try:
                result_cache, result_chunk_idx, chunk_res = future.result()
                chunk_results[result_cache].append((result_chunk_idx, chunk_res))
                completed += 1
                print(
                    f"  [{completed}/{len(tasks)}] Completed chunk {result_chunk_idx} "
                    f"for {result_cache} ({len(chunk_res)} SMILES)",
                    flush=True
                )
                if len(chunk_results[result_cache]) == expected_chunks[result_cache]:
                    _write_aizynth_results(
                        {result_cache: cache_orders[result_cache]},
                        {result_cache: chunk_results[result_cache]},
                    )
                    caches_done.add(result_cache)
            except Exception as e:
                completed += 1
                print(
                    f"  [{completed}/{len(tasks)}] ERROR chunk {chunk_idx} "
                    f"for {cache_path_str}: {e}",
                    flush=True
                )

    missing = set(cache_orders) - caches_done
    if missing:
        print(f"  Warning: AIZynthFinder did not complete {len(missing)} cache file(s).")


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
    parser.add_argument(
        "--stoplight-max-workers",
        type=int,
        default=None,
        help=(
            "Maximum parallel Stoplight worker processes. Defaults to STOPLIGHT_MAX_WORKERS "
            "if set, otherwise SLURM_CPUS_PER_TASK/os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--stoplight-timeout-startup-sec",
        type=int,
        default=int(os.environ.get("STOPLIGHT_TIMEOUT_STARTUP_SEC", "900")),
        help="Fixed Stoplight timeout allowance per cache in seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--stoplight-timeout-per-mol-sec",
        type=int,
        default=int(os.environ.get("STOPLIGHT_TIMEOUT_PER_MOL_SEC", "15")),
        help="Additional Stoplight timeout allowance per molecule in seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--aizynth-max-workers",
        type=int,
        default=None,
        help=(
            "Maximum parallel AIZynthFinder worker processes. Defaults to AIZYNTH_MAX_WORKERS "
            "if set, otherwise 1 (single GPU worker)."
        ),
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
        allocated_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
        env_worker_limit = os.environ.get("STOPLIGHT_MAX_WORKERS")
        worker_limit = args.stoplight_max_workers
        if worker_limit is None and env_worker_limit:
            worker_limit = int(env_worker_limit)
        if worker_limit is None:
            worker_limit = allocated_workers
        num_workers = max(1, min(len(cache_files), allocated_workers, worker_limit))
        print(
            "\nRunning Stoplight ADMET scorer with "
            f"{num_workers} persistent worker processes "
            f"(allocated CPUs: {allocated_workers}, worker cap: {worker_limit})."
        )
        if args.stoplight_timeout_startup_sec > 0 or args.stoplight_timeout_per_mol_sec > 0:
            print(
                "Stoplight timeout budget: "
                f"{args.stoplight_timeout_startup_sec}s startup + "
                f"{args.stoplight_timeout_per_mol_sec}s per molecule."
            )
        _run_stoplight_pool(
            cache_files=cache_files,
            force=args.force,
            stoplight_python=args.stoplight_python,
            stoplight_dir=args.stoplight_dir,
            timeout_startup_sec=args.stoplight_timeout_startup_sec,
            timeout_per_mol_sec=args.stoplight_timeout_per_mol_sec,
            num_workers=num_workers,
        )
    elif args.scorer == "aizynthfinder":
        allocated_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
        env_worker_limit = os.environ.get("AIZYNTH_MAX_WORKERS")
        worker_limit = args.aizynth_max_workers
        if worker_limit is None and env_worker_limit:
            worker_limit = int(env_worker_limit)
        if worker_limit is None:
            worker_limit = 1
        num_workers = max(1, min(len(cache_files), worker_limit))
        print(
            "\nRunning AIZynthFinder scorer with "
            f"{num_workers} persistent worker process(es) "
            f"(allocated CPUs: {allocated_workers}, worker cap: {worker_limit})."
        )
        _run_aizynth_pool(
            cache_files=cache_files,
            force=args.force,
            config_path=DEFAULT_AIZYNTH_CONFIG,
            num_workers=num_workers,
        )
    else:
        for i, cache_path in enumerate(cache_files, 1):
            print(f"\n[{i}/{len(cache_files)}] Processing cache: {cache_path}")
            run_molskill(cache_path, args.force, args.batch_size, args.quiet)

    print(f"\nScoring completed successfully for: {args.scorer.upper()}!")


if __name__ == "__main__":
    main()
