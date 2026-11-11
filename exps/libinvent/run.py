"""
REINVENT4 Lib-INVENT x mockdock Benchmark Evaluation
===================================================
Runs REINVENT4 Lib-INVENT against mockdock benchmarks via a custom scoring
component plugin that calls MDOracle.score().
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import math
import os
import pathlib
import shutil
import subprocess
import sys
from collections import defaultdict

import click
import polars as pl

PROJECT_SRC = (pathlib.Path(__file__).resolve().parents[2] / "src").resolve()
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from mockdock import MDEvaluator, MDOracle

BENCHMARKS = ["DPP4", "CHK1", "ITK", "PEPCK", "TTK", "VEGFR2", "PptT"]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
TEMPLATE_PATH = SCRIPT_DIR / "templates" / "libinvent_base.toml"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _require_scaffold(oracle: MDOracle) -> str:
    scaffold = oracle.libinvent_scaffold_with_dummies or oracle.fragment_smiles_with_dummies
    if not scaffold:
        raise ValueError(
            f"{oracle.benchmark_name}: no LibInvent-compatible scaffold is set in mockdock config."
        )
    if scaffold.count("*") < 2:
        raise ValueError(
            f"{oracle.benchmark_name}: scaffold '{scaffold}' does not contain two attachment points for Lib-INVENT."
        )
    return scaffold


def warmup_oracle(oracle: MDOracle, output_dir: pathlib.Path, n: int = 25) -> None:
    """Pre-score a small set of initial compounds to warm up docking and
    establish baseline data.  Uses at most *n* oracle calls."""
    initial_df = oracle.get_initial_compounds()
    if initial_df.is_empty():
        log.info("No initial compounds available for warmup.")
        return

    smiles_col = "canonical_smiles"
    subset = initial_df.head(n)
    smiles_list = subset[smiles_col].to_list()

    if "score" in subset.columns:
        log.info(
            "Using pre-computed docking scores for %d warmup compounds.",
            len(smiles_list),
        )
        scores = {
            row["canonical_smiles"]: row["score"]
            for row in subset.iter_rows(named=True)
        }
    else:
        log.info("Warming up oracle with %d initial compounds …", len(smiles_list))
        scores = oracle.score(smiles_list)

    nonzero = sum(1 for v in scores.values() if v > 0)
    log.info("Warmup complete: %d/%d compounds scored > 0.", nonzero, len(smiles_list))

    # Save warmup results alongside the main outputs
    warmup_csv = output_dir / "initial_compounds_warmup.csv"
    with open(warmup_csv, "w") as fh:
        fh.write("smiles,score\n")
        for smi, score in scores.items():
            fh.write(f"{smi},{score}\n")
    log.info("Warmup results saved to %s", warmup_csv)


def _render_toml(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(key, value)
    return rendered


def _run_reinvent(
    toml_file: pathlib.Path,
    log_file: pathlib.Path,
    plugin_root: pathlib.Path,
    budget_stop_marker: pathlib.Path,
    dry_run: bool,
):
    reinvent_bin = shutil.which("reinvent") or str(pathlib.Path(sys.executable).parent / "reinvent")
    cmd = [reinvent_bin, "-l", str(log_file), str(toml_file)]
    env = os.environ.copy()
    add_paths = [str(plugin_root), str((SCRIPT_DIR.parents[1] / "src").resolve())]
    env["PYTHONPATH"] = os.pathsep.join(add_paths + [env.get("PYTHONPATH", "")]).strip(
        os.pathsep
    )
    if dry_run:
        log.info("Dry run: would execute: %s", " ".join(cmd))
        return
    proc = subprocess.run(cmd, check=False, env=env)
    if proc.returncode == 0:
        return
    if budget_stop_marker.exists():
        log.info("REINVENT stopped after consuming configured budget.")
        return
    raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_benchmark(
    benchmark: str,
    budget: int,
    out_root: pathlib.Path,
    run_parent: pathlib.Path,
    prior_file: pathlib.Path,
    n_warmup: int,
    device: str,
    docking_backend: str,
    clip_reward_upper_bound: bool,
    require_pose_rmsd: bool | None,
    dry_run: bool,
):
    output_dir = out_root / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_run_dir = run_parent / benchmark
    benchmark_run_dir.mkdir(parents=True, exist_ok=True)

    oracle = MDOracle(
        benchmark_name=benchmark,
        budget=budget,
        run_dir=benchmark_run_dir,
        docking_backend=docking_backend,
        clip_reward_upper_bound=clip_reward_upper_bound,
        require_pose_rmsd=require_pose_rmsd,
        model="libinvent",
    )
    scaffold = _require_scaffold(oracle)

    scaffold_file = output_dir / "scaffolds.smi"
    scaffold_file.write_text(f"{scaffold}\n", encoding="utf-8")

    if n_warmup > 0:
        warmup_oracle(oracle, output_dir, n=n_warmup)

    remaining_budget = oracle.budget_remaining
    log.info("Oracle budget remaining after warmup: %d", remaining_budget)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keep RL running long enough that budget, not step count, controls termination.
    max_steps = max(1_000_000, math.ceil(remaining_budget / 64))
    eff_pose_rmsd = (
        require_pose_rmsd if require_pose_rmsd is not None else oracle.require_pose_rmsd
    )
    toml_values = {
        "__DEVICE__": device,
        "__TB_LOGDIR__": str(output_dir / "tb_logs"),
        "__JSON_OUT_CONFIG__": str(output_dir / f"config_{ts}.json"),
        "__SUMMARY_CSV_PREFIX__": str(output_dir / f"libinvent_{benchmark}_{ts}"),
        "__PRIOR_FILE__": str(prior_file),
        "__AGENT_FILE__": str(prior_file),
        "__SCAFFOLDS_FILE__": str(scaffold_file),
        "__CHKPT_FILE__": str(output_dir / f"libinvent_{benchmark}_{ts}.chkpt"),
        "__MAX_STEPS__": str(max_steps),
        "__BENCHMARK__": benchmark,
        "__BUDGET__": str(remaining_budget),
        "__RUN_DIR__": str(benchmark_run_dir),
        "__DOCKING_BACKEND__": docking_backend,
        "__CLIP_REWARD_UPPER_BOUND__": "true" if clip_reward_upper_bound else "false",
        "__REQUIRE_POSE_RMSD__": "true" if eff_pose_rmsd else "false",
    }
    toml_text = _render_toml(template, toml_values)
    toml_file = output_dir / f"libinvent_{benchmark}_{ts}.toml"
    toml_file.write_text(toml_text, encoding="utf-8")

    run_log = output_dir / f"reinvent_{benchmark}_{ts}.log"
    _run_reinvent(
        toml_file=toml_file,
        log_file=run_log,
        plugin_root=SCRIPT_DIR,
        budget_stop_marker=benchmark_run_dir / "budget_exhausted.json",
        dry_run=dry_run,
    )

    if not dry_run:
        summary_csv_candidates = sorted(output_dir.glob(f"libinvent_{benchmark}_{ts}*.csv"))
        if summary_csv_candidates:
            reconcile_libinvent_run(
                benchmark=benchmark,
                benchmark_run_dir=benchmark_run_dir,
                summary_csv_path=summary_csv_candidates[0],
                budget=budget,
            )
        else:
            log.warning(
                "No summary CSV found for reconciliation matching libinvent_%s_%s*.csv in %s",
                benchmark,
                ts,
                output_dir,
            )

    log.info("Benchmark %s finished. Outputs: %s", benchmark, output_dir)


def reconcile_libinvent_run(
    benchmark: str,
    benchmark_run_dir: pathlib.Path,
    summary_csv_path: pathlib.Path,
    budget: int = 1000,
) -> None:
    """Reconcile LibINVENT results.csv with REINVENT4 raw generation summary.

    Ensures invalid assemblies (state 0) and duplicates (state 2) are counted
    against the oracle budget on identical footing with all other generative models.
    """
    results_csv = benchmark_run_dir / "results.csv"
    if not results_csv.exists() or not summary_csv_path.exists():
        log.warning("Cannot reconcile: missing %s or %s", results_csv, summary_csv_path)
        return

    existing_df = pl.read_csv(results_csv)
    docked_rows_by_smiles = defaultdict(list)
    for row in existing_df.iter_rows(named=True):
        smi = row["original_smiles"] or row["smiles"]
        docked_rows_by_smiles[smi].append(row)

    with open(summary_csv_path, mode="r", encoding="utf-8") as fh:
        raw_summary_rows = list(csv.DictReader(fh))

    reconciled_rows = []
    docked_usage_counter = defaultdict(int)

    for idx, row in enumerate(raw_summary_rows):
        smi = row.get("SMILES", "")
        state = str(row.get("SMILES_state", "1")).strip()
        step = int(row.get("step", 1))
        is_over_budget = idx >= budget

        if is_over_budget:
            reconciled_rows.append(
                {
                    "smiles": smi,
                    "original_smiles": smi,
                    "docking_score": None,
                    "norm_score": 0.0,
                    "reward_score": 0.0,
                    "valid_pose_found": False,
                    "dlg_path": None,
                    "pose_index": None,
                    "best_any_score": None,
                    "skip_reason": "budget_exhausted",
                    "n_conformers": 0,
                    "generation_round": step,
                    "generation_index": idx + 1,
                }
            )
            continue

        if state == "0":  # INVALID
            reconciled_rows.append(
                {
                    "smiles": smi,
                    "original_smiles": smi,
                    "docking_score": None,
                    "norm_score": 0.0,
                    "reward_score": 0.0,
                    "valid_pose_found": False,
                    "dlg_path": None,
                    "pose_index": None,
                    "best_any_score": None,
                    "skip_reason": "invalid_molecule",
                    "n_conformers": 0,
                    "generation_round": step,
                    "generation_index": idx + 1,
                }
            )
        else:  # VALID (1) or DUPLICATE (2)
            candidates = docked_rows_by_smiles.get(smi, [])
            usage = docked_usage_counter[smi]
            if usage < len(candidates):
                base_row = dict(candidates[usage])
                docked_usage_counter[smi] += 1
            elif candidates:
                base_row = dict(candidates[0])
            else:
                base_row = {
                    "smiles": smi,
                    "original_smiles": smi,
                    "docking_score": None,
                    "norm_score": 0.0,
                    "reward_score": 0.0,
                    "valid_pose_found": False,
                    "dlg_path": None,
                    "pose_index": None,
                    "best_any_score": None,
                    "skip_reason": "duplicate" if state == "2" else None,
                    "n_conformers": 0,
                }
            base_row["generation_round"] = step
            base_row["generation_index"] = idx + 1
            reconciled_rows.append(base_row)

    SCHEMA = {
        "smiles": pl.String,
        "original_smiles": pl.String,
        "docking_score": pl.Float64,
        "norm_score": pl.Float64,
        "reward_score": pl.Float64,
        "valid_pose_found": pl.Boolean,
        "dlg_path": pl.String,
        "pose_index": pl.Int64,
        "best_any_score": pl.Float64,
        "skip_reason": pl.String,
        "n_conformers": pl.Int64,
        "generation_round": pl.Int64,
        "generation_index": pl.Int64,
    }

    clean_rows = []
    for r in reconciled_rows:
        clean_rows.append(
            {
                "smiles": str(r.get("smiles", "")) if r.get("smiles") is not None else "",
                "original_smiles": (
                    str(r.get("original_smiles", "")) if r.get("original_smiles") is not None else ""
                ),
                "docking_score": (
                    float(r["docking_score"]) if r.get("docking_score") is not None else None
                ),
                "norm_score": float(r["norm_score"]) if r.get("norm_score") is not None else 0.0,
                "reward_score": (
                    float(r["reward_score"]) if r.get("reward_score") is not None else 0.0
                ),
                "valid_pose_found": bool(r.get("valid_pose_found", False)),
                "dlg_path": str(r["dlg_path"]) if r.get("dlg_path") is not None else None,
                "pose_index": (
                    int(r["pose_index"]) if r.get("pose_index") is not None else None
                ),
                "best_any_score": (
                    float(r["best_any_score"]) if r.get("best_any_score") is not None else None
                ),
                "skip_reason": (
                    str(r["skip_reason"]) if r.get("skip_reason") is not None else None
                ),
                "n_conformers": (
                    int(r.get("n_conformers", 0)) if r.get("n_conformers") is not None else 0
                ),
                "generation_round": int(r.get("generation_round", 1)),
                "generation_index": int(r.get("generation_index", 1)),
            }
        )

    reconciled_df = pl.DataFrame(clean_rows, schema=SCHEMA)
    reconciled_df.write_csv(results_csv)
    log.info(
        "Reconciled %s with %d raw attempts from %s",
        results_csv,
        len(reconciled_df),
        summary_csv_path.name,
    )

    try:
        ev = MDEvaluator(benchmark, model="libinvent")
        eval_metrics = ev.compute_metrics(
            results_csv, output_path=benchmark_run_dir / "eval_metrics.json"
        )
        log.info(
            "Updated eval_metrics.json for %s (validity=%.3f, uniqueness=%.3f, effective_yield=%.3f)",
            benchmark,
            eval_metrics.get("validity", 0.0),
            eval_metrics.get("uniqueness", 0.0),
            eval_metrics.get("effective_yield_rate", 0.0),
        )
    except Exception as exc:
        log.warning("Failed to recompute eval_metrics for %s: %s", benchmark, exc)


@click.command()
@click.option(
    "--benchmark",
    "benchmarks",
    multiple=True,
    type=click.Choice(BENCHMARKS, case_sensitive=False),
    help="Benchmark(s) to run. Defaults to all configured benchmarks.",
)
@click.option("--budget", default=1000, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int, help="Logged only.")
@click.option(
    "--out",
    default=None,
    help="Output directory root. Defaults to <script_dir>/outputs.",
)
@click.option(
    "--run-dir",
    default=None,
    help="Parent directory for benchmark run directories. Defaults to run_<timestamp>.",
)
@click.option(
    "--prior-file",
    default=None,
    help="Path to libinvent.prior. Defaults to $LIBINVENT_PRIOR or ./priors/libinvent.prior.",
)
@click.option(
    "--n-warmup",
    default=25,
    show_default=True,
    help="Number of initial compounds to pre-score as oracle warmup (0 to skip).",
)
@click.option("--device", default="cuda:0", show_default=True)
@click.option(
    "--docking-backend",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "autodock_gpu"], case_sensitive=False),
)
@click.option(
    "--clip-reward-upper-bound/--no-clip-reward-upper-bound",
    default=True,
    show_default=True,
    help="Cap reward_score at 1.0 (minimum 0.0 is always enforced).",
)
@click.option(
    "--require-pose-rmsd/--no-require-pose-rmsd",
    default=None,
    help="Gate rewards by fragment pose RMSD (defaults to benchmark config if not set).",
)
@click.option("--dry-run", is_flag=True, default=False, help="Generate files only.")
def main(
    benchmarks,
    budget,
    seed,
    out,
    run_dir,
    prior_file,
    n_warmup,
    device,
    docking_backend,
    clip_reward_upper_bound,
    require_pose_rmsd,
    dry_run,
):
    selected = list(benchmarks) if benchmarks else BENCHMARKS
    outputs_root = pathlib.Path(out) if out else SCRIPT_DIR / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    if run_dir is not None:
        run_parent = pathlib.Path(run_dir)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_parent = SCRIPT_DIR / f"run_{ts}"
    run_parent.mkdir(parents=True, exist_ok=True)

    prior_env = os.getenv("LIBINVENT_PRIOR")
    prior_path = pathlib.Path(prior_file or prior_env or (SCRIPT_DIR / "priors/libinvent.prior"))
    if not prior_path.exists() and not dry_run:
        raise SystemExit(
            f"Lib-INVENT prior not found: {prior_path}. Set --prior-file or LIBINVENT_PRIOR."
        )

    reinvent_bin = shutil.which("reinvent") or (pathlib.Path(sys.executable).parent / "reinvent")
    if not dry_run and not (shutil.which("reinvent") or pathlib.Path(reinvent_bin).exists()):
        raise SystemExit("`reinvent` executable not found in PATH or venv bin. Activate REINVENT4 environment.")

    log.info("Benchmarks : %s", selected)
    log.info("Budget     : %d", budget)
    log.info("Seed       : %d", seed)
    log.info("Run parent : %s", run_parent)
    log.info("Outputs    : %s", outputs_root)
    log.info("Prior file : %s", prior_path)
    log.info("Clip upper bound: %s", clip_reward_upper_bound)
    log.info("Dry run    : %s", dry_run)

    for bm in selected:
        run_benchmark(
            benchmark=bm,
            budget=budget,
            out_root=outputs_root,
            run_parent=run_parent,
            prior_file=prior_path,
            n_warmup=n_warmup,
            device=device,
            docking_backend=docking_backend,
            clip_reward_upper_bound=clip_reward_upper_bound,
            require_pose_rmsd=require_pose_rmsd,
            dry_run=dry_run,
        )

    log.info("All benchmarks complete.")


if __name__ == "__main__":
    main()
