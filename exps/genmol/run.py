"""
GenMol x mockdock Benchmark Evaluation
========================================
Runs GenMol against the six mockdock benchmarks.

Generation (GenMol) runs inside the Apptainer container via subprocess so
that GenMol's rdkit-pypi is fully isolated from mockdock's newer RDKit/Meeko.
Scoring (mockdock) runs in the host Python environment.
"""

from __future__ import annotations

import logging
import math
import os
import pathlib
import datetime
import random
import subprocess
import sys
import time

import click
import pandas as pd

from mockdock import MDOracle

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
BENCHMARKS = ["DPP4", "CHK1", "ITK", "PEPCK", "TTK", "VEGFR2"]


def _generate_smiles(
    *,
    mode: str,
    model_path: str,
    container_image: str,
    genmol_dir: str,
    workdir: str,
    generate_script: str,
    num_samples: int,
    seed: int,
    scaffold: str | None = None,
    base_smiles: str | None = None,
) -> list[str]:
    """
    Call generate.py inside the Apptainer container and return the list of SMILES
    printed to stdout. All GenMol / rdkit-pypi activity stays inside the container.
    """
    container_model_path = model_path
    if model_path.startswith(genmol_dir):
        container_model_path = model_path.replace(genmol_dir, "/opt/genmol", 1)

    cmd = [
        "apptainer", "exec", "--nv",
        "--env", "PYTHONPATH=/opt/genmol/pkgs:/opt/genmol",
        # Prevent user site-packages (~/.local) from leaking into the container
        # and shadowing the container's rdkit-pypi that GenMol depends on.
        "--env", "PYTHONNOUSERSITE=1",
        "--bind", f"{workdir}:{workdir}",
        "--bind", f"{genmol_dir}:/opt/genmol",
        "--bind", "/nas/longleaf/apps/autodock-gpu/1.6/bin:/opt/adgpu-bin",
        "--pwd", SCRIPT_DIR.as_posix(),
        container_image,
        "python", generate_script,
        "--model-path", container_model_path,
        "--mode", mode,
        "--num-samples", str(num_samples),
        "--seed", str(seed),
    ]
    if scaffold:
        cmd += ["--scaffold", scaffold]
    if base_smiles:
        cmd += ["--base-smiles", base_smiles]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log.warning("Generation subprocess timed out.")
        return []

    if result.returncode != 0:
        # Surface any errors from inside the container
        for line in result.stderr.splitlines():
            log.error("[container] %s", line)

    smiles = [s.strip() for s in result.stdout.splitlines() if s.strip()]
    return smiles


def run_benchmark(
    benchmark: str,
    budget: int,
    seed: int,
    outputs_root: pathlib.Path,
    genmol_model_path: str,
    batch_size: int,
    clip_reward_upper_bound: bool,
    run_parent: pathlib.Path,
    container_image: str,
    genmol_dir: str,
    workdir: str,
):
    log.info("=" * 60)
    log.info(" Benchmark : %s", benchmark)
    log.info(" Budget    : %d", budget)
    log.info(" Seed      : %d", seed)
    log.info("=" * 60)

    output_dir = outputs_root / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_run_dir = run_parent / benchmark
    benchmark_run_dir.mkdir(parents=True, exist_ok=True)

    generate_script = (SCRIPT_DIR / "generate.py").as_posix()

    oracle = MDOracle(
        benchmark,
        budget=budget,
        run_dir=benchmark_run_dir,
        clip_reward_upper_bound=clip_reward_upper_bound,
    )
    log.info("Run directory: %s", oracle.run_dir)

    initial_df = oracle.get_initial_compounds()
    if initial_df.is_empty():
        log.info("No initial compounds available.")
    else:
        # Determine the correct score column name (usually reward_score or score)
        score_col = None
        for col in ["reward_score", "score"]:
            if col in initial_df.columns:
                score_col = col
                break
        if score_col is not None:
            nonzero = (initial_df[score_col] > 0).sum()
            log.info("Initial compounds: %d, pre-computed reward > 0: %d", len(initial_df), nonzero)
        else:
            log.info("Initial compounds: %d", len(initial_df))

    scaffold = oracle.fragment_smiles_with_dummies
    if scaffold:
        scaffold = scaffold.replace("[*]", "*")
        log.info("Scaffold conditioning: %s", scaffold)
    else:
        log.info("No scaffold conditioning for %s", benchmark)

    t0 = time.time()

    population: list[tuple[float, str]] = []
    if not initial_df.is_empty():
        score_col = None
        for col in ["reward_score", "score"]:
            if col in initial_df.columns:
                score_col = col
                break
        if score_col is not None:
            for row in initial_df.iter_rows(named=True):
                smi = row["canonical_smiles"]
                val = row[score_col]
                if val is not None and not math.isnan(val) and val > 0:
                    population.append((val, smi))
            
            # Sort starting population by reward score descending
            population.sort(reverse=True)
            # Limit the initial population to batch_size
            population = population[:batch_size]
            log.info(
                "Initialized starting population with %d ChEMBL seed compounds. Best score: %.4f",
                len(population),
                population[0][0] if population else 0.0,
            )
        else:
            log.warning("No pre-computed score column found in initial compounds.")

    rounds = 0
    while oracle.budget_remaining > 0:
        rounds += 1
        current_batch_size = min(batch_size, oracle.budget_remaining)

        if scaffold:
            if population and rounds > 1:
                log.info("Evolving top %d compounds...", len(population))
                base_smi = random.choice(population)[1]
                generated_smiles = _generate_smiles(
                    mode="evolve",
                    model_path=genmol_model_path,
                    container_image=container_image,
                    genmol_dir=genmol_dir,
                    workdir=workdir,
                    generate_script=generate_script,
                    num_samples=current_batch_size,
                    seed=seed + rounds,
                    base_smiles=base_smi,
                )
            else:
                log.info("Running scaffold decoration on %s", scaffold)
                generated_smiles = _generate_smiles(
                    mode="scaffold",
                    model_path=genmol_model_path,
                    container_image=container_image,
                    genmol_dir=genmol_dir,
                    workdir=workdir,
                    generate_script=generate_script,
                    num_samples=current_batch_size,
                    seed=seed + rounds,
                    scaffold=scaffold,
                )
        else:
            log.info("Running de novo generation...")
            generated_smiles = _generate_smiles(
                mode="denovo",
                model_path=genmol_model_path,
                container_image=container_image,
                genmol_dir=genmol_dir,
                workdir=workdir,
                generate_script=generate_script,
                num_samples=current_batch_size,
                seed=seed + rounds,
            )

        if not generated_smiles:
            log.warning("No SMILES generated this round, breaking.")
            break

        scores = oracle.score(generated_smiles)

        for row in oracle.results_df.iter_rows(named=True):
            if row["reward_score"] > 0:
                population.append((row["reward_score"], row["smiles"]))

        unique_pop: dict[str, float] = {}
        for score, smi in population:
            if smi not in unique_pop or score > unique_pop[smi]:
                unique_pop[smi] = score
        population = [(score, smi) for smi, score in unique_pop.items()]
        population.sort(reverse=True)
        population = population[:batch_size]

        best_score = population[0][0] if population else 0.0
        log.info(
            "Round %d: Scored %d valid molecules. Best reward: %.4f",
            rounds,
            len(generated_smiles),
            best_score,
        )

    total_time = time.time() - t0
    oracle_time = oracle._total_prep_time + oracle._total_dock_time + oracle._total_analysis_time
    total_gen_time_sec = max(0.0, total_time - oracle_time)
    n_gen = max(1, len(oracle.results_df))

    try:
        oracle.export_top_poses(n=10)
    except Exception as exc:
        log.warning("Could not export top poses: %s", exc)

    oracle.save_metrics(
        extra={
            "model": "genmol",
            "seed": seed,
            "total_generation_time_sec": total_gen_time_sec,
            "n_generated_ligands": n_gen,
        }
    )

    log.info(
        "Benchmark %s complete. Budget used: %d/%d, rounds: %d",
        benchmark,
        oracle.budget_used,
        oracle.max_budget,
        oracle.generation_round,
    )


@click.command()
@click.option(
    "--benchmark",
    "benchmarks",
    multiple=True,
    type=click.Choice(BENCHMARKS, case_sensitive=False),
    help="Benchmark(s) to run. Defaults to all six.",
)
@click.option("--budget", default=1000, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--out", default=None)
@click.option(
    "--model-path",
    default="/work/users/s/h/shuhang/genmol/model_v2.ckpt",
    help="Path to the GenMol checkpoint.",
)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--run-dir", default=None)
@click.option(
    "--clip-reward-upper-bound/--no-clip-reward-upper-bound",
    default=False,
    show_default=True,
)
@click.option(
    "--container-image",
    default="/work/users/s/h/shuhang/genmol/genmol.sif",
    help="Path to the Apptainer SIF image for GenMol generation.",
)
@click.option(
    "--genmol-dir",
    default="/work/users/s/h/shuhang/genmol",
    help="GenMol source directory (mounted as /opt/genmol in container).",
)
@click.option(
    "--workdir",
    default="/work/users/s/h/shuhang/benchmark",
    help="Benchmark working directory.",
)
def main(
    benchmarks,
    budget,
    seed,
    out,
    model_path,
    batch_size,
    run_dir,
    clip_reward_upper_bound,
    container_image,
    genmol_dir,
    workdir,
):
    benchmarks = list(benchmarks) if benchmarks else BENCHMARKS
    outputs_root = pathlib.Path(out) if out else SCRIPT_DIR / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    if run_dir is not None:
        run_parent = pathlib.Path(run_dir)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_parent = SCRIPT_DIR / f"run_{ts}"
    run_parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    for bm in benchmarks:
        run_benchmark(
            benchmark=bm,
            budget=budget,
            seed=seed,
            outputs_root=outputs_root,
            genmol_model_path=model_path,
            batch_size=batch_size,
            clip_reward_upper_bound=clip_reward_upper_bound,
            run_parent=run_parent,
            container_image=container_image,
            genmol_dir=genmol_dir,
            workdir=workdir,
        )

    log.info("All benchmarks complete.")


if __name__ == "__main__":
    main()
