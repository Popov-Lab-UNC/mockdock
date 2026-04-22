"""
PrexSyn × mockdock Benchmark Evaluation
======================================
Evaluates PrexSyn on mockdock docking benchmarks using the current PrexSyn
API: AllInOneLoader → MoleculeProjector → fingerprint genetic algorithm
(shortcuts/genetic.py).

The optimization loop is implemented here (not delegated to a black-box
Optimizer) so we have full visibility into every oracle call.

Anti-cheat measures (vs the old Optimizer-based version):
  1. Budget is charged for *every* molecule generated, including duplicates
     that PrexSyn's own population-level dedup would otherwise make free.
  2. The loop runs until the oracle budget is exhausted — no early stopping
     at score 1.0 or any other internal convergence criterion.
"""

from __future__ import annotations

import datetime
import heapq
import logging
import pathlib
import sys
import time
from collections.abc import Sequence
from typing import Optional

import click
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from prexsyn.shortcuts import AllInOneLoader, MoleculeProjector
from prexsyn.shortcuts.genetic import EmbryoSet, History, Population, evolve, hatch, initialize

from prexsyn_engine.chemistry import Molecule
from prexsyn_engine.chemspace import Synthesis

from mockdock import MDOracle


# ──────────────────────────────────────────────────────────────────────────────
# Oracle adapter with anti-cheat deduplication tracking
# ──────────────────────────────────────────────────────────────────────────────


class MDOracleAdapter:
    """
    Wraps MDOracle to satisfy the `_FitnessFunction` protocol expected by
    prexsyn.shortcuts.genetic: takes a list of (Synthesis, Molecule) and
    returns an np.ndarray of fitness scores in [0, 1] (or negative).

    Anti-cheat: every SMILES the adapter has ever yielded a score for is
    cached in `_seen`.  When the model re-generates a molecule it already
    evaluated, we return the cached score but still charge one budget unit
    for it — exactly the same cost as a novel molecule.

    This prevents PrexSyn's population-level dedup from giving repeated
    molecules a free ride through the budget counter.
    """

    def __init__(self, oracle: MDOracle) -> None:
        self._oracle = oracle
        self._seen: dict[str, float] = {}

    @property
    def budget_used(self) -> int:
        return self._oracle.budget_used

    @property
    def budget_exhausted(self) -> bool:
        return self._oracle.budget_used >= self._oracle.max_budget

    def __call__(
        self,
        phenotypes: Sequence[tuple[Synthesis, Molecule]],
    ) -> np.ndarray:
        """Score phenotypes, charging budget for every molecule including repeats."""
        smiles_list = [mol.smiles() for _, mol in phenotypes]
        scores = self._charge_and_cache(smiles_list)
        return np.array([scores[smi] for smi in smiles_list], dtype=np.float32)

    def _charge_and_cache(self, smiles_list: list[str]) -> dict[str, float]:
        novel: list[str] = []
        repeated: list[str] = []
        for smi in smiles_list:
            (repeated if smi in self._seen else novel).append(smi)

        result: dict[str, float] = {}

        # Score novel molecules through the real oracle (consumes budget normally)
        if novel:
            score_map = self._oracle.score(novel)
            result.update(score_map)
            for smi, sc in score_map.items():
                self._seen[smi] = sc

        # Handle repeats: use cached score but charge budget
        if repeated:
            n = len(repeated)
            self._oracle.budget_used = min(
                self._oracle.budget_used + n, self._oracle.max_budget
            )
            print(
                f"[prexsyn-adapter] {n} repeated SMILES charged to budget "
                f"(budget now {self._oracle.budget_used}/{self._oracle.max_budget})"
            )
            for smi in repeated:
                result[smi] = self._seen[smi]

        return result

    def __repr__(self) -> str:
        return f"MDOracleAdapter({self._oracle.benchmark_name})"


# ──────────────────────────────────────────────────────────────────────────────
# AUC-Top10 metric (used for per-run summary)
# ──────────────────────────────────────────────────────────────────────────────


def auc_top10_from_scores(all_scores: list[float], max_evals: int) -> float:
    top10: list[float] = []
    moving_avg: list[float] = []
    for score in all_scores:
        heapq.heappush(top10, score)
        if len(top10) > 10:
            heapq.heappop(top10)
        moving_avg.append(sum(top10) / len(top10) if top10 else 0.0)
    if len(moving_avg) < max_evals:
        last = moving_avg[-1] if moving_avg else 0.0
        moving_avg += [last] * (max_evals - len(moving_avg))
    return float(np.mean(moving_avg[:max_evals]))


# ──────────────────────────────────────────────────────────────────────────────
# Single optimization run
# ──────────────────────────────────────────────────────────────────────────────


def initialize_from_context(
    smiles_list: list[str],
    size: int,
    projector: MoleculeProjector,
    fn,
) -> tuple[Population, History]:
    """Initialize genetic algorithm context using specific mockdock baseline SMILES."""
    mols = [Molecule.from_smiles(s) for s in smiles_list]
    base_genotypes = projector.descriptor_function(mols)

    n_seeds = len(base_genotypes)
    target_size = size * 2

    # Sample to form embryos
    indices = np.random.choice(n_seeds, size=target_size, replace=True)
    seed_genotypes = base_genotypes[indices]
    
    # Apply a light 1% mutation rate for diversity
    mutate_mask = np.random.rand(*seed_genotypes.shape) < 0.01
    new_genotypes = np.where(mutate_mask, ~seed_genotypes, seed_genotypes)

    embryos = EmbryoSet(
        genotypes=new_genotypes,
        unique_identifiers=np.arange(target_size),
        parents=np.full((target_size, 2), -1),
    )

    population = hatch(embryos, projector, projector.descriptor_function, fn)
    history = History()
    history.add_population(population)
    return population, history


def run_optimization(
    loader: AllInOneLoader,
    adapter: MDOracleAdapter,
    budget: int,
    num_init_samples: int,
    bottleneck_size: int,
    bottleneck_temperature: float,
    descriptor: str,
    num_samples_per_query: int,
    logger: logging.Logger,
    time_limit: Optional[int] = None,
) -> tuple[pd.DataFrame, float]:
    """
    Run one optimization episode.

    Returns:
        (tracker_df, auc_top10)  — results DataFrame and scalar AUC-Top10 metric.
    """
    model = loader.model().to("cuda")
    detokenizer = loader.detokenizer()

    projector = MoleculeProjector(
        model=model,
        detokenizer=detokenizer,
        descriptor=descriptor,
        num_samples=num_samples_per_query,
    )

    # Fitness function: wraps the oracle adapter  
    fitness_fn = adapter

    initial_context_df = adapter._oracle.get_initial_compounds()
    initial_smiles = initial_context_df["canonical_smiles"].to_list()
    
    logger.info(f"Initializing population (size={num_init_samples}) from {len(initial_smiles)} mockdock baseline molecules...")
    t_start = time.time()
    population, history = initialize_from_context(
        smiles_list=initial_smiles,
        size=num_init_samples,
        projector=projector,
        fn=fitness_fn,
    )

    # Tracker: list of (smiles, score) in generation order
    tracker_rows: list[dict] = []
    all_scores: list[float] = []
    step = 0

    def _record_population(ppl: Population, gen: int) -> None:
        for (_, mol), fit in zip(ppl.phenotypes, ppl.fitnesses):
            smi = mol.smiles()
            tracker_rows.append({
                "smiles": smi,
                "score": float(fit),
                "step": gen,
            })
            all_scores.append(float(fit))

    _record_population(population, step)

    logger.info(
        f"Init complete. Budget used: {adapter.budget_used}/{budget}. "
        f"Best score: {float(population.fitnesses.max()):.4f}"
    )

    # ── Main optimization loop ────────────────────────────────────────────
    # Runs until budget exhausted. No early stopping at any score threshold.
    while not adapter.budget_exhausted:
        elapsed = time.time() - t_start
        if time_limit is not None and elapsed >= time_limit:
            logger.info(f"Time limit ({time_limit}s) reached. Stopping.")
            break

        step += 1
        evolve(
            ppl=population,
            history=history,
            projector=projector,
            fitness_fn=fitness_fn,
            k=bottleneck_size,
            t=bottleneck_temperature,
        )
        _record_population(population, step)

        best = float(population.fitnesses.max())
        auc = auc_top10_from_scores(all_scores, budget)
        logger.info(
            f"Step {step:4d}: budget={adapter.budget_used}/{budget}, "
            f"best={best:.4f}, auc_top10={auc:.4f}"
        )

    # ── Build results DataFrame ───────────────────────────────────────────
    df = pd.DataFrame(tracker_rows)
    auc_top10 = auc_top10_from_scores(all_scores, budget)
    logger.info(f"Run complete. AUC-Top10({budget//1000}k)={auc_top10:.4f}")
    return df, auc_top10


# ──────────────────────────────────────────────────────────────────────────────
# Task (one benchmark, potentially multiple independent runs)
# ──────────────────────────────────────────────────────────────────────────────


class Task:
    def __init__(
        self,
        benchmark_name: str,
        budget: int = 1000,
        num_runs: int = 1,
        bottleneck_size: int = 50,
        bottleneck_temperature: float = 0.5,
        num_init_samples: int = 25,
        descriptor: str = "ecfp4",
        num_samples_per_query: int = 32,
        run_dir: pathlib.Path | None = None,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.budget = budget
        self.num_runs = num_runs
        self.num_init_samples = num_init_samples
        self.bottleneck_size = bottleneck_size
        self.bottleneck_temperature = bottleneck_temperature
        self.descriptor = descriptor
        self.num_samples_per_query = num_samples_per_query

        self.md_oracle = MDOracle(benchmark_name, budget=budget, run_dir=run_dir)
        self.adapter = MDOracleAdapter(self.md_oracle)

    def run(
        self,
        loader: AllInOneLoader,
        out_root: pathlib.Path,
        time_limit: Optional[int] = None,
    ) -> None:
        task_dir = self.md_oracle.run_dir
        task_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(self.benchmark_name)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(task_dir / "log.txt"))

        logger.info(f"Benchmark   : {self.benchmark_name}")
        logger.info(f"Fragment    : {self.md_oracle.fragment_smiles}")
        logger.info(f"PDB ID      : {self.md_oracle.pdb_id}")
        logger.info(f"Budget      : {self.budget}")
        logger.info(f"Num runs    : {self.num_runs}")
        logger.info(f"Descriptor  : {self.descriptor}")
        logger.info(f"Bottleneck  : {self.bottleneck_size}")
        logger.info(f"Run dir     : {task_dir}")

        auc_top10_all: list[float] = []
        total_time_accum = 0.0

        try:
            for run_id in range(1, self.num_runs + 1):
                logger.info(f"Run {run_id}/{self.num_runs} starting...")
                result_path = task_dir / f"run_{run_id:02d}.df.pkl"

                if result_path.exists():
                    logger.info(f"Skipping existing run: {result_path}")
                    df_result = pd.read_pickle(result_path)
                    auc_top10 = auc_top10_from_scores(
                        df_result["score"].tolist(), self.budget
                    )
                else:
                    # Reset oracle budget for each independent run
                    self.md_oracle.budget_used = 0
                    self.adapter._seen.clear()

                    t0 = time.time()
                    df_result, auc_top10 = run_optimization(
                        loader=loader,
                        adapter=self.adapter,
                        budget=self.budget,
                        num_init_samples=self.num_init_samples,
                        bottleneck_size=self.bottleneck_size,
                        bottleneck_temperature=self.bottleneck_temperature,
                        descriptor=self.descriptor,
                        num_samples_per_query=self.num_samples_per_query,
                        logger=logger,
                        time_limit=time_limit,
                    )
                    total_time_accum += time.time() - t0
                    df_result.to_pickle(result_path)

                auc_top10_all.append(auc_top10)
                logger.info(
                    f"Run {run_id}/{self.num_runs}: "
                    f"AUC-Top10({self.budget // 1000}k)={auc_top10:.4f}"
                )
        finally:
            self.md_oracle.results_df.write_csv(task_dir / "oracle_results.csv")
            try:
                self.md_oracle.export_top_poses(n=10)
            except Exception as exc:
                logger.warning(f"Could not export top poses: {exc}")
            total_oracle_time = self.md_oracle._total_prep_time + self.md_oracle._total_dock_time + self.md_oracle._total_analysis_time
            total_gen_time_sec = max(0.0, total_time_accum - total_oracle_time)
            self.md_oracle.save_metrics(extra={
                "model": "prexsyn",
                "total_generation_time_sec": total_gen_time_sec,
                "n_generated_ligands": max(1, len(self.md_oracle.results_df))
            })

        logger.info("==== Summary ====")
        logger.info(f"Oracle: {self.benchmark_name}")
        logger.info(f"- Runs: {len(auc_top10_all)}")
        if auc_top10_all:
            logger.info(
                f"- AUC-Top10: {np.mean(auc_top10_all):.3f} ± {np.std(auc_top10_all):.3f}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARKS = ["DPP4", "CHK1", "ITK", "PEPCK", "TTK", "VEGFR2"]


@click.command()
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="./data/trained_models/enamine2310_rxn115_202511.yml",
    show_default=True,
    help="Path to PrexSyn model .yaml (checkpoint .ckpt must be alongside it).",
)
@click.option(
    "--out",
    "output_dir",
    type=click.Path(path_type=pathlib.Path),
    default="./outputs",
    show_default=True,
)
@click.option(
    "--budget",
    type=int,
    default=1000,
    show_default=True,
    help="Total oracle scoring budget per benchmark (number of molecules).",
)
@click.option(
    "--num-runs",
    type=int,
    default=1,
    show_default=True,
    help="Number of independent optimization runs per benchmark.",
)
@click.option(
    "--num-init-samples",
    type=int,
    default=25,
    show_default=True,
    help="Population size for random initialization.",
)
@click.option(
    "--bottleneck-size",
    type=int,
    default=50,
    show_default=True,
    help="Elite population size for genetic selection each step.",
)
@click.option(
    "--bottleneck-temperature",
    type=float,
    default=0.5,
    show_default=True,
    help="Softmax temperature for parent selection.",
)
@click.option(
    "--num-samples-per-query",
    type=int,
    default=32,
    show_default=True,
    help="Number of PrexSyn samples generated per fingerprint query.",
)
@click.option(
    "--descriptor",
    type=str,
    default="ecfp4",
    show_default=True,
    help="Molecular descriptor to use for PrexSyn conditioning (e.g. 'ecfp4').",
)
@click.option(
    "--time-limit",
    type=int,
    default=None,
    help="Wall-clock time limit per run in seconds (optional).",
)
@click.option(
    "selected_benchmarks",
    "--benchmark",
    "-b",
    multiple=True,
    type=click.Choice(BENCHMARKS),
    default=None,
    help="Run only these benchmarks (repeat for multiple). Defaults to all six.",
)
@click.option(
    "--run-dir",
    "run_dir",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Parent directory for all benchmark run outputs.",
)
def main(
    model_path: pathlib.Path,
    output_dir: pathlib.Path,
    budget: int,
    num_runs: int,
    num_init_samples: int,
    bottleneck_size: int,
    bottleneck_temperature: float,
    num_samples_per_query: int,
    descriptor: str,
    time_limit: Optional[int],
    selected_benchmarks: tuple[str, ...],
    run_dir: Optional[pathlib.Path],
) -> None:
    torch.set_grad_enabled(False)

    loader = AllInOneLoader(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = pathlib.Path(__file__).resolve().parent
    if run_dir is not None:
        run_parent = run_dir
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_parent = script_dir / f"run_{ts}"
    run_parent.mkdir(parents=True, exist_ok=True)

    benchmarks = list(selected_benchmarks) if selected_benchmarks else BENCHMARKS

    tasks = [
        Task(
            benchmark_name=name,
            budget=budget,
            num_runs=num_runs,
            num_init_samples=num_init_samples,
            bottleneck_size=bottleneck_size,
            bottleneck_temperature=bottleneck_temperature,
            descriptor=descriptor,
            num_samples_per_query=num_samples_per_query,
            run_dir=run_parent / name,
        )
        for name in benchmarks
    ]

    for task in tasks:
        task.run(loader, output_dir, time_limit=time_limit)


if __name__ == "__main__":
    main()
