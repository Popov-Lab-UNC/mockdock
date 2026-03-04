"""
PrexSyn × FCGMB Benchmark Evaluation
======================================
Evaluates PrexSyn on FCGMB docking benchmarks using an exposed
generate → score → update loop that mirrors the structure of
prexsyn/scripts/benchmarks/optim.py as closely as possible.

Key differences from prexsyn's autodock_Mpro_7gaw task:
  - Oracle is FCGMBOracle (6 targets) rather than a bundled AutoDock oracle.
  - The oracle call (score_with_oracle) is factored out so that the
    SMILES list passed to and scores returned from FCGMBOracle are visible.
  - num_runs defaults to 1 (docking is expensive; increase for statistics).
"""

import logging
import pathlib
import sys
from typing import cast

import click
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from prexsyn.applications.optim import Optimizer
from prexsyn.applications.optim.step import FingerprintGenetic
from prexsyn.applications.optim.tracker import OptimTracker
from prexsyn.factories import load_model
from prexsyn.factories.facade import Facade
from prexsyn.models.prexsyn import PrexSyn
from prexsyn.properties import PropertySet
from prexsyn.queries import Query
from prexsyn.utils.oracles import CachedOracle, OracleProtocol

from fcgmb import FCGMBOracle


# ──────────────────────────────────────────────────────────────────────────────
# Queries  (identical to prexsyn/scripts/benchmarks/optim.py)
# ──────────────────────────────────────────────────────────────────────────────


def query_lipinski(ps: PropertySet, pn: str = "rdkit_descriptor_upper_bound") -> Query:
    p = ps[pn]
    return (
        p.lt("amw", 500.0)
        & p.lt("CrippenClogP", 5.0)
        & p.lt("lipinskiHBD", 4)
        & p.lt("lipinskiHBA", 9)
        & p.lt("NumRotatableBonds", 9)
        & p.lt("tpsa", 140.0)
    )


def query_fragment(ps: PropertySet, fragment_smiles: str) -> Query:
    """
    Soft BRICS-based fragment conditioning query.
    Biases PrexSyn toward generating molecules that decompose into
    fragments resembling the benchmark fragment.
    Hard enforcement is always applied by FCGMBOracle regardless.
    """
    fragment_mol = Chem.MolFromSmiles(fragment_smiles)
    if fragment_mol is None:
        raise ValueError(f"Invalid fragment SMILES: {fragment_smiles}")
    return ps["brics"].has(fragment_mol)


def query_initial_context(ps: PropertySet, ref_mols: list[Chem.Mol]) -> Query | None:
    """
    Build an ECFP4 reference query from benchmark-provided initial compounds.
    """
    if not ref_mols:
        return None
    ecfp = ps["ecfp4"]
    query: Query | None = None
    for mol in ref_mols:
        q = ecfp.eq(mol)
        query = q if query is None else (query | q)
    return query


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in columns:
            return cand
    return None


def _extract_initial_context_mols(
    oracle: FCGMBOracle,
    max_refs: int,
) -> list[Chem.Mol]:
    """
    Select representative initial compounds as PrexSyn context references.
    """
    initial_df = oracle.get_initial_compounds()
    if initial_df.is_empty():
        return []

    cols = initial_df.columns
    smiles_col = _pick_column(cols, ["canonical_smiles", "smiles", "SMILES"])
    score_col = _pick_column(cols, ["pchembl_value", "activity", "score"])
    if smiles_col is None:
        return []

    rows = initial_df.select(
        [smiles_col] + ([score_col] if score_col is not None else [])
    ).iter_rows(named=True)

    fragment_mol = Chem.MolFromSmiles(oracle.fragment_smiles)
    seen: set[str] = set()
    candidates: list[tuple[float, Chem.Mol]] = []
    for row in rows:
        smi_raw = row.get(smiles_col)
        if smi_raw is None:
            continue
        smi = str(smi_raw)
        if smi in seen:
            continue
        seen.add(smi)

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if fragment_mol is not None and not mol.HasSubstructMatch(fragment_mol):
            continue

        score = row.get(score_col) if score_col is not None else 0.0
        try:
            val = float(score) if score is not None else 0.0
        except Exception:
            val = 0.0
        candidates.append((val, mol))

    if not candidates:
        return []

    # Use strongest references from the provided initial pool.
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [mol for _, mol in candidates[:max_refs]]


# ──────────────────────────────────────────────────────────────────────────────
# AUC-Top10  (identical to prexsyn/scripts/benchmarks/optim.py)
# ──────────────────────────────────────────────────────────────────────────────


def auc_top10_from_df(df: pd.DataFrame, max_evals: int) -> float:
    import heapq

    scores: list[float] = df["score"].tolist()
    top10: list[float] = []
    moving_top10_avg: list[float] = []
    for score in scores:
        heapq.heappush(top10, score)
        if len(top10) > 10:
            heapq.heappop(top10)
        moving_top10_avg.append(sum(top10) / len(top10) if top10 else 0.0)
    if len(moving_top10_avg) < max_evals:
        moving_top10_avg += [moving_top10_avg[-1]] * (max_evals - len(moving_top10_avg))
    return float(np.mean(moving_top10_avg[:max_evals]))


# ──────────────────────────────────────────────────────────────────────────────
# FCGMBOracle adapter
# ──────────────────────────────────────────────────────────────────────────────


class FCGMBOracleAdapter:
    """
    Wraps FCGMBOracle so it satisfies OracleProtocol (Chem.Mol → float).

    The score() call is made visible here: SMILES are extracted from
    the molecule(s), passed to oracle.score(), and the dict result is
    unpacked back into floats.
    """

    def __init__(self, oracle: FCGMBOracle) -> None:
        self._oracle = oracle

    def __call__(self, mol: Chem.Mol | list[Chem.Mol]) -> float | list[float]:
        if isinstance(mol, list):
            smiles_list = [Chem.MolToSmiles(m) for m in mol]
            # ── Explicit call to FCGMBOracle ──────────────────────────────────
            score_map: dict[str, float] = self._oracle.score(smiles_list)
            return [float(score_map.get(smi, 0.0)) for smi in smiles_list]
        else:
            smi = Chem.MolToSmiles(mol)
            score_map = self._oracle.score([smi])
            return float(score_map.get(smi, 0.0))

    def __repr__(self) -> str:
        return f"FCGMBOracle({self._oracle.benchmark_name})"


# ──────────────────────────────────────────────────────────────────────────────
# Task  (mirrors prexsyn/scripts/benchmarks/optim.py Task)
# ──────────────────────────────────────────────────────────────────────────────


class Task:
    def __init__(
        self,
        benchmark_name: str,
        budget: int = 1000,
        num_runs: int = 1,
        constraint_name: str = "null",
        bottleneck_size: int = 50,
        bottleneck_temperature: float = 0.5,
        num_init_samples: int = 500,
        use_fragment_condition: bool = False,
        use_initial_context: bool = True,
        initial_context_refs: int = 8,
        run_dir: pathlib.Path | None = None,
    ) -> None:
        super().__init__()
        self.benchmark_name = benchmark_name
        self.budget = budget
        self.num_runs = num_runs
        self.num_init_samples = num_init_samples
        self.use_fragment_condition = use_fragment_condition
        self.use_initial_context = use_initial_context
        self.initial_context_refs = initial_context_refs

        # FCGMBOracle — one instance shared across all runs (tracks total budget)
        self.fcgmb = FCGMBOracle(benchmark_name, budget=budget, run_dir=run_dir)

        # PrexSyn-compatible oracle wrapping FCGMBOracle
        self.oracle_fn: OracleProtocol = CachedOracle(FCGMBOracleAdapter(self.fcgmb))
        self.constraint_fn: OracleProtocol = CachedOracle(_get_null_oracle())

        self.step_strategy = FingerprintGenetic(
            bottleneck_size=bottleneck_size,
            bottleneck_temperature=bottleneck_temperature,
        )

    def run(
        self,
        facade: Facade,
        model: PrexSyn,
        out_root: pathlib.Path,
        time_limit: int | None = None,
    ) -> None:
        task_dir = self.fcgmb.run_dir
        task_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(self.benchmark_name)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(task_dir / "log.txt"))

        logger.info(f"Benchmark   : {self.benchmark_name}")
        logger.info(f"Fragment    : {self.fcgmb.fragment_smiles}")
        logger.info(f"PDB ID      : {self.fcgmb.pdb_id}")
        logger.info(f"Budget      : {self.budget}")
        logger.info(f"Num runs    : {self.num_runs}")
        logger.info(f"Frag. cond. : {self.use_fragment_condition}")
        logger.info(
            f"Init ctx    : {self.use_initial_context} (refs={self.initial_context_refs})"
        )
        logger.info(f"Run dir     : {self.fcgmb.run_dir}")

        cond_parts: list[Query] = []
        if self.use_fragment_condition:
            try:
                frag_query = query_fragment(
                    facade.property_set, self.fcgmb.fragment_smiles
                )
                cond_parts.append(frag_query)
                logger.info(f"Fragment query enabled: {frag_query}")
            except Exception as e:
                logger.warning(f"Could not build fragment query ({e}). Skipping.")

        if self.use_initial_context:
            try:
                ref_mols = _extract_initial_context_mols(
                    self.fcgmb, self.initial_context_refs
                )
                ctx_query = query_initial_context(facade.property_set, ref_mols)
                if ctx_query is not None:
                    cond_parts.append(ctx_query)
                    logger.info(
                        f"Initial-compound context enabled with {len(ref_mols)} refs."
                    )
                else:
                    logger.warning(
                        "No valid initial-compound references found for context."
                    )
            except Exception as e:
                logger.warning(
                    f"Could not build initial-compound context ({e}). Skipping."
                )

        cond_query: Query | None = None
        if cond_parts:
            cond_query = cond_parts[0]
            for q in cond_parts[1:]:
                cond_query = cond_query & q

        auc_top10_all: list[float] = []
        df_result_all: list[pd.DataFrame] = []

        try:
            for run_id in range(1, self.num_runs + 1):
                logger.info(
                    f"Running task: {self.benchmark_name}, run {run_id}/{self.num_runs}"
                )
                result_path = task_dir / f"run_{run_id:02d}.df.pkl"

                if result_path.exists():
                    logger.info(f"Skipping existing run: {result_path}")
                    df_result = cast(pd.DataFrame, pd.read_pickle(result_path))
                    auc_top10 = auc_top10_from_df(df_result, self.budget)
                    df_result_all.append(df_result)
                else:
                    optimizer = Optimizer(
                        facade=facade,
                        model=model,
                        init_query=query_lipinski(facade.property_set),
                        num_init_samples=self.num_init_samples,
                        max_evals=self.budget,
                        step_strategy=self.step_strategy,
                        oracle_fn=self.oracle_fn,
                        constraint_fn=self.constraint_fn,
                        cond_query=cond_query,
                        time_limit=time_limit,
                    )
                    tracker: OptimTracker = optimizer.run()
                    df_result = tracker.get_dataframe()
                    auc_top10 = tracker.auc_top10(self.budget)
                    df_result.to_pickle(result_path)
                    df_result_all.append(df_result)

                auc_top10_all.append(auc_top10)
                logger.info(
                    f"Run {run_id}/{self.num_runs}, "
                    f"AUC-Top10({self.budget / 1000:.0f}k): {auc_top10:.4f}, "
                    f"Rounds: {self.fcgmb.generation_round}"
                )
        finally:
            # Save oracle results and run artifacts even if interrupted
            self.fcgmb.results_df.write_csv(task_dir / "oracle_results.csv")
            try:
                self.fcgmb.export_top_poses(n=10)
            except Exception as exc:
                logger.warning(f"Could not export top poses: {exc}")
            self.fcgmb.save_metrics(extra={"model": "prexsyn"})

        logger.info("==== Summary ====")
        logger.info(f"Oracle: {self.benchmark_name}")
        logger.info(f"- Runs: {len(auc_top10_all)}")
        logger.info(
            f"- AUC-Top10: {np.mean(auc_top10_all):.3f} ± {np.std(auc_top10_all):.3f}"
        )
        logger.info(f"- Budget used: {self.fcgmb.budget_used} / {self.budget}")
        logger.info(f"- Rounds: {self.fcgmb.generation_round}")


# ──────────────────────────────────────────────────────────────────────────────
# Null oracle helper
# ──────────────────────────────────────────────────────────────────────────────


def _get_null_oracle() -> OracleProtocol:
    """Returns 0.0 for every molecule (no constraint)."""

    def _null(mol: Chem.Mol | list[Chem.Mol]) -> float | list[float]:
        if isinstance(mol, list):
            return [0.0] * len(mol)
        return 0.0

    return _null  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# CLI  (mirrors prexsyn/scripts/benchmarks/optim.py)
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARKS = ["AKT1", "CHK1", "ITK", "PCK1", "TTK", "VEGFR2"]


@click.command()
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="./data/trained_models/v1_converted.yaml",
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
    help="Total oracle scoring budget per benchmark (number of molecules docked).",
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
    default=500,
    show_default=True,
    help="Number of molecules sampled during initialization.",
)
@click.option(
    "--bottleneck-size",
    type=int,
    default=50,
    show_default=True,
    help="Elite population size passed to FingerprintGenetic each step.",
)
@click.option(
    "--time-limit",
    type=int,
    default=None,
    help="Wall-clock time limit per run in seconds (optional).",
)
@click.option(
    "--fragment-condition",
    is_flag=True,
    default=False,
    help=(
        "Enable soft BRICS fragment conditioning. Biases PrexSyn toward "
        "the benchmark fragment during generation. Hard enforcement is always "
        "applied by FCGMBOracle regardless."
    ),
)
@click.option(
    "--initial-context/--no-initial-context",
    default=True,
    show_default=True,
    help="Use benchmark initial compounds to build PrexSyn ECFP context query.",
)
@click.option(
    "--initial-context-refs",
    type=int,
    default=8,
    show_default=True,
    help="Number of initial compounds used as ECFP context references.",
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
    help="Parent directory for all benchmark run outputs. Defaults to <script_dir>/run_<timestamp>.",
)
def main(
    model_path: pathlib.Path,
    output_dir: pathlib.Path,
    budget: int,
    num_runs: int,
    num_init_samples: int,
    bottleneck_size: int,
    time_limit: int | None,
    fragment_condition: bool,
    initial_context: bool,
    initial_context_refs: int,
    selected_benchmarks: tuple[str, ...],
    run_dir: pathlib.Path | None,
) -> None:
    import datetime

    torch.set_grad_enabled(False)
    facade, model = load_model(model_path, train=False)
    model = model.to("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a single timestamped parent directory shared across all benchmarks.
    # Layout: run_<timestamp>/<BENCHMARK>/poses/, results_full.csv, results.yaml, …
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
            use_fragment_condition=fragment_condition,
            use_initial_context=initial_context,
            initial_context_refs=initial_context_refs,
            run_dir=run_parent / name,
        )
        for name in benchmarks
    ]

    for task in tasks:
        task.run(facade, model, output_dir, time_limit=time_limit)


if __name__ == "__main__":
    main()
