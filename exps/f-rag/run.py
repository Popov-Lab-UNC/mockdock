"""
f-RAG x FCGMB Benchmark Evaluation
==================================
Runs f-RAG's fragment-retrieval generation loop against the six FCGMB oracles.

This implementation keeps the original f-RAG flow:
  - SAFE generation + GA reproduction
  - arm/linker fragment populations
  - retrieval-augmented generation via SAFEFusionDesign

FCGMB-specific changes:
  - Oracle is FCGMBOracle instead of qVina.
  - Initial compound context comes from oracle.get_initial_compounds().
    We use those compounds to seed molecule and fragment populations.
"""

from __future__ import annotations

import logging
import pathlib
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import click
import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

from fcgmb import FCGMBOracle

os_environ = __import__("os").environ
os_environ["TOKENIZERS_PARALLELISM"] = "false"


BENCHMARKS = ["AKT1", "CHK1", "ITK", "PCK1", "TTK", "VEGFR2"]


def _resolve_default_frag_root() -> pathlib.Path:
    # benchmark/exps/f-rag/run.py -> benchmark/exps/f-rag -> benchmark/exps
    # -> benchmark -> <workspace>; f-RAG lives as sibling of benchmark.
    return pathlib.Path(__file__).resolve().parents[3] / "f-RAG"


def _load_f_rag_modules(f_rag_root: pathlib.Path | None) -> dict[str, Any]:
    # Preferred path: import directly from the active Python environment
    # (e.g. when running inside the f-rag virtualenv).
    try:
        import safe as sf
        import ga.crossover as co
        from ga.ga import reproduce
        from fusion.sample import SAFEFusionDesign
        from fusion.slicer import MolSlicer
    except Exception:
        # Fallback: import from local checkout via sys.path.
        if f_rag_root is None:
            f_rag_root = _resolve_default_frag_root()
        if not f_rag_root.exists():
            raise FileNotFoundError(
                "Could not import f-RAG modules from the active environment, and "
                f"f-RAG root was not found at '{f_rag_root}'. "
                "Install f-RAG dependencies in this environment or pass --f-rag-root."
            )
        sys.path.insert(0, str(f_rag_root))
        try:
            import safe as sf
            import ga.crossover as co
            from ga.ga import reproduce
            from fusion.sample import SAFEFusionDesign
            from fusion.slicer import MolSlicer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Could not import f-RAG modules. Install f-RAG dependencies in this "
                "environment or pass --f-rag-root to a valid checkout."
            ) from exc

    return {
        "sf": sf,
        "co": co,
        "reproduce": reproduce,
        "SAFEFusionDesign": SAFEFusionDesign,
        "MolSlicer": MolSlicer,
    }


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in columns:
            return cand
    return None


@dataclass
class HParams:
    injection_model_path: str
    mol_population_size: int
    frag_population_size: int
    mutation_rate: float
    num_ga: int
    num_safe: int
    min_frag_size: int
    max_frag_size: int
    min_mol_size: int
    max_mol_size: int
    max_no_budget_progress_steps: int
    max_generated_multiplier: int

    @classmethod
    def from_yaml(cls, path: pathlib.Path) -> "HParams":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)


class FRAGFCGMB:
    def __init__(
        self,
        benchmark_name: str,
        budget: int,
        seed: int,
        out_dir: pathlib.Path,
        f_rag_root: pathlib.Path | None,
        hparams: HParams,
        safe_model_path: str | None,
        logger: logging.Logger,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.seed = seed
        self.args = hparams
        self.logger = logger
        self.out_dir = out_dir
        self.safe_model_path = safe_model_path

        random.seed(seed)
        np.random.seed(seed)

        mods = _load_f_rag_modules(f_rag_root)
        self.sf = mods["sf"]
        self.co = mods["co"]
        self.reproduce = mods["reproduce"]
        self.SAFEFusionDesign = mods["SAFEFusionDesign"]
        self.MolSlicer = mods["MolSlicer"]

        self.oracle = FCGMBOracle(benchmark_name, budget=budget)
        self.designer = self.SAFEFusionDesign.load_default()
        self.designer.load_fuser(self.args.injection_model_path)
        # Fallback generator: vanilla SAFE-GPT without f-RAG fusion, used when
        # fragment-based generation repeatedly fails to produce valid molecules.
        self.safe_fallback = self._init_safe_fallback()
        self.slicer = self.MolSlicer(shortest_linker=True)
        self._safe_fallback_calls = 0  # incremented each call to vary random_seed
        self.co.MIN_SIZE = self.args.min_mol_size
        self.co.MAX_SIZE = self.args.max_mol_size

        self.results_path = self.out_dir / f"seed_{seed}.csv"
        self._init_from_oracle_context()

    def _init_safe_fallback(self):
        """Initialise the SAFE-only fallback generator, optionally from a custom model path."""
        try:
            if self.safe_model_path:
                from safe.trainer.model import SAFEDoubleHeadsModel
                from safe.tokenizer import SAFETokenizer

                tokenizer = SAFETokenizer.from_pretrained(self.safe_model_path)
                model = SAFEDoubleHeadsModel.from_pretrained(self.safe_model_path)
                designer = self.sf.SAFEDesign(model=model, tokenizer=tokenizer)
            else:
                designer = self.sf.SAFEDesign.load_default(verbose=False)
            return designer
        except Exception as exc:  # pragma: no cover
            self.logger.warning(
                "SAFE-only fallback is unavailable (SAFEDesign initialisation failed: %s).",
                exc,
            )
            return None

    def _init_from_oracle_context(self) -> None:
        initial_df = self.oracle.get_initial_compounds()
        if initial_df.is_empty():
            raise RuntimeError(
                f"No initial compounds returned for {self.benchmark_name}; "
                "cannot initialize f-RAG populations."
            )

        cols = initial_df.columns
        smiles_col = _pick_column(cols, ["canonical_smiles", "smiles", "SMILES"])
        score_col = _pick_column(cols, ["pchembl_value", "activity", "score"])
        if smiles_col is None or score_col is None:
            raise RuntimeError(
                "Initial compounds missing required columns. "
                f"Found columns: {cols}"
            )

        seed_rows = initial_df.select([smiles_col, score_col]).iter_rows(named=True)
        seen_smiles: set[str] = set()
        cleaned_rows: list[tuple[str, float]] = []
        for row in seed_rows:
            smi_raw = row.get(smiles_col)
            score_raw = row.get(score_col)
            if smi_raw is None or score_raw is None:
                continue
            smi = str(smi_raw)
            if smi in seen_smiles:
                continue
            try:
                score = float(score_raw)
            except Exception:
                continue
            seen_smiles.add(smi)
            cleaned_rows.append((smi, score))

        if not cleaned_rows:
            raise RuntimeError("Initial compounds became empty after cleaning.")

        # Log the raw initial compound set for traceability.
        for idx, (smi, proxy_score) in enumerate(cleaned_rows, start=1):
            self.logger.info(
                "Initial compound %d: smiles=%s, proxy_score=%.4f",
                idx,
                smi,
                proxy_score,
            )

        # Use pre-computed docking scores if available, otherwise dock them.
        seed_smiles = [smi for smi, _ in cleaned_rows]
        if "score" in initial_df.columns:
            self.logger.info(
                "Using pre-computed docking scores for %d initial compounds.",
                len(seed_smiles),
            )
            docking_scores = {
                row[smiles_col]: row["score"]
                for row in initial_df.iter_rows(named=True)
                if row[smiles_col] in seed_smiles
            }
        else:
            self.logger.info(
                "Scoring %d initial seed compounds with oracle.", len(seed_smiles)
            )
            docking_scores = self.oracle.score(seed_smiles)

        if not docking_scores:
            raise RuntimeError("Oracle returned no scores for initial compounds.")

        scored = [float(docking_scores[s]) for s in seed_smiles if s in docking_scores]
        if scored:
            self.logger.info(
                "Initial docking scores summary: n_scored=%d, min=%.4f, mean=%.4f, max=%.4f",
                len(scored),
                float(np.min(scored)),
                float(np.mean(scored)),
                float(np.max(scored)),
            )
        else:
            self.logger.warning(
                "Oracle returned scores for %d SMILES, but none matched the cleaned "
                "seed SMILES exactly; all scores will default to 0.0.",
                len(docking_scores),
            )

        mol_population: list[tuple[float, str]] = []
        arm_scores: dict[str, float] = defaultdict(float)
        arm_counts: dict[str, int] = defaultdict(int)
        linker_scores: dict[str, float] = defaultdict(float)
        linker_counts: dict[str, int] = defaultdict(int)

        for smi, _ in cleaned_rows:
            # Normalised scores below 0.0 mean weaker binding than the low_score
            # baseline; clamp to 0.0 for GA probability weights. Scores above 1.0
            # are valid (better than the reference ligand) and are kept as-is.
            score = max(0.0, float(docking_scores.get(smi, 0.0)))
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            n_atoms = mol.GetNumAtoms()
            if self.args.min_mol_size <= n_atoms <= self.args.max_mol_size:
                mol_population.append((float(score), smi))

            frags = self.fragmentize(smi)
            if frags is None:
                continue
            for frag in frags:
                if frag.count("*") == 1:
                    arm_scores[frag] += float(score)
                    arm_counts[frag] += 1
                elif frag.count("*") == 2:
                    linker_scores[frag] += float(score)
                    linker_counts[frag] += 1

        self.mol_population = sorted(set(mol_population), reverse=True)[
            : self.args.mol_population_size
        ]

        arm_population = [
            (arm_scores[f] / max(1, arm_counts[f]), f) for f in arm_scores.keys()
        ]
        linker_population = [
            (linker_scores[f] / max(1, linker_counts[f]), f)
            for f in linker_scores.keys()
        ]
        arm_population.sort(reverse=True)
        linker_population.sort(reverse=True)
        self.arm_population = arm_population[: self.args.frag_population_size]
        self.linker_population = linker_population[: self.args.frag_population_size]

        if len(self.arm_population) < 2 or len(self.linker_population) < 1:
            self.logger.warning(
                "Insufficient initial fragment context to start f-RAG. "
                "arms=%d, linkers=%d; will rely on SAFE-only fallback to bootstrap.",
                len(self.arm_population),
                len(self.linker_population),
            )

        raw_init_best = max(
            [s for s, _ in self.mol_population], default=float("-inf")
        )
        # For logging, treat a non-positive best seed score as "no useful signal yet".
        init_best_score = raw_init_best if raw_init_best > 0 else float("-inf")
        self.logger.info(
            "Initialized from oracle context: %d seed molecules, %d arm frags, %d linker frags "
            "(using docking-based seed scores; budget_used=%d/%d; initial_best_score=%.4f)",
            len(self.mol_population),
            len(self.arm_population),
            len(self.linker_population),
            self.oracle.budget_used,
            self.oracle.max_budget,
            init_best_score,
        )

    def attach(self, frag1: str, frag2: str) -> str:
        rxn = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
        mols = rxn.RunReactants((Chem.MolFromSmiles(frag1), Chem.MolFromSmiles(frag2)))
        idx = np.random.randint(len(mols))
        return Chem.MolToSmiles(mols[idx][0])

    def fragmentize(self, smiles: str) -> list[str] | None:
        try:
            frags = set()
            for safe_frag in self.slicer(smiles):
                if safe_frag is None:
                    continue
                smiles_frag = self.sf.decode(
                    Chem.MolToSmiles(safe_frag), remove_dummies=False
                )
                smiles_frag = re.sub(r"\[\d+\*\]", "[1*]", smiles_frag)
                if smiles_frag.count("*") in {1, 2}:
                    frag_mol = Chem.MolFromSmiles(smiles_frag)
                    if frag_mol is None:
                        continue
                    frag_size = frag_mol.GetNumAtoms()
                    if self.args.min_frag_size <= frag_size <= self.args.max_frag_size:
                        frags.add(smiles_frag)
            return list(frags)
        except KeyboardInterrupt:  # pragma: no cover
            raise
        except Exception:
            return None

    def update_population(self, prop_list: list[float], smiles_list: list[str]) -> None:
        self.mol_population += list(set(zip(prop_list, smiles_list)))
        self.mol_population.sort(reverse=True)
        self.mol_population = self.mol_population[: self.args.mol_population_size]

        arms = {frag for _, frag in self.arm_population}
        linkers = {frag for _, frag in self.linker_population}
        for prop, smiles in zip(prop_list, smiles_list):
            frags = self.fragmentize(smiles)
            if frags is None:
                continue
            for frag in frags:
                if frag.count("*") == 1 and frag not in arms:
                    self.arm_population.append((prop, frag))
                elif frag.count("*") == 2 and frag not in linkers:
                    self.linker_population.append((prop, frag))

        self.arm_population.sort(reverse=True)
        self.linker_population.sort(reverse=True)
        self.arm_population = self.arm_population[: self.args.frag_population_size]
        self.linker_population = self.linker_population[: self.args.frag_population_size]

    def _generate_safe_only(self) -> str | None:
        """Fallback: use vanilla SAFE-GPT (no f-RAG fusion) to propose a molecule.

        Two strategies are tried in order:
          1. scaffold_decoration — constrained to the benchmark fragment scaffold.
          2. de_novo_generation  — unconstrained, if scaffold decoration yields nothing.

        The random seed is incremented on each call so different molecules are
        explored across repeated fallback invocations.
        """
        if self.safe_fallback is None:
            self.logger.warning("SAFE-only fallback is unavailable (SAFEDesign not initialised).")
            return None

        self._safe_fallback_calls += 1
        varied_seed = self.seed + self._safe_fallback_calls
        core = getattr(self.oracle, "fragment_smiles_with_dummies", None)

        # --- Strategy 1: scaffold_decoration ---
        if core:
            try:
                torch.manual_seed(varied_seed)
                candidates = self.safe_fallback.scaffold_decoration(
                    scaffold=core,
                    n_samples_per_trial=10,
                    n_trials=3,
                    sanitize=True,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.logger.warning("SAFE-only scaffold_decoration failed: %s", exc)
                candidates = []

            for smi in candidates or []:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                n_atoms = mol.GetNumAtoms()
                if self.args.min_mol_size <= n_atoms <= self.args.max_mol_size:
                    self.logger.info(
                        "SAFE-only fallback (scaffold_decoration) produced candidate (atoms=%d).",
                        n_atoms,
                    )
                    return smi
                self.logger.debug(
                    "SAFE scaffold_decoration candidate rejected: atoms=%d not in [%d, %d].",
                    n_atoms, self.args.min_mol_size, self.args.max_mol_size,
                )
        else:
            self.logger.warning(
                "SAFE-only fallback: oracle fragment_smiles_with_dummies unavailable; "
                "skipping scaffold_decoration."
            )

        # --- Strategy 2: de_novo_generation (unconstrained) ---
        try:
            torch.manual_seed(varied_seed)
            candidates = self.safe_fallback.de_novo_generation(
                n_samples_per_trial=10,
                n_trials=3,
                sanitize=True,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.logger.warning("SAFE-only de_novo_generation failed: %s", exc)
            candidates = []

        for smi in candidates or []:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            n_atoms = mol.GetNumAtoms()
            if self.args.min_mol_size <= n_atoms <= self.args.max_mol_size:
                self.logger.info(
                    "SAFE-only fallback (de_novo) produced candidate (atoms=%d).",
                    n_atoms,
                )
                return smi
            self.logger.debug(
                "SAFE de_novo candidate rejected: atoms=%d not in [%d, %d].",
                n_atoms, self.args.min_mol_size, self.args.max_mol_size,
            )

        self.logger.info(
            "SAFE-only fallback produced no valid molecule (varied_seed=%d, calls=%d).",
            varied_seed, self._safe_fallback_calls,
        )
        return None

    def generate(self) -> str | None:
        """Return a single valid molecule or None if generation fails.

        This first tries f-RAG's fragment-based SAFE-fusion generation; if that
        fails repeatedly, it falls back to a vanilla SAFE-GPT generation.
        """
        attempts = 0
        for _ in range(1000):
            attempts += 1
            try:
                can_arm_arm = len(self.arm_population) >= 2
                can_arm_linker = (
                    len(self.arm_population) >= 1 and len(self.linker_population) >= 1
                )

                if not (can_arm_arm or can_arm_linker):
                    # No fragment-based generation possible yet; break to SAFE-only.
                    break

                if can_arm_arm and (not can_arm_linker or random.random() < 0.5):  # arm + arm
                    frag1, frag2 = random.sample(
                        [frag for _, frag in self.arm_population], 2
                    )
                    self.designer.frags = [frag for _, frag in self.linker_population]
                    smiles = self.designer.linker_generation(
                        frag1,
                        frag2,
                        n_samples_per_trial=1,
                    )[0]
                else:  # arm + linker
                    frag1 = random.choice([frag for _, frag in self.arm_population])
                    frag2 = random.choice([frag for _, frag in self.linker_population])
                    frag = re.sub(r"\[1\*\]", "[*]", self.attach(frag1, frag2))
                    self.designer.frags = [frag for _, frag in self.arm_population]
                    smiles = self.designer.motif_extension(
                        frag,
                        n_samples_per_trial=1,
                    )[0]
                    smiles = sorted(smiles.split("."), key=len)[-1]

                decoded = self.sf.decode(smiles)
                mol = Chem.MolFromSmiles(decoded)
                if mol is None:
                    # Invalid molecule; try again.
                    continue
                n_atoms = mol.GetNumAtoms()
                if self.args.min_mol_size <= n_atoms <= self.args.max_mol_size:
                    self.logger.info(
                        "Generated candidate molecule (atoms=%d) after %d attempts.",
                        n_atoms,
                        attempts,
                    )
                    return decoded
            except KeyboardInterrupt:  # pragma: no cover
                raise
            except Exception:
                continue

        # Fragment-based generation failed; fall back to SAFE-only generation.
        fallback = self._generate_safe_only()
        if fallback is not None:
            return fallback

        self.logger.warning(
            "Generation failed to produce a valid molecule after %d attempts (including SAFE-only fallback).",
            attempts,
        )
        return None

    def score_batch(self, smiles_list: list[str]) -> list[float]:
        # Keep order stable while deduplicating oracle calls.
        unique = list(dict.fromkeys(smiles_list))
        score_map = self.oracle.score(unique)
        return [float(score_map.get(smi, 0.0)) for smi in smiles_list]

    def record(self, smiles_list: list[str], prop_list: list[float]) -> None:
        rows = [{"smiles": s, "score": p} for s, p in zip(smiles_list, prop_list)]
        df = pd.DataFrame(rows)
        header = not self.results_path.exists()
        df.to_csv(self.results_path, mode="a", header=header, index=False)

    def run(self) -> None:
        generated = 0
        no_progress_steps = 0
        prev_budget_used = self.oracle.budget_used
        max_generated = self.oracle.max_budget * self.args.max_generated_multiplier

        while self.oracle.status == "active":
            safe_smiles_list = [
                smi for smi in (self.generate() for _ in range(self.args.num_safe)) if smi
            ]
            if safe_smiles_list:
                safe_prop_list = self.score_batch(safe_smiles_list)
                self.update_population(safe_prop_list, safe_smiles_list)
                self.record(safe_smiles_list, safe_prop_list)
                generated += len(safe_smiles_list)
            else:
                self.logger.info(
                    "SAFE step produced no valid molecules in this iteration."
                )

            if len(self.mol_population) == self.args.mol_population_size:
                ga_smiles_list = [
                    self.reproduce(self.mol_population, self.args.mutation_rate)
                    for _ in range(self.args.num_ga)
                ]
                ga_smiles_list = [s for s in ga_smiles_list if s]
                if ga_smiles_list:
                    ga_prop_list = self.score_batch(ga_smiles_list)
                    self.update_population(ga_prop_list, ga_smiles_list)
                    self.record(ga_smiles_list, ga_prop_list)
                    generated += len(ga_smiles_list)
                else:
                    self.logger.info(
                        "GA step produced no valid molecules in this iteration."
                    )

            if self.oracle.budget_used == prev_budget_used:
                no_progress_steps += 1
            else:
                prev_budget_used = self.oracle.budget_used
                no_progress_steps = 0

            raw_best = max(
                [s for s, _ in self.mol_population], default=float("-inf")
            )
            # For logging, emphasise only strictly positive oracle scores; otherwise
            # display -inf to indicate no improvement beyond a zero baseline.
            best_score = raw_best if raw_best > 0 else float("-inf")
            self.logger.info(
                "generated=%d, budget_used=%d/%d, round=%d, best_score=%.4f",
                generated,
                self.oracle.budget_used,
                self.oracle.max_budget,
                self.oracle.generation_round,
                best_score,
            )

            if no_progress_steps >= self.args.max_no_budget_progress_steps:
                self.logger.warning(
                    "Stopping due to no oracle budget progress for %d steps.",
                    no_progress_steps,
                )
                break
            if generated >= max_generated:
                self.logger.warning(
                    "Stopping at generated=%d (safety cap max_generated=%d).",
                    generated,
                    max_generated,
                )
                break


@click.command()
@click.option(
    "--out",
    "output_dir",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("./outputs"),
    show_default=True,
)
@click.option(
    "--f-rag-root",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help=(
        "Optional path to f-RAG repository. Usually not needed if running in an "
        "environment where f-RAG modules are already importable."
    ),
)
@click.option(
    "--hparams",
    "hparams_path",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path(__file__).with_name("hparams.yaml"),
    show_default=True,
)
@click.option(
    "--injection-model-path",
    type=str,
    default=None,
    help="Path to f-RAG injection module model.safetensors.",
)
@click.option(
    "--safe-model-path",
    type=str,
    default=None,
    help="Optional path to SAFE-GPT base model directory.",
)
@click.option(
    "--budget",
    type=int,
    default=1000,
    show_default=True,
    help="Oracle scoring budget per benchmark.",
)
@click.option("--num-runs", type=int, default=1, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("num_safe", "--num-safe", type=int, default=None)
@click.option("num_ga", "--num-ga", type=int, default=None)
@click.option("mutation_rate", "--mutation-rate", type=float, default=None)
@click.option("mol_population_size", "--mol-population-size", type=int, default=None)
@click.option("frag_population_size", "--frag-population-size", type=int, default=None)
@click.option("min_frag_size", "--min-frag-size", type=int, default=None)
@click.option("max_frag_size", "--max-frag-size", type=int, default=None)
@click.option("min_mol_size", "--min-mol-size", type=int, default=None)
@click.option("max_mol_size", "--max-mol-size", type=int, default=None)
@click.option("selected_benchmarks", "--benchmark", "-b", multiple=True, type=click.Choice(BENCHMARKS))
def main(
    output_dir: pathlib.Path,
    f_rag_root: pathlib.Path | None,
    hparams_path: pathlib.Path,
    injection_model_path: str | None,
    safe_model_path: str | None,
    budget: int,
    num_runs: int,
    seed: int,
    num_safe: int | None,
    num_ga: int | None,
    mutation_rate: float | None,
    mol_population_size: int | None,
    frag_population_size: int | None,
    min_frag_size: int | None,
    max_frag_size: int | None,
    min_mol_size: int | None,
    max_mol_size: int | None,
    selected_benchmarks: tuple[str, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    hparams = HParams.from_yaml(hparams_path)
    effective_root = f_rag_root or _resolve_default_frag_root()
    if injection_model_path is not None:
        hparams.injection_model_path = injection_model_path
    elif not pathlib.Path(hparams.injection_model_path).exists():
        candidate_ckpts = [
            effective_root / "ckpt" / "model.safetensors",
            pathlib.Path.cwd() / "ckpt" / "model.safetensors",
        ]
        for ckpt in candidate_ckpts:
            if ckpt.exists():
                hparams.injection_model_path = str(ckpt)
                break

    for key, value in {
        "num_safe": num_safe,
        "num_ga": num_ga,
        "mutation_rate": mutation_rate,
        "mol_population_size": mol_population_size,
        "frag_population_size": frag_population_size,
        "min_frag_size": min_frag_size,
        "max_frag_size": max_frag_size,
        "min_mol_size": min_mol_size,
        "max_mol_size": max_mol_size,
    }.items():
        if value is not None:
            setattr(hparams, key, value)

    benchmarks = list(selected_benchmarks) if selected_benchmarks else BENCHMARKS

    for benchmark_name in benchmarks:
        task_dir = output_dir / benchmark_name
        task_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(f"f-rag-{benchmark_name}")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(task_dir / "log.txt"))

        logger.info("Benchmark   : %s", benchmark_name)
        logger.info("Budget      : %d", budget)
        logger.info("Num runs    : %d", num_runs)
        logger.info("Base seed   : %d", seed)
        logger.info("f-RAG root  : %s", effective_root)

        for run_idx in range(num_runs):
            run_seed = seed + run_idx
            logger.info("Run %d/%d (seed=%d)", run_idx + 1, num_runs, run_seed)
            runner = FRAGFCGMB(
                benchmark_name=benchmark_name,
                budget=budget,
                seed=run_seed,
                out_dir=task_dir,
                f_rag_root=f_rag_root,
                hparams=hparams,
                safe_model_path=safe_model_path,
                logger=logger,
            )
            logger.info("Run directory: %s", runner.oracle.run_dir)
            try:
                runner.run()
            finally:
                runner.oracle.results_df.write_csv(
                    task_dir / f"oracle_results_seed_{run_seed}.csv"
                )
                try:
                    runner.oracle.export_top_poses(n=10)
                except Exception as exc:
                    logger.warning("Could not export top poses: %s", exc)
                runner.oracle.save_metrics(extra={
                    "model": "f-rag",
                    "seed": run_seed,
                })
                logger.info(
                    "Finished seed=%d, budget_used=%d/%d, rounds=%d",
                    run_seed,
                    runner.oracle.budget_used,
                    runner.oracle.max_budget,
                    runner.oracle.generation_round,
                )


if __name__ == "__main__":
    main()
