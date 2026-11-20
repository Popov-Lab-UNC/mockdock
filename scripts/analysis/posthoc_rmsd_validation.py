#!/usr/bin/env python3
"""
Post-hoc RMSD pose validation and ablation analysis comparing:
  - exps_capped         (require_pose_rmsd = True,  clipped at 1.0)
  - exps_capped_no_rmsd (require_pose_rmsd = False, clipped at 1.0)

Uses MOCKDOCK's exact DockingAnalyzer (Meeko + RDKit) and BenchmarkLoader
configurations (including model-specific fragment overrides such as PptT
LibINVENT and 8GKF_ligand_corrected.sdf) to evaluate:

  1. Rewarded-pose core RMSD (exact pose recorded by dlg_path + pose_index)
  2. All-poses / all-conformer-states MOCKDOCK oracle replay (scanning all
     sibling chunk_X_lig_Y_s*.dlg files in poses/) to recover:
       - min_rmsd_any_pose
       - any_valid_pose_found (RMSD < 2.0 A)
       - best_valid_docking_score (kcal/mol)
       - best_valid_norm_score (uncapped)
       - best_valid_reward_score (capped [0, 1], 0.0 if no pose < 2.0 A)
       - best_valid_rmsd
  3. Apparent vs. Post-hoc Pose-Validated Top-10 and Top-100 scores
     (fixed-set re-scoring AND full-run re-ranking, both zero-padded and
      MDEvaluator-style mean over passing molecules)
  4. All-EYR (docked) and Top-K pose pass rates, core RMSD quantiles,
     MW / cLogP / Lipinski drift, and chemical quality (QED, SA, MolSkill,
     STOPLIGHT, AIZynthFinder)
  5. Optimization trajectory RMSD & MW drift binned by oracle call quintile.

Outputs (written to --output-dir):
  - molecules_capped.parquet
  - molecules_capped_no_rmsd.parquet
  - trajectory_bins.csv
  - runs_summary.csv
  - target_model_summary.csv
  - macro_summary.csv
  - ablation_comparison_delta.csv
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from meeko import PDBQTMolecule, RDKitMolCreate
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski

RDLogger.DisableLog("rdApp.*")

# Ensure mockdock src is importable
MOCKDOCK_SRC = Path(__file__).resolve().parents[2] / "src"
if str(MOCKDOCK_SRC) not in sys.path:
    sys.path.insert(0, str(MOCKDOCK_SRC))

import mockdock
from mockdock.analysis import DockingAnalyzer, unroll_multiconf_rdkit_mols
from mockdock.loader import BenchmarkLoader

MODEL_RENAME_MAP = {
    "acegen-a2c": "A2C",
    "acegen-ahc": "AHC",
    "acegen-ppo": "PPO",
    "acegen-ppod": "PPOD",
    "acegen-reinforce": "REINFORCE",
    "acegen-reinvent": "REINVENT",
    "libinvent": "LibINVENT",
    "genmol": "GenMol",
    "invirtuogen": "InVirtuoGen",
}

HIGH_YIELD_TARGETS = {"CHK1", "DPP4", "ITK", "PEPCK", "TTK"}
STATE_SUFFIX_RE = re.compile(r"_s\d+\.dlg$", re.IGNORECASE)

# Worker-local cache of (target, frag_smiles) -> (BenchmarkLoader, DockingAnalyzer)
_WORKER_CACHE: dict[tuple[str, str], tuple[BenchmarkLoader, DockingAnalyzer]] = {}


def _get_loader_and_analyzer(target: str, model_raw: str) -> tuple[BenchmarkLoader, DockingAnalyzer]:
    loader = BenchmarkLoader(target)
    frag_smiles = loader.get_fragment_smiles(model_raw)
    key = (target, frag_smiles)
    if key in _WORKER_CACHE:
        return _WORKER_CACHE[key]

    pkg_dir = Path(mockdock.__file__).parent
    grid_dir = pkg_dir / "grids" / loader.structure_id
    ref_path = grid_dir / f"{loader.structure_id}_ligand_corrected.sdf"
    if not ref_path.exists():
        ref_path = grid_dir / f"{loader.structure_id}_ligand.pdb"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference ligand not found for {target} ({loader.structure_id}) in {grid_dir}")

    analyzer = DockingAnalyzer(
        reference_ligand_path=ref_path,
        fragment_smiles=frag_smiles,
        rmsd_threshold=loader.rmsd_threshold,
    )
    _WORKER_CACHE[key] = (loader, analyzer)
    return loader, analyzer


def _normalize_score(energy: float, low_score: float, high_score: float) -> tuple[float, float]:
    """Replicate MDOracle score normalization (uncapped norm_score and [0,1] capped reward_score)."""
    if not math.isfinite(energy):
        return 0.0, 0.0
    denom = low_score - high_score
    if abs(denom) > 1e-6:
        norm = (low_score - energy) / denom
    else:
        norm = 1.0 if energy <= high_score else 0.0
    reward = min(max(norm, 0.0), 1.0)
    return float(norm), float(reward)


def _parse_dlg_poses(dlg_path: Path, analyzer: DockingAnalyzer) -> dict[str, Any]:
    """Parse all poses in a single DLG file and compute core RMSD and energy for each pose."""
    out = {
        "n_poses": 0,
        "energies": [],
        "rmsds": [],
        "min_rmsd": float("nan"),
        "best_valid_energy": float("nan"),
        "best_valid_rmsd": float("nan"),
        "best_any_energy": float("nan"),
        "best_any_idx": -1,
    }
    if not dlg_path.exists():
        return out
    try:
        pdbqt_mol = PDBQTMolecule.from_file(str(dlg_path), is_dlg=True, skip_typing=True)
        rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        if not rdkit_mols:
            return out
        rdkit_mols = unroll_multiconf_rdkit_mols(rdkit_mols)
        energies = list(getattr(pdbqt_mol, "_pose_data", {}).get("free_energies", []))

        rmsds: list[float] = []
        pose_energies: list[float] = []
        min_rmsd = float("inf")
        best_valid_e = float("nan")
        best_valid_r = float("nan")
        best_any_e = float("nan")
        best_any_idx = -1
        thresh = analyzer.rmsd_threshold

        for idx, mol in enumerate(rdkit_mols):
            e = float(energies[idx]) if idx < len(energies) else float("nan")
            r = float(analyzer.calculate_rmsd(mol))
            pose_energies.append(e)
            rmsds.append(r)

            if math.isfinite(e) and (math.isnan(best_any_e) or e < best_any_e):
                best_any_e = e
                best_any_idx = idx
            if math.isfinite(r) and r < min_rmsd:
                min_rmsd = r
            if math.isfinite(r) and r < thresh and math.isfinite(e):
                if math.isnan(best_valid_e) or e < best_valid_e:
                    best_valid_e = e
                    best_valid_r = r

        out["n_poses"] = len(rdkit_mols)
        out["energies"] = pose_energies
        out["rmsds"] = rmsds
        out["min_rmsd"] = min_rmsd if math.isfinite(min_rmsd) else float("nan")
        out["best_valid_energy"] = best_valid_e
        out["best_valid_rmsd"] = best_valid_r
        out["best_any_energy"] = best_any_e
        out["best_any_idx"] = best_any_idx
        return out
    except Exception:
        return out


def _compute_rdkit_physchem(smiles_list: list[str]) -> dict[str, tuple[float, float, bool]]:
    res: dict[str, tuple[float, float, bool]] = {}
    for smi in smiles_list:
        if smi in res:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            res[smi] = (float("nan"), float("nan"), False)
            continue
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        lip = (mw <= 500.0) and (logp <= 5.0) and (hbd <= 5) and (hba <= 10)
        res[smi] = (mw, logp, lip)
    return res


def _load_optional_score_map(path: Path, value_col: str) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        df = pl.read_csv(path, infer_schema_length=10000)
        if "smiles" not in df.columns or value_col not in df.columns:
            return {}
        out: dict[str, float] = {}
        for s, v in zip(df["smiles"].to_list(), df[value_col].to_list()):
            if s is not None and v is not None:
                fv = float(v)
                if math.isfinite(fv):
                    out[str(s)] = fv
        return out
    except Exception:
        return {}


def _safe_stat(arr: list[float], fn: str) -> float:
    vals = [x for x in arr if x is not None and math.isfinite(x)]
    if not vals:
        return float("nan")
    a = np.asarray(vals, dtype=float)
    if fn == "mean":
        return float(np.mean(a))
    if fn == "median":
        return float(np.median(a))
    if fn == "iqr":
        return float(np.percentile(a, 75) - np.percentile(a, 25))
    if fn == "p25":
        return float(np.percentile(a, 25))
    if fn == "p75":
        return float(np.percentile(a, 75))
    return float("nan")


def _cohort_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_rewarded_pass_rate": float("nan"),
            f"{prefix}_anypose_pass_rate": float("nan"),
            f"{prefix}_rewarded_rmsd_median": float("nan"),
            f"{prefix}_rewarded_rmsd_mean": float("nan"),
            f"{prefix}_rewarded_rmsd_iqr": float("nan"),
            f"{prefix}_min_rmsd_median": float("nan"),
            f"{prefix}_min_rmsd_mean": float("nan"),
            f"{prefix}_docking_kcal_mean": float("nan"),
            f"{prefix}_mw_mean": float("nan"),
            f"{prefix}_clogp_mean": float("nan"),
            f"{prefix}_lipinski_rate": float("nan"),
            f"{prefix}_qed_mean": float("nan"),
            f"{prefix}_sa_mean": float("nan"),
            f"{prefix}_molskill_mean": float("nan"),
            f"{prefix}_stoplight_mean": float("nan"),
            f"{prefix}_aizynth_state_mean": float("nan"),
            f"{prefix}_aizynth_solved_rate": float("nan"),
        }

    rew_pass = [1.0 if r["rewarded_passed"] else 0.0 for r in rows]
    any_pass = [1.0 if r["any_valid_pose_found"] else 0.0 for r in rows]
    rew_rmsds = [r["rewarded_rmsd"] for r in rows]
    min_rmsds = [r["min_rmsd_any_pose"] for r in rows]

    return {
        f"{prefix}_n": n,
        f"{prefix}_rewarded_pass_rate": float(np.mean(rew_pass)),
        f"{prefix}_anypose_pass_rate": float(np.mean(any_pass)),
        f"{prefix}_rewarded_rmsd_median": _safe_stat(rew_rmsds, "median"),
        f"{prefix}_rewarded_rmsd_mean": _safe_stat(rew_rmsds, "mean"),
        f"{prefix}_rewarded_rmsd_iqr": _safe_stat(rew_rmsds, "iqr"),
        f"{prefix}_min_rmsd_median": _safe_stat(min_rmsds, "median"),
        f"{prefix}_min_rmsd_mean": _safe_stat(min_rmsds, "mean"),
        f"{prefix}_docking_kcal_mean": _safe_stat([r["docking_score"] for r in rows], "mean"),
        f"{prefix}_mw_mean": _safe_stat([r["mw"] for r in rows], "mean"),
        f"{prefix}_clogp_mean": _safe_stat([r["clogp"] for r in rows], "mean"),
        f"{prefix}_lipinski_rate": float(np.mean([1.0 if r["lipinski_pass"] else 0.0 for r in rows])),
        f"{prefix}_qed_mean": _safe_stat([r["qed"] for r in rows], "mean"),
        f"{prefix}_sa_mean": _safe_stat([r["sa"] for r in rows], "mean"),
        f"{prefix}_molskill_mean": _safe_stat([r["molskill_score"] for r in rows], "mean"),
        f"{prefix}_stoplight_mean": _safe_stat([r["stoplight_score"] for r in rows], "mean"),
        f"{prefix}_aizynth_state_mean": _safe_stat([r["aizynthfinder_state_score"] for r in rows], "mean"),
        f"{prefix}_aizynth_solved_rate": _safe_stat([r["aizynthfinder_score"] for r in rows], "mean"),
    }


def process_single_run(task: tuple[str, str, bool]) -> dict[str, Any]:
    """Process one (experiment, run_target_dir) unit."""
    exp_label, run_target_dir_str, scan_all_siblings = task
    run_dir = Path(run_target_dir_str)
    target = run_dir.name
    seed = run_dir.parent.name
    model_raw = run_dir.parent.parent.name
    model = MODEL_RENAME_MAP.get(model_raw.lower(), model_raw)

    results_csv = run_dir / "results.csv"
    poses_dir = run_dir / "poses"
    loader, analyzer = _get_loader_and_analyzer(target, model_raw)
    low_score = float(loader.low_score)
    high_score = float(loader.high_score)
    thresh = float(loader.rmsd_threshold)

    # Index all .dlg files in poses/ by prefix (chunk_X_lig_Y) in a single scandir pass
    sibling_index: dict[str, list[Path]] = {}
    if poses_dir.exists():
        for entry in os.scandir(poses_dir):
            if entry.name.endswith(".dlg"):
                prefix = STATE_SUFFIX_RE.sub("", entry.name)
                sibling_index.setdefault(prefix, []).append(Path(entry.path))

    # Load molecule_metrics_cache.csv & expensive score maps
    cache_map: dict[str, dict[str, Any]] = {}
    cache_csv = run_dir / "molecule_metrics_cache.csv"
    if cache_csv.exists():
        try:
            cdf = pl.read_csv(cache_csv, infer_schema_length=10000)
            for row in cdf.iter_rows(named=True):
                smi = row.get("smiles")
                if smi:
                    cache_map[str(smi)] = {
                        "qed": float(row["qed"]) if row.get("qed") is not None else float("nan"),
                        "sa": float(row["sa"]) if row.get("sa") is not None else float("nan"),
                        "is_novel": bool(row.get("is_novel", False)),
                        "has_fragment": bool(row.get("has_fragment", False)),
                        "passes_medchem": bool(row.get("passes_medchem", False)),
                    }
        except Exception:
            pass

    molskill_map = _load_optional_score_map(run_dir / "scores_molskill.csv", "molskill_score")
    stoplight_map = _load_optional_score_map(run_dir / "scores_stoplight.csv", "stoplight_score")
    az_state_map = _load_optional_score_map(run_dir / "scores_aizynthfinder.csv", "aizynthfinder_state_score")
    az_bin_map = _load_optional_score_map(run_dir / "scores_aizynthfinder.csv", "aizynthfinder_score")

    # Read results.csv
    res_df = pl.read_csv(
        results_csv,
        infer_schema_length=10000,
        schema_overrides={
            "smiles": pl.Utf8,
            "original_smiles": pl.Utf8,
            "dlg_path": pl.Utf8,
            "skip_reason": pl.Utf8,
            "docking_score": pl.Float64,
            "norm_score": pl.Float64,
            "reward_score": pl.Float64,
            "best_any_score": pl.Float64,
            "pose_index": pl.Int64,
            "generation_index": pl.Int64,
        },
        null_values=["", "NaN", "nan", "null", "None"],
    )

    # Cache parsed DLGs within this run
    dlg_cache: dict[str, dict[str, Any]] = {}

    def get_dlg_parsed(p: Path) -> dict[str, Any]:
        k = str(p)
        if k not in dlg_cache:
            dlg_cache[k] = _parse_dlg_poses(p, analyzer)
        return dlg_cache[k]

    # Collect evaluated (docked) rows: either skip_reason is null (docked & passed),
    # or skip_reason == "failed_rmsd" (docked in exps_capped, failed RMSD).
    evaluated_rows: list[dict[str, Any]] = []
    unique_smiles_seen: set[str] = set()

    for row in res_df.iter_rows(named=True):
        skip_reason = row.get("skip_reason")
        if skip_reason is not None and skip_reason != "failed_rmsd":
            continue
        smi = row.get("smiles")
        if not smi:
            continue
        smi = str(smi)
        unique_smiles_seen.add(smi)

        gen_idx = row.get("generation_index")
        gen_idx = int(gen_idx) if gen_idx is not None else -1
        dock_sc = row.get("docking_score")
        dock_sc = float(dock_sc) if dock_sc is not None and math.isfinite(dock_sc) else float("nan")
        best_any_sc = row.get("best_any_score")
        best_any_sc = float(best_any_sc) if best_any_sc is not None and math.isfinite(best_any_sc) else dock_sc
        norm_sc = row.get("norm_score")
        norm_sc = float(norm_sc) if norm_sc is not None and math.isfinite(norm_sc) else 0.0
        rew_sc = row.get("reward_score")
        rew_sc = float(rew_sc) if rew_sc is not None and math.isfinite(rew_sc) else 0.0
        pose_idx = row.get("pose_index")
        pose_idx = int(pose_idx) if pose_idx is not None else -1
        dlg_raw = row.get("dlg_path")

        if skip_reason == "failed_rmsd":
            # In exps_capped, MOCKDOCK already checked all states and found no pose < 2.0 A.
            evaluated_rows.append(
                {
                    "smiles": smi,
                    "generation_index": gen_idx,
                    "skip_reason": skip_reason,
                    "docking_score": best_any_sc,
                    "norm_score": 0.0,
                    "reward_score": 0.0,
                    "rewarded_rmsd": float("nan"),
                    "rewarded_passed": False,
                    "min_rmsd_any_pose": float("nan"),
                    "any_valid_pose_found": False,
                    "best_valid_docking_score": float("nan"),
                    "best_valid_norm_score": 0.0,
                    "best_valid_reward_score": 0.0,
                    "best_valid_rmsd": float("nan"),
                    "n_states_checked": 0,
                }
            )
            continue

        # Docked OK row with dlg_path
        rew_rmsd = float("nan")
        rew_passed = False
        min_rmsd_all = float("inf")
        best_valid_e = float("nan")
        best_valid_r = float("nan")
        n_states_checked = 0

        if dlg_raw:
            bn = os.path.basename(str(dlg_raw))
            local_dlg = poses_dir / bn
            parsed_main = get_dlg_parsed(local_dlg)
            n_states_checked = 1

            # Rewarded pose RMSD
            rmsds_main = parsed_main["rmsds"]
            if 0 <= pose_idx < len(rmsds_main):
                rew_rmsd = rmsds_main[pose_idx]
            elif 0 <= parsed_main["best_any_idx"] < len(rmsds_main):
                rew_rmsd = rmsds_main[parsed_main["best_any_idx"]]
            rew_passed = math.isfinite(rew_rmsd) and (rew_rmsd < thresh)

            if scan_all_siblings:
                prefix = STATE_SUFFIX_RE.sub("", bn)
                sib_paths = sibling_index.get(prefix, [local_dlg])
                n_states_checked = len(sib_paths)
                for sp in sib_paths:
                    pm = get_dlg_parsed(sp)
                    mr = pm["min_rmsd"]
                    if math.isfinite(mr) and mr < min_rmsd_all:
                        min_rmsd_all = mr
                    bve = pm["best_valid_energy"]
                    bvr = pm["best_valid_rmsd"]
                    if math.isfinite(bve) and (math.isnan(best_valid_e) or bve < best_valid_e):
                        best_valid_e = bve
                        best_valid_r = bvr
            else:
                # exps_capped OK row: already RMSD-validated during the run
                mr = parsed_main["min_rmsd"]
                if math.isfinite(mr):
                    min_rmsd_all = mr
                best_valid_e = dock_sc
                best_valid_r = rew_rmsd

        min_rmsd_out = min_rmsd_all if math.isfinite(min_rmsd_all) else float("nan")
        any_valid = math.isfinite(best_valid_e) if scan_all_siblings else True
        if scan_all_siblings:
            bv_norm, bv_rew = _normalize_score(best_valid_e, low_score, high_score) if any_valid else (0.0, 0.0)
        else:
            bv_norm, bv_rew = norm_sc, rew_sc

        evaluated_rows.append(
            {
                "smiles": smi,
                "generation_index": gen_idx,
                "skip_reason": "",
                "docking_score": dock_sc,
                "norm_score": norm_sc,
                "reward_score": rew_sc,
                "rewarded_rmsd": rew_rmsd,
                "rewarded_passed": rew_passed,
                "min_rmsd_any_pose": min_rmsd_out,
                "any_valid_pose_found": any_valid,
                "best_valid_docking_score": best_valid_e,
                "best_valid_norm_score": bv_norm,
                "best_valid_reward_score": bv_rew,
                "best_valid_rmsd": best_valid_r,
                "n_states_checked": n_states_checked,
            }
        )

    # Compute RDKit physchem for all unique evaluated SMILES
    physchem = _compute_rdkit_physchem(list(unique_smiles_seen))

    # Attach physchem + cache + quality scores to every row
    for r in evaluated_rows:
        smi = r["smiles"]
        mw, clogp, lip = physchem.get(smi, (float("nan"), float("nan"), False))
        cm = cache_map.get(smi, {})
        r["mw"] = mw
        r["clogp"] = clogp
        r["lipinski_pass"] = lip
        r["qed"] = cm.get("qed", float("nan"))
        r["sa"] = cm.get("sa", float("nan"))
        r["is_novel"] = cm.get("is_novel", False)
        r["has_fragment"] = cm.get("has_fragment", False)
        r["passes_medchem"] = cm.get("passes_medchem", False)
        r["molskill_score"] = molskill_map.get(smi, float("nan"))
        r["stoplight_score"] = stoplight_map.get(smi, float("nan"))
        r["aizynthfinder_state_score"] = az_state_map.get(smi, float("nan"))
        r["aizynthfinder_score"] = az_bin_map.get(smi, float("nan"))

    # --- 1. Trajectory bins (all evaluated calls, binned by generation_index) ---
    bin_defs = [(1, 200, "1-200"), (201, 400, "201-400"), (401, 600, "401-600"), (601, 800, "601-800"), (801, 1000, "801-1000")]
    traj_rows: list[dict[str, Any]] = []
    for lo, hi, blabel in bin_defs:
        b_items = [r for r in evaluated_rows if lo <= r["generation_index"] <= hi]
        traj_rows.append(
            {
                "experiment": exp_label,
                "model": model,
                "target": target,
                "seed": seed,
                "bin": blabel,
                "bin_start": lo,
                "n_docked": len(b_items),
                "rewarded_pass_rate": float(np.mean([1.0 if r["rewarded_passed"] else 0.0 for r in b_items])) if b_items else float("nan"),
                "anypose_pass_rate": float(np.mean([1.0 if r["any_valid_pose_found"] else 0.0 for r in b_items])) if b_items else float("nan"),
                "rewarded_rmsd_median": _safe_stat([r["rewarded_rmsd"] for r in b_items], "median"),
                "rewarded_rmsd_mean": _safe_stat([r["rewarded_rmsd"] for r in b_items], "mean"),
                "min_rmsd_median": _safe_stat([r["min_rmsd_any_pose"] for r in b_items], "median"),
                "mw_mean": _safe_stat([r["mw"] for r in b_items], "mean"),
                "reward_score_mean": _safe_stat([r["reward_score"] for r in b_items], "mean"),
                "docking_kcal_mean": _safe_stat([r["docking_score"] for r in b_items], "mean"),
            }
        )

    # --- 2. Deduplicate unique docked molecules ---
    # Apparent ranking: match MDEvaluator (only skip_reason == "" rows; best reward_score, tie-break norm_score)
    best_apparent_by_smi: dict[str, dict[str, Any]] = {}
    # All docked (including failed_rmsd in exps_capped) for All-EYR pose pass rate
    best_all_docked_by_smi: dict[str, dict[str, Any]] = {}
    # Post-hoc re-ranked valid: keep best valid_reward_score per SMILES
    best_valid_by_smi: dict[str, dict[str, Any]] = {}

    for r in evaluated_rows:
        smi = r["smiles"]
        # Track best across ALL docked calls (including failed_rmsd)
        prev_all = best_all_docked_by_smi.get(smi)
        key_all = (
            1 if r["any_valid_pose_found"] else 0,
            r["reward_score"],
            r["norm_score"],
            -r["docking_score"] if math.isfinite(r["docking_score"]) else -999.0,
        )
        if prev_all is None or key_all > prev_all["_k"]:
            d = dict(r)
            d["_k"] = key_all
            best_all_docked_by_smi[smi] = d

        if r["skip_reason"] == "":
            prev_app = best_apparent_by_smi.get(smi)
            key_app = (
                r["reward_score"],
                r["norm_score"],
                -r["docking_score"] if math.isfinite(r["docking_score"]) else -999.0,
            )
            if prev_app is None or key_app > prev_app["_k"]:
                d = dict(r)
                d["_k"] = key_app
                best_apparent_by_smi[smi] = d

        if r["any_valid_pose_found"]:
            prev_val = best_valid_by_smi.get(smi)
            key_val = (
                r["best_valid_reward_score"],
                r["best_valid_norm_score"],
                -r["best_valid_docking_score"] if math.isfinite(r["best_valid_docking_score"]) else -999.0,
            )
            if prev_val is None or key_val > prev_val["_k"]:
                d = dict(r)
                d["_k"] = key_val
                best_valid_by_smi[smi] = d

    apparent_sorted = sorted(best_apparent_by_smi.values(), key=lambda x: x["_k"], reverse=True)
    valid_sorted = sorted(best_valid_by_smi.values(), key=lambda x: x["_k"], reverse=True)
    all_docked_unique = list(best_all_docked_by_smi.values())
    eyr_docked_unique = [r for r in all_docked_unique if r["is_novel"] and r["has_fragment"]]

    top10_app = apparent_sorted[:10]
    top100_app = apparent_sorted[:100]

    # Helper for Top-K score calculations
    def score_topk(k: int, app_list: list[dict[str, Any]], val_list: list[dict[str, Any]]) -> dict[str, float]:
        app_k = app_list[:k]
        val_k = val_list[:k]
        # Apparent
        app_mdeval = float(np.mean([r["reward_score"] for r in app_k])) if app_k else 0.0
        app_zeropad = float(sum(r["reward_score"] for r in app_k) / k)
        # Fixed-set re-scoring of the apparent Top-K
        fixed_rew_zeropad = float(sum(r["reward_score"] if r["rewarded_passed"] else 0.0 for r in app_k) / k)
        fixed_rew_mean = float(np.mean([r["reward_score"] if r["rewarded_passed"] else 0.0 for r in app_k])) if app_k else 0.0
        fixed_any_zeropad = float(sum(r["best_valid_reward_score"] if r["any_valid_pose_found"] else 0.0 for r in app_k) / k)
        fixed_any_mean = float(np.mean([r["best_valid_reward_score"] if r["any_valid_pose_found"] else 0.0 for r in app_k])) if app_k else 0.0
        # Full-run post-hoc re-ranked under MOCKDOCK RMSD < 2.0 A rule
        rerank_zeropad = float(sum(r["best_valid_reward_score"] for r in val_k) / k)
        rerank_mdeval = float(np.mean([r["best_valid_reward_score"] for r in val_k])) if val_k else 0.0

        return {
            f"apparent_top{k}_mdeval": app_mdeval,
            f"apparent_top{k}_zeropad": app_zeropad,
            f"validated_top{k}_fixed_rewarded_zeropad": fixed_rew_zeropad,
            f"validated_top{k}_fixed_rewarded_mean": fixed_rew_mean,
            f"validated_top{k}_fixed_anypose_zeropad": fixed_any_zeropad,
            f"validated_top{k}_fixed_anypose_mean": fixed_any_mean,
            f"validated_top{k}_reranked_zeropad": rerank_zeropad,
            f"validated_top{k}_reranked_mdeval": rerank_mdeval,
        }

    run_summary: dict[str, Any] = {
        "experiment": exp_label,
        "model": model,
        "model_raw": model_raw,
        "target": target,
        "seed": seed,
        "is_high_yield_target": target in HIGH_YIELD_TARGETS,
        "n_docked_unique": len(all_docked_unique),
        "n_apparent_ok_unique": len(apparent_sorted),
        "n_valid_pose_unique": len(valid_sorted),
        "n_eyr_docked_unique": len(eyr_docked_unique),
    }
    run_summary.update(score_topk(10, apparent_sorted, valid_sorted))
    run_summary.update(score_topk(100, apparent_sorted, valid_sorted))
    run_summary.update(_cohort_metrics(top10_app, "top10"))
    run_summary.update(_cohort_metrics(top100_app, "top100"))
    run_summary.update(_cohort_metrics(eyr_docked_unique, "eyr"))

    # Add early vs late trajectory summary columns
    early = traj_rows[0]
    late = traj_rows[-1]
    run_summary["early_rmsd_median"] = early["rewarded_rmsd_median"]
    run_summary["late_rmsd_median"] = late["rewarded_rmsd_median"]
    run_summary["early_rewarded_pass_rate"] = early["rewarded_pass_rate"]
    run_summary["late_rewarded_pass_rate"] = late["rewarded_pass_rate"]
    run_summary["early_mw_mean"] = early["mw_mean"]
    run_summary["late_mw_mean"] = late["mw_mean"]

    # Build slim per-molecule records for parquet export
    top10_smis = {r["smiles"] for r in top10_app}
    top100_smis = {r["smiles"] for r in top100_app}
    mol_records: list[dict[str, Any]] = []
    for r in all_docked_unique:
        smi = r["smiles"]
        # Use apparent row when available so docking_score / rewarded_rmsd match the apparent ranking
        src = best_apparent_by_smi.get(smi, r)
        mol_records.append(
            {
                "experiment": exp_label,
                "model": model,
                "target": target,
                "seed": seed,
                "smiles": smi,
                "generation_index": src["generation_index"],
                "in_top10": smi in top10_smis,
                "in_top100": smi in top100_smis,
                "is_eyr": bool(src["is_novel"] and src["has_fragment"]),
                "passes_medchem": bool(src["passes_medchem"]),
                "skip_reason": src["skip_reason"],
                "docking_score": src["docking_score"],
                "norm_score": src["norm_score"],
                "reward_score": src["reward_score"],
                "rewarded_rmsd": src["rewarded_rmsd"],
                "rewarded_passed": bool(src["rewarded_passed"]),
                "min_rmsd_any_pose": src["min_rmsd_any_pose"],
                "any_valid_pose_found": bool(src["any_valid_pose_found"]),
                "best_valid_docking_score": src["best_valid_docking_score"],
                "best_valid_norm_score": src["best_valid_norm_score"],
                "best_valid_reward_score": src["best_valid_reward_score"],
                "best_valid_rmsd": src["best_valid_rmsd"],
                "mw": src["mw"],
                "clogp": src["clogp"],
                "lipinski_pass": bool(src["lipinski_pass"]),
                "qed": src["qed"],
                "sa": src["sa"],
                "molskill_score": src["molskill_score"],
                "stoplight_score": src["stoplight_score"],
                "aizynthfinder_state_score": src["aizynthfinder_state_score"],
                "aizynthfinder_score": src["aizynthfinder_score"],
            }
        )

    return {
        "run_summary": run_summary,
        "traj_rows": traj_rows,
        "mol_records": mol_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc RMSD validation for MOCKDOCK ablation study.")
    parser.add_argument(
        "--capped-dir",
        type=Path,
        default=Path("/hickory/users/s/h/shuhang/devel/mockdock_data/exps_capped"),
    )
    parser.add_argument(
        "--no-rmsd-dir",
        type=Path,
        default=Path("/hickory/users/s/h/shuhang/devel/mockdock_data/exps_capped_no_rmsd"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/hickory/users/s/h/shuhang/devel/mockdock_data/analysis_ablation_rmsd"),
    )
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on runs per experiment for quick smoke test")
    args = parser.parse_args()

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, bool]] = []
    for exp_label, exp_dir, scan_sibs in [
        ("capped_rmsd", args.capped_dir, False),
        ("capped_no_rmsd", args.no_rmsd_dir, True),
    ]:
        res_files = sorted(exp_dir.glob("*/*/*/results.csv"))
        if args.limit > 0:
            res_files = res_files[: args.limit]
        for rf in res_files:
            tasks.append((exp_label, str(rf.parent), scan_sibs))

    print(f"Discovered {len(tasks)} run directories to validate with {args.workers} workers.", flush=True)

    run_summaries: list[dict[str, Any]] = []
    traj_all: list[dict[str, Any]] = []
    mols_by_exp: dict[str, list[dict[str, Any]]] = {"capped_rmsd": [], "capped_no_rmsd": []}

    done = 0
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx) as ex:
        futures = {ex.submit(process_single_run, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            done += 1
            try:
                res = fut.result()
                rs = res["run_summary"]
                run_summaries.append(rs)
                traj_all.extend(res["traj_rows"])
                mols_by_exp[rs["experiment"]].extend(res["mol_records"])
                if done % 25 == 0 or done == len(tasks):
                    print(
                        f"  [{done}/{len(tasks)}] {rs['experiment']} {rs['model']} {rs['target']} {rs['seed']} "
                        f"| app100={rs['apparent_top100_mdeval']:.3f} "
                        f"val100_rerank_zp={rs['validated_top100_reranked_zeropad']:.3f} "
                        f"top100_pass={rs['top100_rewarded_pass_rate']:.2f}/{rs['top100_anypose_pass_rate']:.2f} "
                        f"({time.time() - t0:.1f}s)",
                        flush=True,
                    )
            except Exception as e:
                print(f"  [ERROR] {t}: {e}", flush=True)

    # Save per-molecule parquet files
    for exp_label, records in mols_by_exp.items():
        if records:
            df_m = pl.DataFrame(records)
            out_pq = args.output_dir / f"molecules_{exp_label}.parquet"
            df_m.write_parquet(out_pq)
            print(f"Saved {len(df_m)} molecule rows to {out_pq}", flush=True)

    # Save trajectory bins
    if traj_all:
        df_traj = pl.DataFrame(traj_all).sort(["experiment", "model", "target", "seed", "bin_start"])
        df_traj.write_csv(args.output_dir / "trajectory_bins.csv")
        print(f"Saved trajectory bins ({len(df_traj)} rows)", flush=True)

    # Save run-level summary
    df_runs = pl.DataFrame(run_summaries).sort(["experiment", "target", "model", "seed"])
    df_runs.write_csv(args.output_dir / "runs_summary.csv")
    print(f"Saved runs_summary.csv ({len(df_runs)} rows)", flush=True)

    # Aggregate target x model summary (mean & sd across seeds)
    numeric_cols = [
        c for c, dt in zip(df_runs.columns, df_runs.dtypes)
        if dt in (pl.Float64, pl.Float32, pl.Int64, pl.Int32) and c not in ("is_high_yield_target",)
    ]
    agg_exprs = []
    for c in numeric_cols:
        agg_exprs.append(pl.col(c).drop_nans().mean().alias(f"{c}_mean"))
        agg_exprs.append(pl.col(c).drop_nans().std().alias(f"{c}_sd"))

    df_tm = (
        df_runs.group_by(["experiment", "target", "model"])
        .agg([pl.len().alias("n_runs"), *agg_exprs])
        .sort(["experiment", "target", "model"])
    )
    df_tm.write_csv(args.output_dir / "target_model_summary.csv")
    print(f"Saved target_model_summary.csv ({len(df_tm)} rows)", flush=True)

    # Macro summaries: (1) all 7 targets, (2) 5 high-yield targets
    macro_frames = []
    for subset_name, target_filter in [
        ("all_7_targets", None),
        ("high_yield_5_targets", list(HIGH_YIELD_TARGETS)),
    ]:
        sub = df_tm if target_filter is None else df_tm.filter(pl.col("target").is_in(target_filter))
        mean_cols = [f"{c}_mean" for c in numeric_cols]
        m_agg = [pl.col(mc).drop_nans().mean().alias(mc) for mc in mean_cols]
        df_mac = (
            sub.group_by(["experiment", "model"])
            .agg(m_agg)
            .with_columns(pl.lit(subset_name).alias("target_subset"))
            .sort(["target_subset", "experiment", "model"])
        )
        macro_frames.append(df_mac)
    df_macro = pl.concat(macro_frames, how="vertical")
    df_macro.write_csv(args.output_dir / "macro_summary.csv")
    print(f"Saved macro_summary.csv ({len(df_macro)} rows)", flush=True)

    # Side-by-side comparison delta table (no_rmsd - rmsd) per target x model
    key_compare_cols = [
        "apparent_top100_mdeval_mean",
        "apparent_top100_zeropad_mean",
        "validated_top100_fixed_rewarded_zeropad_mean",
        "validated_top100_fixed_anypose_zeropad_mean",
        "validated_top100_reranked_zeropad_mean",
        "validated_top100_reranked_mdeval_mean",
        "apparent_top10_mdeval_mean",
        "validated_top10_fixed_rewarded_zeropad_mean",
        "validated_top10_fixed_anypose_zeropad_mean",
        "validated_top10_reranked_zeropad_mean",
        "top10_rewarded_pass_rate_mean",
        "top10_anypose_pass_rate_mean",
        "top10_rewarded_rmsd_median_mean",
        "top100_rewarded_pass_rate_mean",
        "top100_anypose_pass_rate_mean",
        "top100_rewarded_rmsd_median_mean",
        "top100_min_rmsd_median_mean",
        "eyr_rewarded_pass_rate_mean",
        "eyr_anypose_pass_rate_mean",
        "eyr_rewarded_rmsd_median_mean",
        "eyr_min_rmsd_median_mean",
        "top100_docking_kcal_mean_mean",
        "top100_mw_mean_mean",
        "top100_qed_mean_mean",
        "top100_sa_mean_mean",
        "top100_molskill_mean_mean",
        "top100_stoplight_mean_mean",
        "top100_aizynth_state_mean_mean",
        "top100_lipinski_rate_mean",
        "eyr_mw_mean_mean",
        "eyr_qed_mean_mean",
        "eyr_sa_mean_mean",
        "eyr_molskill_mean_mean",
        "eyr_stoplight_mean_mean",
        "eyr_aizynth_state_mean_mean",
        "eyr_lipinski_rate_mean",
        "early_rmsd_median_mean",
        "late_rmsd_median_mean",
        "early_rewarded_pass_rate_mean",
        "late_rewarded_pass_rate_mean",
    ]
    c_df = df_tm.filter(pl.col("experiment") == "capped_rmsd").select(["target", "model", *key_compare_cols])
    n_df = df_tm.filter(pl.col("experiment") == "capped_no_rmsd").select(["target", "model", *key_compare_cols])
    joined = c_df.join(n_df, on=["target", "model"], how="full", suffix="_no_rmsd")
    delta_exprs = [
        (pl.col(f"{c}_no_rmsd") - pl.col(c)).alias(f"delta_{c}")
        for c in key_compare_cols
    ]
    joined = joined.with_columns(delta_exprs).sort(["target", "model"])
    joined.write_csv(args.output_dir / "ablation_comparison_delta.csv")
    print(f"Saved ablation_comparison_delta.csv ({len(joined)} rows)", flush=True)

    print(f"All done in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
