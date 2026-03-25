#!/usr/bin/env python3
"""Generate crystal ligand mapping for variance analysis plots.

Reads benchmark configs (YAML), fetches ChEMBL IDs for each crystal ligand
from the RCSB PDB Data API, and writes a CSV suitable for
analyze_variance_runs.py --mapping.

Output format (crystal_ligand_mapping.csv):
  system_key,molecule_chembl_id,label

The system_key matches the variance run results layout:
  {target_id}_{pdb_id}_{doc_id}_{assay_id}

Run from benchmark root::

    python scripts/misc/generate_crystal_ligand_mapping.py
    python scripts/misc/generate_crystal_ligand_mapping.py -o variance_runs/crystal_ligand_mapping.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Resolve paths relative to benchmark root (parent of scripts/)
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIGS_DIR = BENCHMARK_ROOT / "src" / "fcgmb" / "configs"
DEFAULT_OUTPUT = BENCHMARK_ROOT / "variance_runs" / "crystal_ligand_mapping.csv"

RCSB_CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"


def fetch_ligand_metadata_for_resname(ligand_resname: str) -> tuple[str | None, str | None]:
    """Fetch (ChEMBL ID, SMILES) for a PDB ligand component from RCSB Data API."""
    url = RCSB_CHEMCOMP_URL.format(comp_id=ligand_resname)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"  [WARN] Failed to fetch {ligand_resname}: {e}", file=sys.stderr)
        return None, None

    # RCSB returns rcsb_chem_comp_related at top level
    related = data.get("rcsb_chem_comp_related") or []
    if not isinstance(related, list):
        related = []

    chembl_id = None
    for entry in related:
        if isinstance(entry, dict) and entry.get("resource_name") == "ChEMBL":
            chembl_id = entry.get("resource_accession_code")
            break

    descriptor = data.get("rcsb_chem_comp_descriptor") or {}
    # RCSB descriptor key casing is not consistent across components.
    smiles = (
        descriptor.get("SMILES_stereo")
        or descriptor.get("SMILES")
        or descriptor.get("smilesstereo")
        or descriptor.get("smiles")
    )
    return chembl_id, smiles


def load_cache_compounds(
    cache_dir: Path, target_id: str, doc_id: str, assay_id: str
) -> list[tuple[str, str]]:
    """Load (molecule_chembl_id, canonical_smiles) from assay cache CSV."""
    assay_path = cache_dir / target_id / f"{doc_id}_{assay_id}.csv"
    doc_path = cache_dir / f"{target_id}_{doc_id}.csv"
    source_path = assay_path if assay_path.exists() else doc_path

    if not source_path.exists():
        return []

    rows: list[tuple[str, str]] = []
    with open(source_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mol_id = row.get("molecule_chembl_id")
            smiles = row.get("canonical_smiles")
            if not mol_id or not smiles:
                continue
            if source_path == doc_path and row.get("assay_chembl_id") not in (None, "", assay_id):
                # Document-level cache can contain multiple assays; keep current assay only.
                continue
            rows.append((mol_id, smiles))
    return rows


def find_best_similarity_match(
    query_smiles: str, candidates: list[tuple[str, str]], fpgen
) -> tuple[str | None, float]:
    """Return best matching molecule_chembl_id by Morgan/Tanimoto similarity."""
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        return None, -1.0
    query_fp = fpgen.GetFingerprint(query_mol)

    best_by_id: dict[str, float] = {}
    for mol_id, smiles in candidates:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        sim = DataStructs.TanimotoSimilarity(query_fp, fpgen.GetFingerprint(mol))
        if sim > best_by_id.get(mol_id, -1.0):
            best_by_id[mol_id] = sim

    if not best_by_id:
        return None, -1.0

    best_mol_id = max(best_by_id, key=best_by_id.get)
    return best_mol_id, best_by_id[best_mol_id]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate crystal ligand mapping from benchmark configs"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIGS_DIR,
        help=f"Directory with benchmark YAML configs (default: {DEFAULT_CONFIGS_DIR})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional ChEMBL cache directory (e.g. data/chembl_cache). "
            "When provided, use max-similarity mapping against cache to find/verify molecule_chembl_id."
        ),
    )
    parser.add_argument(
        "--min-cache-similarity",
        type=float,
        default=0.99,
        help=(
            "Minimum Tanimoto similarity required to trust cache-based ID mapping "
            "(default: 0.99)."
        ),
    )
    args = parser.parse_args()

    config_dir = args.config_dir
    output_path = args.output
    cache_dir = args.cache_dir

    if not config_dir.exists():
        print(f"ERROR: Config directory not found: {config_dir}", file=sys.stderr)
        sys.exit(1)
    if cache_dir is not None and not cache_dir.exists():
        print(f"ERROR: Cache directory not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    configs = sorted(config_dir.glob("*.yaml"))
    if not configs:
        print(f"ERROR: No YAML configs found in {config_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for config_path in configs:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        target_id = cfg.get("target_id")
        pdb_id = cfg.get("pdb_id")
        doc_id = cfg.get("doc_id")
        assay_id = cfg.get("assay_id")
        ligand_resname = cfg.get("ligand_resname")
        benchmark_name = cfg.get("benchmark_name", config_path.stem)

        if not all([target_id, pdb_id, doc_id, assay_id, ligand_resname]):
            print(f"  [SKIP] {benchmark_name}: missing required config keys")
            continue

        system_key = f"{target_id}_{pdb_id}_{doc_id}_{assay_id}"
        rcsb_chembl_id, crystal_smiles = fetch_ligand_metadata_for_resname(ligand_resname)

        chembl_id = rcsb_chembl_id
        cache_message = ""
        if cache_dir is not None and crystal_smiles:
            cache_compounds = load_cache_compounds(cache_dir, target_id, doc_id, assay_id)
            if cache_compounds:
                best_cache_id, best_sim = find_best_similarity_match(
                    crystal_smiles, cache_compounds, fpgen
                )
                if best_cache_id is not None and best_sim >= args.min_cache_similarity:
                    chembl_id = best_cache_id
                    if rcsb_chembl_id and rcsb_chembl_id != best_cache_id:
                        cache_message = (
                            f" [WARN mismatch: RCSB={rcsb_chembl_id}, cache_best={best_cache_id}, "
                            f"sim={best_sim:.3f}]"
                        )
                    else:
                        cache_message = f" [cache verified, sim={best_sim:.3f}]"
                elif best_cache_id is not None:
                    if chembl_id is None:
                        # RCSB can miss ChEMBL cross-references for some ligands.
                        chembl_id = best_cache_id
                        cache_message = (
                            f" [cache fallback below threshold: {best_cache_id} "
                            f"sim={best_sim:.3f}]"
                        )
                    else:
                        cache_message = (
                            f" [cache best below threshold: {best_cache_id} sim={best_sim:.3f}]"
                        )
            else:
                cache_message = " [cache miss]"

        if chembl_id is None:
            print(f"  [SKIP] {benchmark_name}: no ChEMBL ID for ligand {ligand_resname}")
            continue

        rows.append({
            "system_key": system_key,
            "molecule_chembl_id": chembl_id,
            "label": "Crystal ligand",
        })
        print(f"  [OK]   {benchmark_name}: {ligand_resname} -> {chembl_id}{cache_message}")

    if not rows:
        print("ERROR: No mappings generated.", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["system_key", "molecule_chembl_id", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} mappings to {output_path}")


if __name__ == "__main__":
    main()
