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

# Resolve paths relative to benchmark root (parent of scripts/)
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIGS_DIR = BENCHMARK_ROOT / "src" / "fcgmb" / "configs"
DEFAULT_OUTPUT = BENCHMARK_ROOT / "variance_runs" / "crystal_ligand_mapping.csv"

RCSB_CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"


def fetch_chembl_id_for_ligand(ligand_resname: str) -> str | None:
    """Fetch ChEMBL ID for a PDB ligand component from RCSB Data API."""
    url = RCSB_CHEMCOMP_URL.format(comp_id=ligand_resname)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"  [WARN] Failed to fetch {ligand_resname}: {e}", file=sys.stderr)
        return None

    # RCSB returns rcsb_chem_comp_related at top level
    related = data.get("rcsb_chem_comp_related") or []
    if not isinstance(related, list):
        related = []

    for entry in related:
        if isinstance(entry, dict) and entry.get("resource_name") == "ChEMBL":
            return entry.get("resource_accession_code")
    return None


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
    args = parser.parse_args()

    config_dir = args.config_dir
    output_path = args.output

    if not config_dir.exists():
        print(f"ERROR: Config directory not found: {config_dir}", file=sys.stderr)
        sys.exit(1)

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
        chembl_id = fetch_chembl_id_for_ligand(ligand_resname)
        if chembl_id is None:
            print(f"  [SKIP] {benchmark_name}: no ChEMBL ID for ligand {ligand_resname}")
            continue

        rows.append({
            "system_key": system_key,
            "molecule_chembl_id": chembl_id,
            "label": "Crystal ligand",
        })
        print(f"  [OK]   {benchmark_name}: {ligand_resname} -> {chembl_id}")

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
