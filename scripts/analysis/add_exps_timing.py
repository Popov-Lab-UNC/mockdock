#!/usr/bin/env python3
"""
Populate timing fields in benchmark exps metrics.json files.

For each exps/{model}/{run}/{target}/metrics.json:
  - evaluation timing is derived from oracle timings already present in metrics.json
  - generation timing is read from an optional sidecar file written by model runners
    outside oracle calls (default: generation_timing.json in the same folder)

Sidecar format example:
{
  "total_generation_time_sec": 123.45,
  "n_generated_ligands": 1000
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def discover_metrics(exps_dir: Path) -> list[Path]:
    return sorted(exps_dir.glob("*/*/*/metrics.json"))


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def canonicalize_metrics_payload(payload: dict) -> dict:
    ordered = {}
    for key in ["benchmark", "budget_used", "budget_total", "generation_rounds", "model", "seed"]:
        if key in payload:
            ordered[key] = payload[key]
    for key in [
        "n_molecules_total",
        "n_molecules_attempted",
        "total_gen_time",
        "avg_gen_time_per_mol",
        "total_eval_time",
        "avg_eval_time_per_mol",
        "total_time",
        "avg_time_per_mol",
    ]:
        if key in payload:
            ordered[key] = payload[key]
    for k, v in payload.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def get_metric(payload: dict, key: str, default=0):
    """Read key from flat payload."""
    return payload.get(key, default)


def enrich_metrics(metrics_path: Path, sidecar_name: str) -> tuple[bool, str]:
    payload = load_json(metrics_path)
    total_eval = float(get_metric(payload, "total_eval_time", 0.0))
    n_generated = int(get_metric(payload, "n_molecules_total", 0))
    n_attempted_docked = int(get_metric(payload, "n_molecules_attempted", 0))
    avg_eval = total_eval / max(1, n_attempted_docked)

    payload["n_molecules_total"] = int(n_generated)
    payload["n_molecules_attempted"] = int(n_attempted_docked)
    payload["total_eval_time"] = round(total_eval, 2)
    payload["avg_eval_time_per_mol"] = round(avg_eval, 4)

    sidecar_path = metrics_path.parent / sidecar_name
    if sidecar_path.exists():
        sidecar = load_json(sidecar_path)
        total_generation = float(sidecar.get("total_generation_time_sec", 0.0))
        n_sidecar = sidecar.get("n_generated_ligands", None)
        if n_sidecar is not None:
            n_generated = int(n_sidecar)
        avg_generation = total_generation / max(1, n_generated)
        payload["n_molecules_total"] = int(n_generated)
        payload["total_gen_time"] = round(total_generation, 2)
        payload["avg_gen_time_per_mol"] = round(avg_generation, 4)
        payload["total_time"] = round(total_generation + total_eval, 2)
        payload["avg_time_per_mol"] = round(
            avg_generation + avg_eval,
            4,
        )
        payload = canonicalize_metrics_payload(payload)
        write_json(metrics_path, payload)
        return True, "updated (with generation + evaluation timing)"

    # Keep generation fields explicit even when sidecar is missing.
    payload["total_gen_time"] = 0.0
    payload["avg_gen_time_per_mol"] = 0.0
    payload["total_time"] = round(total_eval, 2)
    payload["avg_time_per_mol"] = round(avg_eval, 4)
    payload = canonicalize_metrics_payload(payload)
    write_json(metrics_path, payload)
    return True, "updated (evaluation timing only; no generation sidecar)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add generation/evaluation timing to exps metrics.json files."
    )
    parser.add_argument("--exps-dir", type=Path, default=Path("exps"), help="Path to exps folder")
    parser.add_argument(
        "--generation-sidecar",
        type=str,
        default="generation_timing.json",
        help="Sidecar filename expected in each target folder",
    )
    args = parser.parse_args()

    metrics_files = discover_metrics(args.exps_dir)
    print(f"Found {len(metrics_files)} metrics.json files.")
    if not metrics_files:
        return

    updated = 0
    for metrics_path in metrics_files:
        ok, msg = enrich_metrics(metrics_path, args.generation_sidecar)
        if ok:
            updated += 1
        print(f"{metrics_path}: {msg}")

    print(f"Done. Updated {updated}/{len(metrics_files)} files.")


if __name__ == "__main__":
    main()
