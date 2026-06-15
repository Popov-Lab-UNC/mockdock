#!/usr/bin/env python3
"""Long-lived Stoplight worker intended to run with the Stoplight virtualenv."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import sys
from pathlib import Path

STOPLIGHT_DIR = os.environ.get(
    "STOPLIGHT_DIR", "/work/users/s/h/shuhang/stoplight"
)
os.chdir(STOPLIGHT_DIR)
if STOPLIGHT_DIR not in sys.path:
    sys.path.insert(0, STOPLIGHT_DIR)

from Stoplight.main import get_csv_from_smiles

CONVERT_OPTIONS = {
    "fsp3": "FSP3",
    "logp": "ALogP",
    "mw": "Molecular Weight",
    "rot_bonds": "Number of Rotatable Bonds",
    "psa": "Polar Surface Area",
    "esol": "Solubility in Water (mg/L)",
    "hbd": "HBD",
    "hba": "HBA",
    "nha": "Num Heavy Atoms",
    "nrings": "Number of Rings",
    "nsc4": "Num Saturated Quaternary Carbons",
    "blac_agg": "AmpC β-lactamase aggregation",
    "cprot_agg": "Cysteine protease cruzain aggregation",
    "fluc_inter": "Firefly Luciferase interference",
    "nluc_inter": "Nano Luciferase interference",
    "redox_inter": "Redox interference",
    "thiol_inter": "Thiol interference",
    "bbb": "BBB Permeability",
    "caco2": "CACO2",
    "cns": "CNS Activity",
    "hep_stab": "Hepatic Stability",
    "micro_hf_sub": "Microsomal Half-life Sub-cellular",
    "micro_hf_t": "Microsomal Half-life Tissue",
    "micro_clr": "Microsomal Intrinsic Clearance",
    "o_avail": "Oral Bioavailability",
    "pla_hf": "Plasma Half-life",
    "pla_pb": "Plasma Protein Binding",
    "ren_clr": "Renal Clearance",
}

PROPERTY_LITERALS = {
    "all": [
        "fsp3", "logp", "mw", "rot_bonds", "psa", "esol", "blac_agg", "cprot_agg",
        "fluc_inter", "nluc_inter", "redox_inter", "thiol_inter", "bbb", "caco2", "cns",
        "hep_stab", "micro_hf_sub", "micro_hf_t", "micro_clr", "o_avail", "pla_hf",
        "pla_pb", "ren_clr",
    ],
}

OPTIONS = {
    CONVERT_OPTIONS[key]: True
    for key in PROPERTY_LITERALS["all"]
}
OPTIONS["drop_invalid"] = False
OPTIONS["precision"] = 2


def _score_cache(cache_path: str, smiles_list: list[str]) -> None:
    out_path = Path(cache_path).parent / "scores_stoplight.csv"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        csv_text = get_csv_from_smiles(smiles_list=smiles_list, options=OPTIONS)
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if not rows or "SMILES" not in rows[0] or "OverallScore" not in rows[0]:
        raise ValueError("Stoplight output missing required columns.")

    scores_map = {row["SMILES"]: row["OverallScore"] for row in rows}
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["smiles", "stoplight_score"])
        writer.writeheader()
        for smiles in smiles_list:
            writer.writerow({"smiles": smiles, "stoplight_score": scores_map.get(smiles)})


def main() -> None:
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "STOP":
            break
        task = json.loads(line)
        cache_path = task["cache_path"]
        try:
            _score_cache(cache_path, task["smiles_list"])
            print(
                json.dumps(
                    {
                        "ok": True,
                        "cache_path": cache_path,
                        "message": f"Saved Stoplight scores for {cache_path}",
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "cache_path": cache_path,
                        "message": str(exc),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
