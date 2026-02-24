"""
Temporary script to:
1. Identify the crystal ligand residue for each PDB system in variance_runs/init/
2. Fetch the ideal SDF from RCSB Ligand Expo
3. Match the ligand to a ChEMBL ID via canonical SMILES or Tanimoto similarity
   against molecules found in *_cleaned_data.csv files
4. Export results as CSV
"""

import asyncio
import csv
import re
import sys
from pathlib import Path

import aiohttp
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# ── Config ────────────────────────────────────────────────────────────────────

INIT_DIR   = Path(__file__).parent / "variance_runs" / "init"
OUTPUT_CSV = Path(__file__).parent / "variance_runs" / "crystal_ligand_mapping_full.csv"
LIGAND_EXPO_URL = "https://files.rcsb.org/ligands/view/{resname}_ideal.sdf"

# Residue names to skip (common solvents, ions, crystallographic artefacts)
SKIP_RESNAMES = {
    "HOH", "WAT", "H2O",  # water
    "SO4", "PO4", "NO3",  # common salts
    "EDO", "PEG", "GOL", "EOH", "DMS", "MPD", "TRS", "MES", "IMD",  # cryo/buffer
    "NA", "K", "MG", "CA", "ZN", "FE", "MN", "NI", "CU", "CO",      # metals
    "CL", "BR", "IOD", "F",                                            # halogens
    "ACT", "ACY", "FMT",                                               # acetate etc
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_ligand_resname(ligand_pdb: Path) -> str | None:
    """Extract the 3-letter residue name from HETATM lines in a ligand PDB file."""
    with ligand_pdb.open() as fh:
        for line in fh:
            if line.startswith(("HETATM", "ATOM")):
                resname = line[17:20].strip()
                if resname and resname not in SKIP_RESNAMES:
                    return resname
    return None


async def fetch_sdf(resname: str, session: aiohttp.ClientSession) -> str | None:
    """Fetch ideal SDF text for a ligand residue name from RCSB Ligand Expo."""
    url = LIGAND_EXPO_URL.format(resname=resname.upper())
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                return await resp.text()
            print(f"  [WARN] Ligand Expo returned {resp.status} for {resname}")
            return None
    except Exception as exc:
        print(f"  [WARN] Failed to fetch {resname}: {exc}")
        return None


def sdf_to_smiles(sdf_text: str) -> str | None:
    """Parse an SDF block and return the canonical SMILES (no Hs)."""
    mol = Chem.MolFromMolBlock(sdf_text, removeHs=True)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def canonical_smiles(smi: str) -> str | None:
    """Return RDKit canonical SMILES or None if the SMILES is invalid."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def tanimoto(smi_a: str, smi_b: str) -> float:
    """Compute Morgan (ECFP4) Tanimoto similarity between two SMILES."""
    try:
        mol_a = Chem.MolFromSmiles(smi_a)
        mol_b = Chem.MolFromSmiles(smi_b)
        if mol_a is None or mol_b is None:
            return 0.0
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return 0.0


def load_molecules_from_csvs(pdb_dir: Path) -> tuple[list[dict], list[str]]:
    """
    Walk all *_cleaned_data.csv files under pdb_dir.

    Returns:
        molecules  – unique molecules as list of {"chembl_id": ..., "smiles": ...}
        system_keys – one system_key per CSV file found (filename prefix)
    """
    molecules: dict[str, str] = {}   # chembl_id -> canonical_smiles
    system_keys: list[str] = []

    for csv_path in sorted(pdb_dir.rglob("*_cleaned_data.csv")):
        # system_key = filename without the _cleaned_data.csv suffix
        system_keys.append(csv_path.stem.replace("_cleaned_data", ""))
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cid = row.get("molecule_chembl_id", "").strip()
                smi = row.get("canonical_smiles", "").strip()
                if cid and smi and cid not in molecules:
                    molecules[cid] = smi

    return [{"chembl_id": k, "smiles": v} for k, v in molecules.items()], system_keys


def find_best_match(
    query_smi: str,
    molecules: list[dict],
    similarity_threshold: float = 0.5,
) -> dict:
    """
    Try exact canonical SMILES match first; fall back to highest Tanimoto.
    Returns dict with keys: matched_chembl_id, match_type, tanimoto_score.
    """
    query_can = canonical_smiles(query_smi)

    best_id = None
    best_score = 0.0
    perfect_similarity_ids: list[str] = []

    for mol in molecules:
        mol_can = canonical_smiles(mol["smiles"])
        if not mol_can:
            continue

        # Exact match
        if query_can and mol_can == query_can:
            return {
                "matched_chembl_id": mol["chembl_id"],
                "match_type": "exact",
                "tanimoto_score": 1.0,
            }

        # Tanimoto
        score = tanimoto(query_smi, mol["smiles"])
        if score == 1.0:
            perfect_similarity_ids.append(mol["chembl_id"])

        if score > best_score:
            best_score = score
            best_id = mol["chembl_id"]

    if best_score >= similarity_threshold:
        # Guardrail: similarity-based fallback must not choose arbitrarily when
        # multiple compounds are perfect (1.0) Tanimoto matches.
        if best_score == 1.0 and len(perfect_similarity_ids) > 1:
            return {
                "matched_chembl_id": None,
                "match_type": "ambiguous_similarity_multiple_1.0",
                "tanimoto_score": 1.0,
            }
        return {
            "matched_chembl_id": best_id,
            "match_type": "similarity",
            "tanimoto_score": round(best_score, 4),
        }

    return {
        "matched_chembl_id": None,
        "match_type": "no_match",
        "tanimoto_score": round(best_score, 4),
    }


# ── Main async logic ──────────────────────────────────────────────────────────

async def process_system(
    pdb_dir: Path,
    session: aiohttp.ClientSession,
) -> dict | None:
    """Process one CHEMBL{target}_{PDB} directory and return a result row."""
    dir_name = pdb_dir.name                   # e.g. CHEMBL204_1MU6
    parts    = dir_name.split("_", 1)
    if len(parts) != 2:
        print(f"  [SKIP] Unexpected directory name: {dir_name}")
        return None

    target_chembl_id = parts[0]              # CHEMBL204
    pdb_id           = parts[1]              # 1MU6

    # 1. Find ligand residue name
    ligand_pdb = pdb_dir / f"{pdb_id}_ligand.pdb"
    if not ligand_pdb.exists():
        print(f"  [SKIP] No ligand PDB found for {dir_name}")
        _, system_keys = load_molecules_from_csvs(pdb_dir)
        fallback = {
            "system_dir":             dir_name,
            "target_chembl_id":       target_chembl_id,
            "pdb_id":                 pdb_id,
            "ligand_resname":         None,
            "ligand_smiles":          None,
            "molecule_chembl_id":     None,
            "match_type":             "no_ligand_pdb",
            "tanimoto_score":         None,
            "n_molecules_in_dataset": 0,
            "label":                  "Crystal ligand",
        }
        if not system_keys:
            return [{"system_key": None, **fallback}]
        return [{"system_key": sk, **fallback} for sk in system_keys]

    resname = parse_ligand_resname(ligand_pdb)
    if not resname:
        print(f"  [WARN] Could not parse residue name from {ligand_pdb.name}")
        resname = "UNKNOWN"

    print(f"  [{dir_name}] Residue: {resname}")

    # 2. Fetch SDF from Ligand Expo
    sdf_text = await fetch_sdf(resname, session)
    ligand_smiles = None
    if sdf_text:
        ligand_smiles = sdf_to_smiles(sdf_text)
        if ligand_smiles:
            print(f"  [{dir_name}] Ligand SMILES: {ligand_smiles[:60]}...")
        else:
            print(f"  [{dir_name}] Could not parse SMILES from SDF")
    else:
        print(f"  [{dir_name}] No SDF retrieved from Ligand Expo")

    # 3. Load dataset molecules and system keys
    molecules, system_keys = load_molecules_from_csvs(pdb_dir)
    print(f"  [{dir_name}] {len(molecules)} unique molecules across {len(system_keys)} CSV(s)")

    # 4. Match crystal ligand against the pooled dataset
    if ligand_smiles and molecules:
        match = find_best_match(ligand_smiles, molecules)
    else:
        match = {
            "matched_chembl_id": None,
            "match_type": "no_smiles_or_no_data",
            "tanimoto_score": None,
        }

    # 5. Build one row per system_key (one per CSV file found)
    base = {
        "system_dir":             dir_name,
        "target_chembl_id":       target_chembl_id,
        "pdb_id":                 pdb_id,
        "ligand_resname":         resname,
        "ligand_smiles":          ligand_smiles,
        "molecule_chembl_id":     match["matched_chembl_id"],
        "match_type":             match["match_type"],
        "tanimoto_score":         match["tanimoto_score"],
        "n_molecules_in_dataset": len(molecules),
        "label":                  "Crystal ligand",
    }

    if not system_keys:
        # No CSVs found – still emit one row, system_key left blank
        return [{"system_key": None, **base}]

    return [{"system_key": sk, **base} for sk in system_keys]


async def main() -> None:
    if not INIT_DIR.exists():
        sys.exit(f"ERROR: INIT_DIR does not exist: {INIT_DIR}")

    # Collect all target-PDB directories (direct children of INIT_DIR)
    pdb_dirs = sorted(
        d for d in INIT_DIR.iterdir()
        if d.is_dir() and re.match(r"^CHEMBL\d+_[A-Z0-9]{4}$", d.name)
    )
    print(f"Found {len(pdb_dirs)} PDB systems to process.\n")

    results: list[dict] = []
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_system(d, session) for d in pdb_dirs]
        for coro in asyncio.as_completed(tasks):
            rows = await coro
            if rows:
                results.extend(rows)

    # Sort by system_key for a deterministic, join-friendly output
    results.sort(key=lambda r: (r["system_key"] or "", r["system_dir"]))

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system_key",
        "molecule_chembl_id",
        "label",
        "system_dir", "target_chembl_id", "pdb_id",
        "ligand_resname", "ligand_smiles",
        "match_type", "tanimoto_score",
        "n_molecules_in_dataset",
    ]
    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Results written to: {OUTPUT_CSV}")
    print(f"Total rows: {len(results)}")

    # Quick summary
    matched = [r for r in results if r["molecule_chembl_id"]]
    print(f"Matched: {len(matched)} / {len(results)}")
    for r in matched:
        print(f"  {r['system_key']:55s} -> {r['molecule_chembl_id']} ({r['match_type']}, Tanimoto={r['tanimoto_score']})")


if __name__ == "__main__":
    asyncio.run(main())
