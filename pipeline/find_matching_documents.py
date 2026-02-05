#!/usr/bin/env python3
"""
Step 4: Find ChEMBL documents (patents and/or publications) that contain compounds
matching the crystal ligand using fingerprint similarity.

Uses RDKit's FingerprintGenerator for efficient similarity calculation.

Input: data/chembl_pdb_druglike.csv
Output: data/chembl_docking_benchmark.csv
"""
import argparse
import os
import pandas as pd
from tqdm import tqdm
import time
from collections import defaultdict
from statistics import median
import sys
from pathlib import Path
import sqlite3

# Add script directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import get_chunk_output_path

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

import polars as pl
from chembl_cache import write_assay_cache
import requests

def check_api_status():
    """Check if the ChEMBL API is responding."""
    try:
        r = requests.get("https://www.ebi.ac.uk/chembl/api/data/status", timeout=5)
        return r.status_code == 200
    except:
        return False


def get_fingerprint(smiles: str, fpgen):
    """Generate fingerprint for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return fpgen.GetFingerprint(mol)
    except:
        return None


def fetch_assays_for_target(target_chembl_id: str, min_compounds: int = 20,
                            cache_dir: str = "data/chembl_cache"):
    """
    Fetch ChEMBL assays for a target with their associated compounds.
    
    Returns: dict of {(doc_id, assay_id): {'type': str, 'smiles_list': [str], 'pchembl_by_smiles': {}}}
    """
    from chembl_webresource_client.new_client import new_client
    

    try:
        # Fetch all binding activities for the target in one go
        activities = new_client.activity.filter(
            target_chembl_id=target_chembl_id,
            assay_type='B',
            pchembl_value__isnull=False
        ).only('document_chembl_id', 'assay_chembl_id', 'canonical_smiles', 
               'src_id', 'pchembl_value', 'molecule_chembl_id')
        
        # Group by (doc, assay)
        assay_groups = defaultdict(list)
        doc_types = {}
        
        for act in activities:
            doc_id = act.get('document_chembl_id')
            assay_id = act.get('assay_chembl_id')
            src_id = act.get('src_id')
            
            if not doc_id or not assay_id:
                continue
                
            # Identify source for your 'type' key
            if src_id == 1:
                doc_type = 'literature'
            else:
                doc_type = 'other'
            
            assay_groups[(doc_id, assay_id)].append(act)
            doc_types[doc_id] = doc_type
        
        results = {}
        
        for (doc_id, assay_id), acts in assay_groups.items():
            # Extract smiles and pchembl
            smiles_list = []
            pchembl_by_smiles = defaultdict(list)
            
            # Prepare rows for cache
            cache_rows = []
            
            for act in acts:
                smiles = act.get('canonical_smiles')
                pchembl = act.get('pchembl_value')
                mol_id = act.get('molecule_chembl_id')
                
                if not smiles:
                    continue
                    
                smiles_list.append(smiles)
                if pchembl:
                    pchembl_by_smiles[smiles].append(float(pchembl))
                
                cache_rows.append({
                    'molecule_chembl_id': mol_id,
                    'canonical_smiles': smiles,
                    'pchembl_value': pchembl,
                    'assay_chembl_id': assay_id,
                    'document_chembl_id': doc_id,
                    'target_chembl_id': target_chembl_id
                })
            
            # Check min compounds threshold (at assay level)
            unique_smiles = set(smiles_list)
            if len(unique_smiles) < min_compounds:
                continue
                
            # Save to cache
            if cache_rows:
                df_cache = pl.DataFrame(cache_rows)
                write_assay_cache(cache_dir, target_chembl_id, doc_id, assay_id, df_cache)
            
            results[(doc_id, assay_id)] = {
                'type': doc_types.get(doc_id, 'unknown'),
                'smiles_list': list(unique_smiles),
                'pchembl_by_smiles': pchembl_by_smiles
            }
            
        return results
        
    except Exception as e:
        print(f"  [!] Error fetching assays for {target_chembl_id}: {e}")
        return {}


def _chunked_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_assays_for_targets_sqlite(
    target_ids,
    sqlite_path: Path,
    min_compounds: int = 20,
    cache_dir: str = "data/chembl_cache",
):
    """
    Fetch assays for multiple targets in a single (batched) SQLite query.
    Returns: dict[target_id][(doc_id, assay_id)] = {...}
    """
    if len(target_ids) == 0:
        return {}

    all_rows = []
    query = """
    SELECT
        td.chembl_id AS target_chembl_id,
        d.chembl_id AS document_chembl_id,
        a.chembl_id AS assay_chembl_id,
        cs.canonical_smiles,
        d.src_id,
        act.pchembl_value,
        md.chembl_id AS molecule_chembl_id
    FROM activities act
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid
    JOIN docs d ON act.doc_id = d.doc_id
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    LEFT JOIN compound_structures cs ON act.molregno = cs.molregno
    WHERE td.chembl_id IN ({placeholders})
      AND a.assay_type = 'B'
      AND act.pchembl_value IS NOT NULL
    """

    with sqlite3.connect(sqlite_path) as conn:
        # SQLite has a limit on the number of query parameters, so chunk targets if needed.
        for chunk in _chunked_list(list(target_ids), 900):
            placeholders = ",".join(["?"] * len(chunk))
            chunk_query = query.format(placeholders=placeholders)
            chunk_df = pd.read_sql_query(chunk_query, conn, params=chunk)
            all_rows.append(chunk_df)

    if not all_rows:
        return {}

    df = pd.concat(all_rows, ignore_index=True)
    if df.empty:
        return {}

    results_by_target = defaultdict(dict)

    for target_id, tdf in df.groupby("target_chembl_id"):
        for (doc_id, assay_id), group in tdf.groupby(["document_chembl_id", "assay_chembl_id"]):
            if pd.isna(doc_id) or pd.isna(assay_id):
                continue

            smiles_list = []
            pchembl_by_smiles = defaultdict(list)
            cache_rows = []

            src_id = group["src_id"].iloc[0] if "src_id" in group.columns and not group["src_id"].isna().all() else None
            if src_id == 1:
                doc_type = "literature"
            elif src_id is None or pd.isna(src_id):
                doc_type = "unknown"
            else:
                doc_type = "other"

            for _, row in group.iterrows():
                smiles = row.get("canonical_smiles")
                pchembl = row.get("pchembl_value")
                mol_id = row.get("molecule_chembl_id")

                if not smiles or pd.isna(smiles):
                    continue

                smiles_list.append(smiles)
                if pchembl is not None and not pd.isna(pchembl):
                    pchembl_by_smiles[smiles].append(float(pchembl))

                cache_rows.append({
                    "molecule_chembl_id": mol_id,
                    "canonical_smiles": smiles,
                    "pchembl_value": pchembl,
                    "assay_chembl_id": assay_id,
                    "document_chembl_id": doc_id,
                    "target_chembl_id": target_id,
                })

            unique_smiles = set(smiles_list)
            if len(unique_smiles) < min_compounds:
                continue

            if cache_rows:
                df_cache = pl.DataFrame(cache_rows)
                write_assay_cache(cache_dir, target_id, doc_id, assay_id, df_cache)

            results_by_target[target_id][(doc_id, assay_id)] = {
                "type": doc_type,
                "smiles_list": list(unique_smiles),
                "pchembl_by_smiles": pchembl_by_smiles,
            }

    return results_by_target


def find_matching_compounds(crystal_fp, doc_smiles_list: list, fpgen, 
                            similarity_threshold: float = 1.0):
    """
    Find compounds in document that match the crystal ligand.
    
    Returns: list of (smiles, similarity) tuples for matches above threshold
    """
    matches = []
    seen = set()
    
    for smiles in doc_smiles_list:
        if smiles in seen:
            continue
        seen.add(smiles)
        
        fp = get_fingerprint(smiles, fpgen)
        if fp is None:
            continue
        
        similarity = DataStructs.TanimotoSimilarity(crystal_fp, fp)
        if similarity >= similarity_threshold:
            matches.append((smiles, similarity))
    
    return sorted(matches, key=lambda x: x[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Find ChEMBL documents matching crystal ligands")
    parser.add_argument("--input", default="data/chembl_pdb_druglike.csv", help="Input CSV")
    parser.add_argument("--output", default="data/chembl_docking_benchmark.csv", help="Output CSV")

    parser.add_argument("--similarity-threshold", type=float, default=1.0, 
                        help="Tanimoto similarity threshold for matching (default: 1.0 = exact match)")
    parser.add_argument("--min-compounds", type=int, default=20, 
                        help="Minimum compounds in document to consider")
    parser.add_argument("--start", type=int, default=0, help="Start index (for parallelization)")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between ChEMBL API calls")
    parser.add_argument("--cache-dir", default="data/chembl_cache", help="ChEMBL data cache directory")
    parser.add_argument(
        "--chembl-sqlite",
        default=None,
        help="Optional path to local ChEMBL SQLite DB (e.g., /path/to/chembl_36.db)"
    )
    args = parser.parse_args()

    print(f"Configuration:")

    print(f"  Similarity Threshold: {args.similarity_threshold}")
    print(f"  Min Compounds per Assay: {args.min_compounds}")

    def resolve_sqlite_path(cli_path):
        if cli_path:
            return Path(cli_path)
        env_path = os.environ.get("CHEMBL_SQLITE_PATH")
        if env_path:
            return Path(env_path)
        return None

    sqlite_path = resolve_sqlite_path(args.chembl_sqlite)
    use_sqlite = sqlite_path is not None
    if use_sqlite:
        if not sqlite_path.exists():
            raise FileNotFoundError(f"ChEMBL SQLite DB not found: {sqlite_path}")
        print(f"Using local ChEMBL SQLite: {sqlite_path}")
    else:
        # Check API status before starting
        if not check_api_status():
            print("\n[!] ERROR: ChEMBL API is currently down or unreachable.")
            print("Please check: https://www.ebi.ac.uk/chembl/api/data/status")
            print("Aborting to avoid incomplete data.")
            sys.exit(1)

    # Initialize fingerprint generator (Morgan/Circular fingerprints)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # Load drug-like PDB data
    df = pd.read_csv(args.input)
    print(f"\nLoaded {len(df)} drug-like ligand entries from {args.input}")

    # Get unique targets
    unique_targets = df['target_chembl_id'].unique()
    print(f"Unique targets: {len(unique_targets)}")

    # Subset if specified
    if args.end:
        unique_targets = unique_targets[args.start:args.end]
        print(f"Processing subset: indices {args.start} to {args.end} ({len(unique_targets)} targets)")

    results = []

    assays_by_target = {}
    if use_sqlite:
        assays_by_target = fetch_assays_for_targets_sqlite(
            target_ids=unique_targets,
            sqlite_path=sqlite_path,
            min_compounds=args.min_compounds,
            cache_dir=args.cache_dir,
        )

    for target_id in tqdm(unique_targets, desc="Processing targets"):
        # Get all PDBs and ligands for this target
        target_data = df[df['target_chembl_id'] == target_id]
        
        # Pre-compute crystal ligand fingerprints
        crystal_fps = {}
        for _, row in target_data.iterrows():
            pdb_id = row['pdb_id']
            smiles = row['ligand_smiles']
            key = (pdb_id, row['ligand_resname'])
            
            if key not in crystal_fps:
                fp = get_fingerprint(smiles, fpgen)
                if fp:
                    crystal_fps[key] = {
                        'fp': fp,
                        'smiles': smiles,
                        'resname': row['ligand_resname'],
                        'pdb_id': pdb_id,
                        'resolution': row.get('resolution'),
                        'target_name': row['target_name'],
                        'uniprot_id': row['uniprot_id']
                    }

        if not crystal_fps:
            continue

        # Fetch assays for this target
        if use_sqlite:
            assays = assays_by_target.get(target_id, {})
        else:
            assays = fetch_assays_for_target(
                target_id,
                min_compounds=args.min_compounds,
                cache_dir=args.cache_dir
            )

        if not assays:
            if not use_sqlite:
                time.sleep(args.delay)
            continue

        # Check each assay against each crystal ligand
        for (doc_id, assay_id), assay_data in assays.items():
            assay_smiles = assay_data['smiles_list']
            doc_type = assay_data['type']
            n_compounds = len(assay_smiles)
            
            # Get activity stats
            pchembl_by_smiles = assay_data.get('pchembl_by_smiles', {})
            per_smiles_medians = [median(vals) for vals in pchembl_by_smiles.values() if vals]
            median_pchembl = median(per_smiles_medians) if per_smiles_medians else None

            for key, crystal_data in crystal_fps.items():
                matches = find_matching_compounds(
                    crystal_data['fp'],
                    assay_smiles,
                    fpgen,
                    similarity_threshold=args.similarity_threshold
                )

                if matches:
                    best_match_smiles, best_similarity = matches[0]
                    
                    # Check if crystal ligand itself is in the assay (similarity >= 0.99)
                    crystal_in_doc = best_similarity >= 0.99

                    results.append({
                        'target_chembl_id': target_id,
                        'target_name': crystal_data['target_name'],
                        'uniprot_id': crystal_data['uniprot_id'],
                        'pdb_id': crystal_data['pdb_id'],
                        'resolution': crystal_data['resolution'],
                        'ligand_resname': crystal_data['resname'],
                        'crystal_smiles': crystal_data['smiles'],
                        'document_chembl_id': doc_id,
                        'assay_chembl_id': assay_id,
                        'document_type': doc_type,
                        'n_compounds_in_assay': n_compounds,
                        'n_matches': len(matches),
                        'best_match_smiles': best_match_smiles,
                        'best_similarity': best_similarity,
                        'crystal_in_assay': crystal_in_doc,
                        'median_pchembl': median_pchembl
                    })

        if not use_sqlite:
            time.sleep(args.delay)

    # Save results
    df_out = pd.DataFrame(results)
    
    # Sort by resolution (best first) and similarity
    if len(df_out) > 0:
        df_out = df_out.sort_values(['resolution', 'best_similarity'], ascending=[True, False])
    
    # Handle partial runs
    if args.start > 0 or args.end:
        output_path = get_chunk_output_path(args.output, args.start, args.end)
    else:
        output_path = Path(args.output)

    df_out.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"Total benchmark entries: {len(df_out)}")
    print(f"Unique targets with matches: {df_out['target_chembl_id'].nunique()}")
    print(f"Unique PDBs: {df_out['pdb_id'].nunique()}")
    print(f"Unique documents: {df_out['document_chembl_id'].nunique()}")
    print(f"Unique assays: {df_out['assay_chembl_id'].nunique()}")
    if 'document_type' in df_out.columns:
        print(f"\nBy document type:")
        print(df_out['document_type'].value_counts())
    print(f"\nCrystal ligand in assay: {df_out['crystal_in_assay'].sum()}")


if __name__ == "__main__":
    main()
