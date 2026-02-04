#!/usr/bin/env python3
"""
Step 2: Map UniProt IDs to PDB structures and extract ligand information.

This is the slow step - queries RCSB per UniProt to maintain mapping.
Designed to be run via SLURM.

Input: data/chembl_targets.csv
Output: data/chembl_pdb_map.csv
"""
import pandas as pd
import requests
import time
import asyncio
import aiohttp
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

# Add script directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import get_chunk_output_path

# Common crystallization artifacts to skip
SKIP_LIGANDS = {
    'HOH', 'GOL', 'EDO', 'PEG', 'DMS', 'SO4', 'PO4', 'CL', 'NA', 'MG', 'ZN', 
    'CA', 'K', 'ACT', 'BME', 'TRS', 'MPD', 'IOD', 'MES', 'EPE', 'FMT', 'IMD',
    'NO3', 'SCN', 'CO3', 'NH4', 'BR', 'FE', 'NI', 'CU', 'MN', 'CD', 'HG',
    'PG4', 'PE4', '1PE', 'P6G', 'PGE', 'PEO', 'PDO', 'BU3', 'TAR', 'CIT',
    'MLI', 'TLA', 'EOH', 'IPA', 'MOH', 'DOD', 'UNX', 'UNK', 'UNL'
}

def get_pdbs_for_uniprot(uniprot_id: str) -> list:
    """Query RCSB for PDBs containing a specific UniProt accession."""
    search_query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id
            }
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True}
    }
    try:
        resp = requests.post(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            json=search_query,
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            return [x['identifier'] for x in data.get('result_set', [])]
        elif resp.status_code == 204:
            return []  # No results
    except Exception as e:
        print(f"  [!] Error querying {uniprot_id}: {e}")
    return []


def _get_graphql_query(pdb_id: str) -> str:
    return f'''
    query {{
        entry(entry_id: "{pdb_id}") {{
            rcsb_entry_info {{
                resolution_combined
            }}
            nonpolymer_entities {{
                nonpolymer_comp {{
                    chem_comp {{
                        id
                        name
                    }}
                    rcsb_chem_comp_descriptor {{
                        SMILES
                        SMILES_stereo
                    }}
                }}
            }}
        }}
    }}
    '''


def _parse_entry_data(entry_data: dict) -> tuple:
    if not entry_data:
        return [], None

    # Get resolution
    resolution = None
    res_info = entry_data.get('rcsb_entry_info', {})
    if res_info and res_info.get('resolution_combined'):
        resolution = res_info['resolution_combined'][0]

    ligands = []
    if entry_data.get('nonpolymer_entities'):
        for entity in entry_data['nonpolymer_entities']:
            try:
                comp = entity.get('nonpolymer_comp', {})
                chem = comp.get('chem_comp', {})
                desc = comp.get('rcsb_chem_comp_descriptor', {})

                resname = chem.get('id', '')
                smiles = desc.get('SMILES_stereo') or desc.get('SMILES') or ''

                if resname and resname not in SKIP_LIGANDS and smiles and len(smiles) > 5:
                    ligands.append({
                        'resname': resname,
                        'smiles': smiles,
                        'name': chem.get('name', '')
                    })
            except Exception:
                continue

    return ligands, resolution


def get_ligand_info_for_pdb(pdb_id: str) -> list:
    """Get ligand residue names and SMILES for a PDB using GraphQL."""
    query = _get_graphql_query(pdb_id)
    try:
        r = requests.post(
            "https://data.rcsb.org/graphql",
            json={'query': query},
            timeout=30
        )
        if r.status_code != 200:
            return [], None

        entry_data = r.json().get('data', {}).get('entry', {})
        return _parse_entry_data(entry_data)
    except Exception as e:
        return [], None


async def fetch_ligand_info_async(session, pdb_id: str):
    """Async version of get_ligand_info_for_pdb."""
    query = _get_graphql_query(pdb_id)
    try:
        async with session.post("https://data.rcsb.org/graphql", json={'query': query}) as response:
            if response.status != 200:
                return [], None

            data = await response.json()
            entry_data = data.get('data', {}).get('entry', {})
            return _parse_entry_data(entry_data)
    except Exception as e:
        return [], None

async def _get_ligand_info_batch(pdb_ids: list):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_ligand_info_async(session, pdb_id) for pdb_id in pdb_ids]
        results = await asyncio.gather(*tasks)
        return dict(zip(pdb_ids, results))


def get_ligand_info_for_pdbs(pdb_ids: list) -> dict:
    """Batch fetch ligand info for multiple PDBs."""
    if not pdb_ids:
        return {}
    return asyncio.run(_get_ligand_info_batch(pdb_ids))


def main():
    parser = argparse.ArgumentParser(description="Map ChEMBL targets to PDB structures")
    parser.add_argument("--input", default="data/chembl_targets.csv", help="Input targets CSV")
    parser.add_argument("--output", default="data/chembl_pdb_map.csv", help="Output CSV path")
    parser.add_argument("--start", type=int, default=0, help="Start index (for parallelization)")
    parser.add_argument("--end", type=int, default=None, help="End index (for parallelization)")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    # Load targets
    df_targets = pd.read_csv(args.input)
    print(f"Loaded {len(df_targets)} targets from {args.input}")

    # Subset if specified
    if args.end:
        df_targets = df_targets.iloc[args.start:args.end]
        print(f"Processing subset: indices {args.start} to {args.end}")

    results = []
    for _, row in tqdm(df_targets.iterrows(), total=len(df_targets), desc="Mapping"):
        chembl_id = row['chembl_target_id']
        uniprot = row['uniprot_id']
        target_name = row['target_name']

        # Get PDBs for this UniProt
        pdb_ids = get_pdbs_for_uniprot(uniprot)

        if not pdb_ids:
            continue

        # Get ligand info for each PDB
        batch_results = get_ligand_info_for_pdbs(pdb_ids)

        for pdb_id in pdb_ids:
            ligands, resolution = batch_results.get(pdb_id, ([], None))

            if ligands:
                for lig in ligands:
                    results.append({
                        'chembl_target_id': chembl_id,
                        'target_name': target_name,
                        'uniprot_id': uniprot,
                        'pdb_id': pdb_id,
                        'resolution': resolution,
                        'ligand_resname': lig['resname'],
                        'ligand_name': lig['name'],
                        'ligand_smiles': lig['smiles'],
                        'is_holo': True
                    })
            else:
                # Record apo structure
                results.append({
                    'chembl_target_id': chembl_id,
                    'target_name': target_name,
                    'uniprot_id': uniprot,
                    'pdb_id': pdb_id,
                    'resolution': resolution,
                    'ligand_resname': None,
                    'ligand_name': None,
                    'ligand_smiles': None,
                    'is_holo': False
                })

        time.sleep(args.delay)

    # Save
    df_out = pd.DataFrame(results)
    
    # If this is a partial run, save to intermediate folder
    if args.start > 0 or args.end:
        output_path = get_chunk_output_path(args.output, args.start, args.end)
    else:
        output_path = Path(args.output)
    
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved {len(df_out)} rows to {output_path}")
    print(f"Unique targets with structures: {df_out['chembl_target_id'].nunique()}")
    print(f"Unique PDBs: {df_out['pdb_id'].nunique()}")
    print(f"Holo structures: {df_out['is_holo'].sum()}")


if __name__ == "__main__":
    main()
