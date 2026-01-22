#!/usr/bin/env python3
"""
Step 1: Fetch all human single-protein targets from ChEMBL with UniProt IDs.

Output: data/chembl_targets.csv
"""
import pandas as pd
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fetch ChEMBL human protein targets")
    parser.add_argument("--output", default="data/chembl_targets.csv", help="Output CSV path")
    args = parser.parse_args()

    # Import here to fail fast if ChEMBL is down
    from chembl_webresource_client.new_client import new_client

    print("Fetching Human Single Protein Targets from ChEMBL...")
    target_api = new_client.target
    targets = target_api.filter(
        organism='Homo sapiens',
        target_type='SINGLE PROTEIN'
    ).only('target_chembl_id', 'pref_name', 'target_components')

    target_list = []
    for t in tqdm(targets, desc="Processing Targets"):
        try:
            # Get UniProt ID (some targets might not have one)
            uniprot_id = t['target_components'][0]['accession']
            target_list.append({
                'chembl_target_id': t['target_chembl_id'],
                'target_name': t['pref_name'],
                'uniprot_id': uniprot_id
            })
        except (IndexError, KeyError, TypeError):
            continue

    df = pd.DataFrame(target_list)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} targets to {args.output}")

if __name__ == "__main__":
    main()
