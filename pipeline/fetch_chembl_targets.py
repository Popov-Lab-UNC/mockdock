#!/usr/bin/env python3
"""
Step 1: Fetch all human single-protein targets from ChEMBL with UniProt IDs.

Output: data/chembl_targets.csv
"""
import argparse
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Fetch ChEMBL human protein targets")
    parser.add_argument("--output", default="data/chembl_targets.csv", help="Output CSV path")
    parser.add_argument(
        "--chembl-sqlite",
        default=None,
        help="Optional path to local ChEMBL SQLite DB (e.g., /path/to/chembl_36.db)"
    )
    args = parser.parse_args()

    def resolve_sqlite_path(cli_path):
        if cli_path:
            return Path(cli_path)
        env_path = os.environ.get("CHEMBL_SQLITE_PATH")
        if env_path:
            return Path(env_path)
        return None

    sqlite_path = resolve_sqlite_path(args.chembl_sqlite)
    if sqlite_path and not sqlite_path.exists():
        raise FileNotFoundError(f"ChEMBL SQLite DB not found: {sqlite_path}")

    if sqlite_path:
        print(f"Fetching Human Single Protein Targets from local ChEMBL SQLite: {sqlite_path}")
        import sqlite3

        query = """
            SELECT
                td.chembl_id AS target_chembl_id,
                td.pref_name AS target_name,
                cs.accession AS uniprot_id
            FROM target_dictionary td
            JOIN target_components tc ON td.tid = tc.tid
            JOIN component_sequences cs ON tc.component_id = cs.component_id
            WHERE td.organism = 'Homo sapiens'
            AND td.target_type = 'SINGLE PROTEIN'
            AND cs.accession IS NOT NULL;
        """
        with sqlite3.connect(sqlite_path) as conn:
            df = pd.read_sql_query(query, conn)
        df = df.drop_duplicates(subset=["target_chembl_id"])
    else:
        # Import here to fail fast if ChEMBL is down
        from chembl_webresource_client.new_client import new_client

        print("Fetching Human Single Protein Targets from ChEMBL API...")
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
                    'target_chembl_id': t.get('target_chembl_id'),
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
