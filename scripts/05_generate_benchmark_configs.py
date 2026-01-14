#!/usr/bin/env python3
"""
Step 5: Generate YAML workflow configs from the benchmark dataset.

Input: data/chembl_docking_benchmark.csv (must have mcs_smiles column from step 4b)
Output: generated_configs/*.yaml
"""
import pandas as pd
import yaml
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors


def is_valid_mcs(smiles: str) -> bool:
    """
    Check if the MCS is high quality:
    1. Valid SMILES
    2. No generic '~' bonds
    3. Molecular weight > 90.0
    """
    if not smiles or not isinstance(smiles, str):
        return False
    
    if '~' in smiles:
        return False
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
            
        mw = Descriptors.MolWt(mol)
        if mw <= 90.0:
            return False
            
        return True
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate docking workflow configs")
    parser.add_argument("--input", default="data/chembl_docking_benchmark.csv", help="Input CSV")
    parser.add_argument("--output-dir", default="generated_configs", help="Output directory")
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Generate configs for top N entries (set to 0 for no limit / all)",
    )
    parser.add_argument("--require-crystal-in-doc", action="store_true", default=True,
                        help="Only include entries where crystal ligand is in the document (default: True)")
    parser.add_argument("--no-require-crystal-in-doc", dest="require-crystal-in-doc", action="store_false",
                        help="Don't require crystal ligand in doc")
    parser.add_argument("--max-resolution", type=float, default=3.0, help="Maximum resolution (Å)")
    parser.add_argument("--min-compounds", type=int, default=30, help="Minimum compounds in document")
    args = parser.parse_args()

    # Load benchmark data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} benchmark entries from {args.input}")

    # Apply filters
    if args.require_crystal_in_doc:
        df = df[df['crystal_in_document'] == True]
        print(f"After crystal-in-doc filter: {len(df)}")

    if args.max_resolution:
        df = df[df['resolution'] <= args.max_resolution]
        print(f"After resolution filter (<= {args.max_resolution}Å): {len(df)}")

    if args.min_compounds:
        df = df[df['n_compounds_in_doc'] >= args.min_compounds]
        print(f"After min compounds filter (>= {args.min_compounds}): {len(df)}")

    # Load MCS results from separate file
    mcs_results_path = Path(args.input).parent / "mcs_results.csv"
    if mcs_results_path.exists():
        mcs_df = pd.read_csv(mcs_results_path)
        # Join MCS results with benchmark data
        df = df.merge(
            mcs_df[['document_chembl_id', 'mcs_smiles']],
            on='document_chembl_id',
            how='left'
        )
        # Filter to entries with MCS computed
        df = df[df['mcs_smiles'].notna()]
        print(f"After MCS filter (must have mcs_smiles): {len(df)}")
    else:
        print("WARNING: mcs_results.csv not found. Run step 4b first!")
        return

    # Filter by MCS quality BEFORE selecting top N
    df['mcs_is_valid'] = df['mcs_smiles'].apply(is_valid_mcs)
    df = df[df['mcs_is_valid'] == True]
    print(f"After MCS quality filter: {len(df)}")

    # Sort by resolution and select top N
    df = df.sort_values('resolution')
    if args.top_n and args.top_n > 0:
        df = df.head(args.top_n)
        print(f"Selected top {len(df)} entries by resolution")
    else:
        print(f"Selected all {len(df)} entries by resolution (no top-N limit)")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate configs
    generated = 0
    for _, row in df.iterrows():
        target_id = row['chembl_target_id']
        doc_id = row['document_chembl_id']
        pdb_id = row['pdb_id']
        
        # Use MCS from document compounds as fragment constraint
        mcs_smiles = row['mcs_smiles']
        
        # Get activity units from CSV, default to nM if not present
        activity_units = row.get('activity_units', 'nM')
        if pd.isna(activity_units) or activity_units == '':
            activity_units = 'nM'

        config = {
            'pdb_id': pdb_id,
            'output_dir': f"{pdb_id}_workflow",
            'target_id': target_id,
            'doc_id': doc_id,
            'ligand_csv_path': None,
            'activity_units': activity_units,
            'activity_column': 'standard_value',
            'id_column': 'molecule_chembl_id',
            'chain': 'A',
            'ligand_resname': row['ligand_resname'],
            'protein_pdb_path': None,
            'ligand_pdb_path': None,
            'fragment_smiles': mcs_smiles,
            'rmsd_threshold': 2.0,
        }

        # Write YAML
        yaml_path = output_dir / f"{target_id}_{pdb_id}_{doc_id}.yaml"
        with open(yaml_path, 'w') as f:
            f.write(f"# ChEMBL Target: {row['target_name']}\n")
            f.write(f"# UniProt: {row['uniprot_id']}\n")
            f.write(f"# PDB: {pdb_id} (Resolution: {row['resolution']:.2f} Å)\n")
            f.write(f"# Document: {doc_id} ({row['document_type']}, {row['n_compounds_in_doc']} compounds)\n")
            f.write(f"# Crystal Ligand: {row['ligand_resname']} ({row['crystal_smiles'][:60]}...)\n")
            f.write(f"# Best Match Similarity: {row['best_similarity']:.3f}\n")
            f.write(f"# MCS Fragment: {mcs_smiles}\n\n")
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        generated += 1

    print(f"\nGenerated {generated} YAML configs in {output_dir}/")

    # Also save a config index table (this is NOT a benchmark run summary)
    summary_path = output_dir / "config_index.csv"
    summary_cols = ['chembl_target_id', 'target_name', 'pdb_id', 'resolution', 'ligand_resname',
                    'document_chembl_id', 'document_type', 'n_compounds_in_doc', 
                    'best_similarity', 'crystal_in_document']
    if 'mcs_smiles' in df.columns:
        summary_cols.append('mcs_smiles')
    if 'activity_units' in df.columns:
        summary_cols.append('activity_units')
    df[summary_cols].to_csv(summary_path, index=False)
    print(f"Config index saved to {summary_path}")


if __name__ == "__main__":
    main()
