#!/usr/bin/env python3
"""
Step 3: Filter PDB ligands to drug-like molecules only.

Uses RDKit descriptors for drug-likeness filtering.

Input: data/chembl_pdb_map.csv (or merged chunks)
Output: data/chembl_pdb_druglike.csv
"""

import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


HETEROATOM_PATTERN = Chem.MolFromSmarts("[N,O]")


def is_druglike(smiles: str) -> bool:
    """
    Check if a molecule is drug-like using basic filters:
    - MW: 200-800
    - Has at least one ring
    - Contains N or O (heteroatoms)
    - LogP: -2 to 7
    - Rotatable bonds: <= 15
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        mw = Descriptors.MolWt(mol)
        if mw < 200 or mw > 800:
            return False

        n_rings = rdMolDescriptors.CalcNumRings(mol)
        if n_rings < 1:
            return False

        if not mol.HasSubstructMatch(HETEROATOM_PATTERN):
            return False

        logp = Descriptors.MolLogP(mol)
        if logp < -2 or logp > 7:
            return False

        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if rot_bonds > 15:
            return False

        return True
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description="Filter to drug-like ligands")
    parser.add_argument("--input", default="data/chembl_pdb_map.csv", help="Input CSV")
    parser.add_argument(
        "--output", default="data/chembl_pdb_druglike.csv", help="Output CSV"
    )
    parser.add_argument(
        "--merge-chunks", nargs="*", help="Merge multiple chunk files first"
    )
    args = parser.parse_args()

    # Merge chunks if specified
    if args.merge_chunks:
        print(f"Merging {len(args.merge_chunks)} chunk files...")
        dfs = [pd.read_csv(f) for f in args.merge_chunks]
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["pdb_id", "ligand_resname"])
        merged_path = args.input
        df.to_csv(merged_path, index=False)
        print(f"Merged {len(df)} rows to {merged_path}")
    else:
        # If the merged file doesn't exist (common when step2 ran as chunks),
        # auto-merge from the standard intermediate chunk directory.
        input_path = Path(args.input)
        if not input_path.exists():
            intermediate_dir = input_path.parent / "intermediate" / input_path.stem
            chunk_files = sorted(intermediate_dir.glob("chunk_*.csv"))
            if chunk_files:
                print(
                    f"Input {args.input} not found; merging {len(chunk_files)} chunk files from {intermediate_dir}..."
                )
                dfs = [pd.read_csv(f) for f in chunk_files]
                df = pd.concat(dfs, ignore_index=True)
                df = df.drop_duplicates(subset=["pdb_id", "ligand_resname"])
                input_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(input_path, index=False)
                print(f"Merged {len(df)} rows to {input_path}")
            else:
                print(f"ERROR: Input file not found: {args.input}")
                print(f"Also found 0 chunk files under: {intermediate_dir}")
                print(
                    "If you ran step2 as a SLURM array, make sure it produced chunk_*.csv files there."
                )
                sys.exit(1)

        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows from {args.input}")

    # Filter to holo only
    df_holo = df[df["is_holo"] == True].copy()
    print(f"Holo structures: {len(df_holo)}")

    # Apply drug-like filter
    print("Filtering for drug-like molecules...")
    tqdm.pandas(desc="Checking drug-likeness")
    df_holo["is_druglike"] = df_holo["ligand_smiles"].progress_apply(is_druglike)

    df_druglike = df_holo[df_holo["is_druglike"]].drop(columns=["is_druglike"])

    # Deduplicate
    df_druglike = df_druglike.drop_duplicates(
        subset=["target_chembl_id", "pdb_id", "ligand_resname"]
    )

    # Save
    df_druglike.to_csv(args.output, index=False)
    print(f"\nSaved {len(df_druglike)} drug-like ligand entries to {args.output}")
    print(f"Unique targets: {df_druglike['target_chembl_id'].nunique()}")
    print(f"Unique PDBs: {df_druglike['pdb_id'].nunique()}")


if __name__ == "__main__":
    main()
