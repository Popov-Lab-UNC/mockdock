#!/usr/bin/env python3
"""
Step 4: Find ChEMBL documents (patents and/or publications) that contain compounds
matching the crystal ligand using fingerprint similarity.

Uses RDKit's FingerprintGenerator for efficient similarity calculation.

Input: data/chembl_pdb_druglike.csv
Output: data/chembl_docking_benchmark.csv
"""
import pandas as pd
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import sys
from pathlib import Path

# Add script directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import get_chunk_output_path

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs


def get_fingerprint(smiles: str, fpgen):
    """Generate fingerprint for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return fpgen.GetFingerprint(mol)
    except:
        return None


def fetch_documents_for_target(target_chembl_id: str, include_patents: bool = True, 
                                include_publications: bool = True, min_compounds: int = 20):
    """
    Fetch ChEMBL documents for a target with their associated compounds.
    
    Returns: dict of {doc_id: {'type': str, 'smiles_list': [str], 'year': int}}
    """
    from chembl_webresource_client.new_client import new_client
    
    # Build source filter
    src_ids = []
    if include_patents:
        src_ids.append(1)  # Patents
    if include_publications:
        src_ids.extend([0, 2, 3, 4, 5, 6, 7, 8, 9])  # Various publication sources
    
    try:
        activities = new_client.activity.filter(
            target_chembl_id=target_chembl_id,
            assay_type='B',  # Binding assays
            pchembl_value__isnull=False
        ).only('document_chembl_id', 'canonical_smiles', 'src_id', 'pchembl_value')
        
        # Group by document
        doc_data = defaultdict(lambda: {'smiles_list': [], 'type': 'unknown', 'pchembl_values': []})
        
        for act in activities:
            doc_id = act.get('document_chembl_id')
            smiles = act.get('canonical_smiles')
            src_id = act.get('src_id')
            pchembl = act.get('pchembl_value')
            
            if not doc_id or not smiles:
                continue
            
            # Check source filter
            if src_id not in src_ids:
                continue
            
            doc_data[doc_id]['smiles_list'].append(smiles)
            doc_data[doc_id]['type'] = 'patent' if src_id == 1 else 'publication'
            if pchembl:
                doc_data[doc_id]['pchembl_values'].append(float(pchembl))
        
        # Filter by minimum compounds
        filtered = {
            doc_id: data for doc_id, data in doc_data.items() 
            if len(set(data['smiles_list'])) >= min_compounds
        }
        
        return filtered
        
    except Exception as e:
        print(f"  [!] Error fetching docs for {target_chembl_id}: {e}")
        return {}


def find_matching_compounds(crystal_fp, doc_smiles_list: list, fpgen, 
                            similarity_threshold: float = 0.7):
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
    parser.add_argument("--include-patents", action="store_true", default=True, help="Include patents")
    parser.add_argument("--no-patents", dest="include_patents", action="store_false", help="Exclude patents")
    parser.add_argument("--include-publications", action="store_true", default=True, help="Include publications")
    parser.add_argument("--no-publications", dest="include_publications", action="store_false", help="Exclude publications")
    parser.add_argument("--similarity-threshold", type=float, default=0.7, 
                        help="Tanimoto similarity threshold for matching")
    parser.add_argument("--min-compounds", type=int, default=20, 
                        help="Minimum compounds in document to consider")
    parser.add_argument("--start", type=int, default=0, help="Start index (for parallelization)")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between ChEMBL API calls")
    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Include Patents: {args.include_patents}")
    print(f"  Include Publications: {args.include_publications}")
    print(f"  Similarity Threshold: {args.similarity_threshold}")
    print(f"  Min Compounds per Document: {args.min_compounds}")

    # Initialize fingerprint generator (Morgan/Circular fingerprints)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # Load drug-like PDB data
    df = pd.read_csv(args.input)
    print(f"\nLoaded {len(df)} drug-like ligand entries from {args.input}")

    # Get unique targets
    unique_targets = df['chembl_target_id'].unique()
    print(f"Unique targets: {len(unique_targets)}")

    # Subset if specified
    if args.end:
        unique_targets = unique_targets[args.start:args.end]
        print(f"Processing subset: indices {args.start} to {args.end} ({len(unique_targets)} targets)")

    results = []

    for target_id in tqdm(unique_targets, desc="Processing targets"):
        # Get all PDBs and ligands for this target
        target_data = df[df['chembl_target_id'] == target_id]
        
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

        # Fetch documents for this target
        documents = fetch_documents_for_target(
            target_id,
            include_patents=args.include_patents,
            include_publications=args.include_publications,
            min_compounds=args.min_compounds
        )

        if not documents:
            time.sleep(args.delay)
            continue

        # Check each document against each crystal ligand
        for doc_id, doc_data in documents.items():
            doc_smiles = list(set(doc_data['smiles_list']))
            doc_type = doc_data['type']
            n_compounds = len(doc_smiles)
            
            # Get activity stats
            pchembl_values = doc_data.get('pchembl_values', [])
            avg_pchembl = sum(pchembl_values) / len(pchembl_values) if pchembl_values else None

            for key, crystal_data in crystal_fps.items():
                matches = find_matching_compounds(
                    crystal_data['fp'],
                    doc_smiles,
                    fpgen,
                    similarity_threshold=args.similarity_threshold
                )

                if matches:
                    best_match_smiles, best_similarity = matches[0]
                    
                    # Check if crystal ligand itself is in the document (similarity >= 0.99)
                    crystal_in_doc = best_similarity >= 0.99

                    results.append({
                        'chembl_target_id': target_id,
                        'target_name': crystal_data['target_name'],
                        'uniprot_id': crystal_data['uniprot_id'],
                        'pdb_id': crystal_data['pdb_id'],
                        'resolution': crystal_data['resolution'],
                        'ligand_resname': crystal_data['resname'],
                        'crystal_smiles': crystal_data['smiles'],
                        'document_chembl_id': doc_id,
                        'document_type': doc_type,
                        'n_compounds_in_doc': n_compounds,
                        'n_matches': len(matches),
                        'best_match_smiles': best_match_smiles,
                        'best_similarity': best_similarity,
                        'crystal_in_document': crystal_in_doc,
                        'avg_pchembl': avg_pchembl
                    })

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
    print(f"Unique targets with matches: {df_out['chembl_target_id'].nunique()}")
    print(f"Unique PDBs: {df_out['pdb_id'].nunique()}")
    print(f"Unique documents: {df_out['document_chembl_id'].nunique()}")
    if 'document_type' in df_out.columns:
        print(f"\nBy document type:")
        print(df_out['document_type'].value_counts())
    print(f"\nCrystal ligand in document: {df_out['crystal_in_document'].sum()}")


if __name__ == "__main__":
    main()
