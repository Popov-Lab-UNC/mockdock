#!/usr/bin/env python3
"""
Step 4b: Compute Maximum Common Substructure (MCS) for each document.

For each unique document in the benchmark dataset, fetches all compounds
and computes the MCS across them. This MCS will be used as the fragment
constraint in docking workflows.

Input: data/chembl_docking_benchmark.csv
Output: data/mcs_results.csv (separate document with document_chembl_id and mcs_smiles)
"""
import pandas as pd
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
from datetime import datetime
import multiprocessing
import sys
from pathlib import Path

# Add script directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import get_chunk_output_path, run_with_timeout

from rdkit import Chem
from rdkit.Chem import rdFMCS

from chembl_cache import read_assay_cache
# Add parent directory to path to import fcgmb modules
script_dir = Path(__file__).parent
benchmark_dir = script_dir.parent
sys.path.insert(0, str(benchmark_dir))

# Try both import styles
try:
    from fcgmb.data import standardize_smiles
except ImportError:
    # Fallback: add the parent directory if not already there
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))
    from fcgmb.data import standardize_smiles


# Removed fetch functions since we now use the unified cache from step 4


def compute_mcs(smiles_list: list, reference_smiles: str = None, max_compounds: int = 100, timeout: int = 10):
    """
    Compute Maximum Common Substructure across a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        reference_smiles: SMILES string to use as template for SMILES extraction
        max_compounds: Maximum number of compounds to use (for performance)
        timeout: Timeout in seconds for MCS computation
    
    Returns: Clean SMILES string of the MCS, or None if computation fails
    """
    if not smiles_list:
        return None
    
    # Limit to first max_compounds for performance
    if len(smiles_list) > max_compounds:
        smiles_list = smiles_list[:max_compounds]
    
    # Convert SMILES to molecules
    mols = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            mols.append(mol)
        except:
            continue
    
    if len(mols) < 2:
        return None
    
    try:
        # Compute MCS with strict parameters
        mcs = rdFMCS.FindMCS(
            mols,
            threshold=1.0,
            timeout=timeout,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            matchValences=True,
            ringMatchesRingOnly=True,
            completeRingsOnly=True
        )
        
        if mcs.numAtoms == 0:
            return None
        
        # 2. Convert result SMARTS to a Query Molecule
        mcs_query = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_query is None:
            return None

        # 3. Match against reference (crystal ligand) to extract clean SMILES
        # Fall back to first molecule in list if no reference provided
        ref_mol = None
        if reference_smiles:
            ref_mol = Chem.MolFromSmiles(reference_smiles)
        
        if ref_mol is None:
            ref_mol = mols[0]

        match_atoms = ref_mol.GetSubstructMatch(mcs_query)
        if not match_atoms:
            # Fallback: just return SMARTS if match fails
            return Chem.MolToSmiles(mcs.queryMol) if mcs.queryMol else None

        # 4. Extract specific atoms from the real molecule to generate SMILES
        # We Kekulize a copy of the reference molecule first to ensure 
        # the resulting fragment SMILES uses standard bonds (C, N) 
        # instead of aromatic markers (c, n) which can be invalid in fragments.
        ref_copy = Chem.Mol(ref_mol)
        Chem.Kekulize(ref_copy, clearAromaticFlags=True)
        
        fragment_smiles = Chem.MolFragmentToSmiles(
            ref_copy, 
            atomsToUse=match_atoms, 
            canonical=True, 
            isomericSmiles=False,
            kekuleSmiles=True
        )
        
        # Final validation: ensure the SMILES is parseable
        if fragment_smiles and Chem.MolFromSmiles(fragment_smiles):
            return fragment_smiles
        
        # Fallback to original method if Kekulization fails for some reason
        return Chem.MolFragmentToSmiles(
            ref_mol, 
            atomsToUse=match_atoms, 
            canonical=True, 
            isomericSmiles=False
        )
        
    except Exception as e:
        print(f"  [!] Error computing MCS: {e}")
        return None


def merge_mcs_results(mcs_results_csv: str, mapping_pattern: str, output_csv: str):
    """
    Merge MCS mapping files into the MCS results CSV.
    
    Args:
        mcs_results_csv: Path to existing MCS results CSV (or will create new)
        mapping_pattern: Glob pattern for mapping files (e.g., "data/mcs_mapping_*.csv")
        output_csv: Output path for merged MCS results CSV
    """
    import glob
    from pathlib import Path
    
    dfs = []

    # Load existing MCS results if they exist
    if Path(mcs_results_csv).exists():
        try:
            existing_df = pd.read_csv(mcs_results_csv)
            if 'document_chembl_id' in existing_df.columns:
                dfs.append(existing_df)
                print(f"Loaded {len(existing_df)} existing MCS results")
        except Exception as e:
            print(f"  [!] Could not load existing MCS results: {e}")
    
    # Load all mapping files
    mapping_files = sorted(glob.glob(mapping_pattern))
    print(f"Found {len(mapping_files)} mapping files to merge")
    
    current_time = datetime.now().isoformat()

    for mapping_file in mapping_files:
        try:
            mapping_df = pd.read_csv(mapping_file)
            if 'document_chembl_id' not in mapping_df.columns or 'assay_chembl_id' not in mapping_df.columns or 'mcs_smiles' not in mapping_df.columns:
                print(f"  [!] Skipping {mapping_file}: missing required columns")
                continue
            
            # Prepare the DataFrame to match structure
            mapping_df['n_compounds'] = None
            mapping_df['status'] = 'merged'
            mapping_df['timestamp'] = current_time

            # Select only necessary columns
            cols_to_keep = ['document_chembl_id', 'assay_chembl_id', 'mcs_smiles', 'n_compounds', 'status', 'timestamp']
            mapping_df = mapping_df[cols_to_keep]

            dfs.append(mapping_df)
            
            print(f"  Merged {mapping_file}")
        except Exception as e:
            print(f"  [!] Error merging {mapping_file}: {e}")
    
    # Save merged results
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        # Deduplicate
        merged_df = merged_df.drop_duplicates(subset=['document_chembl_id', 'assay_chembl_id'], keep='last')
        merged_df = merged_df.sort_values(['document_chembl_id', 'assay_chembl_id'])
        merged_df.to_csv(output_csv, index=False)
        print(f"\nMerged results saved to {output_csv}")
        print(f"Total assays with MCS: {merged_df['mcs_smiles'].notna().sum()}")
        print(f"Total assays: {len(merged_df)}")
    else:
        print("\nNo results to save")


def prefetch_compounds(docs_to_fetch, batch_size=20, max_workers=5):
    """
    Prefetch compounds for a list of documents in parallel batches.

    Args:
        docs_to_fetch: List of document IDs
        batch_size: Number of documents per batch
        max_workers: Number of parallel workers

    Returns:
        Dictionary mapping document ID to list of SMILES
    """
    import concurrent.futures

    doc_compounds_cache = {}

    if not docs_to_fetch:
        return doc_compounds_cache

    print("Pre-fetching compounds in batches...")
    batches = []
    for i in range(0, len(docs_to_fetch), batch_size):
        batches.append(docs_to_fetch[i:i + batch_size])

    print(f"  Fetching {len(batches)} batches in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(fetch_batch_with_timeout, batch, timeout=120): batch
            for batch in batches
        }

        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            completed_count += 1
            try:
                batch_results = future.result()

                # Update cache
                if batch_results is not None:
                    doc_compounds_cache.update(batch_results)

                    # Explicitly mark docs that returned no results as having empty list
                    # only if the batch fetch itself was successful (not None)
                    for doc_id in batch:
                        if doc_id not in doc_compounds_cache:
                            doc_compounds_cache[doc_id] = []

                print(f"  Fetched batch {completed_count}/{len(batches)} ({len(batch)} docs)")

            except Exception as exc:
                print(f"  [!] Batch fetch generated an exception: {exc}")

    return doc_compounds_cache
  
def mcs_worker(args):
    """
    Worker function for parallel MCS computation.

    Args:
        args: Tuple containing (doc_id, compounds, crystal_smiles, max_compounds, timeout)

    Returns:
        tuple: (doc_id, mcs_smiles, n_compounds)
    """
    doc_id, compounds, crystal_smiles, max_compounds, timeout = args
    mcs_smiles = compute_mcs(compounds, reference_smiles=crystal_smiles,
                            max_compounds=max_compounds, timeout=timeout)
    return doc_id, mcs_smiles, len(compounds)


def main():
    parser = argparse.ArgumentParser(description="Compute MCS for benchmark documents")
    parser.add_argument("--input", default="data/chembl_docking_benchmark.csv", help="Input CSV")
    parser.add_argument("--output", default="data/mcs_results.csv", help="Output CSV for MCS results")
    parser.add_argument("--max-compounds", type=int, default=100, 
                        help="Maximum compounds per document to use for MCS")
    parser.add_argument("--timeout", type=int, default=10, 
                        help="Timeout in seconds for MCS computation")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay (deprecated, no API calls)")
    parser.add_argument("--cache-dir", default="data/chembl_cache", help="ChEMBL data cache directory")
    parser.add_argument("--start", type=int, default=0, help="Start index (for parallelization)")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--merge", action="store_true", 
                        help="Merge mapping files instead of computing MCS")
    parser.add_argument(
        "--mapping-pattern",
        default="data/intermediate/mcs_results/chunk_*.csv",
                        help="Glob pattern for mapping files (used with --merge)")
    parser.add_argument("--n-cpus", type=int, default=None, help="Number of CPUs to use")
    args = parser.parse_args()
    
    # Handle merge mode
    if args.merge:
        merge_mcs_results(args.input, args.mapping_pattern, args.output)
        return

    print(f"Configuration:")
    print(f"  Max Compounds per Document: {args.max_compounds}")
    print(f"  MCS Timeout: {args.timeout}s")
    print(f"  API Delay: {args.delay}s")

    # Check if input file exists
    from pathlib import Path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\nERROR: Input file does not exist: {args.input}")
        print("Please run step 4 first and merge the results.")
        return
    
    if input_path.stat().st_size == 0:
        print(f"\nERROR: Input file is empty: {args.input}")
        print("Please run step 4 first and merge the results.")
        return

    # Load benchmark data
    try:
        df = pd.read_csv(args.input)
        if len(df) == 0:
            print(f"\nERROR: Input file has no data rows: {args.input}")
            return
    except pd.errors.EmptyDataError:
        print(f"\nERROR: Input file has no columns or is empty: {args.input}")
        print("Please run step 4 first and merge the results.")
        return
    except Exception as e:
        print(f"\nERROR: Failed to read input file: {e}")
        return
    
    print(f"\nLoaded {len(df)} benchmark entries from {args.input}")

    # Check for required columns
    if 'document_chembl_id' not in df.columns or 'assay_chembl_id' not in df.columns:
        print(f"\nERROR: Required columns 'document_chembl_id' or 'assay_chembl_id' not found in input file.")
        print(f"Available columns: {', '.join(df.columns)}")
        return

    # Get unique (doc, assay) pairs
    unique_pairs = df[['document_chembl_id', 'assay_chembl_id', 'target_chembl_id']].drop_duplicates()
    print(f"Unique (document, assay) pairs: {len(unique_pairs)}")

    # Subset if specified
    if args.end:
        unique_pairs = unique_pairs.iloc[args.start:args.end]
        print(f"Processing subset: indices {args.start} to {args.end} ({len(unique_pairs)} pairs)")

    # Load existing MCS results if they exist
    output_path = Path(args.output)
    existing_mcs = set()
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            if 'document_chembl_id' in existing_df.columns and 'assay_chembl_id' in existing_df.columns:
                for _, row in existing_df.iterrows():
                    if pd.notna(row['mcs_smiles']):
                        existing_mcs.add((row['document_chembl_id'], row['assay_chembl_id']))
                print(f"Loaded {len(existing_mcs)} existing MCS results from {output_path}")
        except Exception as e:
            print(f"  [!] Could not load existing MCS results: {e}")

    # Track results for new document
    mcs_results = []
    
    # Process each pair
    processed = 0
    skipped = 0
    for _, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Processing assays"):
        doc_id = row['document_chembl_id']
        assay_id = row['assay_chembl_id']
        target_id = row['target_chembl_id']

        # Check if already computed
        if (doc_id, assay_id) in existing_mcs:
            skipped += 1
            continue

        # Get crystal ligand SMILES for this entry
        doc_rows = df[(df['document_chembl_id'] == doc_id) & (df['assay_chembl_id'] == assay_id)]
        crystal_smiles = doc_rows['crystal_smiles'].iloc[0] if 'crystal_smiles' in doc_rows.columns else None
        
        # Read compounds from cache
        df_assay = read_assay_cache(args.cache_dir, target_id, doc_id, assay_id)
        
        if df_assay is None or len(df_assay) == 0:
            print(f"  [!] No cached data found or empty for {doc_id} / {assay_id}")
            mcs_results.append({
                'document_chembl_id': doc_id,
                'assay_chembl_id': assay_id,
                'mcs_smiles': None,
                'n_compounds': 0,
                'status': 'no_compounds',
                'timestamp': datetime.now().isoformat()
            })
            continue
        
        compounds = df_assay['canonical_smiles'].to_list()
        
        # Standardize SMILES before MCS
        standardized = []
        for s in compounds:
            try:
                s_std = standardize_smiles(s)
                if s_std:
                    standardized.append(s_std)
            except:
                continue
        
        if not standardized:
            print(f"  [!] No valid standardized SMILES for {doc_id} / {assay_id}")
            mcs_results.append({
                'document_chembl_id': doc_id,
                'assay_chembl_id': assay_id,
                'mcs_smiles': None,
                'n_compounds': 0,
                'status': 'standardization_failed',
                'timestamp': datetime.now().isoformat()
            })
            continue

        # Compute MCS
        mcs_smiles = compute_mcs(standardized, reference_smiles=crystal_smiles, 
                                max_compounds=args.max_compounds, timeout=args.timeout)
        
        if mcs_smiles:
            mcs_results.append({
                'document_chembl_id': doc_id,
                'assay_chembl_id': assay_id,
                'mcs_smiles': mcs_smiles,
                'n_compounds': len(standardized),
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            processed += 1
        else:
            print(f"  [!] Could not compute MCS for assay {assay_id} in {doc_id}")
            mcs_results.append({
                'document_chembl_id': doc_id,
                'assay_chembl_id': assay_id,
                'mcs_smiles': None,
                'n_compounds': len(standardized),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            })

    print(f"\nComputed MCS for {processed} new documents")
    if skipped > 0:
        print(f"Skipped {skipped} documents (already computed)")

    # Combine existing and new results
    if mcs_results:
        new_df = pd.DataFrame(mcs_results)
        
        # Merge with existing results
        if existing_mcs:
            # Reconstruct existing_df format for merging
            existing_rows = []
            for d_id, a_id in existing_mcs:
                existing_rows.append({
                    'document_chembl_id': d_id,
                    'assay_chembl_id': a_id,
                    'mcs_smiles': 'EXISTING_PLACEHOLDER', # We just need the key
                    'n_compounds': None,
                    'status': 'existing',
                    'timestamp': None
                })
            existing_df = pd.DataFrame(existing_rows)
            # Combine and deduplicate (new results take precedence)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['document_chembl_id', 'assay_chembl_id'], keep='last')
        else:
            combined_df = new_df
        
        # Handle partial runs - write mapping file for merging
        if args.start > 0 or args.end:
            # For parallel runs, write a mapping file
            mapping_path = get_chunk_output_path(args.output, args.start, args.end)
            mapping_df = new_df[['document_chembl_id', 'assay_chembl_id', 'mcs_smiles']].copy()
            mapping_df.to_csv(mapping_path, index=False)
            print(f"\nMCS mapping saved to {mapping_path}")
            print(f"Run merge to combine all partial results into {args.output}")
        else:
            # For sequential runs, write full results CSV
            combined_df = combined_df.sort_values(['document_chembl_id', 'assay_chembl_id'])
            combined_df.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
            print(f"Total assays with MCS: {combined_df['mcs_smiles'].notna().sum()}")
            print(f"Total assays processed: {len(combined_df)}")
    else:
        print("\nNo new MCS results to save")


if __name__ == "__main__":
    main()
