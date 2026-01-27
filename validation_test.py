import os
import sys
import polars as pl
from fcgmb.oracle import FCGMBOracle

def run_validation():
    # The 3 best benchmarks
    benchmarks = [
        #"CHEMBL4482_5XVA_CHEMBL4028914",
        "CHEMBL4630_2R0U_CHEMBL1140535",
        "CHEMBL3313835_7KCF_CHEMBL4834423"
    ]
    
    # Path to AutoDock-GPU
    adgpu_exe = os.environ.get("ADGPU_EXE", "adgpu")
    
    summary = []
    
    for bm_name in benchmarks:
        print(f"\n{'='*60}")
        print(f"VALIDATION TEST: {bm_name}")
        print(f"{'='*60}")
        
        try:
            # Initialize Oracle with a large budget for validation
            oracle = FCGMBOracle(bm_name, budget=1000, adgpu_executable=adgpu_exe)
            
            # Get validation compounds (upper 3 quartiles)
            val_df = oracle.get_validation_compounds()
            
            if val_df.is_empty():
                print(f"No validation compounds found for {bm_name}")
                continue
            
            smiles_to_score = val_df.get_column("canonical_smiles").to_list()
            print(f"Found {len(smiles_to_score)} validation compounds to score.")
            
            # Score them
            results = oracle.score(smiles_to_score)
            
            # Aggregate results aligned to input order (keep per-compound)
            scores = [results.get(smi, float("nan")) for smi in smiles_to_score]
            n_scored = len(scores)
            n_success = int(oracle.results_df.get_column("valid_pose_found").sum())
            valid_scores = [s for s in scores if s == s]
            min_score = min(valid_scores) if valid_scores else 0
            max_score = max(valid_scores) if valid_scores else 0
            
            print(f"\nResults for {bm_name}:")
            print(f"  Scored: {n_scored}")
            print(f"  Success (RMSD passed): {n_success}")
            print(f"  Normalized Score Range: {min_score:.3f} - {max_score:.3f}")
            
            summary.append({
                "benchmark": bm_name,
                "n_scored": n_scored,
                "n_success": n_success,
                "min_score": min_score,
                "max_score": max_score
            })
            
            # Save detailed results
            out_file = f"validation_results_{bm_name}.csv"
            oracle.results_df.write_csv(out_file)
            print(f"Detailed results saved to {out_file}")
            
        except Exception as e:
            print(f"Error running validation for {bm_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "="*60)
    print("FINAL VALIDATION SUMMARY")
    print("="*60)
    for res in summary:
        print(f"{res['benchmark']}: Scored={res['n_scored']}, Passed={res['n_success']}, Range={res['min_score']:.3f}-{res['max_score']:.3f}")

if __name__ == "__main__":
    run_validation()
