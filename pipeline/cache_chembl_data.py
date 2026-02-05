#!/usr/bin/env python3
"""
Fetch all ChEMBL data for all configs and cache it to avoid rate limiting.

This script:
1. Reads all YAML configs
2. Extracts unique (target_id, doc_id) pairs
3. Fetches data from ChEMBL with rate limiting (1 request/second)
4. Saves to a cache directory structure

Usage:
    python cache_chembl_data.py --config-dir configs --cache-dir data/chembl_cache
"""
import argparse
import time
import yaml
from pathlib import Path
from typing import Set, Tuple
import polars as pl
import sys
import os

# Add parent directory to path to import fcgmb modules
# Add parent directory to path to import fcgmb modules
script_dir = Path(__file__).parent
benchmark_dir = script_dir.parent
sys.path.insert(0, str(benchmark_dir))

# Try both import styles
try:
    from fcgmb.data import fetch_chembl_data
except ImportError:
    # Fallback: add the parent directory (which is benchmark_dir)
    sys.path.insert(0, str(benchmark_dir))
    from fcgmb.data import fetch_chembl_data


def get_unique_config_triplets(config_dir: Path) -> Set[Tuple[str, str, Optional[str]]]:
    """Extract unique (target_id, doc_id, assay_id) triplets from all YAML configs."""
    triplets = set()
    for config_file in config_dir.glob("*.yaml"):
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                target_id = config.get("target_id")
                doc_id = config.get("doc_id")
                assay_id = config.get("assay_id")
                if target_id and doc_id:
                    triplets.add((target_id, doc_id, assay_id))
        except Exception as e:
            print(f"Warning: Could not parse {config_file}: {e}")
            continue
    return triplets


def cache_all_chembl_data(
    config_dir: Path,
    cache_dir: Path,
    rate_limit_seconds: float = 1.0,
    retry_on_failure: bool = True,
    max_retries: int = 3
):
    """
    Fetch all ChEMBL data for all configs and cache it.
    
    Args:
        config_dir: Directory containing YAML config files
        cache_dir: Directory to store cached CSV files
        rate_limit_seconds: Seconds to wait between API requests (default 1.0 for 1 req/sec)
        retry_on_failure: Whether to retry failed requests
        max_retries: Maximum number of retries for failed requests
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all unique (target_id, doc_id, assay_id) triplets
    triplets = get_unique_config_triplets(config_dir)
    print(f"Found {len(triplets)} unique (target, doc, assay) combinations to fetch")
    
    # Track success/failure
    successful = []
    failed = []
    
    for idx, (target_id, doc_id, assay_id) in enumerate(sorted(triplets, key=lambda x: (x[0], x[1], x[2] or "")), 1):
        if assay_id:
            # New granular structure
            target_dir = cache_dir / target_id
            target_dir.mkdir(exist_ok=True, parents=True)
            cache_file = target_dir / f"{doc_id}_{assay_id}.csv"
        else:
            # Legacy flat structure
            cache_file = cache_dir / f"{target_id}_{doc_id}.csv"
        
        # Skip if already cached
        if cache_file.exists():
            print(f"[{idx}/{len(triplets)}] Skipping {target_id}_{doc_id}{'_' + assay_id if assay_id else ''} (already cached)")
            successful.append((target_id, doc_id, assay_id))
            continue
        
        # Fetch with rate limiting and retries
        print(f"[{idx}/{len(triplets)}] Fetching {target_id}_{doc_id}{'_' + assay_id if assay_id else ''}...")
        retries = 0
        success = False
        
        while retries <= max_retries:
            try:
                # Rate limiting: wait before each request (except the first)
                if idx > 1 or retries > 0:
                    time.sleep(rate_limit_seconds)
                
                df, stats = fetch_chembl_data(target_id, doc_id, assay_chembl_id=assay_id, return_stats=True)
                
                if df.is_empty():
                    print(f"   Warning: No data returned")
                    failed.append((target_id, doc_id, assay_id, "No data"))
                    break
                
                # Save to cache
                df.write_csv(cache_file)
                print(f"   Saved {len(df)} compounds to {cache_file}")
                successful.append((target_id, doc_id, assay_id))
                success = True
                break
                
            except Exception as e:
                retries += 1
                error_msg = str(e)
                print(f"   Error (attempt {retries}/{max_retries + 1}): {error_msg}")
                
                if retries > max_retries:
                    print(f"   Failed after {max_retries + 1} attempts")
                    failed.append((target_id, doc_id, error_msg))
                    if not retry_on_failure:
                        break
                else:
                    # Exponential backoff for retries
                    wait_time = rate_limit_seconds * (2 ** retries)
                    print(f"   Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed combinations:")
        for target_id, doc_id, assay_id, error in failed:
            print(f"  {target_id}_{doc_id}{'_' + assay_id if assay_id else ''}: {error}")
    
    # Create an index file for quick lookup
    index_file = cache_dir / "index.csv"
    index_data = [
        {"target_id": t, "doc_id": d, "assay_id": a, "cached": True}
        for t, d, a in successful
    ]
    if index_data:
        pl.DataFrame(index_data).write_csv(index_file)
        print(f"\nCreated index file: {index_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Cache all ChEMBL data for benchmark configs"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing YAML config files"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/chembl_cache"),
        help="Directory to store cached CSV files"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds to wait between API requests (default: 1.0 for 1 req/sec)"
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Don't retry failed requests"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed requests (default: 3)"
    )
    
    args = parser.parse_args()
    
    cache_all_chembl_data(
        config_dir=args.config_dir,
        cache_dir=args.cache_dir,
        rate_limit_seconds=args.rate_limit,
        retry_on_failure=not args.no_retry,
        max_retries=args.max_retries
    )


if __name__ == "__main__":
    main()
