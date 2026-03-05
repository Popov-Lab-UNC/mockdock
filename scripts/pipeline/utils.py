import glob
import multiprocessing
from pathlib import Path
from typing import Optional

import pandas as pd


def get_chunk_output_path(base_output_path: str, start: int, end: int) -> Path:
    """
    Generate a standardized path for chunk output files.

    Args:
        base_output_path: The main output path (e.g., 'data/results.csv')
        start: Start index
        end: End index

    Returns:
        Path object for the chunk file (e.g., 'data/intermediate/results/chunk_0_100.csv')
    """
    output_path_obj = Path(base_output_path)
    intermediate_dir = output_path_obj.parent / "intermediate" / output_path_obj.stem
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    return intermediate_dir / f"chunk_{start}_{end}.csv"


def merge_csv_files(
    pattern: str, output_path: str, dedup_cols: Optional[list[str]] = None
) -> None:
    """
    Merge CSV files matching a pattern into a single output file.

    Args:
        pattern: Glob pattern for input files
        output_path: Path to save the merged CSV
        dedup_cols: List of columns to use for deduplication
    """
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} files matching '{pattern}'")

    if not files:
        print("No files found to merge.")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from {f}")
        except Exception as e:
            print(f"  [!] Error loading {f}: {e}")

    if not dfs:
        print("No data loaded.")
        return

    df_merged = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after merge: {len(df_merged)}")

    if dedup_cols:
        before = len(df_merged)
        df_merged = df_merged.drop_duplicates(subset=dedup_cols)
        print(
            f"After deduplication: {len(df_merged)} (removed {before - len(df_merged)})"
        )

    df_merged.to_csv(output_path, index=False)
    print(f"Saved merged results to {output_path}")


def run_with_timeout(func, args=(), kwargs=None, timeout=60):
    """
    Run a function with a timeout using multiprocessing.

    Args:
        func: Function to run
        args: Positional arguments
        kwargs: Keyword arguments
        timeout: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError/Exception
    """
    if kwargs is None:
        kwargs = {}

    with multiprocessing.Pool(processes=1) as pool:
        result = pool.apply_async(func, args, kwargs)
        try:
            return result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout}s")
        except Exception as e:
            raise e
