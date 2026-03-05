# Standard library imports
from pathlib import Path
from typing import Optional, Union

# Third-party imports
import polars as pl

from .utils import standardize_smiles  # noqa: F401 – re-exported for backwards compat


def _find_cache_directory(
    cache_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    Find the ChEMBL cache directory by checking common locations.

    Returns:
        Path to cache directory if found, None otherwise
    """
    if cache_dir:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            return cache_path
        return None

    # Try common locations
    possible_cache_dirs = [
        Path("data/chembl_cache"),
        Path.cwd() / "data" / "chembl_cache",
        Path(__file__).parent.parent.parent / "data" / "chembl_cache",
    ]

    # Also check parent directories (in case we're in a subdirectory)
    current = Path.cwd()
    for _ in range(3):  # Check up to 3 levels up
        possible_cache_dirs.append(current / "data" / "chembl_cache")
        current = current.parent

    for cache_path in possible_cache_dirs:
        if cache_path.exists() and cache_path.is_dir():
            return cache_path

    return None


def fetch_chembl_data(
    target_chembl_id: str,
    document_chembl_id: str,
    assay_chembl_id: Optional[str] = None,
    return_stats: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
    cache_only: bool = False,
) -> Union[pl.DataFrame, tuple[pl.DataFrame, dict]]:
    """
    Fetch bioactivity data from ChEMBL for a specific target and document.

    Returns:
        DataFrame with canonical_smiles, molecule_chembl_id, and pchembl_value.
    """

    def process_data(temp_df):
        # 0. Initial raw count
        n_retrieved = len(temp_df)

        # 1. Require pchembl_value (unit-agnostic)
        if "pchembl_value" not in temp_df.columns:
            # If it's cached data, it should have it, but just in case
            raise RuntimeError("ChEMBL data missing pchembl_value.")

        # 2. Basic cleanup and casting
        temp_df = temp_df.with_columns(pl.col("pchembl_value").cast(pl.Float64, strict=False))
        temp_df = temp_df.drop_nulls(subset=["canonical_smiles", "pchembl_value"])

        n_orig = len(temp_df)

        # 3. Apply standardization (strip salts, neutralize, canonicalize)
        temp_df = temp_df.with_columns(
            pl.col("canonical_smiles").map_elements(standardize_smiles, return_dtype=pl.String)
        ).drop_nulls(subset=["canonical_smiles"])

        n_clean = len(temp_df)
        if n_orig > n_clean:
            print(f"   Standardization: Removed {n_orig - n_clean} invalid/failed compounds.")

        # 4. Deduplicate by canonical_smiles using median pchembl_value
        n_before = len(temp_df)
        agg_exprs = [
            pl.first("molecule_chembl_id").alias("molecule_chembl_id"),
            pl.median("pchembl_value").alias("pchembl_value"),
        ]
        if "assay_chembl_id" in temp_df.columns:
            agg_exprs.append(pl.first("assay_chembl_id").alias("assay_chembl_id"))

        temp_df = temp_df.group_by("canonical_smiles").agg(agg_exprs)
        n_after = len(temp_df)
        if n_after < n_before:
            print(f"   Deduplicated by canonical_smiles using median: {n_before} -> {n_after}")

        return temp_df, n_retrieved, n_orig, n_clean, n_after

    # Check cache first if enabled - this avoids API calls entirely
    if use_cache:
        found_cache_dir = _find_cache_directory(cache_dir)

        if found_cache_dir:
            # 1. Try granular assay-level cache
            # Path: found_cache_dir/{target_id}/{doc_id}_{assay_id}.csv
            if assay_chembl_id:
                assay_cache_file = (
                    found_cache_dir
                    / target_chembl_id
                    / f"{document_chembl_id}_{assay_chembl_id}.csv"
                )
                if assay_cache_file.exists():
                    try:
                        df = pl.read_csv(assay_cache_file)
                        print(f"   Loaded {len(df)} compounds from assay cache: {assay_cache_file}")
                        df, n_ret, n_orig, n_clean, n_after = process_data(df)
                        if return_stats:
                            stats = {
                                "n_retrieved": n_ret,
                                "n_total": n_orig,
                                "n_standardized": n_clean,
                                "n_deduplicated": n_after,
                            }
                            return (df, stats)
                        return df
                    except Exception as e:
                        print(
                            f"   Warning: Could not read assay cache file {assay_cache_file}: {e}"
                        )

            # 2. Try document-level cache (legacy or fallback)
            # Path: found_cache_dir/{target_id}_{document_id}.csv
            doc_cache_file = found_cache_dir / f"{target_chembl_id}_{document_chembl_id}.csv"
            if doc_cache_file.exists():
                try:
                    df = pl.read_csv(doc_cache_file)
                    # If we need assay-specific data but only have doc-level cache, filter it
                    if assay_chembl_id and "assay_chembl_id" in df.columns:
                        n_before = len(df)
                        df = df.filter(pl.col("assay_chembl_id") == assay_chembl_id)
                        print(
                            f"   Filtered document cache to assay {assay_chembl_id}: {n_before} -> {len(df)} compounds"
                        )

                    print(f"   Loaded {len(df)} compounds from document cache: {doc_cache_file}")
                    df, n_ret, n_orig, n_clean, n_after = process_data(df)
                    if return_stats:
                        stats = {
                            "n_retrieved": n_ret,
                            "n_total": n_orig,
                            "n_standardized": n_clean,
                            "n_deduplicated": n_after,
                        }
                        return (df, stats)
                    return df
                except Exception as e:
                    print(f"   Warning: Could not read document cache file {doc_cache_file}: {e}")

            # If nothing found and cache_only, fail
            if cache_only:
                target_file = assay_cache_file if assay_chembl_id else doc_cache_file
                raise RuntimeError(
                    f"Cache-only mode enabled but cache file not found: {target_file}\n"
                    f"Build the cache first using fcgmb/pipeline/find_matching_documents.py"
                )
            # Fallback message
            print(f"   Cache file not found for {target_chembl_id}_{document_chembl_id}")
            print("   Will attempt API call...")
        else:
            # Cache directory doesn't exist - suggest building cache
            if cache_only:
                raise RuntimeError(
                    "Cache-only mode enabled but cache directory not found.\n"
                    "Build the cache first:\n"
                    "python fcgmb/pipeline/cache_chembl_data.py --config-dir configs --cache-dir data/chembl_cache"
                )
            print("   Cache directory not found. To avoid API rate limiting, build cache first:")
            print(
                "   python fcgmb/pipeline/cache_chembl_data.py --config-dir configs --cache-dir data/chembl_cache"
            )

    # If cache miss or disabled, fetch from API (unless cache_only is True)
    if cache_only:
        raise RuntimeError(
            f"Cache-only mode enabled but cache not found for {target_chembl_id}_{document_chembl_id}.\n"
            f"Build the cache first using fcgmb/pipeline/find_matching_documents.py"
        )
    # Use lazy import with better error handling
    try:
        # Try to import the client - this may fail if API is down

        # Try importing with error suppression
        try:
            from chembl_webresource_client.new_client import new_client
        except Exception as import_error:
            # If import fails, check if it's because API is down
            error_str = str(import_error).lower()
            if (
                "500" in error_str
                or "unavailable" in error_str
                or "error getting schema" in error_str
            ):
                # API is down - provide helpful error message
                cache_hint = ""
                if use_cache:
                    cache_hint = "\n   To avoid this error, build the cache first:\n   python fcgmb/pipeline/cache_chembl_data.py --config-dir configs --cache-dir data/chembl_cache"
                raise RuntimeError(
                    f"ChEMBL API is currently unavailable (HTTP 500 error). "
                    f"The API may be experiencing issues or rate limiting.{cache_hint}\n"
                    f"Original error: {import_error}"
                )
            else:
                # Some other import error
                raise RuntimeError(
                    f"Could not import ChEMBL client. Detail: {import_error}\n"
                    f"If the API is down, build the cache first:\n"
                    f"python fcgmb/pipeline/cache_chembl_data.py --config-dir configs --cache-dir data/chembl_cache"
                )
    except RuntimeError:
        # Re-raise our custom RuntimeError
        raise
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Error: Could not connect to ChEMBL API. Detail: {e}")
        raise RuntimeError(
            "ChEMBL API is currently unavailable. Please try again later or use a local CSV file.\n"
            "To avoid API issues, build the cache first:\n"
            "python fcgmb/pipeline/cache_chembl_data.py --config-dir configs --cache-dir data/chembl_cache"
        )

    activity = new_client.activity

    # Filter by target and document (and assay if provided)
    # Align with find_matching_documents.py: Binding assays (assay_type='B') and non-null pChEMBL
    filter_params = {
        "target_chembl_id": target_chembl_id,
        "document_chembl_id": document_chembl_id,
        "assay_type": "B",
        "pchembl_value__isnull": False,
    }
    if assay_chembl_id:
        filter_params["assay_chembl_id"] = assay_chembl_id

    res = activity.filter(**filter_params)

    data = list(res)

    if not data:
        stats = {"n_retrieved": 0, "n_total": 0, "n_standardized": 0, "n_deduplicated": 0}
        return (pl.DataFrame(), stats) if return_stats else pl.DataFrame()

    df = pl.from_dicts(data, infer_schema_length=None)

    # If a cache directory is available, write raw data so future runs use cache
    write_cache_dir = (
        Path(cache_dir) if cache_dir and Path(cache_dir).exists() else _find_cache_directory(None)
    )
    if write_cache_dir is None and cache_dir:
        write_cache_dir = Path(cache_dir)
        write_cache_dir.mkdir(parents=True, exist_ok=True)
    if write_cache_dir is not None:
        if "target_chembl_id" not in df.columns:
            df = df.with_columns(pl.lit(target_chembl_id).alias("target_chembl_id"))
        if assay_chembl_id:
            cache_file = (
                write_cache_dir / target_chembl_id / f"{document_chembl_id}_{assay_chembl_id}.csv"
            )
            cache_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            cache_file = write_cache_dir / f"{target_chembl_id}_{document_chembl_id}.csv"
        try:
            df.write_csv(cache_file)
            print(f"   Cached {len(df)} compounds to {cache_file}")
        except Exception as e:
            print(f"   Warning: Could not write cache to {cache_file}: {e}")

    df, n_ret, n_orig, n_clean, n_after = process_data(df)
    stats = {
        "n_retrieved": n_ret,
        "n_total": n_orig,
        "n_standardized": n_clean,
        "n_deduplicated": n_after,
    }
    print("   Using pchembl_value (unit-agnostic) from ChEMBL")

    return (df, stats) if return_stats else df
