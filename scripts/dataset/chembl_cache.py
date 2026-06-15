from pathlib import Path
from typing import Optional, Union

import polars as pl


def get_cache_path(
    cache_dir: Union[str, Path], target_id: str, document_id: str, assay_id: str
) -> Path:
    """Get the standard path for a cached ChEMBL assay file."""
    # Structure: cache_dir/TARGET/DOC_ASSAY.csv
    target_dir = Path(cache_dir) / target_id
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{document_id}_{assay_id}.csv"


def write_assay_cache(
    cache_dir: Union[str, Path],
    target_id: str,
    document_id: str,
    assay_id: str,
    df: pl.DataFrame,
):
    """Write raw bioactivity data for a specific assay to cache."""
    path = get_cache_path(cache_dir, target_id, document_id, assay_id)
    df.write_csv(path)


def read_assay_cache(
    cache_dir: Union[str, Path], target_id: str, document_id: str, assay_id: str
) -> Optional[pl.DataFrame]:
    """Read raw bioactivity data for a specific assay from cache."""
    path = get_cache_path(cache_dir, target_id, document_id, assay_id)
    if path.exists():
        try:
            return pl.read_csv(path)
        except Exception:
            return None
    return None


def is_assay_cached(
    cache_dir: Union[str, Path], target_id: str, document_id: str, assay_id: str
) -> bool:
    """Check if an assay is already in cache."""
    return get_cache_path(cache_dir, target_id, document_id, assay_id).exists()
