# Standard library imports
from typing import Optional, Tuple, Union

# Third-party imports
import polars as pl
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

def standardize_smiles(smiles: str) -> Optional[str]:
    """
    Strip salts, neutralize, and canonicalize a SMILES string.
    Returns None if the SMILES is invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 1. Keep largest fragment (removes [Na+], [Cl-], etc.)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        
        # 2. Uncharge (neutralize where chemically sensible)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        
        # 3. Return canonical SMILES
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def fetch_chembl_data(target_chembl_id: str, document_chembl_id: str, units: str = "nM", return_stats: bool = False) -> Union[pl.DataFrame, Tuple[pl.DataFrame, dict]]:
    """
    Fetch bioactivity data from ChEMBL for a specific target and document.
    
    For ChEMBL data, prefers pchembl_value (unit-agnostic) when available.
    For custom user data, units parameter is used to convert standard_value.
    
    Args:
        target_chembl_id: ChEMBL target ID
        document_chembl_id: ChEMBL document ID
        units: Only used for custom data fallback (default: "nM")
        return_stats: If True, return (DataFrame, stats_dict)
    
    Returns:
        DataFrame with canonical_smiles, molecule_chembl_id, and either:
        - pchembl_value (preferred, unit-agnostic) OR
        - standard_value (requires units parameter for conversion)
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except Exception as e:
        print(f"Error: Could not connect to ChEMBL API. It might be down. Detail: {e}")
        raise RuntimeError("ChEMBL API is currently unavailable. Please try again later or use a local CSV file.")

    activity = new_client.activity
    
    # Filter by target and document
    res = activity.filter(
        target_chembl_id=target_chembl_id,
        document_chembl_id=document_chembl_id
    )

    data = list(res)
    stats = {"n_total": 0, "n_standardized": 0}
    
    if not data:
        return (pl.DataFrame(), stats) if return_stats else pl.DataFrame()

    df = pl.from_dicts(data, infer_schema_length=None)

    # Prefer pchembl_value (unit-agnostic) when available
    # Fall back to standard_value if pchembl_value is not present
    preferred_cols = ['molecule_chembl_id', 'canonical_smiles', 'pchembl_value']
    fallback_cols = ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units']
    
    # Check what columns are available
    has_pchembl = 'pchembl_value' in df.columns
    has_standard = 'standard_value' in df.columns
    
    def apply_standardization(temp_df):
        n_orig = len(temp_df)
        temp_df = temp_df.with_columns(
            pl.col("canonical_smiles").map_elements(standardize_smiles, return_dtype=pl.String)
        ).drop_nulls(subset=["canonical_smiles"])
        n_clean = len(temp_df)
        if n_orig > n_clean:
            print(f"   Standardization: Removed {n_orig - n_clean} invalid/failed compounds.")
        return temp_df, n_orig, n_clean

    def aggregate_by_smiles(temp_df, value_col):
        n_before = len(temp_df)
        agg_exprs = [
            pl.first("molecule_chembl_id").alias("molecule_chembl_id"),
            pl.median(value_col).alias(value_col),
        ]
        if "standard_units" in temp_df.columns:
            agg_exprs.append(pl.first("standard_units").alias("standard_units"))
        temp_df = temp_df.group_by("canonical_smiles").agg(agg_exprs)
        n_after = len(temp_df)
        if n_after < n_before:
            print(f"   Deduplicated by canonical_smiles using median: {n_before} -> {n_after}")
        return temp_df

    if has_pchembl:
        # Use pchembl_value (preferred - unit-agnostic)
        existing_cols = [c for c in preferred_cols if c in df.columns]
        df = df.select(existing_cols)
        if 'pchembl_value' in df.columns:
            df = df.with_columns(
                pl.col('pchembl_value').cast(pl.Float64, strict=False)
            )
        # Drop rows with missing SMILES or pchembl
        df = df.drop_nulls(subset=['canonical_smiles', 'pchembl_value'])
        
        # Apply standardization
        df, n_orig, n_clean = apply_standardization(df)
        stats["n_total"] = n_orig
        stats["n_standardized"] = n_clean

        # Deduplicate by canonical_smiles using median pchembl_value
        df = aggregate_by_smiles(df, "pchembl_value")
            
        # Also create standard_value column for compatibility (pchembl is already in pActivity units)
        df = df.with_columns(
            pl.col('pchembl_value').alias('standard_value')
        )
        print(f"   Using pchembl_value (unit-agnostic) from ChEMBL")
    elif has_standard:
        # Fall back to standard_value (requires units)
        existing_cols = [c for c in fallback_cols if c in df.columns]
        df = df.select(existing_cols)
        if 'standard_value' in df.columns:
            df = df.with_columns(
                pl.col('standard_value').cast(pl.Float64, strict=False)
            )
        # Drop rows with missing SMILES or Activity
        df = df.drop_nulls(subset=['canonical_smiles', 'standard_value'])
        
        # Apply standardization
        df, n_orig, n_clean = apply_standardization(df)
        stats["n_total"] = n_orig
        stats["n_standardized"] = n_clean
            
        # Filter for specified units if standard_units column exists
        if 'standard_units' in df.columns:
            df = df.filter(pl.col('standard_units') == units)
            print(f"   Using standard_value with units filter: {units}")
        else:
            print(f"   Using standard_value (no units column, assuming {units})")

        # Deduplicate by canonical_smiles using median standard_value
        df = aggregate_by_smiles(df, "standard_value")
    else:
        print("   WARNING: No activity data (pchembl_value or standard_value) found")
        return (pl.DataFrame(), stats) if return_stats else pl.DataFrame()

    return (df, stats) if return_stats else df
