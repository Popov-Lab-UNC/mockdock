import polars as pl

def fetch_chembl_data(target_chembl_id: str, document_chembl_id: str, units: str = "nM") -> pl.DataFrame:
    """
    Fetch bioactivity data from ChEMBL for a specific target and document.
    
    For ChEMBL data, prefers pchembl_value (unit-agnostic) when available.
    For custom user data, units parameter is used to convert standard_value.
    
    Args:
        target_chembl_id: ChEMBL target ID
        document_chembl_id: ChEMBL document ID
        units: Only used for custom data fallback (default: "nM")
    
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
    
    if not data:
        return pl.DataFrame()

    df = pl.from_dicts(data, infer_schema_length=None)

    # Prefer pchembl_value (unit-agnostic) when available
    # Fall back to standard_value if pchembl_value is not present
    preferred_cols = ['molecule_chembl_id', 'canonical_smiles', 'pchembl_value']
    fallback_cols = ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units']
    
    # Check what columns are available
    has_pchembl = 'pchembl_value' in df.columns
    has_standard = 'standard_value' in df.columns
    
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
        # Filter for specified units if standard_units column exists
        if 'standard_units' in df.columns:
            df = df.filter(pl.col('standard_units') == units)
            print(f"   Using standard_value with units filter: {units}")
        else:
            print(f"   Using standard_value (no units column, assuming {units})")
    else:
        print("   WARNING: No activity data (pchembl_value or standard_value) found")
        return pl.DataFrame()

    return df
