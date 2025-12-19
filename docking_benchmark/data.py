import polars as pl
from chembl_webresource_client.new_client import new_client

def fetch_chembl_data(target_chembl_id: str, document_chembl_id: str, units: str = "nM") -> pl.DataFrame:
    """
    Fetch bioactivity data from ChEMBL for a specific target and document.

    Args:
        target_chembl_id (str): The ChEMBL ID of the target (e.g., 'CHEMBL301').
        document_chembl_id (str): The ChEMBL ID of the document (e.g., 'CHEMBL5728430').
        units (str): The units to filter for (e.g., 'nM', 'uM'). Defaults to 'nM'.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the SMILES and standard activity values.
    """
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

    # Convert standard_value to float and handle potential missing values
    # We are interested in standard_type, standard_value, canonical_smiles
    
    # Check if columns exist
    required_cols = ['molecule_chembl_id', 'standard_type', 'standard_value', 'standard_units', 'canonical_smiles']
    existing_cols = [c for c in required_cols if c in df.columns]
    
    df = df.select(existing_cols)
    
    if 'standard_value' in df.columns:
        df = df.with_columns(
            pl.col('standard_value').cast(pl.Float64, strict=False)
        )
        
    # Drop rows with missing SMILES or Activity
    df = df.drop_nulls(subset=['canonical_smiles', 'standard_value'])
    
    # Filter for specified units as requested
    if 'standard_units' in df.columns:
        df = df.filter(pl.col('standard_units') == units)

    return df
