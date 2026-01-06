import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Optional
import requests
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

def plot_docking_results(
    df: pl.DataFrame,
    score_col: str = "docking_score",
    activity_col: str = "standard_value", # Assumed to be log-activity or activity
    valid_col: str = "valid_pose_found",
    output_path: Optional[str] = None,
    activity_units: str = "nM"
):
    """
    Plot docking scores vs activity values.
    
    Args:
        df: Polars DataFrame containing the results.
        score_col: Column name for docking scores.
        activity_col: Column name for activity values.
        valid_col: Column name for boolean validity (used for coloring).
        output_path: If provided, save the plot to this path.
        activity_units: Units of the activity values ('nM', 'uM', 'mM', 'M').
    """
    
    # Filter out failed scores (999.9), nulls, and NaNs
    clean_df = df.filter(
        (pl.col(score_col).is_not_null()) &
        (pl.col(score_col).is_not_nan()) &
        (pl.col(activity_col).is_not_null()) &
        (pl.col(score_col) < 999.0)
    )
    
    if len(clean_df) < 2:
        print("Not enough data points to plot after filtering failed scores/NaNs.")
        return

    # Extract columns
    scores = clean_df.get_column(score_col).to_numpy()
    activities = clean_df.get_column(activity_col).to_numpy()
    
    if valid_col in clean_df.columns:
        is_valid = clean_df.get_column(valid_col).to_numpy()
    else:
        is_valid = np.ones(len(scores), dtype=bool)

    # Log transform activity (pActivity = -log10(Molar))
    # Adjust based on units
    unit_offsets = {
        "nM": 9,
        "uM": 6,
        "mM": 3,
        "M": 0
    }
    offset = unit_offsets.get(activity_units, 9) # Default to nM if unknown
    
    p_activities = offset - np.log10(activities + 1e-12)

    plt.figure(figsize=(10, 6))
    
    # Plot valid points (Blue)
    valid_mask = is_valid == True
    if np.any(valid_mask):
        sns.scatterplot(x=p_activities[valid_mask], y=scores[valid_mask], color='blue', alpha=0.6, label='RMSD < Threshold')

    # Plot invalid points (Red)
    invalid_mask = is_valid == False
    if np.any(invalid_mask):
        sns.scatterplot(x=p_activities[invalid_mask], y=scores[invalid_mask], color='red', alpha=0.6, marker='X', label='RMSD > Threshold')

    # Calculate correlations
    pearson_corr = 0.0
    spearman_corr = 0.0
    r_squared = 0.0

    if np.var(scores) > 0 and np.var(p_activities) > 0:
        pearson_corr, _ = pearsonr(scores, p_activities)
        spearman_corr, _ = spearmanr(scores, p_activities)
        r_squared = pearson_corr ** 2
    else:
        print("Warning: Constant input detected, correlation set to 0.0")
    
    plt.title(f"Docking Score vs pActivity\nR²: {r_squared:.3f}, Pearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}")
    plt.ylabel("Docking Score (Predicted)")
    plt.xlabel(f"pActivity (-log10 experimental {activity_units} -> M)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_activity_distribution(
    df: pl.DataFrame,
    activity_col: str = "standard_value",
    output_path: Optional[str] = None,
    activity_units: str = "nM"
):
    """
    Plot the distribution of bioactivity values.
    """
    # Filter nulls
    clean_df = df.filter(pl.col(activity_col).is_not_null())
    
    if len(clean_df) == 0:
        print("No activity data to plot distribution.")
        return

    activities = clean_df.get_column(activity_col).to_numpy()
    
    # Calculate pActivity for the histogram
    unit_offsets = {"nM": 9, "uM": 6, "mM": 3, "M": 0}
    offset = unit_offsets.get(activity_units, 9)
    p_activities = offset - np.log10(activities + 1e-12)

    plt.figure(figsize=(10, 6))
    sns.histplot(p_activities, kde=True, bins=30, color='skyblue')
    
    plt.title(f"Distribution of Experimental Activity (converted to pActivity)\nUnits: {activity_units}, Total compounds: {len(p_activities)}")
    plt.xlabel(f"pActivity (offset={offset})")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Activity distribution plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def fetch_ligand_expo_sdf(resname: str, output_dir: Path) -> Optional[Path]:
    """
    Fetch the ideal SDF for a ligand from RCSB Ligand Expo.
    """
    # Sanitize resname
    resname = resname.upper()
    url = f"https://files.rcsb.org/ligands/view/{resname}_ideal.sdf"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            out_path = output_dir / f"{resname}_ideal.sdf"
            out_path.write_text(response.text)
            return out_path
        else:
            print(f"Failed to fetch SDF for {resname} from Ligand Expo: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching SDF for {resname}: {e}")
        return None

def assign_bond_orders_from_template(pdb_mol: Chem.Mol, template_mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Assign bond orders to a PDB molecule using a template molecule (with bond orders).
    """
    try:
        # Remove Hs from template if the PDB mol doesn't have them
        # PDB mols from MolFromPDBFile usually don't have Hs
        if pdb_mol.GetNumAtoms() < template_mol.GetNumAtoms():
            template_mol = Chem.RemoveHs(template_mol)
            
        new_mol = AllChem.AssignBondOrdersFromTemplate(template_mol, pdb_mol)
        return new_mol
    except Exception as e:
        print(f"Failed to assign bond orders from template: {e}")
        return None
