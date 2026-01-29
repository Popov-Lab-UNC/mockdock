# Standard library imports
import asyncio
from pathlib import Path
from typing import Optional

# Third-party imports
import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr

def plot_docking_results(
    df: pl.DataFrame,
    score_col: str = "docking_score",
    activity_col: str = "pchembl_value", 
    valid_col: str = "valid_pose_found",
    output_path: Optional[str] = None
):
    """
    Plot docking scores vs pChEMBL values.
    Assumes activity_col is already log-scaled (e.g., pchembl_value).
    """
    # 1. Filter out failed scores (999.9), nulls, and NaNs
    clean_df = df.filter(
        (pl.col(score_col).is_not_null()) &
        (pl.col(score_col).is_not_nan()) &
        (pl.col(activity_col).is_not_null()) &
        (pl.col(activity_col).is_not_nan()) &
        (pl.col(score_col) < 999.0)
    )
    
    if len(clean_df) < 2:
        print("Not enough data points to plot.")
        return None

    # 2. Extract columns (No math, just extraction)
    scores = clean_df.get_column(score_col).to_numpy()
    activities = clean_df.get_column(activity_col).to_numpy()
    
    # 3. Handle validity column safely (Fill nulls with False)
    if valid_col in clean_df.columns:
        is_valid = clean_df.get_column(valid_col).fill_null(False).to_numpy()
    else:
        # If column missing, assume everything is valid (or invalid, depending on preference)
        is_valid = np.ones(len(scores), dtype=bool)

    plt.figure(figsize=(10, 6))
    
    valid_mask = (is_valid == True)
    invalid_mask = (is_valid == False)

    # 4. Fix Plotting Order: Plot Noise (Red) FIRST, Signal (Blue) SECOND
    if np.any(invalid_mask):
        sns.scatterplot(
            x=scores[invalid_mask], 
            y=activities[invalid_mask], 
            color='red', 
            alpha=0.5, 
            label='RMSD > Threshold'
        )

    if np.any(valid_mask):
        sns.scatterplot(
            x=scores[valid_mask], 
            y=activities[valid_mask], 
            color='blue', 
            alpha=0.7, 
            label='RMSD < Threshold'
        )

    # 5. Compute Stats
    def _compute_stats(x_vals, y_vals):
        stats = {"n": int(len(x_vals)), "pearson": 0.0, "spearman": 0.0, "r2": 0.0}
        if len(x_vals) < 2:
            return stats
        if np.var(x_vals) > 0 and np.var(y_vals) > 0:
            p_corr, _ = pearsonr(x_vals, y_vals)
            s_corr, _ = spearmanr(x_vals, y_vals)
            stats["pearson"] = float(p_corr)
            stats["spearman"] = float(s_corr)
            stats["r2"] = float(p_corr ** 2)
        return stats

    valid_stats = _compute_stats(scores[valid_mask], activities[valid_mask])
    all_stats = _compute_stats(scores, activities)
    
    # Calculate pass percentage
    pass_pct = 100.0 * float(valid_stats["n"]) / float(len(scores))

    stats_text = (
        f"Pass RMSD: {pass_pct:.1f}%\n"
        f"Blue (n={valid_stats['n']}): R² {valid_stats['r2']:.3f}, "
        f"Pearson {valid_stats['pearson']:.3f}, Spearman {valid_stats['spearman']:.3f}\n"
        f"All (n={all_stats['n']}): R² {all_stats['r2']:.3f}, "
        f"Pearson {all_stats['pearson']:.3f}, Spearman {all_stats['spearman']:.3f}"
    )

    plt.title(f"{activity_col} vs Docking Score")
    plt.xlabel("Docking Score (Predicted)")
    plt.ylabel(f"{activity_col} (Experimental)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.text(
        0.02, 0.98, stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=9,
    )
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        "score_col": score_col,
        "activity_col": activity_col,
        "n_points": int(len(clean_df)),
        "pass_pct": pass_pct,
        "stats_valid": valid_stats,
        "stats_all": all_stats,
    }

def plot_activity_distribution(
    df: pl.DataFrame,
    activity_col: str = "pchembl_value",
    output_path: Optional[str] = None
):
    """
    Plot the distribution of bioactivity values.
    
    Args:
        df: Polars DataFrame containing activity data.
        activity_col: Column name for activity values.
        output_path: If provided, save the plot to this path.
    """
    # Filter nulls
    clean_df = df.filter(pl.col(activity_col).is_not_null())
    
    if len(clean_df) == 0:
        print("No activity data to plot distribution.")
        return

    activities = clean_df.get_column(activity_col).to_numpy()
    
    if activity_col != "pchembl_value":
        raise ValueError("pchembl_value is required for activity plots.")

    p_activities = activities
    activity_label = "pActivity (from pchembl_value)"
    title_suffix = "pchembl_value"

    plt.figure(figsize=(10, 6))
    sns.histplot(p_activities, kde=True, bins=30, color='skyblue')
    
    plt.title(f"Distribution of Experimental Activity (converted to pActivity)\n{title_suffix}, Total compounds: {len(p_activities)}")
    plt.xlabel(activity_label)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Activity distribution plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

async def fetch_ligand_expo_sdf(
    resname: str,
    output_dir: Path,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[Path]:
    """
    Fetch the ideal SDF for a ligand from RCSB Ligand Expo.

    Args:
        resname: The 3-letter ligand residue name (e.g., 'ATP').
        output_dir: Directory where the SDF file should be saved.
        session: Optional aiohttp ClientSession to reuse connections.
    """
    # Sanitize resname
    resname = resname.upper()
    url = f"https://files.rcsb.org/ligands/view/{resname}_ideal.sdf"

    should_close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        should_close_session = True

    try:
        async with session.get(url) as response:
            if response.status == 200:
                out_path = output_dir / f"{resname}_ideal.sdf"
                text = await response.text()
                # Run file I/O in executor to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, out_path.write_text, text)
                return out_path
            else:
                print(f"Failed to fetch SDF for {resname} from Ligand Expo: {response.status}")
                return None
    except Exception as e:
        print(f"Error fetching SDF for {resname}: {e}")
        return None
    finally:
        if should_close_session:
            await session.close()

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
