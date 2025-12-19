import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Optional

def plot_docking_results(
    df: pl.DataFrame,
    score_col: str = "docking_score",
    activity_col: str = "standard_value", # Assumed to be log-activity or activity
    output_path: Optional[str] = None,
    activity_units: str = "nM"
):
    """
    Plot docking scores vs activity values.
    
    Args:
        df: Polars DataFrame containing the results.
        score_col: Column name for docking scores.
        activity_col: Column name for activity values.
        output_path: If provided, save the plot to this path.
        activity_units: Units of the activity values ('nM', 'uM', 'mM', 'M').
    """
    
    # Filter out failed scores (999.9) or nulls
    clean_df = df.filter(
        (pl.col(score_col).is_not_null()) &
        (pl.col(activity_col).is_not_null()) &
        (pl.col(score_col) < 999.0)
    )
    
    if len(clean_df) < 2:
        print("Not enough data points to plot after filtering failed scores.")
        return

    # Extract columns
    scores = clean_df.get_column(score_col).to_numpy()
    activities = clean_df.get_column(activity_col).to_numpy()
    
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
    # X is predicted (docking), Y is experimental (pActivity)
    sns.scatterplot(x=scores, y=p_activities, alpha=0.6)
    
    # Calculate correlations only if we have variation in both axes
    pearson_corr = 0.0
    spearman_corr = 0.0
    if np.var(scores) > 0 and np.var(p_activities) > 0:
        pearson_corr, _ = pearsonr(scores, p_activities)
        spearman_corr, _ = spearmanr(scores, p_activities)
    else:
        print("Warning: Constant input detected, correlation set to 0.0")
    
    plt.title(f"pActivity vs Docking Score ({activity_units} to pActivity)\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}")
    plt.xlabel("Docking Score (Predicted)")
    plt.ylabel(f"pActivity (-log10 experimental {activity_units} -> M)")
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
