from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    INTERPRETATION_AGG_FUNCS,
    INTERPRETATION_STATS_CSV,
    NUMERIC_FEATURES_FOR_ANALYSIS,
    PLOT_DPI,
    TARGET_COLUMN_NAME,
)
from src.logger import get_logger
from src.utils import banner

logger = get_logger(__name__)

def compute_descriptive_stats(
    df: pd.DataFrame,
    labels: np.ndarray,
    numeric_cols: list[str] | None = None,
    agg_funcs: list[str] | None = None,
) -> pd.DataFrame:
    banner("Step 1 — Descriptive Statistics per Cluster (mean / min / max)")

    cols  = numeric_cols  if numeric_cols  is not None else NUMERIC_FEATURES_FOR_ANALYSIS
    funcs = agg_funcs     if agg_funcs     is not None else INTERPRETATION_AGG_FUNCS

    present = [c for c in cols if c in df.columns]
    absent  = [c for c in cols if c not in df.columns]
    if absent:
        logger.warning("Columns not found, skipped: %s", absent)

    work = df[present].copy()
    work[TARGET_COLUMN_NAME] = labels

    stats: pd.DataFrame = work.groupby(TARGET_COLUMN_NAME)[present].agg(funcs)

    n_clusters = int(labels.max()) + 1
    for k in range(n_clusters):
        row = stats.loc[k]
        logger.info("")
        logger.info("── Cluster %d ──────────────────────────────────────────────", k)
        logger.info(
            "  %-25s  %10s  %10s  %10s  %10s  %10s",
            "Feature", "mean", "min", "max", "std", "median",
        )
        logger.info("  " + "-" * 77)
        for col in present:
            vals = {fn: row[(col, fn)] for fn in funcs}
            logger.info(
                "  %-25s  %10.2f  %10.2f  %10.2f  %10.2f  %10.2f",
                col,
                vals.get("mean",   np.nan),
                vals.get("min",    np.nan),
                vals.get("max",    np.nan),
                vals.get("std",    np.nan),
                vals.get("median", np.nan),
            )
    
    INTERPRETATION_STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(INTERPRETATION_STATS_CSV)
    logger.info("")
    logger.info("Descriptive stats saved → %s", INTERPRETATION_STATS_CSV)

    return stats

def describe_cluster_profiles(
    stats: pd.DataFrame,
    df: pd.DataFrame,
    labels: np.ndarray,
) -> dict[int, str]:
    banner("Step 2 — Cluster Profile Descriptions")

    try:
        mean_df = stats.xs("mean", axis=1, level=1)
    except KeyError:
        mean_df = stats

    numeric_cols = list(mean_df.columns)

    global_mean = df[numeric_cols].mean()
    global_std  = df[numeric_cols].std().replace(0, 1)

    n_clusters = len(mean_df)
    profiles: dict[int, str] = {}

    for k in range(n_clusters):
        cluster_mean = mean_df.loc[k]
        z_scores     = (cluster_mean - global_mean) / global_std

        high_features = [c for c in numeric_cols if z_scores[c] >  0.5]
        low_features  = [c for c in numeric_cols if z_scores[c] < -0.5]
        avg_features  = [c for c in numeric_cols
                         if c not in high_features and c not in low_features]
        
        size        = int((labels == k).sum())
        size_pct    = size / len(labels) * 100

        parts: list[str] = [f"Cluster {k}  ({size} customers, {size_pct:.1f}% of total)"]

        if high_features:
            parts.append(
                "  HIGH   : " + ", ".join(
                    f"{c} ({cluster_mean[c]:.1f})" for c in high_features
                )
            )
        if low_features:
            parts.append(
                "  LOW    : " + ", ".join(
                    f"{c} ({cluster_mean[c]:.1f})" for c in low_features
                )
            )
        if avg_features:
            parts.append(
                "  AVERAGE: " + ", ".join(
                    f"{c} ({cluster_mean[c]:.1f})" for c in avg_features
                )
            )

        if "annual_spend_usd" in high_features and "num_purchases" in high_features:
            # label = "High-Value Active Shoppers"
            label = "Confirm Fraud Transaction"
        # elif "annual_spend_usd" in low_features and "returns_count" in high_features:
        #     label = "High-Return Low-Spend Customers"
        # elif "tenure_months" in high_features and "annual_spend_usd" in low_features:
        #     label = "Loyal Low-Spend Veterans"
        # elif "last_purchase_days" in high_features:
        #     label = "At-Risk / Lapsed Customers"
        elif "num_purchases" in low_features and "avg_order_value" in high_features:
            # label = "Infrequent Big-Ticket Buyers"
            label = "Suspect Fraud Transaction"
        else:
            # label = "Mixed / Moderate Profile"
            label = "Normal Transaction"

        parts.insert(1, f"  LABEL  : {labels}")

        narrative   = "\n".join(parts)
        profiles[k] = narrative

        logger.info("")
        logger.info(narrative)

    return profiles

def run_full_interpretation(
        df: pd.DataFrame,
        labels: np.ndarray,
        numeric_cols: list[str] | None = None,
        export_path: Path | None = None,
) -> dict:
    banner("Cluster Interpretation Pipeline  —  START")

    stats = compute_descriptive_stats(df, labels, numeric_cols=numeric_cols)

    profiles = describe_cluster_profiles(stats, df, labels)

    # labeled_df = export_labeled_dataset(df, labels, save_path=export_path)

    # plot_interpretation_heatmap(stats)

    # plot_radar_chart(stats)

    # banner("Cluster Interpretation Pipeline  —  COMPLETE")
    # logger.info("Outputs written to: %s", INTERPRETATION_STATS_CSV.parent)

    return {
        "descriptive_stats": stats,
        "cluster_profiles":  profiles,
        # "labeled_df":        labeled_df,
    }
    