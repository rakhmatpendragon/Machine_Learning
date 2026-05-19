"""
cluster_interpreter.py — Cluster Interpretation pipeline.
Uses banner() and save_fig() from src.utils (no local copies).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (INTERPRETATION_AGG_FUNCS, INTERPRETATION_HEATMAP_PNG,
                         INTERPRETATION_PROFILE_CSV, INTERPRETATION_RADAR_PNG,
                         INTERPRETATION_STATS_CSV, LABELED_DATASET_PATH,
                         NUMERIC_FEATURES_FOR_ANALYSIS, PLOT_STYLE, TARGET_COLUMN_NAME)
from src.logger import get_logger
from src.utils import banner, save_fig          # ← single shared source

logger = get_logger(__name__)
plt.style.use(PLOT_STYLE)


def compute_descriptive_stats(df, labels, numeric_cols=None, agg_funcs=None):
    banner("Step 1 — Descriptive Statistics per Cluster  (mean / min / max)", logger)
    cols  = numeric_cols or NUMERIC_FEATURES_FOR_ANALYSIS
    funcs = agg_funcs    or INTERPRETATION_AGG_FUNCS
    present = [c for c in cols if c in df.columns]
    absent  = [c for c in cols if c not in df.columns]
    if absent:
        logger.warning("Columns not found, skipped: %s", absent)

    work  = df[present].copy()
    work[TARGET_COLUMN_NAME] = labels
    stats = work.groupby(TARGET_COLUMN_NAME)[present].agg(funcs)

    for k in range(int(labels.max()) + 1):
        row = stats.loc[k]
        logger.info("\n── Cluster %d ──", k)
        logger.info("  %-25s  %10s  %10s  %10s  %10s  %10s",
                    "Feature","mean","min","max","std","median")
        logger.info("  " + "-"*77)
        for col in present:
            logger.info("  %-25s  %10.2f  %10.2f  %10.2f  %10.2f  %10.2f",
                        col, *[row[(col, fn)] for fn in funcs])

    INTERPRETATION_STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(INTERPRETATION_STATS_CSV)
    logger.info("Stats saved → %s", INTERPRETATION_STATS_CSV)
    return stats


def describe_cluster_profiles(stats, df, labels) -> dict[int, str]:
    banner("Step 2 — Cluster Profile Descriptions", logger)
    try:
        mean_df = stats.xs("mean", axis=1, level=1)
    except KeyError:
        mean_df = stats

    numeric_cols = list(mean_df.columns)
    global_mean  = df[numeric_cols].mean()
    global_std   = df[numeric_cols].std().replace(0, 1)
    profiles: dict[int, str] = {}

    for k in range(len(mean_df)):
        z = (mean_df.loc[k] - global_mean) / global_std
        high = [c for c in numeric_cols if z[c] >  0.5]
        low  = [c for c in numeric_cols if z[c] < -0.5]
        avg  = [c for c in numeric_cols if c not in high and c not in low]
        size = int((labels == k).sum())

        if   "annual_spend_usd" in high and "num_purchases" in high: label = "High-Value Active Shoppers"
        elif "annual_spend_usd" in low  and "returns_count" in high: label = "High-Return Low-Spend Customers"
        elif "tenure_months"    in high and "annual_spend_usd" in low: label = "Loyal Low-Spend Veterans"
        elif "last_purchase_days" in high:                             label = "At-Risk / Lapsed Customers"
        elif "num_purchases" in low and "avg_order_value" in high:    label = "Infrequent Big-Ticket Buyers"
        else:                                                           label = "Mixed / Moderate Profile"

        parts = [
            f"Cluster {k}  ({size} customers, {size/len(labels)*100:.1f}% of total)",
            f"  LABEL  : {label}",
            "  HIGH   : " + ", ".join(f"{c} ({mean_df.loc[k,c]:.1f})" for c in high) if high else "",
            "  LOW    : " + ", ".join(f"{c} ({mean_df.loc[k,c]:.1f})" for c in low)  if low  else "",
            "  AVERAGE: " + ", ".join(f"{c} ({mean_df.loc[k,c]:.1f})" for c in avg)  if avg  else "",
        ]
        narrative = "\n".join(p for p in parts if p)
        profiles[k] = narrative
        logger.info("\n%s", narrative)

    INTERPRETATION_PROFILE_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"cluster": k, "description": v} for k, v in profiles.items()]).set_index("cluster").to_csv(INTERPRETATION_PROFILE_CSV)
    logger.info("Profile table saved → %s", INTERPRETATION_PROFILE_CSV)
    return profiles


def export_labeled_dataset(df, labels, save_path=None):
    banner(f'Step 3 — Export Labeled Dataset  (column: "{TARGET_COLUMN_NAME}")', logger)
    if len(labels) != len(df):
        raise ValueError(f"Length mismatch: df={len(df)}, labels={len(labels)}.")
    labeled = df.copy()
    labeled[TARGET_COLUMN_NAME] = labels
    path = save_path or LABELED_DATASET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(path, index=False)
    logger.info("Shape: %d rows × %d cols  |  Target column: \"%s\"  |  Saved → %s",
                *labeled.shape, TARGET_COLUMN_NAME, path)
    return labeled


def plot_interpretation_heatmap(stats, save_path=None):
    banner("Step 4 — Cluster Heatmap  (normalised feature means)", logger)
    try:
        mean_df = stats.xs("mean", axis=1, level=1).copy()
    except KeyError:
        mean_df = stats.copy()
    norm = (mean_df - mean_df.min()) / (mean_df.max() - mean_df.min() + 1e-9)
    norm.index = [f"Cluster {k}" for k in norm.index]
    fig, ax = plt.subplots(figsize=(max(10, len(norm.columns)*1.1), 4))
    sns.heatmap(norm, annot=mean_df.round(1), fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Normalised mean (0=lowest, 1=highest)"})
    ax.set_title("Cluster Feature Means — Normalised Heatmap",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    save_fig(fig, save_path or INTERPRETATION_HEATMAP_PNG, tight=False, logger=logger)


def plot_radar_chart(stats, save_path=None):
    banner("Step 5 — Radar Chart  (cluster profiles)", logger)
    try:
        mean_df = stats.xs("mean", axis=1, level=1).copy()
    except KeyError:
        mean_df = stats.copy()
    norm     = (mean_df - mean_df.min()) / (mean_df.max() - mean_df.min() + 1e-9)
    features = list(norm.columns)
    if len(features) < 3:
        logger.warning("Radar chart requires ≥ 3 features; skipping.")
        return
    n_clusters = len(norm)
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    palette = plt.cm.tab10.colors[:n_clusters]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for k in range(n_clusters):
        vals = norm.iloc[k].tolist() + norm.iloc[k].tolist()[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=palette[k], label=f"Cluster {k}")
        ax.fill(angles, vals, alpha=0.12, color=palette[k])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_","\n") for f in features], fontsize=9)
    ax.set_title("Cluster Profiles — Radar Chart\n(normalised feature means)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), title="Cluster")
    save_fig(fig, save_path or INTERPRETATION_RADAR_PNG, logger=logger)


def run_full_interpretation(df, labels, numeric_cols=None, export_path=None) -> dict:
    banner("Cluster Interpretation Pipeline  —  START", logger)
    stats      = compute_descriptive_stats(df, labels, numeric_cols=numeric_cols)
    profiles   = describe_cluster_profiles(stats, df, labels)
    labeled_df = export_labeled_dataset(df, labels, save_path=export_path)
    plot_interpretation_heatmap(stats)
    plot_radar_chart(stats)
    banner("Cluster Interpretation Pipeline  —  COMPLETE", logger)
    return {"descriptive_stats": stats, "cluster_profiles": profiles,
            "labeled_df": labeled_df}
