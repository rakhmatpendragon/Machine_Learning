"""
cluster_interpreter.py
-----------------------
Cluster Interpretation pipeline for the Retail Customer Churn dataset.

Steps (in order)
----------------
1. compute_descriptive_stats()   — mean, min, max, std, median per cluster
                                   for every numeric feature
2. describe_cluster_profiles()   — human-readable characteristic summary
                                   for each cluster based on aggregation
3. export_labeled_dataset()      — attach cluster labels as column "Target"
                                   and save to CSV
4. plot_interpretation_heatmap() — heatmap of normalised cluster means
5. plot_radar_chart()            — radar/spider chart comparing clusters
6. run_full_interpretation()     — orchestrator: all steps end-to-end

Design principles
-----------------
* Pure functions — every function takes explicit inputs, returns results.
* No in-place mutation of any DataFrame passed in.
* All file paths and column names come from config.py.
* Plots use Agg backend → safe in scripts, CI, and notebooks alike.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    INTERPRETATION_AGG_FUNCS,
    INTERPRETATION_HEATMAP_PNG,
    INTERPRETATION_PROFILE_CSV,
    INTERPRETATION_RADAR_PNG,
    INTERPRETATION_STATS_CSV,
    LABELED_DATASET_PATH,
    NUMERIC_FEATURES_FOR_ANALYSIS,
    PLOT_DPI,
    PLOT_STYLE,
    TARGET_COLUMN_NAME,
)
from src.logger import get_logger

logger = get_logger(__name__)
plt.style.use(PLOT_STYLE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    w = 70
    logger.info("")
    logger.info("=" * w)
    logger.info("  %s", title.upper())
    logger.info("=" * w)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", path)


# ── Step 1 — Descriptive Statistics per Cluster ───────────────────────────────

def compute_descriptive_stats(
    df: pd.DataFrame,
    labels: np.ndarray,
    numeric_cols: list[str] | None = None,
    agg_funcs: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute descriptive statistics (mean, min, max, std, median) for every
    numeric feature, broken down by cluster.

    Parameters
    ----------
    df          : pd.DataFrame  — preprocessed dataset (unscaled values)
    labels      : np.ndarray    — cluster label per row  (shape: n_samples,)
    numeric_cols: list[str] | None
        Columns to analyse.  Defaults to NUMERIC_FEATURES_FOR_ANALYSIS.
    agg_funcs   : list[str] | None
        Aggregation functions to apply.  Defaults to INTERPRETATION_AGG_FUNCS.

    Returns
    -------
    pd.DataFrame
        Multi-level column DataFrame:
            index   → cluster id  (0, 1, 2, …)
            columns → (feature, stat)  e.g. ("annual_spend_usd", "mean")
    """
    _banner("Step 1 — Descriptive Statistics per Cluster  (mean / min / max)")

    cols  = numeric_cols  if numeric_cols  is not None else NUMERIC_FEATURES_FOR_ANALYSIS
    funcs = agg_funcs     if agg_funcs     is not None else INTERPRETATION_AGG_FUNCS

    # Keep only columns that actually exist in df
    present = [c for c in cols if c in df.columns]
    absent  = [c for c in cols if c not in df.columns]
    if absent:
        logger.warning("Columns not found, skipped: %s", absent)

    # Build a copy with the cluster label attached
    work = df[present].copy()
    work[TARGET_COLUMN_NAME] = labels

    # Aggregate
    stats: pd.DataFrame = work.groupby(TARGET_COLUMN_NAME)[present].agg(funcs)

    # ── Log a clean per-cluster table ──────────────────────────────────────
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

    # Persist to CSV
    INTERPRETATION_STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(INTERPRETATION_STATS_CSV)
    logger.info("")
    logger.info("Descriptive stats saved → %s", INTERPRETATION_STATS_CSV)

    return stats


# ── Step 2 — Human-Readable Cluster Profiles ─────────────────────────────────

def describe_cluster_profiles(
    stats: pd.DataFrame,
    df: pd.DataFrame,
    labels: np.ndarray,
) -> dict[int, str]:
    """
    Generate a concise plain-language characteristic summary for each cluster
    by comparing each cluster's feature means against the overall dataset mean.

    Interpretation logic
    --------------------
    For every numeric feature the cluster mean is compared to the global mean:
      > +0.5 σ  →  labelled "High"
      < -0.5 σ  →  labelled "Low"
      otherwise →  labelled "Average"

    A short narrative sentence is assembled from the high/low standouts.

    Parameters
    ----------
    stats  : pd.DataFrame  — output of compute_descriptive_stats()
    df     : pd.DataFrame  — preprocessed dataset (for global statistics)
    labels : np.ndarray    — cluster labels

    Returns
    -------
    dict[int, str]
        {cluster_id: narrative_string}
    """
    _banner("Step 2 — Cluster Profile Descriptions")

    # Identify columns available at the "mean" level
    try:
        mean_df = stats.xs("mean", axis=1, level=1)
    except KeyError:
        # Fallback: stats might only have one level (no multi-index)
        mean_df = stats

    numeric_cols = list(mean_df.columns)

    global_mean = df[numeric_cols].mean()
    global_std  = df[numeric_cols].std().replace(0, 1)   # avoid division by zero

    n_clusters = len(mean_df)
    profiles: dict[int, str] = {}

    for k in range(n_clusters):
        cluster_mean = mean_df.loc[k]
        z_scores     = (cluster_mean - global_mean) / global_std

        high_features = [c for c in numeric_cols if z_scores[c] >  0.5]
        low_features  = [c for c in numeric_cols if z_scores[c] < -0.5]
        avg_features  = [c for c in numeric_cols
                         if c not in high_features and c not in low_features]

        size       = int((labels == k).sum())
        size_pct   = size / len(labels) * 100

        # Build a readable narrative
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

        # One-line label for the cluster
        if "annual_spend_usd" in high_features and "num_purchases" in high_features:
            label = "High-Value Active Shoppers"
        elif "annual_spend_usd" in low_features and "returns_count" in high_features:
            label = "High-Return Low-Spend Customers"
        elif "tenure_months" in high_features and "annual_spend_usd" in low_features:
            label = "Loyal Low-Spend Veterans"
        elif "last_purchase_days" in high_features:
            label = "At-Risk / Lapsed Customers"
        elif "num_purchases" in low_features and "avg_order_value" in high_features:
            label = "Infrequent Big-Ticket Buyers"
        else:
            label = "Mixed / Moderate Profile"

        parts.insert(1, f"  LABEL  : {label}")

        narrative = "\n".join(parts)
        profiles[k] = narrative

        logger.info("")
        logger.info(narrative)

    # Persist full profile table
    profile_rows = []
    for k, text in profiles.items():
        profile_rows.append({"cluster": k, "description": text})
    profile_df = pd.DataFrame(profile_rows).set_index("cluster")
    INTERPRETATION_PROFILE_CSV.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(INTERPRETATION_PROFILE_CSV)
    logger.info("")
    logger.info("Full profile table saved → %s", INTERPRETATION_PROFILE_CSV)

    return profiles


# ── Step 3 — Export Labeled Dataset ───────────────────────────────────────────

def export_labeled_dataset(
    df: pd.DataFrame,
    labels: np.ndarray,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Attach cluster labels to the preprocessed DataFrame as a column named
    ``Target`` and export to CSV.

    Parameters
    ----------
    df        : pd.DataFrame — preprocessed dataset (no cluster column yet)
    labels    : np.ndarray   — cluster label per row
    save_path : Path | None  — override default LABELED_DATASET_PATH

    Returns
    -------
    pd.DataFrame — df with the new "Target" column appended (last column)
    """
    _banner(f'Step 3 — Export Labeled Dataset  (column: "{TARGET_COLUMN_NAME}")')

    if len(labels) != len(df):
        raise ValueError(
            f"Length mismatch: df has {len(df)} rows but labels has {len(labels)} entries."
        )

    labeled = df.copy()
    labeled[TARGET_COLUMN_NAME] = labels

    path = save_path if save_path is not None else LABELED_DATASET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(path, index=False)

    logger.info("Shape          : %d rows × %d columns", *labeled.shape)
    logger.info('Target column  : "%s"  (last column)', TARGET_COLUMN_NAME)
    logger.info("Cluster counts :")
    for k, cnt in sorted(
        zip(*np.unique(labels, return_counts=True)),
        key=lambda x: x[0],
    ):
        logger.info("  Cluster %d → %d rows  (%.1f%%)", k, cnt, cnt / len(labels) * 100)
    logger.info("Exported to    : %s", path)

    return labeled


# ── Step 4 — Heatmap of Cluster Means ────────────────────────────────────────

def plot_interpretation_heatmap(
    stats: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """
    Plot a heatmap of normalised feature means per cluster.

    Each column is normalised to [0, 1] across clusters so all features
    share the same colour scale.

    Parameters
    ----------
    stats     : pd.DataFrame — output of compute_descriptive_stats()
    save_path : Path | None  — override default INTERPRETATION_HEATMAP_PNG
    """
    import seaborn as sns

    _banner("Step 4 — Cluster Heatmap  (normalised feature means)")

    try:
        mean_df = stats.xs("mean", axis=1, level=1).copy()
    except KeyError:
        mean_df = stats.copy()

    # Normalise each column to [0, 1]
    norm = (mean_df - mean_df.min()) / (mean_df.max() - mean_df.min() + 1e-9)
    norm.index = [f"Cluster {k}" for k in norm.index]

    fig, ax = plt.subplots(figsize=(max(10, len(norm.columns) * 1.1), 4))
    sns.heatmap(
        norm,
        annot=mean_df.round(1),    # show raw means as annotations
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Normalised mean (0 = lowest cluster, 1 = highest)"},
    )
    ax.set_title(
        "Cluster Feature Means — Normalised Heatmap",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Cluster")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()

    path = save_path if save_path is not None else INTERPRETATION_HEATMAP_PNG
    _save_fig(fig, path)


# ── Step 5 — Radar Chart ─────────────────────────────────────────────────────

def plot_radar_chart(
    stats: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """
    Draw a radar / spider chart comparing normalised cluster means across
    all numeric features.

    Parameters
    ----------
    stats     : pd.DataFrame — output of compute_descriptive_stats()
    save_path : Path | None  — override default INTERPRETATION_RADAR_PNG
    """
    _banner("Step 5 — Radar Chart  (cluster profiles)")

    try:
        mean_df = stats.xs("mean", axis=1, level=1).copy()
    except KeyError:
        mean_df = stats.copy()

    # Normalise columns to [0, 1]
    norm = (mean_df - mean_df.min()) / (mean_df.max() - mean_df.min() + 1e-9)

    features   = list(norm.columns)
    n_features = len(features)
    n_clusters = len(norm)

    if n_features < 3:
        logger.warning("Radar chart requires ≥ 3 features; skipping.")
        return

    # Angles for each feature spoke (evenly spaced, closed loop)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    palette = plt.cm.tab10.colors[:n_clusters]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for k in range(n_clusters):
        values = norm.iloc[k].tolist()
        values += values[:1]    # close polygon
        ax.plot(angles, values, "o-", linewidth=2, color=palette[k], label=f"Cluster {k}")
        ax.fill(angles, values, alpha=0.12, color=palette[k])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [f.replace("_", "\n") for f in features],
        fontsize=9,
    )
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="grey")
    ax.set_title(
        "Cluster Profiles — Radar Chart\n(normalised feature means)",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.30, 1.10),
        title="Cluster",
        framealpha=0.8,
    )
    fig.tight_layout()

    path = save_path if save_path is not None else INTERPRETATION_RADAR_PNG
    _save_fig(fig, path)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_full_interpretation(
    df: pd.DataFrame,
    labels: np.ndarray,
    numeric_cols: list[str] | None = None,
    export_path: Path | None = None,
) -> dict:
    """
    Execute the complete Cluster Interpretation pipeline.

    Steps
    -----
    1. compute_descriptive_stats  — mean / min / max / std / median per cluster
    2. describe_cluster_profiles  — human-readable characteristics per cluster
    3. export_labeled_dataset     — df + "Target" column → CSV
    4. plot_interpretation_heatmap — normalised heatmap of cluster means
    5. plot_radar_chart            — spider chart comparing cluster shapes

    Parameters
    ----------
    df           : pd.DataFrame     — preprocessed dataset (unscaled)
    labels       : np.ndarray       — cluster label per row
    numeric_cols : list[str] | None — override feature list for descriptive stats
    export_path  : Path | None      — override path for labeled CSV export

    Returns
    -------
    dict with keys:
        descriptive_stats  : pd.DataFrame  — aggregated stats table
        cluster_profiles   : dict[int, str] — narrative per cluster
        labeled_df         : pd.DataFrame  — df with "Target" column
    """
    _banner("Cluster Interpretation Pipeline  —  START")

    # 1. Descriptive statistics
    stats = compute_descriptive_stats(df, labels, numeric_cols=numeric_cols)

    # 2. Profile descriptions
    profiles = describe_cluster_profiles(stats, df, labels)

    # 3. Export labeled dataset
    labeled_df = export_labeled_dataset(df, labels, save_path=export_path)

    # 4. Heatmap
    plot_interpretation_heatmap(stats)

    # 5. Radar chart
    plot_radar_chart(stats)

    _banner("Cluster Interpretation Pipeline  —  COMPLETE")
    logger.info("Outputs written to: %s", INTERPRETATION_STATS_CSV.parent)

    return {
        "descriptive_stats": stats,
        "cluster_profiles":  profiles,
        "labeled_df":        labeled_df,
    }
