"""
eda.py
------
Exploratory Data Analysis for the retail customer churn dataset.

Design principles
-----------------
* Each analysis concern lives in its own function → easy to test / extend.
* run_full_eda() is the single public orchestrator.
* No global state; every function takes a DataFrame and returns results.
* Plots are saved to OUTPUT_DIR (never shown interactively) so the module
  works equally well in notebooks, scripts, and CI pipelines.
"""

from __future__ import annotations

import io
import textwrap

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in any env
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    DESCRIBE_PERCENTILES,
    HEAD_ROWS,
    OUTPUT_DIR,
    PLOT_DPI,
    PLOT_FIGSIZE_SQ,
    PLOT_FIGSIZE_WIDE,
    PLOT_STYLE,
)
from src.logger import get_logger

logger = get_logger(__name__)

plt.style.use(PLOT_STYLE)

# ── Section banner helper ─────────────────────────────────────────────────────

def _banner(title: str) -> None:
    width = 70
    logger.info("")
    logger.info("=" * width)
    logger.info("  %s", title.upper())
    logger.info("=" * width)


# ── 1. head() ────────────────────────────────────────────────────────────────

def display_head(df: pd.DataFrame, n: int = HEAD_ROWS) -> pd.DataFrame:
    """
    Return and log the first *n* rows of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
    n  : int  — number of rows to display (default from config)

    Returns
    -------
    pd.DataFrame  — first n rows
    """
    _banner(f"1. Dataset Preview — head({n})")
    head = df.head(n)
    logger.info("\n%s", head.to_string())
    return head


# ── 2. info() ────────────────────────────────────────────────────────────────

def display_info(df: pd.DataFrame) -> dict:
    """
    Log df.info() output and return a structured summary dict.

    Returns
    -------
    dict with keys: shape, dtypes, missing_counts, missing_pct
    """
    _banner("2. Dataset Information — info()")

    # Capture df.info() string (it writes to a buffer by default)
    buf = io.StringIO()
    df.info(buf=buf)
    logger.info("\n%s", buf.getvalue())

    missing_counts = df.isnull().sum()
    missing_pct    = (missing_counts / len(df) * 100).round(2)

    summary = {
        "shape":          df.shape,
        "dtypes":         df.dtypes.to_dict(),
        "missing_counts": missing_counts.to_dict(),
        "missing_pct":    missing_pct.to_dict(),
    }

    logger.info("Shape          : %d rows × %d columns", *df.shape)
    logger.info(
        "Missing values :\n%s",
        missing_pct[missing_pct > 0].to_string() or "  None",
    )
    return summary


# ── 3. describe() ────────────────────────────────────────────────────────────

def display_describe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log and return descriptive statistics (numeric + categorical).

    Returns
    -------
    pd.DataFrame — describe() output for numeric columns
    """
    _banner("3. Descriptive Statistics — describe()")

    numeric_desc = df.describe(percentiles=DESCRIBE_PERCENTILES)
    logger.info("\n── Numeric columns ──\n%s", numeric_desc.to_string())

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols):
        cat_desc = df[cat_cols].describe()
        logger.info("\n── Categorical / boolean columns ──\n%s", cat_desc.to_string())

    return numeric_desc


# ── 4. Additional EDA analyses ────────────────────────────────────────────────

def analyze_churn_distribution(df: pd.DataFrame) -> pd.Series:
    """Log and return churn rate by count and percentage."""
    _banner("4. Target Distribution — Churn")
    dist = df["churned"].value_counts()
    pct  = df["churned"].value_counts(normalize=True).mul(100).round(2)
    result = pd.DataFrame({"count": dist, "pct_%": pct})
    logger.info("\n%s", result.to_string())
    return dist


def analyze_numeric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Log and return Pearson correlation matrix for numeric features."""
    _banner("5. Numeric Feature Correlations")
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    logger.info("\n%s", corr.round(2).to_string())
    return corr


def analyze_churn_by_group(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Log churn rate broken down by key categorical columns."""
    _banner("6. Churn Rate by Categorical Group")
    cat_cols  = ["gender", "region", "loyalty_tier"]
    results   = {}
    for col in cat_cols:
        grp = (
            df.groupby(col)["churned"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "churned", "count": "total"})
        )
        grp["churn_rate_%"] = (grp["churned"] / grp["total"] * 100).round(2)
        logger.info("\nChurn by %s:\n%s", col, grp.to_string())
        results[col] = grp
    return results


# ── 5. Visualisations ────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", path)


def plot_churn_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=PLOT_FIGSIZE_WIDE)
    fig.suptitle("Churn Distribution", fontsize=14, fontweight="bold")

    labels = ["Retained (0)", "Churned (1)"]
    counts = df["churned"].value_counts().sort_index()

    axes[0].pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        colors=["#4CAF50", "#F44336"],
        startangle=140,
    )
    axes[0].set_title("Share")

    plot_df = df.copy()
    plot_df["Status"] = plot_df["churned"].map({0: "Retained", 1: "Churned"})
    sns.countplot(
        x="Status", data=plot_df, hue="Status", legend=False,
        palette={"Retained": "#4CAF50", "Churned": "#F44336"}, ax=axes[1],
        order=["Retained", "Churned"],
    )
    axes[1].set_title("Count")
    axes[1].set_xlabel("")
    axes[1].set_xlabel("")
    _save(fig, "01_churn_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    num_cols = ["age", "tenure_months", "annual_spend_usd",
                "num_purchases", "avg_order_value", "last_purchase_days"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold")

    for ax, col in zip(axes.flatten(), num_cols):
        df[col].dropna().plot.hist(bins=30, ax=ax, color="#1976D2", edgecolor="white", alpha=0.85)
        ax.set_title(col)
        ax.set_xlabel("")

    plt.tight_layout()
    _save(fig, "02_numeric_distributions.png")


def plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SQ)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_correlation_heatmap.png")


def plot_churn_by_loyalty(df: pd.DataFrame) -> None:
    order  = ["Bronze", "Silver", "Gold", "Platinum"]
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SQ)
    churn_rate = (
        df.groupby("loyalty_tier")["churned"]
        .mean()
        .mul(100)
        .reindex(order)
    )
    churn_rate.plot.bar(color="#FF7043", edgecolor="white", ax=ax)
    ax.set_title("Churn Rate by Loyalty Tier (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Loyalty Tier")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xticklabels(order, rotation=0)
    plt.tight_layout()
    _save(fig, "04_churn_by_loyalty_tier.png")


def plot_spend_vs_tenure(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SQ)
    colors = {0: "#4CAF50", 1: "#F44336"}
    for churn_val, grp in df.dropna(subset=["annual_spend_usd"]).groupby("churned"):
        ax.scatter(
            grp["tenure_months"],
            grp["annual_spend_usd"],
            label=f"{'Churned' if churn_val else 'Retained'}",
            alpha=0.4,
            s=18,
            color=colors[churn_val],
        )
    ax.set_title("Annual Spend vs Tenure", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Annual Spend (USD)")
    ax.legend()
    plt.tight_layout()
    _save(fig, "05_spend_vs_tenure.png")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame) -> dict:
    """
    Execute all EDA steps in sequence.

    Parameters
    ----------
    df : pd.DataFrame — raw dataset

    Returns
    -------
    dict containing all computed artefacts (head, info summary,
    describe, distributions, correlations).
    """
    head         = display_head(df)
    info_summary = display_info(df)
    stats        = display_describe(df)
    churn_dist   = analyze_churn_distribution(df)
    corr         = analyze_numeric_correlations(df)
    churn_by_grp = analyze_churn_by_group(df)

    _banner("Generating Visualisations")
    plot_churn_distribution(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(corr)
    plot_churn_by_loyalty(df)
    plot_spend_vs_tenure(df)

    _banner("EDA Complete")
    logger.info("All outputs saved to: %s", OUTPUT_DIR)

    return {
        "head":          head,
        "info_summary":  info_summary,
        "describe":      stats,
        "churn_dist":    churn_dist,
        "correlations":  corr,
        "churn_by_group": churn_by_grp,
    }
