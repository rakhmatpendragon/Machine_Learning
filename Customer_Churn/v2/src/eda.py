"""
eda.py — Exploratory Data Analysis.
Uses banner() and save_fig() from src.utils (no local copies).
"""
from __future__ import annotations

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (DESCRIBE_PERCENTILES, HEAD_ROWS, OUTPUT_DIR,
                         PLOT_DPI, PLOT_FIGSIZE_SQ, PLOT_FIGSIZE_WIDE, PLOT_STYLE)
from src.logger import get_logger
from src.utils import banner, save_fig          # ← single shared source

logger = get_logger(__name__)
plt.style.use(PLOT_STYLE)


def display_head(df: pd.DataFrame, n: int = HEAD_ROWS) -> pd.DataFrame:
    banner("1. Dataset Preview — head()", logger)
    head = df.head(n)
    logger.info("\n%s", head.to_string())
    return head


def display_info(df: pd.DataFrame) -> dict:
    banner("2. Dataset Information — info()", logger)
    buf = io.StringIO()
    df.info(buf=buf)
    logger.info("\n%s", buf.getvalue())
    missing_counts = df.isnull().sum()
    missing_pct    = (missing_counts / len(df) * 100).round(2)
    logger.info("Missing values:\n%s", missing_pct[missing_pct > 0].to_string() or "  None")
    return {"shape": df.shape, "dtypes": df.dtypes.to_dict(),
            "missing_counts": missing_counts.to_dict(),
            "missing_pct":    missing_pct.to_dict()}


def display_describe(df: pd.DataFrame) -> pd.DataFrame:
    banner("3. Descriptive Statistics — describe()", logger)
    numeric_desc = df.describe(percentiles=DESCRIBE_PERCENTILES)
    logger.info("\n── Numeric columns ──\n%s", numeric_desc.to_string())
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols):
        logger.info("\n── Categorical / boolean columns ──\n%s",
                    df[cat_cols].describe().to_string())
    return numeric_desc


def analyze_churn_distribution(df: pd.DataFrame) -> pd.Series:
    banner("4. Target Distribution — Churn", logger)
    dist   = df["churned"].value_counts()
    pct    = df["churned"].value_counts(normalize=True).mul(100).round(2)
    result = pd.DataFrame({"count": dist, "pct_%": pct})
    logger.info("\n%s", result.to_string())
    return dist


def analyze_numeric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    banner("5. Numeric Feature Correlations", logger)
    corr = df.select_dtypes(include="number").corr()
    logger.info("\n%s", corr.round(2).to_string())
    return corr


def analyze_churn_by_group(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    banner("6. Churn Rate by Categorical Group", logger)
    results = {}
    for col in ["gender", "region", "loyalty_tier"]:
        grp = (df.groupby(col)["churned"].agg(["sum","count"])
                 .rename(columns={"sum":"churned","count":"total"}))
        grp["churn_rate_%"] = (grp["churned"] / grp["total"] * 100).round(2)
        logger.info("\nChurn by %s:\n%s", col, grp.to_string())
        results[col] = grp
    return results


def plot_churn_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=PLOT_FIGSIZE_WIDE)
    fig.suptitle("Churn Distribution", fontsize=14, fontweight="bold")
    labels = ["Retained (0)", "Churned (1)"]
    counts = df["churned"].value_counts().sort_index()
    axes[0].pie(counts, labels=labels, autopct="%1.1f%%",
                colors=["#4CAF50","#F44336"], startangle=140)
    axes[0].set_title("Share")
    plot_df = df.copy()
    plot_df["Status"] = plot_df["churned"].map({0:"Retained",1:"Churned"})
    sns.countplot(x="Status", data=plot_df, hue="Status", legend=False,
                  palette={"Retained":"#4CAF50","Churned":"#F44336"},
                  ax=axes[1], order=["Retained","Churned"])
    axes[1].set_title("Count")
    axes[1].set_xlabel("")
    save_fig(fig, OUTPUT_DIR / "01_churn_distribution.png", logger=logger)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    num_cols = ["age","tenure_months","annual_spend_usd",
                "num_purchases","avg_order_value","last_purchase_days"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold")
    for ax, col in zip(axes.flatten(), num_cols):
        df[col].dropna().plot.hist(bins=30, ax=ax, color="#1976D2",
                                    edgecolor="white", alpha=0.85)
        ax.set_title(col)
    save_fig(fig, OUTPUT_DIR / "02_numeric_distributions.png", logger=logger)


def plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SQ)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Pearson Correlation Matrix", fontsize=13, fontweight="bold")
    save_fig(fig, OUTPUT_DIR / "03_correlation_heatmap.png", logger=logger)


def plot_churn_by_loyalty(df: pd.DataFrame) -> None:
    order = ["Bronze","Silver","Gold","Platinum"]
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SQ)
    (df.groupby("loyalty_tier")["churned"].mean().mul(100).reindex(order)
       .plot.bar(color="#FF7043", edgecolor="white", ax=ax))
    ax.set_title("Churn Rate by Loyalty Tier (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Loyalty Tier")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xticklabels(order, rotation=0)
    save_fig(fig, OUTPUT_DIR / "04_churn_by_loyalty_tier.png", logger=logger)


def plot_spend_vs_tenure(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SQ)
    colors = {0:"#4CAF50", 1:"#F44336"}
    for churn_val, grp in df.dropna(subset=["annual_spend_usd"]).groupby("churned"):
        ax.scatter(grp["tenure_months"], grp["annual_spend_usd"],
                   label=f"{'Churned' if churn_val else 'Retained'}",
                   alpha=0.4, s=18, color=colors[churn_val])
    ax.set_title("Annual Spend vs Tenure", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Annual Spend (USD)")
    ax.legend()
    save_fig(fig, OUTPUT_DIR / "05_spend_vs_tenure.png", logger=logger)


def run_full_eda(df: pd.DataFrame) -> dict:
    head    = display_head(df)
    info    = display_info(df)
    stats   = display_describe(df)
    churn   = analyze_churn_distribution(df)
    corr    = analyze_numeric_correlations(df)
    by_grp  = analyze_churn_by_group(df)
    banner("Generating Visualisations", logger)
    plot_churn_distribution(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(corr)
    plot_churn_by_loyalty(df)
    plot_spend_vs_tenure(df)
    banner("EDA Complete", logger)
    return {"head": head, "info_summary": info, "describe": stats,
            "churn_dist": churn, "correlations": corr, "churn_by_group": by_grp}
