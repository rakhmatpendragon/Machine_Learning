import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    HEAD_ROWS,
    OUTPUT_DIR
)
from src.logger import get_logger
from src.utils import banner

logger = get_logger(__name__)

def display_head(df: pd.DataFrame, n: int = HEAD_ROWS) -> pd.DataFrame:
    banner(f"1. Dataset Preview — head({n})")
    head = df.head(n)
    logger.info("\n%s", head.to_string())
    return head

def display_info(df: pd.DataFrame) -> dict:
    banner("2. Dataset Information — info()")
    info = df.info()
    return info

def display_describe(df: pd.DataFrame) -> dict:
    banner("3. Descriptive Statistics — describe()")
    describe = df.describe()
    logger.info("\n%s", describe.to_string())
    return describe

def display_describe_all(df: pd.DataFrame) -> dict:
    banner("4. Descriptive Statistics All - describe(include=all)")
    describe_all = df.describe(include="all")
    logger.info("\n%s", describe_all.to_string())
    return describe_all

def display_correlation_matrix(df: pd.DataFrame, n: int = HEAD_ROWS) -> dict:
    banner("5. Correlation Matriks")

    numeric_cols_eda = df.select_dtypes(include=[n]).columns.to_list
    corr_matrix = df[numeric_cols_eda].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", centar=0,
        linewidths=0.5, linecolor="white", square=True, ax=ax, annot_kws={"size": 10}
    )

    ax.set_title("Correlation Matrixs Feature Numeric", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", labelsize=9, rotation=30)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=120, bbox_inches="tight")
    plt.show()
    logger.info("Correlation Matrix showed success")

    return corr_matrix

def display_histogram(df: pd.DataFrame) -> None:
    id_date_cols = ['TransactionID', 'AccountID', 'DeviceID', 'IPAddress', 'MerchantID', 'TransactionDate']
    cols_viz = [c for c in df.columns if c not in id_date_cols]
    n_plots  = len(cols_viz)
    n_rows   = (n_plots + 2) // 3

    palette = ['#2196F3','#E91E63','#4CAF50','#FF9800','#9C27B0','#00BCD4']

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cols_viz):
        ax = axes[i]
        color = palette[i % len(palette)]

        if df[col].dtype == 'object':
            counts = df[col].value_counts(dropna=False)
            bars = ax.bar(counts.index.astype(str), counts.values, color=color, edgecolor='white', linewidth=0.8)
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_ylim(0, counts.max() * 1.22)
        else:
            ax.hist(df[col].dropna(), bins=20, color=color, edgecolor='white', linewidth=0.4, alpha=0.9)
            mean_val = df[col].mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
            ax.legend(fontsize=8, framealpha=0.8)

        ax.set_title(f'Distribusi: {col}', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel('Frekuensi', fontsize=9)
        ax.tick_params(axis='x', labelsize=8, rotation=15)
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Sembunyikan subplot kosong
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Distribusi Semua Fitur Dataset (Tanpa Kolom ID/Date)',
                fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout(pad=2.5)
    plt.savefig('histogram_semua_kolom.png', dpi=120, bbox_inches='tight')
    plt.show()
    logger.info('Histogram semua kolom berhasil ditampilkan (tanpa label overlap).')
    

def run_full_eda(df: pd.DataFrame) -> dict:
    head         = display_head(df)
    info_summary = display_info(df)
    stats        = display_describe(df)
    all_stats    = display_describe_all(df)
    # corr_matrix  = display_correlation_matrix(df)

    banner("EDA Complete")
    logger.info("All outputs saved to: %s", OUTPUT_DIR)

    return {
        "head":         head,
        "info_summary": info_summary,
        "describe":     stats,
        "describe_all": all_stats,
    }