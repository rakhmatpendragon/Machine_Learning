"""
preprocessor.py
---------------
Data Cleaning and Preprocessing pipeline for the Retail Customer Churn dataset.

Steps (in order)
----------------
1. audit_quality()        — isnull().sum() + duplicated().sum()  (read-only report)
2. drop_missing()         — dropna()  on feature rows with nulls
3. drop_duplicates()      — drop_duplicates() on the full row
4. drop_id_columns()      — remove ID / address / date columns
5. encode_categoricals()  — LabelEncoder on every categorical column
6. run_full_preprocessing() — orchestrates all steps and returns cleaned DataFrame

Design principles
-----------------
* Every step is a pure function: takes a DataFrame, returns a new DataFrame.
* No in-place mutation — callers always receive a fresh copy.
* LabelEncoder objects are returned so callers can inspect or inverse-transform.
* All constants (column names) come from config — nothing is hard-coded here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    CATEGORICAL_COLUMNS,
    COLUMNS_TO_DROP,
    PROCESSED_DATASET_PATH,
)
from src.logger import get_logger

if TYPE_CHECKING:
    pass   # keep imports clean for type checkers

logger = get_logger(__name__)


# ── Section banner helper ─────────────────────────────────────────────────────

def _banner(title: str) -> None:
    width = 70
    logger.info("")
    logger.info("=" * width)
    logger.info("  %s", title.upper())
    logger.info("=" * width)


# ── Step 1 — Quality Audit ────────────────────────────────────────────────────

def audit_quality(df: pd.DataFrame) -> dict:
    """
    Report missing values and duplicate rows without modifying the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame — raw dataset

    Returns
    -------
    dict with keys:
        missing_per_column : pd.Series  — null count per column
        total_missing      : int        — total null cells
        duplicate_rows     : int        — number of fully duplicate rows
    """
    _banner("Step 1 — Data Quality Audit")

    missing_per_column: pd.Series = df.isnull().sum()
    total_missing: int             = int(missing_per_column.sum())
    duplicate_rows: int            = int(df.duplicated().sum())

    logger.info("── isnull().sum() ──")
    logger.info("\n%s", missing_per_column.to_string())
    logger.info("")
    logger.info("Total missing cells : %d", total_missing)
    logger.info("Duplicate rows      : %d  (out of %d)", duplicate_rows, len(df))

    return {
        "missing_per_column": missing_per_column,
        "total_missing":      total_missing,
        "duplicate_rows":     duplicate_rows,
    }


# ── Step 2 — Drop Missing Values ─────────────────────────────────────────────

def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that contain at least one null value (dropna).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame — copy with null rows removed
    """
    _banner("Step 2 — Handle Missing Values  (dropna)")

    before = len(df)
    cleaned = df.dropna().reset_index(drop=True)
    removed = before - len(cleaned)

    logger.info("Rows before : %d", before)
    logger.info("Rows removed: %d  (%.1f%%)", removed, removed / before * 100)
    logger.info("Rows after  : %d", len(cleaned))

    return cleaned


# ── Step 3 — Remove Duplicates ────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fully duplicate rows (drop_duplicates).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame — copy with duplicate rows removed
    """
    _banner("Step 3 — Remove Duplicate Rows  (drop_duplicates)")

    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)

    logger.info("Rows before : %d", before)
    logger.info("Duplicates  : %d  (%.1f%%)", removed, removed / before * 100)
    logger.info("Rows after  : %d", len(cleaned))

    return cleaned


# ── Step 4 — Drop ID / Address / Date Columns ─────────────────────────────────

def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that carry no predictive value:
    IDs, addresses, and date fields (defined in config.COLUMNS_TO_DROP).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame — copy without the dropped columns
    """
    _banner("Step 4 — Drop ID / Address / Date Columns")

    present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    absent  = [c for c in COLUMNS_TO_DROP if c not in df.columns]

    if absent:
        logger.info("Columns not found (skipped): %s", absent)

    cleaned = df.drop(columns=present)

    logger.info("Dropped  : %s", present)
    logger.info("Remaining: %d columns → %s", len(cleaned.columns), list(cleaned.columns))

    return cleaned


# ── Step 5 — Label Encode Categorical Columns ─────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Apply sklearn LabelEncoder to each categorical column defined in
    config.CATEGORICAL_COLUMNS.  Boolean columns are cast to int (0/1).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[pd.DataFrame, dict[str, LabelEncoder]]
        encoded_df  — DataFrame with encoded columns replacing originals
        encoders    — mapping {column_name: fitted LabelEncoder}
                      Use encoder.inverse_transform([...]) to decode later.
    """
    _banner("Step 5 — Label Encode Categorical Features  (LabelEncoder)")

    encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    # Handle boolean columns first (email_opt_in → 0/1)
    bool_cols = encoded.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        encoded[col] = encoded[col].astype(int)
        logger.info("Bool cast  : %-25s  False→0  True→1", col)

    # Label-encode configured categorical columns
    present_cats = [c for c in CATEGORICAL_COLUMNS if c in encoded.columns]
    absent_cats  = [c for c in CATEGORICAL_COLUMNS if c not in encoded.columns]

    if absent_cats:
        logger.info("Categorical columns not found (skipped): %s", absent_cats)

    for col in present_cats:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(encoded[col].astype(str))
        encoders[col] = le
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        logger.info("Encoded    : %-25s  mapping → %s", col, mapping)

    logger.info("")
    logger.info("Final dtypes after encoding:")
    logger.info("\n%s", encoded.dtypes.to_string())

    return encoded, encoders


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_full_preprocessing(
    df: pd.DataFrame,
    persist: bool = True,
) -> dict:
    """
    Execute the complete Data Cleaning & Preprocessing pipeline.

    Steps
    -----
    1. audit_quality       — report nulls & duplicates (read-only)
    2. drop_missing        — dropna()
    3. remove_duplicates   — drop_duplicates()
    4. drop_id_columns     — remove ID / address / date columns
    5. encode_categoricals — LabelEncoder for categorical features

    Parameters
    ----------
    df      : pd.DataFrame — raw dataset (not modified)
    persist : bool         — if True, saves processed CSV to PROCESSED_DATASET_PATH

    Returns
    -------
    dict with keys:
        audit       : quality audit report (pre-cleaning)
        df_cleaned  : cleaned + encoded DataFrame
        encoders    : {column: LabelEncoder} for inverse-transform
        shape_log   : list of (step, rows, cols) tuples for traceability
    """
    _banner("Data Cleaning & Preprocessing Pipeline  —  START")
    shape_log: list[tuple[str, int, int]] = [("raw", *df.shape)]

    # 1. Audit (read-only — must run on the raw df)
    audit = audit_quality(df)

    # 2. Drop nulls
    df_step = drop_missing(df)
    shape_log.append(("after_dropna", *df_step.shape))

    # 3. Drop duplicates
    df_step = remove_duplicates(df_step)
    shape_log.append(("after_drop_duplicates", *df_step.shape))

    # 4. Drop ID / date columns
    df_step = drop_id_columns(df_step)
    shape_log.append(("after_drop_id_cols", *df_step.shape))

    # 5. Encode categoricals
    df_encoded, encoders = encode_categoricals(df_step)
    shape_log.append(("after_encoding", *df_encoded.shape))

    # ── Shape summary ─────────────────────────────────────────────────────────
    _banner("Preprocessing Summary")
    logger.info("%-30s  %6s  %6s", "Stage", "Rows", "Cols")
    logger.info("-" * 46)
    for stage, rows, cols in shape_log:
        logger.info("%-30s  %6d  %6d", stage, rows, cols)

    # ── Persist ───────────────────────────────────────────────────────────────
    if persist:
        PROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_encoded.to_csv(PROCESSED_DATASET_PATH, index=False)
        logger.info("")
        logger.info("Processed dataset saved → %s", PROCESSED_DATASET_PATH)

    _banner("Preprocessing Pipeline  —  COMPLETE")

    return {
        "audit":      audit,
        "df_cleaned": df_encoded,
        "encoders":   encoders,
        "shape_log":  shape_log,
    }
