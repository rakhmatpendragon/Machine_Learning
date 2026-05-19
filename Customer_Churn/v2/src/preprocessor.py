"""
preprocessor.py — Data Cleaning & Preprocessing pipeline.
Uses banner() from src.utils (no local copy).
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (CATEGORICAL_COLUMNS, COLUMNS_TO_DROP,
                         PROCESSED_DATASET_PATH)
from src.logger import get_logger
from src.utils import banner                    # ← single shared source

logger = get_logger(__name__)


def audit_quality(df: pd.DataFrame) -> dict:
    banner("Step 1 — Data Quality Audit", logger)
    missing = df.isnull().sum()
    dupes   = int(df.duplicated().sum())
    logger.info("── isnull().sum() ──\n%s", missing.to_string())
    logger.info("Total missing : %d  |  Duplicate rows : %d", missing.sum(), dupes)
    return {"missing_per_column": missing,
            "total_missing": int(missing.sum()),
            "duplicate_rows": dupes}


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    banner("Step 2 — Handle Missing Values  (dropna)", logger)
    before  = len(df)
    cleaned = df.dropna().reset_index(drop=True)
    removed = before - len(cleaned)
    logger.info("Rows before/removed/after : %d / %d / %d", before, removed, len(cleaned))
    return cleaned


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    banner("Step 3 — Remove Duplicate Rows  (drop_duplicates)", logger)
    before  = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    logger.info("Rows before/removed/after : %d / %d / %d", before, removed, len(cleaned))
    return cleaned


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    banner("Step 4 — Drop ID / Address / Date Columns", logger)
    present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    absent  = [c for c in COLUMNS_TO_DROP if c not in df.columns]
    if absent:
        logger.info("Not found (skipped): %s", absent)
    cleaned = df.drop(columns=present)
    logger.info("Dropped: %s  |  Remaining columns: %d", present, len(cleaned.columns))
    return cleaned


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    banner("Step 5 — Label Encode Categorical Features  (LabelEncoder)", logger)
    encoded  = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in encoded.select_dtypes(include="bool").columns:
        encoded[col] = encoded[col].astype(int)
        logger.info("Bool cast  : %s → int", col)

    for col in [c for c in CATEGORICAL_COLUMNS if c in encoded.columns]:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(encoded[col].astype(str))
        encoders[col] = le
        logger.info("Encoded    : %-20s  %s", col,
                    dict(zip(le.classes_, le.transform(le.classes_))))

    logger.info("Final dtypes:\n%s", encoded.dtypes.to_string())
    return encoded, encoders


def run_full_preprocessing(df: pd.DataFrame, persist: bool = True) -> dict:
    banner("Preprocessing Pipeline  —  START", logger)
    shape_log = [("raw", *df.shape)]

    audit    = audit_quality(df)
    df_step  = drop_missing(df);       shape_log.append(("after_dropna",          *df_step.shape))
    df_step  = remove_duplicates(df_step); shape_log.append(("after_drop_duplicates", *df_step.shape))
    df_step  = drop_id_columns(df_step);   shape_log.append(("after_drop_id_cols",    *df_step.shape))
    df_enc, encoders = encode_categoricals(df_step)
    shape_log.append(("after_encoding", *df_enc.shape))

    banner("Preprocessing Summary", logger)
    logger.info("%-30s  %6s  %6s", "Stage", "Rows", "Cols")
    logger.info("-" * 46)
    for stage, rows, cols in shape_log:
        logger.info("%-30s  %6d  %6d", stage, rows, cols)

    if persist:
        PROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_enc.to_csv(PROCESSED_DATASET_PATH, index=False)
        logger.info("Processed dataset saved → %s", PROCESSED_DATASET_PATH)

    banner("Preprocessing Pipeline  —  COMPLETE", logger)
    return {"audit": audit, "df_cleaned": df_enc,
            "encoders": encoders, "shape_log": shape_log}
