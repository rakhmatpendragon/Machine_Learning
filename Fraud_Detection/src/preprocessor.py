import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils import _banner
from src.logger import get_logger
from src.config import (
    COLUMNS_TO_DROP,
    CATEGORICAL_COLUMNS,
    PROCESSED_DATASET_PATH,
)

logger = get_logger(__name__)

def audit_quality(df: pd.DataFrame) -> dict:
    _banner("Step 1 — Data Quality Audit")

    missing_per_column: pd.Series = df.isnull().sum()
    total_missing: int            = int(missing_per_column.sum())
    duplicate_row: int            = int(df.duplicated().sum())

    logger.info("—— isnull().sum() ——")
    logger.info("\n%s", missing_per_column.to_string())
    logger.info("")
    logger.info("Total missing cells : %d", total_missing)
    logger.info("Duplicate rows      : %d  (out of %d)", duplicate_row, len(df))

    return {
        "missing_per_column": missing_per_column,
        "total_missing"     : total_missing,
        "duplicate_rows"    : duplicate_row,
    }

def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    _banner("Step 2 — Handle Missing Values  (dropna)")    

    before  = len(df)
    cleaned = df.dropna().reset_index(drop=True)
    remove  = before - len(cleaned)

    logger.info("Rows before : %d", before)
    logger.info("Rows removed: %d  (%.1f%%)", remove, remove / before)
    logger.info("Rows after  : %d", len(cleaned))

    return cleaned

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    _banner("Step 3 — Remove Duplicate Rows  (drop_duplicates)")

    before  = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)

    logger.info("Rows before : %d", before)
    logger.info("Duplicates  : %d  (%.1f%%)", removed, removed / before)
    logger.info("Rows after  : %d", len(cleaned))

    return cleaned

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    _banner("Step 4 — Drop ID / Address / Date Columns")

    present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    absent  = [c for c in COLUMNS_TO_DROP if c not in df.columns]

    if absent:
        logger.info("Columns not found (skipped): %s", absent)

    cleaned = df.drop(columns=present)

    logger.info("Dropped    : %s", present)
    logger.info("Remaining: %d columns → %s", len(cleaned.columns), list(cleaned.columns))

    return cleaned

def label_encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    _banner("Step 5 — Label Encode Categorical Features  (LabelEncoder)")

    encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    bool_cols = encoded.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        encoded[col] = encoded[col].astype(int)
        logger.info("Bool cast  : %-25s  False→0  True→1", col)

    present_cats = [c for c in CATEGORICAL_COLUMNS if c in encoded.columns]
    absent_cats  = [c for c in CATEGORICAL_COLUMNS if c not in encoded.columns]

    if absent_cats:
        logger.info("Categorical columns not fount(skipped): %s", absent_cats)

    for col in present_cats:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(encoded[col].astype(str))
        encoders[col] = le
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        logger.info("Encoded    : %-25s mapping → %s", col, mapping)

    logger.info("")
    logger.info("Final dtypes after encoding:")
    logger.info("\n%s", encoded.dtypes.to_string())

    return encoded, encoders

def run_full_preprocessing(
        df: pd.DataFrame,
        persist: bool = True,
) -> dict:
    _banner("Data Cleaning & Preprocessing Pipeline  —  START")
    shape_log: list[tuple[str, int, int]] = [("raw", *df.shape)]

    audit = audit_quality(df)

    df_step = drop_missing(df)
    shape_log.append(("after_dropna", *df_step.shape))

    df_step = remove_duplicates(df_step)
    shape_log.append(("after_drop_duplicates", *df_step.shape))

    df_step = drop_columns(df_step)
    shape_log.append(("after_drop_cols", *df_step.shape))

    df_encoded, encoders = label_encode(df_step)
    shape_log.append(("after encoding", *df_encoded.shape))

    _banner("Preprocessing Summary")
    logger.info("%-30s %6s %6s", "Stage", "Rows", "Cols")
    logger.info("-" * 46)
    for stage, rows, cols in shape_log:
        logger.info("%-30s %6d %6d", stage, rows, cols)

    if persist:
        PROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_encoded.to_csv(PROCESSED_DATASET_PATH, index=False)
        logger.info("")
        logger.info("Processed dataset saved → %s", PROCESSED_DATASET_PATH)

    _banner("Preprocessing Pipeline  — COMPLETE")

    return {
        "audit":        audit,
        "df_cleaned":   df_encoded,
        "encoders":     encoders,
        "shape_log":    shape_log,
    }
