"""
tests/test_preprocessor.py
---------------------------
Unit tests for src/preprocessor.py
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import _generate_synthetic_data
from src.preprocessor import (
    audit_quality,
    drop_missing,
    remove_duplicates,
    drop_id_columns,
    encode_categoricals,
    run_full_preprocessing,
)
from src import config


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Full synthetic raw dataset (1 000 base rows + 20 dupes)."""
    return _generate_synthetic_data()


# ── audit_quality ─────────────────────────────────────────────────────────────

class TestAuditQuality:

    def test_returns_dict(self, raw_df):
        result = audit_quality(raw_df)
        assert isinstance(result, dict)

    def test_required_keys(self, raw_df):
        result = audit_quality(raw_df)
        assert {"missing_per_column", "total_missing", "duplicate_rows"}.issubset(result)

    def test_total_missing_is_int(self, raw_df):
        result = audit_quality(raw_df)
        assert isinstance(result["total_missing"], int)
        assert result["total_missing"] >= 0

    def test_detects_injected_duplicates(self, raw_df):
        result = audit_quality(raw_df)
        assert result["duplicate_rows"] == config.SYNTHETIC_DUPE_ROWS

    def test_detects_missing_values(self, raw_df):
        result = audit_quality(raw_df)
        assert result["total_missing"] > 0

    def test_does_not_mutate_df(self, raw_df):
        shape_before = raw_df.shape
        audit_quality(raw_df)
        assert raw_df.shape == shape_before


# ── drop_missing ──────────────────────────────────────────────────────────────

class TestDropMissing:

    def test_no_nulls_remain(self, raw_df):
        cleaned = drop_missing(raw_df)
        assert cleaned.isnull().sum().sum() == 0

    def test_returns_new_dataframe(self, raw_df):
        cleaned = drop_missing(raw_df)
        assert cleaned is not raw_df

    def test_row_count_reduced(self, raw_df):
        cleaned = drop_missing(raw_df)
        assert len(cleaned) < len(raw_df)

    def test_columns_unchanged(self, raw_df):
        cleaned = drop_missing(raw_df)
        assert list(cleaned.columns) == list(raw_df.columns)

    def test_idempotent(self, raw_df):
        once  = drop_missing(raw_df)
        twice = drop_missing(once)
        assert len(once) == len(twice)

    def test_input_not_mutated(self, raw_df):
        nulls_before = raw_df.isnull().sum().sum()
        drop_missing(raw_df)
        assert raw_df.isnull().sum().sum() == nulls_before


# ── remove_duplicates ─────────────────────────────────────────────────────────

class TestRemoveDuplicates:

    def test_no_duplicates_remain(self, raw_df):
        cleaned = remove_duplicates(raw_df)
        assert cleaned.duplicated().sum() == 0

    def test_returns_new_dataframe(self, raw_df):
        cleaned = remove_duplicates(raw_df)
        assert cleaned is not raw_df

    def test_row_count_reduced_by_dupe_count(self, raw_df):
        dupes   = int(raw_df.duplicated().sum())
        cleaned = remove_duplicates(raw_df)
        assert len(cleaned) == len(raw_df) - dupes

    def test_idempotent(self, raw_df):
        once  = remove_duplicates(raw_df)
        twice = remove_duplicates(once)
        assert len(once) == len(twice)

    def test_input_not_mutated(self, raw_df):
        dupes_before = raw_df.duplicated().sum()
        remove_duplicates(raw_df)
        assert raw_df.duplicated().sum() == dupes_before


# ── drop_id_columns ───────────────────────────────────────────────────────────

class TestDropIdColumns:

    def test_id_columns_removed(self, raw_df):
        cleaned = drop_id_columns(raw_df)
        for col in config.COLUMNS_TO_DROP:
            assert col not in cleaned.columns

    def test_feature_columns_preserved(self, raw_df):
        feature_cols = [c for c in raw_df.columns if c not in config.COLUMNS_TO_DROP]
        cleaned      = drop_id_columns(raw_df)
        for col in feature_cols:
            assert col in cleaned.columns

    def test_returns_new_dataframe(self, raw_df):
        cleaned = drop_id_columns(raw_df)
        assert cleaned is not raw_df

    def test_row_count_unchanged(self, raw_df):
        cleaned = drop_id_columns(raw_df)
        assert len(cleaned) == len(raw_df)

    def test_tolerates_missing_columns(self):
        """Should not raise if a configured drop-column is absent."""
        mini = pd.DataFrame({"age": [25, 30], "churned": [0, 1]})
        result = drop_id_columns(mini)
        assert list(result.columns) == ["age", "churned"]

    def test_input_not_mutated(self, raw_df):
        cols_before = list(raw_df.columns)
        drop_id_columns(raw_df)
        assert list(raw_df.columns) == cols_before


# ── encode_categoricals ───────────────────────────────────────────────────────

class TestEncodeCategoricals:

    @pytest.fixture(scope="class")
    def clean_no_ids(self, raw_df):
        """Pipeline up to step 4 — ready for encoding."""
        df = drop_missing(raw_df)
        df = remove_duplicates(df)
        df = drop_id_columns(df)
        return df

    def test_returns_tuple(self, clean_no_ids):
        result = encode_categoricals(clean_no_ids)
        assert isinstance(result, tuple) and len(result) == 2

    def test_encoded_df_is_dataframe(self, clean_no_ids):
        encoded, _ = encode_categoricals(clean_no_ids)
        assert isinstance(encoded, pd.DataFrame)

    def test_encoders_dict_has_correct_keys(self, clean_no_ids):
        _, encoders = encode_categoricals(clean_no_ids)
        for col in config.CATEGORICAL_COLUMNS:
            if col in clean_no_ids.columns:
                assert col in encoders

    def test_categorical_columns_are_numeric(self, clean_no_ids):
        encoded, _ = encode_categoricals(clean_no_ids)
        for col in config.CATEGORICAL_COLUMNS:
            if col in encoded.columns:
                assert pd.api.types.is_numeric_dtype(encoded[col]), \
                    f"{col} should be numeric after encoding"

    def test_bool_columns_cast_to_int(self, clean_no_ids):
        encoded, _ = encode_categoricals(clean_no_ids)
        bool_cols  = clean_no_ids.select_dtypes(include="bool").columns
        for col in bool_cols:
            assert encoded[col].dtype in [int, "int32", "int64"]

    def test_row_count_unchanged(self, clean_no_ids):
        encoded, _ = encode_categoricals(clean_no_ids)
        assert len(encoded) == len(clean_no_ids)

    def test_inverse_transform_restores_labels(self, clean_no_ids):
        encoded, encoders = encode_categoricals(clean_no_ids)
        for col, le in encoders.items():
            restored = le.inverse_transform(encoded[col])
            original = clean_no_ids[col].astype(str).values
            assert (restored == original).all(), f"Inverse transform failed for {col}"

    def test_input_not_mutated(self, clean_no_ids):
        dtypes_before = clean_no_ids.dtypes.to_dict()
        encode_categoricals(clean_no_ids)
        assert clean_no_ids.dtypes.to_dict() == dtypes_before


# ── run_full_preprocessing (integration) ──────────────────────────────────────

class TestRunFullPreprocessing:

    @pytest.fixture(scope="class")
    def result(self, raw_df, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("data")
        import src.config as cfg
        original_path = cfg.PROCESSED_DATASET_PATH
        cfg.PROCESSED_DATASET_PATH = tmp / "processed.csv"
        out = run_full_preprocessing(raw_df, persist=True)
        cfg.PROCESSED_DATASET_PATH = original_path
        return out

    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    def test_required_keys(self, result):
        assert {"audit", "df_cleaned", "encoders", "shape_log"}.issubset(result)

    def test_df_cleaned_has_no_nulls(self, result):
        assert result["df_cleaned"].isnull().sum().sum() == 0

    def test_df_cleaned_has_no_duplicates(self, result):
        assert result["df_cleaned"].duplicated().sum() == 0

    def test_id_columns_absent(self, result):
        for col in config.COLUMNS_TO_DROP:
            assert col not in result["df_cleaned"].columns

    def test_all_columns_numeric(self, result):
        for col in result["df_cleaned"].columns:
            assert pd.api.types.is_numeric_dtype(result["df_cleaned"][col]), \
                f"Column '{col}' is not numeric after preprocessing"

    def test_shape_log_has_five_stages(self, result):
        assert len(result["shape_log"]) == 5

    def test_rows_decrease_monotonically_until_encoding(self, result):
        # rows should never increase from raw → dropna → drop_dupes
        rows = [r for _, r, _ in result["shape_log"][:3]]
        assert rows == sorted(rows, reverse=True)

    def test_raw_df_not_mutated(self, raw_df, result):
        # raw_df should still have its original nulls and duplicates
        assert raw_df.isnull().sum().sum() > 0
        assert raw_df.duplicated().sum() > 0
