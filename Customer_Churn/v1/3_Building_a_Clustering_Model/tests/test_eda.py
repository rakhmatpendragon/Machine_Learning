"""
tests/test_eda.py
-----------------
Unit tests for src/eda.py
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import _generate_synthetic_data
from src.eda import (
    display_head,
    display_info,
    display_describe,
    analyze_churn_distribution,
    analyze_numeric_correlations,
    analyze_churn_by_group,
)


@pytest.fixture(scope="module")
def sample_df():
    """Reusable small synthetic DataFrame for all EDA tests."""
    return _generate_synthetic_data()


# ── display_head ──────────────────────────────────────────────────────────────

class TestDisplayHead:

    def test_default_rows(self, sample_df):
        head = display_head(sample_df)
        assert isinstance(head, pd.DataFrame)
        assert len(head) == 10            # config.HEAD_ROWS default

    def test_custom_rows(self, sample_df):
        head = display_head(sample_df, n=5)
        assert len(head) == 5

    def test_columns_preserved(self, sample_df):
        head = display_head(sample_df)
        assert list(head.columns) == list(sample_df.columns)

    def test_does_not_mutate(self, sample_df):
        original_shape = sample_df.shape
        display_head(sample_df)
        assert sample_df.shape == original_shape


# ── display_info ──────────────────────────────────────────────────────────────

class TestDisplayInfo:

    def test_returns_dict(self, sample_df):
        result = display_info(sample_df)
        assert isinstance(result, dict)

    def test_required_keys(self, sample_df):
        result = display_info(sample_df)
        assert {"shape", "dtypes", "missing_counts", "missing_pct"}.issubset(result)

    def test_shape_correct(self, sample_df):
        result = display_info(sample_df)
        assert result["shape"] == sample_df.shape

    def test_missing_pct_between_0_and_100(self, sample_df):
        result = display_info(sample_df)
        for col, pct in result["missing_pct"].items():
            assert 0.0 <= pct <= 100.0, f"Invalid missing% for {col}: {pct}"


# ── display_describe ──────────────────────────────────────────────────────────

class TestDisplayDescribe:

    def test_returns_dataframe(self, sample_df):
        result = display_describe(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_contains_standard_stats(self, sample_df):
        result = display_describe(sample_df)
        for stat in ["mean", "std", "min", "max"]:
            assert stat in result.index

    def test_only_numeric_columns(self, sample_df):
        result = display_describe(sample_df)
        numeric_cols = sample_df.select_dtypes(include="number").columns
        for col in result.columns:
            assert col in numeric_cols


# ── analyze_churn_distribution ────────────────────────────────────────────────

class TestAnalyzeChurnDistribution:

    def test_returns_series(self, sample_df):
        result = analyze_churn_distribution(sample_df)
        assert isinstance(result, pd.Series)

    def test_values_are_binary(self, sample_df):
        result = analyze_churn_distribution(sample_df)
        assert set(result.index).issubset({0, 1})

    def test_counts_sum_to_total(self, sample_df):
        result = analyze_churn_distribution(sample_df)
        assert result.sum() == len(sample_df)


# ── analyze_numeric_correlations ──────────────────────────────────────────────

class TestAnalyzeNumericCorrelations:

    def test_returns_dataframe(self, sample_df):
        corr = analyze_numeric_correlations(sample_df)
        assert isinstance(corr, pd.DataFrame)

    def test_diagonal_is_one(self, sample_df):
        corr = analyze_numeric_correlations(sample_df)
        diag = np.diag(corr.values)
        assert np.allclose(diag, 1.0, atol=1e-6)

    def test_symmetric(self, sample_df):
        corr = analyze_numeric_correlations(sample_df)
        assert np.allclose(corr.values, corr.values.T, atol=1e-10)

    def test_values_between_minus1_and_1(self, sample_df):
        corr = analyze_numeric_correlations(sample_df)
        assert corr.values.min() >= -1.0 - 1e-9
        assert corr.values.max() <=  1.0 + 1e-9


# ── analyze_churn_by_group ────────────────────────────────────────────────────

class TestAnalyzeChurnByGroup:

    def test_returns_dict(self, sample_df):
        result = analyze_churn_by_group(sample_df)
        assert isinstance(result, dict)

    def test_expected_keys(self, sample_df):
        result = analyze_churn_by_group(sample_df)
        assert {"gender", "region", "loyalty_tier"}.issubset(result)

    def test_churn_rate_between_0_and_100(self, sample_df):
        result = analyze_churn_by_group(sample_df)
        for col, grp_df in result.items():
            assert grp_df["churn_rate_%"].between(0, 100).all(), \
                f"churn_rate_% out of range for {col}"

    def test_totals_match_dataset_length(self, sample_df):
        result = analyze_churn_by_group(sample_df)
        for col, grp_df in result.items():
            assert grp_df["total"].sum() == len(sample_df), \
                f"Row totals don't sum correctly for {col}"
