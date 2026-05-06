"""
tests/test_data_loader.py
--------------------------
Unit tests for src/data_loader.py
"""

import numpy as np
import pandas as pd
import pytest

# Patch the dataset path so tests never touch the real filesystem
from unittest.mock import patch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import _generate_synthetic_data, load_dataset
from src import config


# ── _generate_synthetic_data ─────────────────────────────────────────────────

class TestGenerateSyntheticData:

    def setup_method(self):
        self.df = _generate_synthetic_data()

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_row_count(self):
        assert len(self.df) == config.SYNTHETIC_N_ROWS + config.SYNTHETIC_DUPE_ROWS

    def test_expected_columns(self):
        expected = {
            "customer_id", "age", "gender", "region", "tenure_months",
            "annual_spend_usd", "num_purchases", "avg_order_value",
            "returns_count", "loyalty_tier", "last_purchase_days",
            "email_opt_in", "churned",
        }
        assert expected.issubset(set(self.df.columns))

    def test_churned_is_binary(self):
        assert set(self.df["churned"].unique()).issubset({0, 1})

    def test_age_range(self):
        valid = self.df["age"].dropna()
        assert valid.between(18, 75).all()

    def test_tenure_range(self):
        assert self.df["tenure_months"].between(1, 120).all()

    def test_loyalty_tiers(self):
        valid_tiers = {"Bronze", "Silver", "Gold", "Platinum"}
        assert set(self.df["loyalty_tier"].unique()).issubset(valid_tiers)

    def test_customer_ids_unique(self):
        assert self.df["customer_id"].nunique() == config.SYNTHETIC_N_ROWS

    def test_missing_values_injected(self):
        # At least one of the three columns should have some NaNs
        cols_with_nulls = ["age", "annual_spend_usd", "last_purchase_days"]
        total_missing = self.df[cols_with_nulls].isnull().sum().sum()
        assert total_missing > 0

    def test_reproducible_with_same_seed(self):
        df2 = _generate_synthetic_data()
        pd.testing.assert_frame_equal(self.df, df2)


# ── load_dataset (synthetic path — no real file) ─────────────────────────────

class TestLoadDataset:

    def test_load_returns_dataframe(self, tmp_path):
        """When no CSV exists, load_dataset() should generate synthetic data."""
        fake_path = tmp_path / "nonexistent.csv"
        with patch.object(config, "DATASET_PATH", fake_path):
            # Also redirect the save location
            with patch("src.data_loader.DATASET_PATH", fake_path):
                df = load_dataset()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == config.SYNTHETIC_N_ROWS + config.SYNTHETIC_DUPE_ROWS

    def test_load_from_csv(self, tmp_path):
        """When a CSV exists, it should be loaded instead of generating data."""
        sample = pd.DataFrame({
            "customer_id": ["C00001"],
            "age": [30],
            "churned": [0],
        })
        csv_path = tmp_path / "retail_customers.csv"
        sample.to_csv(csv_path, index=False)

        with patch("src.data_loader.DATASET_PATH", csv_path):
            df = load_dataset()

        assert len(df) == 1
        assert "churned" in df.columns
