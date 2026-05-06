"""
tests/test_cluster_interpreter.py
----------------------------------
Unit and integration tests for src/cluster_interpreter.py

All tests use a small in-memory DataFrame for speed and isolation.
File-writing tests redirect output to tmp_path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.cluster_interpreter import (
    compute_descriptive_stats,
    describe_cluster_profiles,
    export_labeled_dataset,
    plot_interpretation_heatmap,
    plot_radar_chart,
    run_full_interpretation,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

N_ROWS     = 300
N_CLUSTERS = 4

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Fully-numeric DataFrame that mirrors preprocessor output."""
    rng = np.random.default_rng(7)
    n   = N_ROWS
    return pd.DataFrame({
        "age":                rng.integers(18, 76, size=n).astype(float),
        "tenure_months":      rng.integers(1, 121, size=n),
        "annual_spend_usd":   rng.uniform(100, 50_000, size=n).round(2),
        "num_purchases":      rng.integers(1, 101, size=n),
        "avg_order_value":    rng.uniform(10, 500, size=n).round(2),
        "returns_count":      rng.integers(0, 11, size=n),
        "last_purchase_days": rng.integers(1, 366, size=n).astype(float),
        "email_opt_in":       rng.integers(0, 2, size=n),
        "gender":             rng.integers(0, 3, size=n),
        "region":             rng.integers(0, 4, size=n),
        "loyalty_tier":       rng.integers(0, 4, size=n),
        "churned":            rng.integers(0, 2, size=n),
    })


@pytest.fixture(scope="module")
def labels(sample_df) -> np.ndarray:
    """Balanced cluster labels cycling 0…N_CLUSTERS-1."""
    return np.tile(np.arange(N_CLUSTERS), N_ROWS // N_CLUSTERS + 1)[:N_ROWS]


@pytest.fixture(scope="module")
def stats(sample_df, labels) -> pd.DataFrame:
    """Pre-computed descriptive stats (shared across stat-dependent tests)."""
    return compute_descriptive_stats(sample_df, labels)


# ── compute_descriptive_stats ─────────────────────────────────────────────────

class TestComputeDescriptiveStats:

    def test_returns_dataframe(self, stats):
        assert isinstance(stats, pd.DataFrame)

    def test_index_equals_cluster_ids(self, stats):
        n_clusters = stats.index.nunique()
        assert set(stats.index) == set(range(n_clusters))

    def test_has_mean_min_max_columns(self, stats):
        top_level_funcs = stats.columns.get_level_values(1).unique().tolist()
        for fn in ["mean", "min", "max"]:
            assert fn in top_level_funcs, f"Missing aggregation: {fn}"

    def test_mean_between_min_and_max(self, stats):
        for col in config.NUMERIC_FEATURES_FOR_ANALYSIS:
            if col not in stats.columns.get_level_values(0):
                continue
            assert (stats[col]["min"] <= stats[col]["mean"]).all()
            assert (stats[col]["mean"] <= stats[col]["max"]).all()

    def test_std_non_negative(self, stats):
        for col in config.NUMERIC_FEATURES_FOR_ANALYSIS:
            if col not in stats.columns.get_level_values(0):
                continue
            assert (stats[col]["std"] >= 0).all()

    def test_custom_numeric_cols(self, sample_df, labels):
        subset = ["age", "returns_count"]
        result = compute_descriptive_stats(sample_df, labels, numeric_cols=subset)
        feat_cols = result.columns.get_level_values(0).unique().tolist()
        assert set(feat_cols) == set(subset)

    def test_custom_agg_funcs(self, sample_df, labels):
        result = compute_descriptive_stats(
            sample_df, labels, agg_funcs=["mean", "max"]
        )
        funcs = result.columns.get_level_values(1).unique().tolist()
        assert "mean" in funcs and "max" in funcs
        assert "min" not in funcs

    def test_saves_csv(self, sample_df, labels, tmp_path):
        csv_path = tmp_path / "stats.csv"
        with patch("src.cluster_interpreter.INTERPRETATION_STATS_CSV", csv_path):
            compute_descriptive_stats(sample_df, labels)
        assert csv_path.exists() and csv_path.stat().st_size > 0

    def test_input_not_mutated(self, sample_df, labels):
        cols_before = list(sample_df.columns)
        compute_descriptive_stats(sample_df, labels)
        assert list(sample_df.columns) == cols_before


# ── describe_cluster_profiles ─────────────────────────────────────────────────

class TestDescribeClusterProfiles:

    def test_returns_dict(self, stats, sample_df, labels):
        result = describe_cluster_profiles(stats, sample_df, labels)
        assert isinstance(result, dict)

    def test_keys_equal_cluster_ids(self, stats, sample_df, labels):
        result = describe_cluster_profiles(stats, sample_df, labels)
        n = len(np.unique(labels))
        assert set(result.keys()) == set(range(n))

    def test_each_value_is_string(self, stats, sample_df, labels):
        result = describe_cluster_profiles(stats, sample_df, labels)
        for k, text in result.items():
            assert isinstance(text, str), f"Cluster {k} profile is not a string"

    def test_each_profile_mentions_cluster_id(self, stats, sample_df, labels):
        result = describe_cluster_profiles(stats, sample_df, labels)
        for k, text in result.items():
            assert f"Cluster {k}" in text

    def test_each_profile_has_label(self, stats, sample_df, labels):
        result = describe_cluster_profiles(stats, sample_df, labels)
        for k, text in result.items():
            assert "LABEL" in text, f"No LABEL in Cluster {k} profile"

    def test_each_profile_mentions_customer_count(self, stats, sample_df, labels):
        result = describe_cluster_profiles(stats, sample_df, labels)
        for k, text in result.items():
            assert "customers" in text.lower()

    def test_saves_profile_csv(self, stats, sample_df, labels, tmp_path):
        csv_path = tmp_path / "profiles.csv"
        with (
            patch("src.cluster_interpreter.INTERPRETATION_PROFILE_CSV", csv_path),
            patch("src.cluster_interpreter.INTERPRETATION_STATS_CSV", tmp_path / "stats.csv"),
        ):
            describe_cluster_profiles(stats, sample_df, labels)
        assert csv_path.exists() and csv_path.stat().st_size > 0

    def test_input_not_mutated(self, stats, sample_df, labels):
        cols_before = list(sample_df.columns)
        describe_cluster_profiles(stats, sample_df, labels)
        assert list(sample_df.columns) == cols_before


# ── export_labeled_dataset ────────────────────────────────────────────────────

class TestExportLabeledDataset:

    def test_returns_dataframe(self, sample_df, labels, tmp_path):
        result = export_labeled_dataset(sample_df, labels, save_path=tmp_path / "out.csv")
        assert isinstance(result, pd.DataFrame)

    def test_target_column_present(self, sample_df, labels, tmp_path):
        result = export_labeled_dataset(sample_df, labels, save_path=tmp_path / "out.csv")
        assert config.TARGET_COLUMN_NAME in result.columns

    def test_target_column_is_last(self, sample_df, labels, tmp_path):
        result = export_labeled_dataset(sample_df, labels, save_path=tmp_path / "out.csv")
        assert result.columns[-1] == config.TARGET_COLUMN_NAME

    def test_target_values_match_labels(self, sample_df, labels, tmp_path):
        result = export_labeled_dataset(sample_df, labels, save_path=tmp_path / "out.csv")
        np.testing.assert_array_equal(result[config.TARGET_COLUMN_NAME].values, labels)

    def test_row_count_unchanged(self, sample_df, labels, tmp_path):
        result = export_labeled_dataset(sample_df, labels, save_path=tmp_path / "out.csv")
        assert len(result) == len(sample_df)

    def test_original_columns_preserved(self, sample_df, labels, tmp_path):
        result = export_labeled_dataset(sample_df, labels, save_path=tmp_path / "out.csv")
        for col in sample_df.columns:
            assert col in result.columns

    def test_csv_created(self, sample_df, labels, tmp_path):
        csv_path = tmp_path / "labeled.csv"
        export_labeled_dataset(sample_df, labels, save_path=csv_path)
        assert csv_path.exists() and csv_path.stat().st_size > 0

    def test_csv_contains_target_column(self, sample_df, labels, tmp_path):
        csv_path = tmp_path / "labeled2.csv"
        export_labeled_dataset(sample_df, labels, save_path=csv_path)
        reloaded = pd.read_csv(csv_path)
        assert config.TARGET_COLUMN_NAME in reloaded.columns

    def test_raises_on_length_mismatch(self, sample_df, tmp_path):
        bad_labels = np.zeros(len(sample_df) + 5, dtype=int)
        with pytest.raises(ValueError, match="Length mismatch"):
            export_labeled_dataset(sample_df, bad_labels, save_path=tmp_path / "bad.csv")

    def test_input_df_not_mutated(self, sample_df, labels, tmp_path):
        cols_before = list(sample_df.columns)
        export_labeled_dataset(sample_df, labels, save_path=tmp_path / "out3.csv")
        assert list(sample_df.columns) == cols_before


# ── plot_interpretation_heatmap ───────────────────────────────────────────────

class TestPlotInterpretationHeatmap:

    def test_creates_png(self, stats, tmp_path):
        path = tmp_path / "heatmap.png"
        plot_interpretation_heatmap(stats, save_path=path)
        assert path.exists() and path.stat().st_size > 0

    def test_does_not_raise_with_single_cluster(self, sample_df):
        """Should handle edge-case of 1 cluster without crashing."""
        single_labels = np.zeros(len(sample_df), dtype=int)
        s = compute_descriptive_stats(sample_df, single_labels)
        # Should not raise
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = Path(f.name)
        try:
            plot_interpretation_heatmap(s, save_path=path)
        finally:
            if path.exists():
                os.unlink(path)


# ── plot_radar_chart ──────────────────────────────────────────────────────────

class TestPlotRadarChart:

    def test_creates_png(self, stats, tmp_path):
        path = tmp_path / "radar.png"
        plot_radar_chart(stats, save_path=path)
        assert path.exists() and path.stat().st_size > 0

    def test_handles_too_few_features_gracefully(self, sample_df, labels, tmp_path):
        """With < 3 features, function should log a warning and return."""
        s = compute_descriptive_stats(
            sample_df, labels,
            numeric_cols=["age", "returns_count"],  # only 2
        )
        path = tmp_path / "radar_skip.png"
        plot_radar_chart(s, save_path=path)
        # File should NOT be created (radar skipped)
        assert not path.exists()


# ── run_full_interpretation  (integration) ────────────────────────────────────

class TestRunFullInterpretation:

    @pytest.fixture(scope="class")
    def interp_result(self, sample_df, labels, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("interp")
        with (
            patch("src.cluster_interpreter.INTERPRETATION_STATS_CSV",  tmp / "stats.csv"),
            patch("src.cluster_interpreter.INTERPRETATION_PROFILE_CSV", tmp / "profiles.csv"),
            patch("src.cluster_interpreter.INTERPRETATION_HEATMAP_PNG", tmp / "heatmap.png"),
            patch("src.cluster_interpreter.INTERPRETATION_RADAR_PNG",   tmp / "radar.png"),
            patch("src.cluster_interpreter.LABELED_DATASET_PATH",       tmp / "labeled.csv"),
        ):
            result = run_full_interpretation(sample_df, labels)
        return result, tmp

    def test_returns_dict(self, interp_result):
        result, _ = interp_result
        assert isinstance(result, dict)

    def test_required_keys(self, interp_result):
        result, _ = interp_result
        assert {"descriptive_stats", "cluster_profiles", "labeled_df"}.issubset(result)

    def test_descriptive_stats_is_dataframe(self, interp_result):
        result, _ = interp_result
        assert isinstance(result["descriptive_stats"], pd.DataFrame)

    def test_cluster_profiles_is_dict(self, interp_result):
        result, _ = interp_result
        assert isinstance(result["cluster_profiles"], dict)

    def test_labeled_df_has_target_column(self, interp_result):
        result, _ = interp_result
        assert config.TARGET_COLUMN_NAME in result["labeled_df"].columns

    def test_labeled_df_row_count_matches_input(self, interp_result, sample_df):
        result, _ = interp_result
        assert len(result["labeled_df"]) == len(sample_df)

    def test_all_output_files_created(self, interp_result):
        _, tmp = interp_result
        assert (tmp / "stats.csv").exists()
        assert (tmp / "profiles.csv").exists()
        assert (tmp / "heatmap.png").exists()
        assert (tmp / "radar.png").exists()
        assert (tmp / "labeled.csv").exists()

    def test_input_df_not_mutated(self, interp_result, sample_df):
        assert config.TARGET_COLUMN_NAME not in sample_df.columns
