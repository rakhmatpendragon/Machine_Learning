"""
tests/test_clustering.py
-------------------------
Unit and integration tests for src/clustering.py

All tests use a small in-memory DataFrame so they run fast and never
depend on disk state.  The few tests that write files use tmp_path.
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
from src.clustering import (
    load_processed_data,
    prepare_features,
    run_elbow_method,
    train_kmeans,
    evaluate_clusters,
    plot_clusters,
    save_model,
    run_full_clustering,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """
    Small fully-numeric DataFrame that mirrors preprocessor output.
    200 rows is enough for KMeans to converge without being slow.
    """
    rng = np.random.default_rng(0)
    n   = 200
    return pd.DataFrame({
        "age":                rng.integers(18, 76, size=n).astype(float),
        "tenure_months":      rng.integers(1, 121, size=n),
        "annual_spend_usd":   rng.uniform(100, 50000, size=n).round(2),
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
def feature_cols(sample_df) -> list[str]:
    present = [c for c in config.CLUSTERING_FEATURES if c in sample_df.columns]
    return present


@pytest.fixture(scope="module")
def scaled_output(sample_df, feature_cols):
    return prepare_features(sample_df, feature_cols)


@pytest.fixture(scope="module")
def X_scaled(scaled_output):
    return scaled_output[0]


@pytest.fixture(scope="module")
def trained_model(X_scaled):
    """Fit once, share across tests that need a model."""
    return train_kmeans(X_scaled, n_clusters=3)


# ── load_processed_data ───────────────────────────────────────────────────────

class TestLoadProcessedData:

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Processed dataset not found"):
            load_processed_data(path=tmp_path / "nonexistent.csv")

    def test_loads_csv(self, tmp_path, sample_df):
        csv_path = tmp_path / "processed.csv"
        sample_df.to_csv(csv_path, index=False)
        loaded = load_processed_data(path=csv_path)
        assert isinstance(loaded, pd.DataFrame)
        assert loaded.shape == sample_df.shape

    def test_preserves_columns(self, tmp_path, sample_df):
        csv_path = tmp_path / "processed.csv"
        sample_df.to_csv(csv_path, index=False)
        loaded = load_processed_data(path=csv_path)
        assert list(loaded.columns) == list(sample_df.columns)


# ── prepare_features ──────────────────────────────────────────────────────────

class TestPrepareFeatures:

    def test_returns_tuple_of_three(self, sample_df, feature_cols):
        result = prepare_features(sample_df, feature_cols)
        assert isinstance(result, tuple) and len(result) == 3

    def test_X_is_ndarray(self, X_scaled):
        assert isinstance(X_scaled, np.ndarray)

    def test_shape_matches_rows_and_features(self, sample_df, X_scaled, feature_cols):
        assert X_scaled.shape == (len(sample_df), len(feature_cols))

    def test_mean_near_zero(self, X_scaled):
        # StandardScaler should centre each feature around 0
        assert np.abs(X_scaled.mean(axis=0)).max() < 1e-9

    def test_std_near_one(self, X_scaled):
        assert np.abs(X_scaled.std(axis=0) - 1).max() < 1e-6

    def test_raises_on_no_valid_columns(self, sample_df):
        with pytest.raises(ValueError, match="No valid feature columns"):
            prepare_features(sample_df, ["nonexistent_col"])

    def test_skips_absent_columns_gracefully(self, sample_df, feature_cols):
        cols_with_fake = feature_cols + ["__fake__"]
        X, used, _ = prepare_features(sample_df, cols_with_fake)
        assert "__fake__" not in used
        assert X.shape[1] == len(feature_cols)

    def test_input_not_mutated(self, sample_df, feature_cols):
        shape_before = sample_df.shape
        prepare_features(sample_df, feature_cols)
        assert sample_df.shape == shape_before


# ── run_elbow_method ──────────────────────────────────────────────────────────

class TestRunElbowMethod:

    def test_returns_int(self, X_scaled, tmp_path):
        k = run_elbow_method(X_scaled, k_min=2, k_max=6, save_path=tmp_path / "elbow.png")
        assert isinstance(k, int)

    def test_k_within_valid_range(self, X_scaled, tmp_path):
        k = run_elbow_method(X_scaled, k_min=2, k_max=6, save_path=tmp_path / "elbow.png")
        assert 2 <= k <= 5

    def test_saves_plot_file(self, X_scaled, tmp_path):
        plot_path = tmp_path / "elbow.png"
        run_elbow_method(X_scaled, k_min=2, k_max=5, save_path=plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0


# ── train_kmeans ──────────────────────────────────────────────────────────────

class TestTrainKmeans:

    def test_returns_tuple(self, X_scaled):
        result = train_kmeans(X_scaled, n_clusters=3)
        assert isinstance(result, tuple) and len(result) == 2

    def test_model_is_kmeans(self, trained_model):
        from sklearn.cluster import KMeans as _KMeans
        model, _ = trained_model
        assert isinstance(model, _KMeans)

    def test_labels_shape(self, X_scaled, trained_model):
        _, labels = trained_model
        assert labels.shape == (len(X_scaled),)

    def test_labels_unique_count_equals_k(self, trained_model):
        model, labels = trained_model
        assert len(np.unique(labels)) == model.n_clusters

    def test_label_values_are_valid(self, trained_model):
        model, labels = trained_model
        assert set(np.unique(labels)).issubset(set(range(model.n_clusters)))

    def test_inertia_is_positive(self, trained_model):
        model, _ = trained_model
        assert model.inertia_ > 0

    def test_model_is_fitted(self, trained_model):
        from sklearn.utils.validation import check_is_fitted
        model, _ = trained_model
        check_is_fitted(model)   # raises if not fitted


# ── evaluate_clusters ─────────────────────────────────────────────────────────

class TestEvaluateClusters:

    def test_returns_dataframe(self, sample_df, trained_model, feature_cols):
        _, labels = trained_model
        result = evaluate_clusters(sample_df, labels, feature_cols)
        assert isinstance(result, pd.DataFrame)

    def test_rows_equal_n_clusters(self, sample_df, trained_model, feature_cols):
        model, labels = trained_model
        result = evaluate_clusters(sample_df, labels, feature_cols)
        assert len(result) == model.n_clusters

    def test_count_column_sums_to_n_rows(self, sample_df, trained_model, feature_cols):
        _, labels = trained_model
        result = evaluate_clusters(sample_df, labels, feature_cols)
        # each feature has a _count column
        count_cols = [c for c in result.columns if c.endswith("_count")]
        assert len(count_cols) > 0
        # counts in any one feature_count column should sum to total rows
        assert result[count_cols[0]].sum() == len(sample_df)

    def test_input_not_mutated(self, sample_df, trained_model, feature_cols):
        _, labels = trained_model
        shape_before = sample_df.shape
        evaluate_clusters(sample_df, labels, feature_cols)
        assert sample_df.shape == shape_before


# ── plot_clusters ─────────────────────────────────────────────────────────────

class TestPlotClusters:

    def test_creates_scatter_plot(self, tmp_path, X_scaled, trained_model, sample_df, feature_cols):
        _, labels = trained_model
        scatter_path = tmp_path / "scatter.png"
        bar_path     = tmp_path / "bar.png"
        with (
            patch.object(config, "CLUSTER_PLOT_2D",  scatter_path),
            patch("src.clustering.CLUSTER_PLOT_2D",  scatter_path),
            patch.object(config, "CLUSTER_PLOT_BAR", bar_path),
            patch("src.clustering.CLUSTER_PLOT_BAR", bar_path),
        ):
            plot_clusters(X_scaled, labels, sample_df, feature_cols)
        assert scatter_path.exists() and scatter_path.stat().st_size > 0

    def test_creates_bar_plot(self, tmp_path, X_scaled, trained_model, sample_df, feature_cols):
        _, labels = trained_model
        scatter_path = tmp_path / "scatter2.png"
        bar_path     = tmp_path / "bar2.png"
        with (
            patch.object(config, "CLUSTER_PLOT_2D",  scatter_path),
            patch("src.clustering.CLUSTER_PLOT_2D",  scatter_path),
            patch.object(config, "CLUSTER_PLOT_BAR", bar_path),
            patch("src.clustering.CLUSTER_PLOT_BAR", bar_path),
        ):
            plot_clusters(X_scaled, labels, sample_df, feature_cols)
        assert bar_path.exists() and bar_path.stat().st_size > 0


# ── save_model ────────────────────────────────────────────────────────────────

class TestSaveModel:

    def test_creates_pkl_file(self, tmp_path, trained_model, scaled_output):
        import joblib
        model, _ = trained_model
        _, _, scaler = scaled_output
        pkl_path = tmp_path / "model_clustering.pkl"
        with (
            patch.object(config, "MODEL_CLUSTERING_PATH", pkl_path),
            patch("src.clustering.MODEL_CLUSTERING_PATH", pkl_path),
            patch.object(config, "MODELS_DIR", tmp_path),
            patch("src.clustering.MODELS_DIR",  tmp_path),
        ):
            save_model(model, scaler)
        assert pkl_path.exists()

    def test_pkl_contains_expected_keys(self, tmp_path, trained_model, scaled_output):
        import joblib
        model, _ = trained_model
        _, _, scaler = scaled_output
        pkl_path = tmp_path / "model_clustering2.pkl"
        with (
            patch.object(config, "MODEL_CLUSTERING_PATH", pkl_path),
            patch("src.clustering.MODEL_CLUSTERING_PATH", pkl_path),
            patch.object(config, "MODELS_DIR", tmp_path),
            patch("src.clustering.MODELS_DIR",  tmp_path),
        ):
            save_model(model, scaler)
        artefact = joblib.load(pkl_path)
        assert {"model", "scaler", "k", "features"}.issubset(artefact)

    def test_loaded_model_can_predict(self, tmp_path, trained_model, scaled_output):
        import joblib
        model, _ = trained_model
        X_scaled, _, scaler = scaled_output
        pkl_path = tmp_path / "model_clustering3.pkl"
        with (
            patch.object(config, "MODEL_CLUSTERING_PATH", pkl_path),
            patch("src.clustering.MODEL_CLUSTERING_PATH", pkl_path),
            patch.object(config, "MODELS_DIR", tmp_path),
            patch("src.clustering.MODELS_DIR",  tmp_path),
        ):
            save_model(model, scaler)
        loaded    = joblib.load(pkl_path)
        preds     = loaded["model"].predict(X_scaled)
        assert preds.shape == (X_scaled.shape[0],)


# ── run_full_clustering (integration) ─────────────────────────────────────────

class TestRunFullClustering:

    @pytest.fixture(scope="class")
    def clustering_result(self, sample_df, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("models")
        with (
            patch.object(config, "MODEL_CLUSTERING_PATH", tmp / "model_clustering.pkl"),
            patch("src.clustering.MODEL_CLUSTERING_PATH", tmp / "model_clustering.pkl"),
            patch.object(config, "MODELS_DIR", tmp),
            patch("src.clustering.MODELS_DIR",  tmp),
            patch.object(config, "CLUSTER_PLOT_2D",  tmp / "scatter.png"),
            patch("src.clustering.CLUSTER_PLOT_2D",  tmp / "scatter.png"),
            patch.object(config, "CLUSTER_PLOT_BAR", tmp / "bar.png"),
            patch("src.clustering.CLUSTER_PLOT_BAR", tmp / "bar.png"),
            patch.object(config, "ELBOW_PLOT_PATH",  tmp / "elbow.png"),
            patch("src.clustering.ELBOW_PLOT_PATH",  tmp / "elbow.png"),
        ):
            result = run_full_clustering(df=sample_df, use_elbow=False)
        return result

    def test_returns_dict(self, clustering_result):
        assert isinstance(clustering_result, dict)

    def test_required_keys(self, clustering_result):
        required = {"df", "X_scaled", "feature_cols", "scaler",
                    "optimal_k", "model", "labels", "cluster_profile"}
        assert required.issubset(clustering_result)

    def test_labels_shape_matches_df(self, clustering_result):
        assert len(clustering_result["labels"]) == len(clustering_result["df"])

    def test_optimal_k_is_positive_int(self, clustering_result):
        k = clustering_result["optimal_k"]
        assert isinstance(k, int) and k >= 2

    def test_cluster_profile_rows_equal_k(self, clustering_result):
        k       = clustering_result["optimal_k"]
        profile = clustering_result["cluster_profile"]
        assert len(profile) == k

    def test_model_inertia_positive(self, clustering_result):
        assert clustering_result["model"].inertia_ > 0

    def test_input_df_not_mutated(self, sample_df, clustering_result):
        # 'churned' column should still be present in the original df
        assert "churned" in sample_df.columns
