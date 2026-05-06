"""
tests/test_classifier.py
-------------------------
Unit and integration tests for src/classifier.py

All tests use a small fully in-memory DataFrame for speed and isolation.
File-writing tests redirect to tmp_path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
from sklearn.tree import DecisionTreeClassifier
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.classifier import (
    load_labeled_data,
    split_dataset,
    train_decision_tree,
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_tree_diagram,
    save_model,
    run_full_classification,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

N_ROWS     = 400
N_CLASSES  = 4

@pytest.fixture(scope="module")
def labeled_df() -> pd.DataFrame:
    """
    Synthetic labeled DataFrame that mirrors cluster_interpreter output.
    Uses 4 balanced classes (Target 0–3) with 100 rows each.
    """
    rng = np.random.default_rng(99)
    n   = N_ROWS
    # Create features with some real signal so DT can actually learn
    target = np.repeat(np.arange(N_CLASSES), n // N_CLASSES)
    return pd.DataFrame({
        "age":                (rng.integers(18, 76, size=n) + target * 5).clip(18, 90).astype(float),
        "tenure_months":      rng.integers(1, 121, size=n),
        "annual_spend_usd":   (rng.uniform(100, 20_000, size=n) + target * 5_000).round(2),
        "num_purchases":      rng.integers(1, 101, size=n),
        "avg_order_value":    rng.uniform(10, 500, size=n).round(2),
        "returns_count":      rng.integers(0, 11, size=n),
        "last_purchase_days": rng.integers(1, 366, size=n).astype(float),
        "email_opt_in":       rng.integers(0, 2, size=n),
        "gender":             rng.integers(0, 3, size=n),
        "region":             rng.integers(0, 4, size=n),
        "loyalty_tier":       rng.integers(0, 4, size=n),
        "churned":            rng.integers(0, 2, size=n),
        config.CLASSIFICATION_LABEL_COL: target,
    })


@pytest.fixture(scope="module")
def split_data(labeled_df):
    return split_dataset(labeled_df)


@pytest.fixture(scope="module")
def trained_model(split_data):
    X_train, _, y_train, _ = split_data
    return train_decision_tree(X_train, y_train)


@pytest.fixture(scope="module")
def evaluation(trained_model, split_data):
    X_train, X_test, y_train, y_test = split_data
    return evaluate_model(trained_model, X_train, X_test, y_train, y_test)


# ── load_labeled_data ─────────────────────────────────────────────────────────

class TestLoadLabeledData:

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Labeled dataset not found"):
            load_labeled_data(path=tmp_path / "nonexistent.csv")

    def test_loads_csv(self, tmp_path, labeled_df):
        csv_path = tmp_path / "labeled.csv"
        labeled_df.to_csv(csv_path, index=False)
        loaded = load_labeled_data(path=csv_path)
        assert isinstance(loaded, pd.DataFrame)
        assert loaded.shape == labeled_df.shape

    def test_target_column_present(self, tmp_path, labeled_df):
        csv_path = tmp_path / "labeled2.csv"
        labeled_df.to_csv(csv_path, index=False)
        loaded = load_labeled_data(path=csv_path)
        assert config.CLASSIFICATION_LABEL_COL in loaded.columns

    def test_preserves_row_count(self, tmp_path, labeled_df):
        csv_path = tmp_path / "labeled3.csv"
        labeled_df.to_csv(csv_path, index=False)
        loaded = load_labeled_data(path=csv_path)
        assert len(loaded) == len(labeled_df)


# ── split_dataset ─────────────────────────────────────────────────────────────

class TestSplitDataset:

    def test_returns_four_elements(self, split_data):
        assert len(split_data) == 4

    def test_X_train_is_dataframe(self, split_data):
        X_train, _, _, _ = split_data
        assert isinstance(X_train, pd.DataFrame)

    def test_X_test_is_dataframe(self, split_data):
        _, X_test, _, _ = split_data
        assert isinstance(X_test, pd.DataFrame)

    def test_y_train_is_series(self, split_data):
        _, _, y_train, _ = split_data
        assert isinstance(y_train, pd.Series)

    def test_y_test_is_series(self, split_data):
        _, _, _, y_test = split_data
        assert isinstance(y_test, pd.Series)

    def test_target_not_in_features(self, split_data):
        X_train, X_test, _, _ = split_data
        assert config.CLASSIFICATION_LABEL_COL not in X_train.columns
        assert config.CLASSIFICATION_LABEL_COL not in X_test.columns

    def test_rows_sum_to_total(self, labeled_df, split_data):
        X_train, X_test, _, _ = split_data
        assert len(X_train) + len(X_test) == len(labeled_df)

    def test_test_size_approx(self, labeled_df, split_data):
        _, X_test, _, _ = split_data
        actual_ratio = len(X_test) / len(labeled_df)
        assert abs(actual_ratio - config.TEST_SIZE) < 0.05

    def test_class_balance_preserved(self, split_data):
        """Stratification should keep roughly equal class proportions."""
        _, _, y_train, y_test = split_data
        train_props = y_train.value_counts(normalize=True).sort_index()
        test_props  = y_test.value_counts(normalize=True).sort_index()
        # Allow ±5% difference per class
        for cls in train_props.index:
            assert abs(train_props[cls] - test_props.get(cls, 0)) < 0.05

    def test_raises_on_missing_label_col(self, labeled_df):
        df_no_label = labeled_df.drop(columns=[config.CLASSIFICATION_LABEL_COL])
        with pytest.raises(ValueError, match="Label column"):
            split_dataset(df_no_label)

    def test_reproducible(self, labeled_df):
        split1 = split_dataset(labeled_df, random_state=0)
        split2 = split_dataset(labeled_df, random_state=0)
        pd.testing.assert_frame_equal(split1[0].reset_index(drop=True),
                                      split2[0].reset_index(drop=True))

    def test_different_seed_gives_different_split(self, labeled_df):
        X_train_a, *_ = split_dataset(labeled_df, random_state=0)
        X_train_b, *_ = split_dataset(labeled_df, random_state=999)
        # At least some indices differ
        assert not X_train_a.index.equals(X_train_b.index)


# ── train_decision_tree ───────────────────────────────────────────────────────

class TestTrainDecisionTree:

    def test_returns_decision_tree(self, trained_model):
        assert isinstance(trained_model, DecisionTreeClassifier)

    def test_model_is_fitted(self, trained_model):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(trained_model)

    def test_classes_match_y_train(self, trained_model, split_data):
        _, _, y_train, _ = split_data
        assert set(trained_model.classes_) == set(y_train.unique())

    def test_n_features_matches_X_train(self, trained_model, split_data):
        X_train, _, _, _ = split_data
        assert trained_model.n_features_in_ == X_train.shape[1]

    def test_depth_positive(self, trained_model):
        assert trained_model.get_depth() > 0

    def test_n_leaves_at_least_n_classes(self, trained_model):
        assert trained_model.get_n_leaves() >= trained_model.n_classes_

    def test_custom_max_depth_respected(self, split_data):
        X_train, _, y_train, _ = split_data
        m = train_decision_tree(X_train, y_train, max_depth=3)
        assert m.get_depth() <= 3

    def test_criterion_gini(self, split_data):
        X_train, _, y_train, _ = split_data
        m = train_decision_tree(X_train, y_train, criterion="gini")
        assert m.criterion == "gini"

    def test_criterion_entropy(self, split_data):
        X_train, _, y_train, _ = split_data
        m = train_decision_tree(X_train, y_train, criterion="entropy")
        assert m.criterion == "entropy"

    def test_input_not_mutated(self, split_data):
        X_train, _, y_train, _ = split_data
        shape_before = X_train.shape
        train_decision_tree(X_train, y_train)
        assert X_train.shape == shape_before


# ── evaluate_model ────────────────────────────────────────────────────────────

class TestEvaluateModel:

    def test_returns_dict(self, evaluation):
        assert isinstance(evaluation, dict)

    def test_required_keys(self, evaluation):
        required = {
            "train_accuracy", "test_accuracy",
            "report_dict", "report_df",
            "confusion_matrix", "y_pred",
        }
        assert required.issubset(evaluation)

    def test_train_accuracy_in_range(self, evaluation):
        acc = evaluation["train_accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_test_accuracy_in_range(self, evaluation):
        acc = evaluation["test_accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_confusion_matrix_is_square(self, evaluation):
        cm = evaluation["confusion_matrix"]
        assert cm.ndim == 2
        assert cm.shape[0] == cm.shape[1]

    def test_confusion_matrix_rows_match_classes(self, evaluation, trained_model):
        cm = evaluation["confusion_matrix"]
        assert cm.shape[0] == trained_model.n_classes_

    def test_confusion_matrix_sum_matches_test_size(self, evaluation, split_data):
        _, X_test, _, _ = split_data
        cm = evaluation["confusion_matrix"]
        assert cm.sum() == len(X_test)

    def test_y_pred_length_matches_test_set(self, evaluation, split_data):
        _, X_test, _, _ = split_data
        assert len(evaluation["y_pred"]) == len(X_test)

    def test_report_df_is_dataframe(self, evaluation):
        assert isinstance(evaluation["report_df"], pd.DataFrame)

    def test_report_df_has_accuracy_row(self, evaluation):
        classes_col = evaluation["report_df"]["class"].tolist()
        assert "accuracy" in classes_col

    def test_saves_report_csv(self, trained_model, split_data, tmp_path):
        X_train, X_test, y_train, y_test = split_data
        csv_path = tmp_path / "report.csv"
        with patch("src.classifier.DT_REPORT_CSV", csv_path):
            evaluate_model(trained_model, X_train, X_test, y_train, y_test)
        assert csv_path.exists() and csv_path.stat().st_size > 0


# ── plot_confusion_matrix ─────────────────────────────────────────────────────

class TestPlotConfusionMatrix:

    def test_creates_png(self, evaluation, trained_model, tmp_path):
        path = tmp_path / "cm.png"
        plot_confusion_matrix(
            cm=evaluation["confusion_matrix"],
            class_labels=[f"C{c}" for c in trained_model.classes_],
            save_path=path,
        )
        assert path.exists() and path.stat().st_size > 0

    def test_works_with_binary_classes(self, tmp_path):
        cm   = np.array([[10, 2], [3, 8]])
        path = tmp_path / "cm_binary.png"
        plot_confusion_matrix(cm=cm, class_labels=["A", "B"], save_path=path)
        assert path.exists()


# ── plot_feature_importance ───────────────────────────────────────────────────

class TestPlotFeatureImportance:

    def test_returns_dataframe(self, trained_model, split_data, tmp_path):
        X_train, *_ = split_data
        path = tmp_path / "fi.png"
        result = plot_feature_importance(
            trained_model, list(X_train.columns), save_path=path
        )
        assert isinstance(result, pd.DataFrame)

    def test_sorted_descending(self, trained_model, split_data, tmp_path):
        X_train, *_ = split_data
        path = tmp_path / "fi2.png"
        result = plot_feature_importance(
            trained_model, list(X_train.columns), save_path=path
        )
        imps = result["importance"].tolist()
        assert imps == sorted(imps, reverse=True)

    def test_sum_near_one(self, trained_model, split_data, tmp_path):
        X_train, *_ = split_data
        path = tmp_path / "fi3.png"
        result = plot_feature_importance(
            trained_model, list(X_train.columns), save_path=path
        )
        assert abs(result["importance"].sum() - 1.0) < 1e-6

    def test_creates_png(self, trained_model, split_data, tmp_path):
        X_train, *_ = split_data
        path = tmp_path / "fi4.png"
        plot_feature_importance(trained_model, list(X_train.columns), save_path=path)
        assert path.exists() and path.stat().st_size > 0

    def test_all_features_present(self, trained_model, split_data, tmp_path):
        X_train, *_ = split_data
        path = tmp_path / "fi5.png"
        result = plot_feature_importance(
            trained_model, list(X_train.columns), save_path=path
        )
        assert set(result["feature"]) == set(X_train.columns)


# ── plot_tree_diagram ─────────────────────────────────────────────────────────

class TestPlotTreeDiagram:

    def test_creates_png(self, trained_model, split_data, tmp_path):
        X_train, *_ = split_data
        path = tmp_path / "tree.png"
        plot_tree_diagram(
            trained_model, list(X_train.columns), max_depth=2, save_path=path
        )
        assert path.exists() and path.stat().st_size > 0


# ── save_model ────────────────────────────────────────────────────────────────

class TestSaveModel:

    def test_creates_h5_file(self, trained_model, split_data, evaluation, tmp_path):
        X_train, *_ = split_data
        path = tmp_path / "decision_tree_model.h5"
        save_model(trained_model, list(X_train.columns), evaluation, save_path=path)
        assert path.exists() and path.stat().st_size > 0

    def test_filename_is_h5(self):
        assert config.DT_MODEL_PATH.suffix == ".h5"

    def test_filename_matches_requirement(self):
        assert config.DT_MODEL_FILENAME == "decision_tree_model.h5"

    def test_artefact_has_required_keys(self, trained_model, split_data, evaluation, tmp_path):
        X_train, *_ = split_data
        path = tmp_path / "decision_tree_model.h5"
        save_model(trained_model, list(X_train.columns), evaluation, save_path=path)
        artefact = joblib.load(path)
        required = {"model", "feature_names", "classes", "test_accuracy", "train_accuracy"}
        assert required.issubset(artefact)

    def test_loaded_model_can_predict(self, trained_model, split_data, evaluation, tmp_path):
        X_train, X_test, _, _ = split_data
        path = tmp_path / "decision_tree_model.h5"
        save_model(trained_model, list(X_train.columns), evaluation, save_path=path)
        loaded   = joblib.load(path)
        preds    = loaded["model"].predict(X_test)
        assert len(preds) == len(X_test)

    def test_saved_accuracy_matches_evaluation(self, trained_model, split_data, evaluation, tmp_path):
        X_train, *_ = split_data
        path = tmp_path / "decision_tree_model.h5"
        save_model(trained_model, list(X_train.columns), evaluation, save_path=path)
        artefact = joblib.load(path)
        assert abs(artefact["test_accuracy"] - evaluation["test_accuracy"]) < 1e-9


import joblib  # needed inside TestSaveModel methods


# ── run_full_classification (integration) ─────────────────────────────────────

class TestRunFullClassification:

    @pytest.fixture(scope="class")
    def clf_result(self, labeled_df, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("clf")
        with (
            patch.object(config, "DT_MODEL_PATH",      tmp / "decision_tree_model.h5"),
            patch("src.classifier.DT_MODEL_PATH",       tmp / "decision_tree_model.h5"),
            patch.object(config, "MODELS_DIR",          tmp),
            patch("src.classifier.MODELS_DIR",          tmp),
            patch.object(config, "DT_REPORT_CSV",       tmp / "report.csv"),
            patch("src.classifier.DT_REPORT_CSV",       tmp / "report.csv"),
            patch.object(config, "DT_CONFUSION_PNG",    tmp / "cm.png"),
            patch("src.classifier.DT_CONFUSION_PNG",    tmp / "cm.png"),
            patch.object(config, "DT_FEATURE_IMP_PNG",  tmp / "fi.png"),
            patch("src.classifier.DT_FEATURE_IMP_PNG",  tmp / "fi.png"),
            patch.object(config, "DT_TREE_PNG",         tmp / "tree.png"),
            patch("src.classifier.DT_TREE_PNG",         tmp / "tree.png"),
        ):
            result = run_full_classification(df=labeled_df)
        return result, tmp

    def test_returns_dict(self, clf_result):
        result, _ = clf_result
        assert isinstance(result, dict)

    def test_required_keys(self, clf_result):
        result, _ = clf_result
        required = {
            "df", "X_train", "X_test", "y_train", "y_test",
            "model", "evaluation", "feature_importance",
        }
        assert required.issubset(result)

    def test_model_is_decision_tree(self, clf_result):
        result, _ = clf_result
        assert isinstance(result["model"], DecisionTreeClassifier)

    def test_test_accuracy_above_random(self, clf_result):
        """Should comfortably beat random (25% for 4 classes)."""
        result, _ = clf_result
        assert result["evaluation"]["test_accuracy"] > 0.25

    def test_train_test_rows_sum_to_total(self, clf_result, labeled_df):
        result, _ = clf_result
        assert len(result["X_train"]) + len(result["X_test"]) == len(labeled_df)

    def test_all_output_files_created(self, clf_result):
        _, tmp = clf_result
        assert (tmp / "decision_tree_model.h5").exists()
        assert (tmp / "report.csv").exists()
        assert (tmp / "cm.png").exists()
        assert (tmp / "fi.png").exists()
        assert (tmp / "tree.png").exists()

    def test_feature_importance_is_dataframe(self, clf_result):
        result, _ = clf_result
        assert isinstance(result["feature_importance"], pd.DataFrame)

    def test_input_df_not_mutated(self, clf_result, labeled_df):
        assert config.CLASSIFICATION_LABEL_COL in labeled_df.columns
