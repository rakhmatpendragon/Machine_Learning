"""
classifier.py — Decision Tree Classification pipeline.
Uses banner() and save_fig() from src.utils (no local copies).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                              classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from src.config import (CLASSIFICATION_INPUT_PATH, CLASSIFICATION_LABEL_COL,
                         DT_CONFUSION_PNG, DT_CRITERION, DT_FEATURE_IMP_PNG,
                         DT_MAX_DEPTH, DT_MIN_SAMPLES_LEAF, DT_MIN_SAMPLES_SPLIT,
                         DT_MODEL_PATH, DT_REPORT_CSV, DT_SEED, DT_TREE_PNG,
                         MODELS_DIR, PLOT_STYLE, RANDOM_STATE, STRATIFY, TEST_SIZE)
from src.logger import get_logger
from src.utils import banner, save_fig          # ← single shared source

logger = get_logger(__name__)
plt.style.use(PLOT_STYLE)


def load_labeled_data(path=None) -> pd.DataFrame:
    banner("Step 1 — Load Labeled Dataset", logger)
    target = path or CLASSIFICATION_INPUT_PATH
    if not target.exists():
        raise FileNotFoundError(f"Labeled dataset not found at {target}.")
    df = pd.read_csv(target)
    logger.info("Loaded %d rows × %d columns from %s", *df.shape, target)
    logger.info("Target '%s' — classes: %s\n%s", CLASSIFICATION_LABEL_COL,
                sorted(df[CLASSIFICATION_LABEL_COL].unique().tolist()),
                df[CLASSIFICATION_LABEL_COL].value_counts().sort_index().to_string())
    return df


def split_dataset(df, label_col=CLASSIFICATION_LABEL_COL,
                  test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=STRATIFY):
    banner(f"Step 2 — Train / Test Split  (test_size={test_size}, stratify={stratify})", logger)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None)
    logger.info("Total/Train/Test : %d / %d / %d", len(df), len(X_train), len(X_test))
    logger.info("Features         : %d → %s", len(X_train.columns), list(X_train.columns))
    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train, y_train, max_depth=DT_MAX_DEPTH,
                        min_samples_split=DT_MIN_SAMPLES_SPLIT,
                        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
                        criterion=DT_CRITERION, random_state=DT_SEED):
    banner(f"Step 3 — Train Decision Tree  (criterion={criterion}, max_depth={max_depth})", logger)
    model = DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Fitted  |  depth=%d  |  leaves=%d  |  features=%d  |  classes=%s",
                model.get_depth(), model.get_n_leaves(),
                model.n_features_in_, list(model.classes_))
    logger.info("Tree (top 3 levels):\n%s",
                export_text(model, feature_names=list(X_train.columns), max_depth=3))
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    banner("Step 4 — Model Evaluation", logger)
    train_acc = float(accuracy_score(y_train, model.predict(X_train)))
    y_pred    = model.predict(X_test)
    test_acc  = float(accuracy_score(y_test, y_pred))
    logger.info("Train accuracy : %.4f (%.2f%%)", train_acc, train_acc*100)
    logger.info("Test  accuracy : %.4f (%.2f%%)", test_acc,  test_acc *100)
    logger.info("Overfit gap    : %+.4f  (%s)", train_acc - test_acc,
                "⚠ possible overfit" if (train_acc-test_acc) > 0.10 else "✓ acceptable")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    report_df = (pd.DataFrame(report_dict).T.reset_index()
                   .rename(columns={"index": "class"}))
    DT_REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(DT_REPORT_CSV, index=False)
    logger.info("Report saved → %s", DT_REPORT_CSV)
    cm = confusion_matrix(y_test, y_pred)
    return {"train_accuracy": train_acc, "test_accuracy": test_acc,
            "report_dict": report_dict, "report_df": report_df,
            "confusion_matrix": cm, "y_pred": y_pred}


def plot_confusion_matrix(cm, class_labels, save_path=None):
    banner("Step 5 — Confusion Matrix Plot", logger)
    n = len(class_labels)
    fig, ax = plt.subplots(figsize=(max(6, n*1.4), max(5, n*1.2)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(
        cmap="Blues", colorbar=True, ax=ax, values_format="d")
    ax.set_title("Confusion Matrix — Decision Tree (test set)",
                 fontsize=13, fontweight="bold")
    save_fig(fig, save_path or DT_CONFUSION_PNG, tight=False, logger=logger)


def plot_feature_importance(model, feature_names, top_n=15, save_path=None) -> pd.DataFrame:
    banner("Step 6 — Feature Importance", logger)
    imp_df = (pd.DataFrame({"feature": feature_names,
                             "importance": model.feature_importances_})
                .sort_values("importance", ascending=False).reset_index(drop=True))
    logger.info("Importances:\n%s", imp_df.to_string(index=False))
    top    = imp_df.head(top_n)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.85, len(top)))
    fig, ax = plt.subplots(figsize=(9, max(4, len(top)*0.45)))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Gini Importance")
    ax.set_title(f"Decision Tree — Top {len(top)} Feature Importances",
                 fontsize=13, fontweight="bold")
    for i, (_, row) in enumerate(top[::-1].iterrows()):
        ax.text(row["importance"]+0.002, i, f"{row['importance']:.3f}", va="center", fontsize=8)
    save_fig(fig, save_path or DT_FEATURE_IMP_PNG, logger=logger)
    return imp_df


def plot_tree_diagram(model, feature_names, class_names=None,
                      max_depth=3, save_path=None):
    banner(f"Step 7 — Decision Tree Diagram  (max_depth={max_depth})", logger)
    fig, ax = plt.subplots(figsize=(min(30, max(12, 2**max_depth*2.5)),
                                    max(8, max_depth*3.5)))
    plot_tree(model, feature_names=feature_names,
              class_names=[str(c) for c in (class_names or model.classes_)],
              max_depth=max_depth, filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title(f"Decision Tree (top {max_depth} levels, {model.n_classes_} classes)",
                 fontsize=13, fontweight="bold", pad=12)
    save_fig(fig, save_path or DT_TREE_PNG, logger=logger)


def save_model(model, feature_names, evaluation, save_path=None):
    banner("Step 8 — Save Model  (joblib.dump → decision_tree_model.h5)", logger)
    path = save_path or DT_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    artefact = {"model": model, "feature_names": feature_names,
                "classes": list(model.classes_),
                "test_accuracy": evaluation["test_accuracy"],
                "train_accuracy": evaluation["train_accuracy"]}
    joblib.dump(artefact, path)
    logger.info("Model saved → %s  (test_acc=%.4f, depth=%d)",
                path, evaluation["test_accuracy"], model.get_depth())


def run_full_classification(df=None, input_path=None) -> dict:
    banner("Classification Pipeline  —  START", logger)
    if df is None:
        df = load_labeled_data(path=input_path)
    X_train, X_test, y_train, y_test = split_dataset(df)
    model        = train_decision_tree(X_train, y_train)
    evaluation   = evaluate_model(model, X_train, X_test, y_train, y_test)
    plot_confusion_matrix(evaluation["confusion_matrix"],
                          [f"Cluster {c}" for c in model.classes_])
    feature_imp  = plot_feature_importance(model, list(X_train.columns))
    plot_tree_diagram(model, list(X_train.columns),
                      [f"Cluster {c}" for c in model.classes_])
    save_model(model, list(X_train.columns), evaluation)
    banner("Classification Pipeline  —  COMPLETE", logger)
    logger.info("Model: %s  |  Test accuracy: %.4f", DT_MODEL_PATH,
                evaluation["test_accuracy"])
    return {"df": df, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test, "model": model,
            "evaluation": evaluation, "feature_importance": feature_imp}
