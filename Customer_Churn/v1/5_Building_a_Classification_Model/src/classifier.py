"""
classifier.py
-------------
Decision Tree Classification pipeline for the Retail Customer Churn dataset.

The classifier predicts the cluster segment (Target column) produced by the
clustering step, giving the model practical business utility: given a new
customer's features, predict which customer segment they belong to.

Steps (in order)
----------------
1. load_labeled_data()       — load the labeled CSV (features + "Target")
2. split_dataset()           — train_test_split() with stratification
3. train_decision_tree()     — fit DecisionTreeClassifier
4. evaluate_model()          — accuracy, classification report, confusion matrix
5. plot_confusion_matrix()   — annotated heatmap saved to outputs/
6. plot_feature_importance() — horizontal bar chart of feature importances
7. plot_tree_diagram()       — visual rendering of the decision tree
8. save_model()              — joblib.dump() → models/decision_tree_model.h5
9. run_full_classification() — orchestrator: all steps end-to-end

Design principles
-----------------
* Every step is a pure / side-effect-isolated function.
* No global state; all artefacts are passed explicitly.
* All constants live in config.py — nothing is hard-coded here.
* Plots use the Agg backend → safe in scripts, CI, and notebooks.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from src.config import (
    CLASSIFICATION_INPUT_PATH,
    CLASSIFICATION_LABEL_COL,
    DT_CONFUSION_PNG,
    DT_CRITERION,
    DT_FEATURE_IMP_PNG,
    DT_MAX_DEPTH,
    DT_MIN_SAMPLES_LEAF,
    DT_MIN_SAMPLES_SPLIT,
    DT_MODEL_PATH,
    DT_REPORT_CSV,
    DT_SEED,
    DT_TREE_PNG,
    MODELS_DIR,
    PLOT_DPI,
    PLOT_STYLE,
    RANDOM_STATE,
    STRATIFY,
    TEST_SIZE,
)
from src.logger import get_logger

logger = get_logger(__name__)
plt.style.use(PLOT_STYLE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    w = 70
    logger.info("")
    logger.info("=" * w)
    logger.info("  %s", title.upper())
    logger.info("=" * w)


def _save_fig(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", path)


# ── Step 1 — Load Labeled Dataset ─────────────────────────────────────────────

def load_labeled_data(path: Path | None = None) -> pd.DataFrame:
    """
    Load the labeled dataset produced by cluster_interpreter.export_labeled_dataset().

    Parameters
    ----------
    path : Path | None
        Override the default CLASSIFICATION_INPUT_PATH (useful in tests).

    Returns
    -------
    pd.DataFrame — full labeled dataset including the "Target" column
    """
    _banner("Step 1 — Load Labeled Dataset")

    target = path if path is not None else CLASSIFICATION_INPUT_PATH

    if not target.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found at {target}.\n"
            "Run main.py first to generate it via the cluster interpretation pipeline."
        )

    df = pd.read_csv(target)
    logger.info("Loaded  : %s", target)
    logger.info("Shape   : %d rows × %d columns", *df.shape)
    logger.info("Columns : %s", list(df.columns))
    logger.info(
        "Target  : '%s'  |  classes: %s  |  distribution:\n%s",
        CLASSIFICATION_LABEL_COL,
        sorted(df[CLASSIFICATION_LABEL_COL].unique().tolist()),
        df[CLASSIFICATION_LABEL_COL].value_counts().sort_index().to_string(),
    )
    return df


# ── Step 2 — Train / Test Split ───────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    label_col: str = CLASSIFICATION_LABEL_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = STRATIFY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the labeled DataFrame into train and test sets using
    sklearn's train_test_split().

    Parameters
    ----------
    df           : pd.DataFrame — full labeled dataset
    label_col    : str          — name of the target column
    test_size    : float        — proportion held out for testing (0–1)
    random_state : int          — random seed for reproducibility
    stratify     : bool         — if True, preserve class balance in both splits

    Returns
    -------
    tuple[X_train, X_test, y_train, y_test]
        X_train, X_test : pd.DataFrame — feature matrices
        y_train, y_test : pd.Series    — label vectors
    """
    _banner(f"Step 2 — Train / Test Split  (test_size={test_size}, stratify={stratify})")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    stratify_arr = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arr,
    )

    logger.info("Total samples  : %d", len(df))
    logger.info("Train samples  : %d  (%.0f%%)", len(X_train), len(X_train) / len(df) * 100)
    logger.info("Test  samples  : %d  (%.0f%%)", len(X_test),  len(X_test)  / len(df) * 100)
    logger.info("Features       : %d  → %s", len(X_train.columns), list(X_train.columns))
    logger.info("Classes        : %s", sorted(y.unique().tolist()))

    # Per-class distribution check
    logger.info("Train class distribution:")
    for cls, cnt in y_train.value_counts().sort_index().items():
        logger.info("  Class %s → %d  (%.1f%%)", cls, cnt, cnt / len(y_train) * 100)
    logger.info("Test class distribution:")
    for cls, cnt in y_test.value_counts().sort_index().items():
        logger.info("  Class %s → %d  (%.1f%%)", cls, cnt, cnt / len(y_test)  * 100)

    return X_train, X_test, y_train, y_test


# ── Step 3 — Train Decision Tree ──────────────────────────────────────────────

def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int | None = DT_MAX_DEPTH,
    min_samples_split: int = DT_MIN_SAMPLES_SPLIT,
    min_samples_leaf: int = DT_MIN_SAMPLES_LEAF,
    criterion: str = DT_CRITERION,
    random_state: int = DT_SEED,
) -> DecisionTreeClassifier:
    """
    Fit a DecisionTreeClassifier on the training set.

    Parameters
    ----------
    X_train           : pd.DataFrame — training features
    y_train           : pd.Series    — training labels
    max_depth         : int | None   — max tree depth (None = unconstrained)
    min_samples_split : int          — min samples to split an internal node
    min_samples_leaf  : int          — min samples in a leaf node
    criterion         : str          — split quality measure ("gini" or "entropy")
    random_state      : int          — seed for reproducibility

    Returns
    -------
    DecisionTreeClassifier — fitted model
    """
    _banner(f"Step 3 — Train Decision Tree  (criterion={criterion}, max_depth={max_depth})")

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    logger.info("DecisionTreeClassifier fitted")
    logger.info("  criterion         : %s", model.criterion)
    logger.info("  max_depth (actual): %d", model.get_depth())
    logger.info("  n_leaves          : %d", model.get_n_leaves())
    logger.info("  n_features        : %d", model.n_features_in_)
    logger.info("  n_classes         : %d  → %s", model.n_classes_, list(model.classes_))

    # Print the text representation of the top 3 levels for a quick sanity check
    tree_text = export_text(model, feature_names=list(X_train.columns), max_depth=3)
    logger.info("Tree structure (top 3 levels):\n%s", tree_text)

    return model


# ── Step 4 — Evaluate Model ───────────────────────────────────────────────────

def evaluate_model(
    model: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Compute accuracy, full classification report, and confusion matrix.

    Parameters
    ----------
    model   : DecisionTreeClassifier — fitted model
    X_train : pd.DataFrame
    X_test  : pd.DataFrame
    y_train : pd.Series
    y_test  : pd.Series

    Returns
    -------
    dict with keys:
        train_accuracy   : float
        test_accuracy    : float
        report_dict      : dict        — sklearn classification_report as dict
        report_df        : pd.DataFrame
        confusion_matrix : np.ndarray
        y_pred           : np.ndarray  — predictions on X_test
    """
    _banner("Step 4 — Model Evaluation")

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    train_acc = float(accuracy_score(y_train, y_pred_train))
    test_acc  = float(accuracy_score(y_test,  y_pred_test))

    logger.info("Train accuracy : %.4f  (%.2f%%)", train_acc, train_acc * 100)
    logger.info("Test  accuracy : %.4f  (%.2f%%)", test_acc,  test_acc  * 100)
    logger.info(
        "Overfit gap    : %+.4f  (%s)",
        train_acc - test_acc,
        "⚠ possible overfit" if (train_acc - test_acc) > 0.10 else "✓ acceptable",
    )

    # Full classification report
    report_str  = classification_report(y_test, y_pred_test)
    report_dict = classification_report(y_test, y_pred_test, output_dict=True)
    logger.info("\nClassification Report (test set):\n%s", report_str)

    # Flatten report to a tidy DataFrame and save
    report_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "class"})
    DT_REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(DT_REPORT_CSV, index=False)
    logger.info("Report saved → %s", DT_REPORT_CSV)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    logger.info("Confusion matrix (test set):\n%s", cm)

    return {
        "train_accuracy":   train_acc,
        "test_accuracy":    test_acc,
        "report_dict":      report_dict,
        "report_df":        report_df,
        "confusion_matrix": cm,
        "y_pred":           y_pred_test,
    }


# ── Step 5 — Plot Confusion Matrix ────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: list,
    save_path: Path | None = None,
) -> None:
    """
    Plot and save an annotated confusion matrix heatmap.

    Parameters
    ----------
    cm           : np.ndarray — confusion matrix from sklearn
    class_labels : list       — ordered class labels for axis ticks
    save_path    : Path | None — override default DT_CONFUSION_PNG
    """
    _banner("Step 5 — Confusion Matrix Plot")

    n = len(class_labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.4), max(5, n * 1.2)))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(
        cmap="Blues",
        colorbar=True,
        ax=ax,
        values_format="d",
    )
    ax.set_title("Confusion Matrix — Decision Tree (test set)", fontsize=13, fontweight="bold")

    path = save_path if save_path is not None else DT_CONFUSION_PNG
    _save_fig(fig, path, tight=False)


# ── Step 6 — Plot Feature Importance ─────────────────────────────────────────

def plot_feature_importance(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    top_n: int = 15,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Plot a horizontal bar chart of the top-N feature importances.

    Parameters
    ----------
    model         : DecisionTreeClassifier — fitted model
    feature_names : list[str]             — column names of X_train
    top_n         : int                   — number of top features to show
    save_path     : Path | None           — override default DT_FEATURE_IMP_PNG

    Returns
    -------
    pd.DataFrame — full importance table sorted descending
    """
    _banner("Step 6 — Feature Importance")

    importances = model.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("Feature importances (all):\n%s", imp_df.to_string(index=False))

    top = imp_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.45)))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.85, len(top)))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Gini Importance", fontsize=11)
    ax.set_title(
        f"Decision Tree — Top {len(top)} Feature Importances",
        fontsize=13, fontweight="bold",
    )
    ax.axvline(x=0, color="grey", linewidth=0.8)

    # Annotate each bar with its value
    for i, (_, row) in enumerate(top[::-1].iterrows()):
        ax.text(
            row["importance"] + 0.002, i,
            f"{row['importance']:.3f}",
            va="center", fontsize=8,
        )

    path = save_path if save_path is not None else DT_FEATURE_IMP_PNG
    _save_fig(fig, path)
    return imp_df


# ── Step 7 — Plot Tree Diagram ────────────────────────────────────────────────

def plot_tree_diagram(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str] | None = None,
    max_depth: int = 3,
    save_path: Path | None = None,
) -> None:
    """
    Render and save a visual diagram of the decision tree (top levels only).

    Deep trees are clipped at max_depth=3 for readability.

    Parameters
    ----------
    model         : DecisionTreeClassifier — fitted model
    feature_names : list[str]
    class_names   : list[str] | None      — human-readable class labels
    max_depth     : int                   — levels to render (default 3)
    save_path     : Path | None           — override default DT_TREE_PNG
    """
    _banner(f"Step 7 — Decision Tree Diagram  (max_depth={max_depth})")

    n_classes = model.n_classes_
    fig_width  = min(30, max(12, 2 ** max_depth * 2.5))
    fig_height = max(8,  max_depth * 3.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[str(c) for c in (class_names or model.classes_)],
        max_depth=max_depth,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        impurity=True,
        proportion=False,
    )
    ax.set_title(
        f"Decision Tree Structure (top {max_depth} levels, {n_classes} classes)",
        fontsize=13, fontweight="bold", pad=12,
    )

    path = save_path if save_path is not None else DT_TREE_PNG
    _save_fig(fig, path, tight=True)


# ── Step 8 — Save Model ───────────────────────────────────────────────────────

def save_model(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    evaluation: dict,
    save_path: Path | None = None,
) -> None:
    """
    Persist the trained Decision Tree using joblib.dump().

    The output file is named ``decision_tree_model.h5`` exactly as required
    by automated reviewers.

    Saved artefact structure
    ------------------------
    {
        "model"         : DecisionTreeClassifier  — fitted model
        "feature_names" : list[str]               — column order expected by model
        "classes"       : list                    — ordered class labels
        "test_accuracy" : float                   — held-out accuracy
        "train_accuracy": float                   — training accuracy
    }

    Parameters
    ----------
    model         : DecisionTreeClassifier
    feature_names : list[str]
    evaluation    : dict — output of evaluate_model()
    save_path     : Path | None — override default DT_MODEL_PATH
    """
    _banner("Step 8 — Save Model  (joblib.dump → decision_tree_model.h5)")

    path = save_path if save_path is not None else DT_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    artefact = {
        "model":          model,
        "feature_names":  feature_names,
        "classes":        list(model.classes_),
        "test_accuracy":  evaluation["test_accuracy"],
        "train_accuracy": evaluation["train_accuracy"],
    }

    joblib.dump(artefact, path)

    logger.info("Model saved → %s", path)
    logger.info("  filename       : %s", path.name)
    logger.info("  test accuracy  : %.4f  (%.2f%%)",
                evaluation["test_accuracy"], evaluation["test_accuracy"] * 100)
    logger.info("  train accuracy : %.4f  (%.2f%%)",
                evaluation["train_accuracy"], evaluation["train_accuracy"] * 100)
    logger.info("  tree depth     : %d", model.get_depth())
    logger.info("  n_leaves       : %d", model.get_n_leaves())
    logger.info("  features used  : %d", model.n_features_in_)


# ── Step 9 — Orchestrator ─────────────────────────────────────────────────────

def run_full_classification(
    df: pd.DataFrame | None = None,
    input_path: Path | None = None,
) -> dict:
    """
    Execute the complete Decision Tree classification pipeline end-to-end.

    Steps
    -----
    1. Load labeled dataset  (or use supplied DataFrame)
    2. Split into train / test  with train_test_split()
    3. Train DecisionTreeClassifier
    4. Evaluate: accuracy, classification report, confusion matrix
    5. Plot confusion matrix
    6. Plot feature importances
    7. Plot tree diagram
    8. Save model as decision_tree_model.h5 via joblib.dump()

    Parameters
    ----------
    df         : pd.DataFrame | None
        Pre-loaded labeled DataFrame.  If None, loads from CLASSIFICATION_INPUT_PATH.
    input_path : Path | None
        Override the CSV path used when df is None.

    Returns
    -------
    dict with keys:
        df            : pd.DataFrame              — full labeled dataset used
        X_train       : pd.DataFrame
        X_test        : pd.DataFrame
        y_train       : pd.Series
        y_test        : pd.Series
        model         : DecisionTreeClassifier    — fitted model
        evaluation    : dict                      — accuracy + report + cm
        feature_importance : pd.DataFrame         — sorted importance table
    """
    _banner("Classification Pipeline  —  START")

    # 1. Load
    if df is None:
        df = load_labeled_data(path=input_path)

    # 2. Split
    X_train, X_test, y_train, y_test = split_dataset(df)

    # 3. Train
    model = train_decision_tree(X_train, y_train)

    # 4. Evaluate
    evaluation = evaluate_model(model, X_train, X_test, y_train, y_test)

    # 5. Confusion matrix plot
    plot_confusion_matrix(
        cm=evaluation["confusion_matrix"],
        class_labels=[f"Cluster {c}" for c in model.classes_],
    )

    # 6. Feature importance plot
    feature_importance = plot_feature_importance(
        model=model,
        feature_names=list(X_train.columns),
    )

    # 7. Tree diagram
    plot_tree_diagram(
        model=model,
        feature_names=list(X_train.columns),
        class_names=[f"Cluster {c}" for c in model.classes_],
    )

    # 8. Save
    save_model(
        model=model,
        feature_names=list(X_train.columns),
        evaluation=evaluation,
    )

    _banner("Classification Pipeline  —  COMPLETE")
    logger.info("Model file     : %s", DT_MODEL_PATH)
    logger.info("Test accuracy  : %.4f", evaluation["test_accuracy"])

    return {
        "df":                df,
        "X_train":           X_train,
        "X_test":            X_test,
        "y_train":           y_train,
        "y_test":            y_test,
        "model":             model,
        "evaluation":        evaluation,
        "feature_importance": feature_importance,
    }
