"""
main.py
-------
Entry point for the Retail Customer Churn ML pipeline.

Pipeline stages
---------------
  Step 1 — Load Dataset
  Step 2 — Exploratory Data Analysis      (EDA)
  Step 3 — Data Cleaning & Preprocessing
  Step 4 — Clustering                     (KMeans + Elbow + joblib.dump)
  Step 5 — Cluster Interpretation         (descriptive stats + profiles + labeled export)
  Step 6 — Classification                 (Decision Tree + train_test_split + joblib.dump)

Run:
    python main.py
"""

from src.data_loader import load_dataset
from src.eda import run_full_eda
from src.preprocessor import run_full_preprocessing
from src.clustering import run_full_clustering
from src.cluster_interpreter import run_full_interpretation
from src.classifier import run_full_classification
from src.logger import get_logger

logger = get_logger("main")


def main() -> dict:
    logger.info("━━━ Retail Customer Churn — ML Pipeline ━━━")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    df_raw = load_dataset()

    # ── Step 2: EDA ───────────────────────────────────────────────────────────
    eda_results = run_full_eda(df_raw)

    # ── Step 3: Preprocessing ─────────────────────────────────────────────────
    prep_results = run_full_preprocessing(df_raw, persist=True)
    df_clean     = prep_results["df_cleaned"]

    # ── Step 4: Clustering ────────────────────────────────────────────────────
    cluster_results = run_full_clustering(df=df_clean, use_elbow=True)

    # ── Step 5: Cluster Interpretation ───────────────────────────────────────
    interp_results = run_full_interpretation(
        df     = cluster_results["df"],
        labels = cluster_results["labels"],
    )

    # ── Step 6: Classification ────────────────────────────────────────────────
    clf_results = run_full_classification(df=interp_results["labeled_df"])

    # ── Summary ───────────────────────────────────────────────────────────────
    eval_info = clf_results["evaluation"]
    logger.info("")
    logger.info("━━━ Pipeline Complete ━━━")
    logger.info("Clean dataset     : %d rows × %d columns", *df_clean.shape)
    logger.info("Clusters found    : %d",  cluster_results["optimal_k"])
    logger.info("Clustering model  : models/model_clustering.pkl")
    logger.info("Labeled export    : data/retail_customers_labeled.csv")
    logger.info("DT train accuracy : %.4f  (%.2f%%)",
                eval_info["train_accuracy"], eval_info["train_accuracy"] * 100)
    logger.info("DT test  accuracy : %.4f  (%.2f%%)",
                eval_info["test_accuracy"],  eval_info["test_accuracy"]  * 100)
    logger.info("Decision Tree     : models/decision_tree_model.h5")

    return {
        "eda":            eda_results,
        "preprocessing":  prep_results,
        "clustering":     cluster_results,
        "interpretation": interp_results,
        "classification": clf_results,
    }


if __name__ == "__main__":
    main()
