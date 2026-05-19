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

Run:
    python main.py
"""

from src.data_loader import load_dataset
from src.eda import run_full_eda
from src.preprocessor import run_full_preprocessing
from src.clustering import run_full_clustering
from src.cluster_interpreter import run_full_interpretation
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

    # ── Summary ───────────────────────────────────────────────────────────────
    labeled_df = interp_results["labeled_df"]
    logger.info("")
    logger.info("━━━ Pipeline Complete ━━━")
    logger.info("Clean dataset     : %d rows × %d columns", *df_clean.shape)
    logger.info("Clusters found    : %d",  cluster_results["optimal_k"])
    logger.info("Model saved       : models/model_clustering.pkl")
    logger.info("Labeled export    : data/retail_customers_labeled.csv  (%d rows × %d cols)",
                *labeled_df.shape)
    logger.info("Interpretation    : outputs/09… 12_*.{csv,png}")

    return {
        "eda":            eda_results,
        "preprocessing":  prep_results,
        "clustering":     cluster_results,
        "interpretation": interp_results,
    }


if __name__ == "__main__":
    main()
