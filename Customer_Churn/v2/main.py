"""
main.py — Entry point for the Retail Customer Churn ML pipeline.

Pipeline stages
---------------
  Step 1 — Load Dataset
  Step 2 — Exploratory Data Analysis
  Step 3 — Data Cleaning & Preprocessing
  Step 4 — Clustering  (KMeans + Elbow + joblib.dump)
  Step 5 — Cluster Interpretation  (stats + profiles + labeled export)
  Step 6 — Classification  (Decision Tree + train_test_split + joblib.dump)

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
    df_raw          = load_dataset()
    eda_results     = run_full_eda(df_raw)
    prep_results    = run_full_preprocessing(df_raw, persist=True)
    cluster_results = run_full_clustering(df=prep_results["df_cleaned"], use_elbow=True)
    interp_results  = run_full_interpretation(df=cluster_results["df"],
                                              labels=cluster_results["labels"])
    clf_results     = run_full_classification(df=interp_results["labeled_df"])

    ev = clf_results["evaluation"]
    logger.info("")
    logger.info("━━━ Pipeline Complete ━━━")
    logger.info("Clusters          : %d", cluster_results["optimal_k"])
    logger.info("DT train accuracy : %.4f", ev["train_accuracy"])
    logger.info("DT test  accuracy : %.4f", ev["test_accuracy"])
    logger.info("Models saved      : models/model_clustering.pkl  |  models/decision_tree_model.h5")
    return {"eda": eda_results, "preprocessing": prep_results,
            "clustering": cluster_results, "interpretation": interp_results,
            "classification": clf_results}


if __name__ == "__main__":
    main()
