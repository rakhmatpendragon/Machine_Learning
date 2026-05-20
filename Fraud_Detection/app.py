import pandas as pd
import sys

from src.data_loader import load_dataset
from src.eda import run_full_eda
from src.preprocessor import run_full_preprocessing
from src.clustering import run_full_clustering
from src.cluster_interpreter import run_full_interpretation
from src.logger import get_logger

logger = get_logger("utama")

def main() -> dict:
    logger.info("━━━ Fraud Detection ━ ML Pipeline ━━━")

    df_raw = load_dataset()

    result = run_full_eda(df_raw)

    pre_result  = run_full_preprocessing(df_raw, persist=True)
    df_clean    = pre_result["df_cleaned"]

    cluster_results = run_full_clustering(df=df_clean, use_elbow=True)

    interp_result = run_full_interpretation(
        df     = cluster_results["df"],
        labels = cluster_results["labels"],
    )

    # labeled_df = interp_result["labeled_df"]
    logger.info("")
    logger.info("━━━ Pipeline Complete ━━━")
    logger.info("Clean dataset      : %d rows x %d colums", *df_clean.shape)
    logger.info("Clusters found     : %d", cluster_results["optimal_k"])
    logger.info("Model saved        : model/model_clustering.pkl")
    # logger.info("Labeled export     : data/retail_customers_labeled.csv  (%d rows x %d cols)",
    #             *labeled_df.shape)
    logger.info("Interpretation     : outputs/09… 12_*.{csv, png}")

    return {
        "eda"           : result,
        "preprocessing" : pre_result,
        "clustering"    : cluster_results,
        "interpretation": interp_result,
    }

if __name__ == "__main__":
    main()