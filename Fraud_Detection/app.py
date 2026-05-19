import pandas as pd
import sys

from src.data_loader import load_dataset
from src.eda import run_full_eda
from src.preprocessor import run_full_preprocessing
from src.clustering import run_full_clustering
from src.logger import get_logger

logger = get_logger("utama")

def main() -> dict:
    df_raw = load_dataset()

    result = run_full_eda(df_raw)

    pre_result  = run_full_preprocessing(df_raw, persist=True)
    df_clean    = pre_result["df_cleaned"]

    cluster_results = run_full_clustering(df=df_clean, use_elbow=True)

    logger.info("")
    logger.info("━━━ Pipeline Complete ━━━")
    logger.info(
        "Clean dataeset: %d rows x %d colums",
        *pre_result["df_cleaned"].shape,
    )

    return {
        "eda"           : result,
        "preprocessing" : pre_result,
        "clustering"    : cluster_results,
    }

if __name__ == "__main__":
    main()