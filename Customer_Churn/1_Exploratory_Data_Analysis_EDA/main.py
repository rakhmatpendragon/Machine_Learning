"""
main.py
-------
Entry point for the Retail Customer Churn EDA project.

Run:
    python main.py
"""

from src.data_loader import load_dataset
from src.eda import run_full_eda
from src.logger import get_logger

logger = get_logger("main")


def main() -> None:
    logger.info("━━━ Retail Customer Churn — EDA Pipeline ━━━")

    # Step 1: Load data
    df = load_dataset()

    # Step 2: Run full EDA (head → info → describe → deeper analysis → plots)
    results = run_full_eda(df)

    logger.info("Pipeline finished successfully.")
    return results


if __name__ == "__main__":
    main()
