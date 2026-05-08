import pandas as pd

from src.config import DATASET_PATH
from src.logger import get_logger

logger = get_logger("utama")

def main() -> None:
    logger.info("START")
    df = pd.read_csv(DATASET_PATH)

    print("Dataset Full Version")
    print(df)

    print("========== HEAD ==========")
    print(df.head())

    print("========== INFO ==========")
    print(df.info())

    print("========== DESCRIBE ==========")
    print(df.describe())

    logger.info("FINSIH")

if __name__ == "__main__":
    main()