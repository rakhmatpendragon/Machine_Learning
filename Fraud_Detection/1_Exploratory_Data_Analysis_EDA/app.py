import pandas as pd
import sys

from src.data_loader import load_dataset
from src.eda import run_full_eda
from src.logger import get_logger

logger = get_logger("utama")

def main() -> dict:
    df = load_dataset()

    result = run_full_eda(df)

if __name__ == "__main__":
    main()