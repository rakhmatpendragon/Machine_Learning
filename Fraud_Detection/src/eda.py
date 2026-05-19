import pandas as pd

from src.config import (
    HEAD_ROWS,
    OUTPUT_DIR
)
from src.logger import get_logger
from src.utils import banner

logger = get_logger(__name__)

def display_head(df: pd.DataFrame, n: int = HEAD_ROWS) -> pd.DataFrame:
    banner(f"1. Dataset Preview — head({n})")
    head = df.head(n)
    logger.info("\n%s", head.to_string())
    return head

def display_info(df: pd.DataFrame) -> dict:
    banner(f"2. Dataset Information — info()")
    info = df.info()
    return info

def display_describe(df: pd.DataFrame) -> dict:
    banner(f"3. Descriptive Statistics — describe()")
    describe = df.describe()
    logger.info("\n%s", describe.to_string())
    return describe


def run_full_eda(df: pd.DataFrame) -> dict:
    head         = display_head(df)
    info_summary = display_info(df)
    stats        = display_describe(df)

    banner("EDA Complete")
    logger.info("All outputs saved to: %s", OUTPUT_DIR)

    return {
        "head":         head,
        "info_summary": info_summary,
        "describe":     stats
    }