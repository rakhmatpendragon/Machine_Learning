import numpy as np
import pandas as pd

from src.config import DATASET_PATH
from src.logger import get_logger

logger = get_logger(__name__)

def load_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        logger.info("Loading dataset from %s", DATASET_PATH)
        return _load_csv(DATASET_PATH)


def _load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows x %d columns", * df.shape)
    return df