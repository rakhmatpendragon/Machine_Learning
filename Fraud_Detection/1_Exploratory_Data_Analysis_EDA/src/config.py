from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = BASE_DIR / "outputs"

DATASET_FILENAME = "bank_transactions_data_edited.csv"
DATASET_PATH     = DATA_DIR / DATASET_FILENAME

HEAD_ROWS = 5

COLUMNS_TO_DROP = [
    "TransactionID",
    "AccountID",
    "DeviceID",
    "IP Address",
    "MerchantID",
    "TransactionDate",
]

CATEGORICAL_COLUMNS = [
    "TransactionType",
    "Channel",
    "CustomerOccupation",
]

PROCESSED_DATASET_PATH = DATA_DIR / "bank_transactions_data_processed.csv"