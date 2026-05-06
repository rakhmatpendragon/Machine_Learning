"""
config.py
---------
Central configuration for the Retail EDA project.
All tuneable constants live here — nothing is hardcoded elsewhere.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_FILENAME = "retail_customers.csv"
DATASET_PATH     = DATA_DIR / DATASET_FILENAME

# Number of rows shown by head()
HEAD_ROWS = 10

# ── Synthetic data generation (used when no real CSV is present) ──────────────
SYNTHETIC_N_ROWS    = 1_000
SYNTHETIC_SEED      = 42
SYNTHETIC_DUPE_ROWS = 20        # intentional duplicate rows injected for demo

# ── EDA display ──────────────────────────────────────────────────────────────
DESCRIBE_PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# ── Plotting ──────────────────────────────────────────────────────────────────
PLOT_STYLE        = "seaborn-v0_8-whitegrid"
PLOT_DPI          = 150
PLOT_FIGSIZE_WIDE = (14, 5)
PLOT_FIGSIZE_SQ   = (8, 6)

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Columns to drop: IDs, addresses, and date fields (no predictive value)
COLUMNS_TO_DROP = [
    "customer_id",
    "transaction_id",
    "account_id",
    "device_id",
    "ip_address",
    "merchant_id",
    "transaction_date",
]

# Categorical columns that will be label-encoded
CATEGORICAL_COLUMNS = ["gender", "region", "loyalty_tier"]

# Where to persist the cleaned + encoded dataset
PROCESSED_DATASET_PATH = DATA_DIR / "retail_customers_processed.csv"
