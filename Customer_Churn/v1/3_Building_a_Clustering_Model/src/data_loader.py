"""
data_loader.py
--------------
Responsible for one thing only: returning a raw pandas DataFrame.

Strategy
--------
1. If DATASET_PATH exists → load it.
2. Otherwise → generate a realistic synthetic retail-customer dataset
   and persist it to DATASET_PATH so subsequent runs reuse it.

The raw dataset intentionally includes:
  - ID / address / date columns (dropped during preprocessing)
  - ~3% missing values in selected columns
  - a small number of duplicate rows
"""

import uuid
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import (
    DATASET_PATH,
    SYNTHETIC_DUPE_ROWS,
    SYNTHETIC_N_ROWS,
    SYNTHETIC_SEED,
)
from src.logger import get_logger

logger = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """
    Load the retail customer dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataset ready for EDA.
    """
    if DATASET_PATH.exists():
        logger.info("Loading dataset from %s", DATASET_PATH)
        return _load_csv(DATASET_PATH)

    logger.warning(
        "Dataset not found at %s — generating synthetic data.", DATASET_PATH
    )
    df = _generate_synthetic_data()
    _persist(df)
    return df


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns.", *df.shape)
    return df


def _generate_synthetic_data() -> pd.DataFrame:
    """
    Generate a realistic retail customer churn dataset.

    Columns
    -------
    customer_id         Unique customer identifier  (to be dropped)
    transaction_id      Unique transaction ID        (to be dropped)
    account_id          Account identifier           (to be dropped)
    device_id           Device identifier            (to be dropped)
    ip_address          IPv4 address string          (to be dropped)
    merchant_id         Merchant identifier          (to be dropped)
    transaction_date    Date of last transaction     (to be dropped)
    age                 Customer age (18–75)
    gender              Male / Female / Other
    region              North / South / East / West
    tenure_months       How long they have been a customer (1–120)
    annual_spend_usd    Total spend in the past 12 months
    num_purchases       Number of transactions in the past 12 months
    avg_order_value     Average value per order (USD)
    returns_count       Number of returned items
    loyalty_tier        Bronze / Silver / Gold / Platinum
    last_purchase_days  Days since last purchase
    email_opt_in        Whether subscribed to marketing emails
    churned             Target label — 1 = churned, 0 = retained
    """
    rng = np.random.default_rng(SYNTHETIC_SEED)
    n   = SYNTHETIC_N_ROWS

    tenure          = rng.integers(1, 121, size=n)
    num_purchases   = rng.integers(1, 101, size=n)
    avg_order_value = rng.uniform(10, 500, size=n).round(2)
    annual_spend    = (num_purchases * avg_order_value * rng.uniform(0.8, 1.2, size=n)).round(2)
    returns_count   = rng.integers(0, 11, size=n)

    churn_prob = (
        0.05
        + 0.30 * (tenure < 12)
        + 0.20 * (num_purchases < 5)
        + 0.15 * (returns_count > 5)
        - 0.10 * (tenure > 60)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.90)
    churned    = rng.binomial(1, churn_prob)

    loyalty_tiers = rng.choice(
        ["Bronze", "Silver", "Gold", "Platinum"],
        size=n,
        p=[0.40, 0.30, 0.20, 0.10],
    )

    # ── Generate ID / date columns ────────────────────────────────────────────
    base_date = date(2023, 1, 1)
    tx_dates  = [
        (base_date + timedelta(days=int(d))).isoformat()
        for d in rng.integers(0, 365, size=n)
    ]

    def _fake_ip(r):
        return ".".join(str(r.integers(1, 255)) for _ in range(4))

    ip_addresses = [_fake_ip(rng) for _ in range(n)]

    df = pd.DataFrame(
        {
            # ── ID / date columns (will be dropped in preprocessing) ──────────
            "customer_id":      [f"C{i:05d}" for i in range(1, n + 1)],
            "transaction_id":   [f"TXN{i:06d}" for i in range(1, n + 1)],
            "account_id":       [f"ACC{i:05d}" for i in range(1, n + 1)],
            "device_id":        [f"DEV{i:05d}" for i in range(1, n + 1)],
            "ip_address":       ip_addresses,
            "merchant_id":      [f"MER{rng.integers(1, 51):03d}" for _ in range(n)],
            "transaction_date": tx_dates,
            # ── Feature columns ───────────────────────────────────────────────
            "age":              rng.integers(18, 76, size=n).astype(float),
            "gender":           rng.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.48, 0.04]),
            "region":           rng.choice(["North", "South", "East", "West"], size=n),
            "tenure_months":    tenure,
            "annual_spend_usd": annual_spend,
            "num_purchases":    num_purchases,
            "avg_order_value":  avg_order_value,
            "returns_count":    returns_count,
            "loyalty_tier":     loyalty_tiers,
            "last_purchase_days": rng.integers(1, 366, size=n).astype(float),
            "email_opt_in":     rng.choice([True, False], size=n, p=[0.65, 0.35]),
            "churned":          churned,
        }
    )

    # ── Inject ~3% missing values ─────────────────────────────────────────────
    for col in ["age", "annual_spend_usd", "last_purchase_days"]:
        mask = rng.random(size=n) < 0.03
        df.loc[mask, col] = np.nan

    # ── Inject duplicate rows ─────────────────────────────────────────────────
    dupe_idx = rng.choice(n, size=SYNTHETIC_DUPE_ROWS, replace=False)
    dupes    = df.iloc[dupe_idx].copy()
    df       = pd.concat([df, dupes], ignore_index=True)

    logger.info(
        "Generated synthetic dataset: %d rows × %d columns "
        "(%d intentional duplicates, ~3%% missing in 3 columns).",
        *df.shape,
        SYNTHETIC_DUPE_ROWS,
    )
    return df


def _persist(df: pd.DataFrame) -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    logger.info("Saved dataset to %s", DATASET_PATH)
