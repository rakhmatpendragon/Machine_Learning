"""
data_loader.py — Load or generate the retail customer dataset.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import (DATASET_PATH, SYNTHETIC_DUPE_ROWS,
                         SYNTHETIC_N_ROWS, SYNTHETIC_SEED)
from src.logger import get_logger

logger = get_logger(__name__)


def load_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        logger.info("Loading dataset from %s", DATASET_PATH)
        return _load_csv(DATASET_PATH)
    logger.warning("Dataset not found — generating synthetic data.")
    df = _generate_synthetic_data()
    _persist(df)
    return df


def _load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns.", *df.shape)
    return df


def _generate_synthetic_data() -> pd.DataFrame:
    rng = np.random.default_rng(SYNTHETIC_SEED)
    n   = SYNTHETIC_N_ROWS

    tenure          = rng.integers(1, 121, size=n)
    num_purchases   = rng.integers(1, 101, size=n)
    avg_order_value = rng.uniform(10, 500, size=n).round(2)
    annual_spend    = (num_purchases * avg_order_value * rng.uniform(0.8, 1.2, size=n)).round(2)
    returns_count   = rng.integers(0, 11, size=n)

    churn_prob = np.clip(
        0.05 + 0.30*(tenure<12) + 0.20*(num_purchases<5)
             + 0.15*(returns_count>5) - 0.10*(tenure>60),
        0.02, 0.90,
    )
    churned = rng.binomial(1, churn_prob)

    loyalty_tiers = rng.choice(["Bronze","Silver","Gold","Platinum"], size=n,
                                p=[0.40,0.30,0.20,0.10])
    base_date = date(2023, 1, 1)
    tx_dates  = [(base_date + timedelta(days=int(d))).isoformat()
                 for d in rng.integers(0, 365, size=n)]
    ip_addresses = [".".join(str(rng.integers(1,255)) for _ in range(4)) for _ in range(n)]

    df = pd.DataFrame({
        "customer_id":      [f"C{i:05d}" for i in range(1, n+1)],
        "transaction_id":   [f"TXN{i:06d}" for i in range(1, n+1)],
        "account_id":       [f"ACC{i:05d}" for i in range(1, n+1)],
        "device_id":        [f"DEV{i:05d}" for i in range(1, n+1)],
        "ip_address":       ip_addresses,
        "merchant_id":      [f"MER{rng.integers(1,51):03d}" for _ in range(n)],
        "transaction_date": tx_dates,
        "age":              rng.integers(18, 76, size=n).astype(float),
        "gender":           rng.choice(["Male","Female","Other"], size=n, p=[0.48,0.48,0.04]),
        "region":           rng.choice(["North","South","East","West"], size=n),
        "tenure_months":    tenure,
        "annual_spend_usd": annual_spend,
        "num_purchases":    num_purchases,
        "avg_order_value":  avg_order_value,
        "returns_count":    returns_count,
        "loyalty_tier":     loyalty_tiers,
        "last_purchase_days": rng.integers(1, 366, size=n).astype(float),
        "email_opt_in":     rng.choice([True, False], size=n, p=[0.65, 0.35]),
        "churned":          churned,
    })

    for col in ["age", "annual_spend_usd", "last_purchase_days"]:
        mask = rng.random(size=n) < 0.03
        df.loc[mask, col] = np.nan

    dupes = df.iloc[rng.choice(n, size=SYNTHETIC_DUPE_ROWS, replace=False)].copy()
    df    = pd.concat([df, dupes], ignore_index=True)

    logger.info("Generated %d rows × %d columns (%d dupes, ~3%% missing).",
                *df.shape, SYNTHETIC_DUPE_ROWS)
    return df


def _persist(df: pd.DataFrame) -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    logger.info("Saved dataset to %s", DATASET_PATH)
