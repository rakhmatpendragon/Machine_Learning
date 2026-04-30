# Retail Customer Churn — EDA Case Study
A production-ready, scalable Python project for Exploratory Data Analysis
on a retail customer churn dataset.

## Project Structure
retail_eda/
├── main.py               ← Single entry point: python main.py
├── requirements.txt
├── src/
│   ├── config.py         ← All constants in one place (paths, seeds, settings)
│   ├── logger.py         ← Shared structured logger (no duplicated handlers)
│   ├── data_loader.py    ← Loads CSV or auto-generates synthetic retail data
│   └── eda.py            ← EDA logic: head / info / describe + analysis + plots
├── tests/
│   ├── test_data_loader.py  ← 12 tests (schema, ranges, uniqueness, reproducibility)
│   └── test_eda.py          ← 22 tests (return types, math invariants, mutations)
├── data/                 ← Dataset auto-created on first run
└── outputs/              ← 5 plots auto-saved here

## The 3 Required EDA Steps
|    Step    |       Function       | : What it does :                                    |
| ---------- | -------------------- | --------------------------------------------------- |
| head()     | display_head(df)     | Logs first 10 rows, returns DataFrame               |
| info()     | display_info(df)     | Logs dtypes, nulls, memory; returns structured dict |
| describe() | display_describe(df) | Logs numeric + categorical stats; returns DataFrame |

## Scalability Design Decisions
* config.py as single source of truth — change a path or seed in one place, everywhere updates
* Each function has one job — data_loader only loads, eda.py only analyses; adding a new data source (S3, DB) touches only data_loader.py
* Pure functions in EDA — every function takes a DataFrame and returns a value; no hidden state, easy to unit test
* run_full_eda() as orchestrator — swap steps in/out or reorder without touching internals
* Agg matplotlib backend — plots save silently to disk, works in scripts, notebooks, and CI equally

## Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## Run Tests
```bash
python -m pytest tests/ -v
```