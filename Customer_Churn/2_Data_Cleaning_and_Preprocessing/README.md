# Retail Customer Churn — EDA Case Study

A production-ready, scalable Python project for Exploratory Data Analysis
on a retail customer churn dataset.

## Project Structure

```
retail_eda/
├── data/                  # Raw and processed datasets
├── src/
│   ├── config.py          # Central configuration
│   ├── data_loader.py     # Dataset loading logic
│   ├── eda.py             # EDA: head / info / describe + analysis
│   └── logger.py          # Shared logger
├── tests/
│   ├── test_data_loader.py
│   └── test_eda.py
├── outputs/               # EDA reports / plots (auto-generated)
├── main.py                # Entry point
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Run Tests

```bash
python -m pytest tests/ -v
```
