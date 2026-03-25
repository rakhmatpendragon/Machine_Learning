# 📊 Basic Machine Learning Workflow (Step-by-Step)

This project demonstrates a simple, structured approach to building a Machine Learning pipeline, starting from dataset creation to preprocessing. It is designed for beginners who want to understand the fundamental steps before applying ML to real-world problems.

---

## 🚀 Project Overview

This repository covers:

1. Dataset Creation (manual & random)
2. Exploratory Data Analysis (EDA)
3. Data Visualization
4. Data Cleaning & Preprocessing

---

## 📁 1. Dataset Creation

### Option A: Manual Dataset

Create a dataset using a Python dictionary and convert it into a DataFrame.

### Option B: Random Dataset

Generate realistic synthetic data using NumPy:

* Numerical features: `age`, `salary`
* Categorical features: `gender`
* Target variable: `purchased` (based on simple logic + randomness)

---

## 🔍 2. Exploratory Data Analysis (EDA)

Basic inspection of dataset:

* `head()` → View first rows
* `info()` → Check data types and non-null values
* `describe()` → Summary statistics

---

## 📊 3. Data Visualization

### Histogram (All Columns)

* Numerical → Histogram
* Categorical → Bar chart (frequency)

### Key Visualizations:

* Distribution of numerical features (age, salary)
* Category distribution (gender, purchased)
* Correlation heatmap
* Boxplot (outlier detection)

### Improvement:

* Use `rwidth` in histogram to add spacing between bars

---

## 🧹 4. Data Cleaning & Preprocessing

### ✔ Check Data Quality

* Missing values → `isnull().sum()`
* Duplicate data → `duplicated().sum()`

### ✔ Handle Issues

* Remove missing values → `dropna()`
* Remove duplicates → `drop_duplicates()`

### ✔ Drop Irrelevant Columns

Remove columns like:

* TransactionID
* AccountID
* DeviceID
* IPAddress
* MerchantID
* TransactionDate

### ✔ Encode Categorical Data

Convert categorical features into numeric using Label Encoding.

---

## 🎯 Final Output

After preprocessing:

* Clean dataset (no nulls, no duplicates)
* Only relevant features
* All data in numeric format (ready for ML models)

---

## 🧠 Next Steps (Future Work)

* Feature selection
* Train/test split
* Model training (e.g., classification)
* Model evaluation

---

## 💡 Notes

* This project focuses on **fundamentals**
* Uses simple logic to simulate real-world data
* Designed to be extended into real ML applications

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib / Seaborn
* Scikit-learn

---

## 📌 Purpose

To build a **strong foundation in Machine Learning workflow** before moving into more complex, real-world systems.

---
