# Telco Customer Churn – IBM Dataset (EDA Phase)

## 📌 Project Overview

This repository contains the Exploratory Data Analysis (EDA) phase of a customer churn project based on a fictional dataset provided by IBM. The goal is to prepare the data for machine learning, understand customer behavior, and assess potential predictors of churn.

---

## 📊 Dataset Information

- **Source**: [IBM Community – Telco Customer Churn](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
- **License**: *Other (specified in description)* – no explicit license. Dataset is fictional and used for educational/demo purposes.
- **File**: `Telco_customer_churn.xlsx`
- **Size**: 7043 observations × 33 columns

ℹ️ The dataset simulates a telco company with customer-level details on contracts, services, and churn-related data.

---

## 📁 Project Structure


---

## 🧪 Exploratory Data Analysis Summary

All steps are included in `notebook/01_eda.ipynb`. Below is a professional summary:

### 1. Initial Checks
- Verified column types, missing values, duplicates, and cardinality.
- Generated summary table with:
  - Data type
  - Unique value count
  - Nulls (absolute & %)
  - Column-level duplicates (added manually)

### 2. Preliminary Feature Assessment
Based on initial review:

- ❌ **Dropped** features:
  - ID-like: `customer_id`, `count`
  - Location: `country`, `state`, `city`, `zip_code`, `lat_long`, `latitude`, `longitude`
  - Redundant / unusable text: `churn_reason`, `churn_label` (duplicate of target), high-cardinality text
- ⚠️ **To cast**: 
  - `total_charges` – from `object` to `float`
- ✅ **Useful**:
  - Demographic and service features for modeling
  - `cltv`, `churn_score`, `tenure_months`, etc.

### 3. Feature Engineering
- Created:
  - `tenure_bin` – binned version of `tenure_months` for categorical analysis

### 4. Data Quality Insights
- Only `churn_reason` had significant missing values (73%) – aligned with domain (only churners have reasons).
- No full-row duplicates.
- Feature types and distributions reviewed via histograms and barplots.
- Correlation heatmaps used to assess relationships.

### 5. Variable Categorization

| Type        | Variables |
|-------------|-----------|
| **Numerical** | `tenure_months`, `monthly_charges`, `total_charges`, `churn_score`, `cltv` |
| **Categorical** | `gender`, `partner`, `contract`, `payment_method`, `internet_service`, etc. |
| **Dropped** | `customer_id`, `country`, `state`, `zip_code`, `latitude`, `longitude`, `churn_reason`, `churn_label`, etc. |

---

## 🧾 Output

- Cleaned dataset saved as `.parquet`:  
  `data/processed/Telco_customer_churn_ML.parquet`

- Ready for input into ML pipeline (modeling phase).

---

## 📦 Setup & Requirements

Install necessary libraries:

```bash
pip install -r requirements.txt
