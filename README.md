# Telco Customer Churn – IBM Dataset

Predicting customer churn using structured data related to customer service from a fictional telecommunications provider. The goal of this project is to analyze customer behavior and identify the key factors that lead to churn. It represents a standard classification problem, commonly used in customer retention scenarios. The project includes data understanding, cleaning, exploratory data analysis (EDA), feature engineering, and machine learning (ML) modeling.

---

## Project Status

**Status:** In Progress – EDA completed, ML phase in preparation.  
The Machine Learning stage is currently in progress and will be added upon completion.

## Objectives and Research Questions

The main objective of this project is to predict whether a customer will churn based on behavioral, demographic, and service-related variables. The project aims to support customer retention strategies by identifying the most relevant features influencing churn.

**Research questions:**
- What are the main factors contributing to customer churn in the telecommunications sector?  
- How do customer demographics and service characteristics affect churn probability?  
- Can a predictive model accurately classify customers at risk of leaving?  
- Which features provide the strongest predictive power for churn detection?

## Data

**Source:** [Telco Customer Churn – IBM Dataset (Kaggle)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data)  
**Original reference:** [IBM Cognos Analytics Sample Data – Telco Churn](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

The dataset represents a fictional telecommunications company providing phone and Internet services to **7,043 customers in California** during Q3. It contains **33 variables**, including demographic, geographic, service-related, and account-level information, along with churn indicators and customer lifetime metrics.

- **Total records:** 7,043  
- **Total features:** 33  
- **Target variable:** `Churn Label` - indicates whether the customer left (`Yes`) or stayed (`No`) during the quarter.  

**Variable types (with examples):**
- **Numerical (e.g.):** `Tenure Months`, `Monthly Charge`, `Total Charges`, `CLTV`, `Churn Score`  
- **Categorical (e.g.):** `Gender`, `Internet Service`, `Contract`, `Payment Method`, `Dependents`, `Partner`  
- **Binary indicators (e.g.):** `Online Security`, `Tech Support`, `Device Protection`, `Streaming TV`, `Streaming Movies`  
- **Geographical (e.g.):** `State`, `City`, `Latitude`, `Longitude`, `Zip Code`

**Key features (preliminary):**
- `Tenure Months` – duration of the customer’s relationship with the company  
- `Contract` – type of subscription contract (Month-to-Month, One Year, Two Year)  
- `Internet Service` – category of internet service provided (DSL, Fiber Optic, Cable)  
- `Payment Method` – payment type selected by the customer  
- `Churn Score` – churn likelihood predicted by IBM SPSS Modeler  
- `CLTV` – predicted Customer Lifetime Value used to assess customer importance  

> These features were considered potentially relevant prior to EDA and model-based selection.  
> Their actual importance will be confirmed through exploratory analysis and machine learning experiments.

## Exploratory Data Analysis (EDA)

### Main Steps of the Analysis

- **Data completeness check:**  
  Missing values were analyzed (`isnull().sum()`) to identify columns with incomplete or irrelevant information (e.g., constant values, IDs, or high-cardinality features).

- **Data types and uniqueness:**  
  A summary table was created to review data types, the number of unique values, and potential column duplicates.

- **Distribution of numerical features:**  
  Histograms, boxplots, and descriptive statistics (mean, std, IQR) were used to identify outliers and understand variable distributions.

- **Correlation analysis:**  
  Pearson correlation heatmaps were computed for numerical features such as `tenure`, `monthly_charges`, `churn_score`, `cltv`, and `total_charges`.

- **Target vs. feature relationships:**  
  - For **numerical features:** mean and distribution comparisons across churned vs. retained customers.  
  - For **categorical features:** churn percentages per category (e.g., contract type, internet service, payment method).

---

### Key Insights

- Customers with **month-to-month contracts** show a significantly higher churn probability.  
- **Fiber optic users** are more likely to churn compared to DSL or non-internet users.  
- Lack of **additional services** such as *Online Security* or *Tech Support* correlates with higher churn rates.  
- Higher **Churn Scores** and lower **CLTV** values are strongly associated with churn, as expected.  
- The feature `churn_label` was removed due to redundancy with the target variable `churn_value`.  
- Location-related features (`zip_code`, `lat_long`, `state`) and identifiers (`customer_id`) were dropped as non-predictive.
More in notebook [`01_eda.ipynb`](./notebook/01_eda.ipynb)

---

### Visualizations Performed

- **Histograms and boxplots** for numerical features (`tenure`, `monthly_charges`, `cltv`, `total_charges`) – used to detect outliers and understand distributions.  
- **Correlation heatmap** – illustrated linear dependencies among numerical features and the target.  
- **Countplots and bar charts** for categorical features (`contract`, `internet_service`, `payment_method`) – compared churn vs. non-churn proportions.  
- **Target distribution plot (`churn_value`)** – showed that the dataset is slightly imbalanced but not severely skewed.

### EDA Summary and Feature Strategy

After completing exploratory analysis, a structured feature strategy was defined to prepare the dataset for modeling.

---

#### Columns Dropped
Features removed due to redundancy, lack of variance, or high cardinality:

- `customer_id` – unique identifier, no predictive value  
- `churn_label` – duplicate of the target variable (`churn_value`)  
- `churn_reason` – free-text feature with missing values; NLP not included in scope  
- `count`, `country`, `state` – constant or non-informative  
- `zip_code`, `lat_long`, `latitude`, `longitude` – unstructured geographic data, excluded from modeling  

```python
columns_to_drop = [
    'customer_id', 'count', 'country', 'state', 
    'zip_code', 'lat_long', 'latitude', 
    'longitude', 'churn_label', 'churn_reason'
```

---

**Feature Engineering**:
- `tenure_bins` - created by discretizing (binning) the variable tenure_months into 4 time intervals. (Type: categorical ('object'))

---

**Columns with type corrections**:
- `total_charges`: converted from `object` to `float` for numerical analysis

---

**Categorical features** (to be encoded):
- `gender`, `senior_citizen`, `partner`, `dependents`
- `phone_service`, `multiple_lines`
- `internet_service`, `online_security`, `online_backup`, `device_protection`, `tech_support`
- `streaming_tv`, `streaming_movies`
- `contract`, `payment_method`

```python
categorical_features = [
    'gender', 'senior_citizen', 'partner', 'dependents',
    'phone_service', 'multiple_lines', 'internet_service',
    'online_security', 'online_backup', 'device_protection',
    'tech_support', 'streaming_tv', 'streaming_movies',
    'contract', 'payment_method'
]

```

---

**Numerical features** (to be scaled):
- `tenure_months`, `monthly_charges`, `total_charges`, `cltv`, `churn_score`
```python
numerical_features = [
    'tenure_months', 'monthly_charges', 'total_charges', 'cltv', 'churn_score'
]
```

## Machine Learning

After completing the exploratory data analysis (EDA), the dataset was prepared for modeling. The initial experimentation was conducted in notebooks using a classical approach (manual transformations, model training), which was later structured into a modular pipeline for production-level automation.

### Key Steps:

- **Train/Test Split:**
  The dataset was split into training and test sets using `train_test_split` with stratification to preserve class balance.

  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
  ```
- **Handling Imbalanced Data**:
  The dataset exhibited a moderate class imbalance. To address this, the pipeline integrates **SMOTE** (Synthetic Minority Over-sampling Technique) within the training phase to oversample the minority class (`churn = 1`).

- **Feature Scaling**:
  Two scaling strategies were tested:
  - `StandardScaler` for z-score normalization
  - `MinMaxScaler` for range-based normalization

- **Categorical Encoding**:
  Categorical variables were encoded using:
  - `OneHotEncoder(drop='first')` to prevent multicollinearity
  - Unknown categories were handled with `handle_unknown='ignore'` to ensure pipeline stability on unseen data

All preprocessing steps were encapsulated in a `ColumnTransformer` and integrated with the model using a modular **Imbalanced-Learn Pipeline** (`ImbPipeline`), allowing seamless transformation, oversampling, and classification.

## Modeling

An experimental approach was used in the modeling phase, testing different classifiers and preprocessing configurations to find the best predictive pipeline. A modular solution was implemented with automated result comparison.

### Models tested:
The following classifiers were used with various combinations of scaling (`StandardScaler`, `MinMaxScaler`) and categorical encoding (`OneHotEncoder`):

- `Logistic Regression`
- `K-Nearest Neighbors`
- `Support Vector Classifier (SVC)`
- `Gaussian Naive Bayes`
- `Decision Tree`
- `Random Forest`
- `XGBoost`

### Validation and pipeline configuration:

- Data balanced with **SMOTE**
- Used an **`ImbPipeline`**-type pipeline
- Data preprocessing: `ColumnTransformer` with separate transformations for numerical and categorical features
- Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Metrics: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`

The full procedure and result comparison (CV + test) are available in the notebook: [`02_ml.ipynb`](./notebook/02_ml.ipynb)

**Best model:** Random Forest with MinMaxScaler + OneHotEncoder  
**Accuracy:** 93% | **F1:** 0.87 | **ROC AUC:** 0.92  (focus on F1 socre because we have unbalanced data, accuracy can be misleading

Confusion matrix and plots: see [`04_results_analysis.ipynb`](./notebook/04_results_analysis.ipynb)

## Final Conclusions

Based on the conducted analyses and modeling experiments, the following conclusions can be drawn:

### Key observations:
- Customers on **Month-to-Month** contracts have a significantly higher likelihood of churn than those on longer-term contracts.
- The absence of add-on services (e.g., `Online Security`, `Tech Support`) correlates with a higher churn rate.
- Users with **Fiber Optic** internet exhibit higher churn compared to DSL users or those without internet service.
- Low **CLTV (Customer Lifetime Value)** and a high **Churn Score** are strong predictors of churn.

## Project Structure

The project is organized in a modular and scalable way, following good practices of data engineering and machine learning workflows.

### Folder overview

```text
.
├─ data/                         # Raw and processed datasets
│  ├─ raw/                       # Original source data
│  ├─ processed/                 # Cleaned / transformed datasets
│  └─ splits/                    # Train–test split for modeling
│
├─ models/                       # Trained model(s) (.pkl files)
├─ reports/                      # Evaluation results, plots, predictions
├─ notebook/                     # EDA, ML training, and analysis notebooks
├─ src/                          # Core Python scripts (modular pipeline)
│
├─ requirements.txt              # Environment and dependencies
├─ .gitignore
└─ README.md

```


### src/ – Modular Scripts

| File | Description |
|------|-------------|
| `config.py` | Centralized config: file paths, column lists, global params |
| `etl.py` | ETL pipeline – cleaning, type conversion, feature engineering |
| `train_models.py` | Script for training multiple ML models with CV |
| `predict_best_model.py` | Loads the trained model and generates predictions |
| `utils.py` | Helper functions used across ETL and ML scripts |

---

### Modular architecture & automation

After initial prototyping in notebooks, the pipeline was migrated into **modular Python scripts**, allowing for:

- automation and reproducibility,
- scalable code structure,
- standalone execution of ETL, training, and prediction stages.

Each stage (ETL, ML, inference) is configurable via a central `config.py` file, making it easy to reuse or extend.

> This architecture makes the project reusable as a template for other classification tasks and ready for future expansion (e.g. with REST API or dashboards).



