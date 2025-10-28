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

---

### Visualizations Performed

- **Histograms and boxplots** for numerical features (`tenure`, `monthly_charges`, `cltv`, `total_charges`) – used to detect outliers and understand distributions.  
- **Correlation heatmap** – illustrated linear dependencies among numerical features and the target.  
- **Countplots and bar charts** for categorical features (`contract`, `internet_service`, `payment_method`) – compared churn vs. non-churn proportions.  
- **Target distribution plot (`churn_value`)** – showed that the dataset is slightly imbalanced but not severely skewed.

### EDA Summary and Feature Strategy

After completing exploratory analysis, a structured feature strategy was defined to prepare the dataset for modeling.

---

#### 🔻 Columns Dropped
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
