# config.py

# Ścieżki do plików
RAW_DATA_PATH = 'data/raw/Telco_customer_churn.xlsx'
PROCESSED_DATA_PATH = 'data/processed/Telco_customer_churn_ML.parquet'
MODEL_OUTPUT_PATH = 'models/best_ImbPipeline.pkl'
PREDICTION_DATA_PATH = 'data/raw/new_customers_to_predict_src.xlsx'


# Kolumny do usunięcia
COLUMNS_TO_DROP = [
    'customer_id', 'count', 'country', 'state',
    'city', 'zip_code', 'lat_long', 'latitude',
    'longitude', 'churn_label', 'churn_reason'
]

# Cechy numeryczne (do skalowania)
NUMERICAL_FEATURES = [
    'tenure_months', 'monthly_charges',
    'total_charges', 'cltv', 'churn_score'
]

# Cechy kategoryczne (do enkodowania)
CATEGORICAL_FEATURES = [
    'gender', 'senior_citizen', 'partner', 'dependents',
    'phone_service', 'multiple_lines', 'internet_service',
    'online_security', 'online_backup', 'device_protection',
    'tech_support', 'streaming_tv', 'streaming_movies',
    'contract', 'payment_method'
]

# Target
TARGET = 'churn_value'