import pandas as pd
import joblib
import os

from config import (
    PREDICTION_DATA_PATH,
    MODEL_OUTPUT_PATH,
    COLUMNS_TO_DROP
)
from ogman import clean_columns

# 1. Wczytaj model
print('Loading model...')
model = joblib.load(MODEL_OUTPUT_PATH)

# 2. Wczytaj dane
print('Loading new customer data...')
df = pd.read_excel(PREDICTION_DATA_PATH)

# 3. Przetwórz dane (jak w ETL)
df = clean_columns(df)

if 'total_charges' in df.columns:
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')

if 'tenure_months' in df.columns:
    df['tenure_bin'] = pd.cut(df['tenure_months'], bins=4).astype(str)

df = df.dropna()

# 4. Predykcja
print('Predicting churn...')
preds = model.predict(df)
probs = model.predict_proba(df)[:, 1]

df['churn_prediction'] = preds
df['churn_probability'] = probs.round(3)

# 5. Zapis wyników
os.makedirs('reports', exist_ok=True)
df.to_csv('reports/predictions_src.csv', index=False)
print('Predictions saved to /reports/predictions_src.csv')
