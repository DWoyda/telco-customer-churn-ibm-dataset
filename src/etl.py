"""
ETL SCRIPT — Telco Customer Churn Project
-----------------------------------------
Purpose:
Automatically process raw Excel data (IBM Telco Churn)
into a clean, ready-to-use dataset in Parquet format.

Includes:
- column name cleaning
- type conversions
- removal of unnecessary columns
- feature engineering
- saving the data to a .parquet file

Author: OscarGoldman
"""
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
import pandas as pd
import numpy as np
from ogman import clean_columns
import os


def run_etl(input_path: str, output_path: str) -> None:
    """
    Runs the ETL process:
    1. Loads the source data
    2. Cleans and transforms the data
    3. Saves the final .parquet file

    Parameters
    ----------
    input_path : str
        Path to the input file (.xlsx)
    output_path : str
        Path to save the processed file (.parquet)
    """

    print('ETL started...')

    df = pd.read_excel(input_path)
    print(f'Data loaded. Shape: {df.shape}')

    df = clean_columns(df)
    print('Column names cleaned.')

    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    print(f'Converted "total_charges" to numeric. Missing values: {df['total_charges'].isna().sum()}')

    columns_to_drop = [
        'customer_id', 'count', 'country', 'state',
        'city', 'zip_code', 'lat_long', 'latitude',
        'longitude', 'churn_label', 'churn_reason'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    print(f'Dropped unnecessary columns. Remaining columns: {len(df.columns)}')

    df['tenure_bin'] = pd.cut(df['tenure_months'], bins=4).astype(str)
    print('Created "tenure_bin" feature.')

    df = df.dropna(subset=['total_charges'])
    print(f'Dropped rows with missing total_charges. Final shape: {df.shape}')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f'Data saved to {output_path}')

    print('ETL completed successfully.')

if __name__ == '__main__':
    run_etl(
        input_path=RAW_DATA_PATH,
        output_path=PROCESSED_DATA_PATH
    )
