import pandas as pd
from sklearn.model_selection import train_test_split

def load_parquet_dataset(path: str) -> pd.DataFrame:
    """Load dataset from a parquet file."""
    return pd.read_parquet(path)

def drop_irrelevant_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """Remove columns that are not relevant for modeling."""
    return df.drop(columns=columns_to_drop)

def split_data(df: pd.DataFrame, target: str, test_size=0.2, random_state=42):
    """Split the dataset into train and test sets with stratification."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def quick_summary(df: pd.DataFrame):
    """Return basic summary of the dataset (types, nulls, duplicates)."""
    summary = (
        df.dtypes.astype(str).to_frame('dtype')
          .join(df.nunique(dropna=False).to_frame('nunique'))
          .join(df.isnull().sum().to_frame('nulls'))
          .join((df.isnull().mean()*100).round(2).to_frame('nulls_percent'))
          .join(df.duplicated().to_frame('duplicated'))
    )
    return summary