import pandas as pd
import numpy as np
import joblib
import time

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from config import (
    PROCESSED_DATA_PATH, MODEL_OUTPUT_PATH,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET
)
from utils import load_parquet_dataset, drop_irrelevant_columns
import os

print('Loading and splitting dataset...')
df = load_parquet_dataset(PROCESSED_DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

os.makedirs('data/splits', exist_ok=True)
X_train.to_parquet('data/splits/X_train.parquet', index=False)
X_test.to_parquet('data/splits/X_test.parquet', index=False)
y_train.to_frame().to_parquet('data/splits/y_train.parquet', index=False)
y_test.to_frame().to_parquet('data/splits/y_test.parquet', index=False)



classifiers = [
    LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    KNeighborsClassifier(n_neighbors=11, weights='distance'),
    SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    GaussianNB(),
    DecisionTreeClassifier(max_depth=10, min_samples_leaf=3, random_state=42),
    RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=42),
    XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=5,
                 subsample=0.8, colsample_bytree=0.8,
                 tree_method='hist', eval_metric='logloss',
                 random_state=42, n_jobs=-1),
]

num_transformers = [StandardScaler(), MinMaxScaler()]
cat_transformers = [OneHotEncoder(drop='first', handle_unknown='ignore')]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorers = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

results = []
best_model = None
best_score = 0

print('Training and evaluating models...')
for clf in classifiers:
    for num_tr in num_transformers:
        for cat_tr in cat_transformers:

            preprocessor = ColumnTransformer([
                ('num', num_tr, NUMERICAL_FEATURES),
                ('cat', cat_tr, CATEGORICAL_FEATURES)
            ])

            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', clf)
            ])

            cv_res = cross_validate(
                pipeline, X_train, y_train,
                cv=cv, scoring=scorers, n_jobs=-1, return_estimator=False
            )

            start = time.time()
            pipeline.fit(X_train, y_train)
            end = time.time()

            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred)

            model_name = f'{clf.__class__.__name__} | {num_tr.__class__.__name__} | {cat_tr.__class__.__name__}'

            results.append({
                'model': model_name,
                'cv_accuracy_mean': cv_res['test_accuracy'].mean(),
                'cv_precision_mean': cv_res['test_precision'].mean(),
                'cv_recall_mean': cv_res['test_recall'].mean(),
                'cv_f1_mean': cv_res['test_f1'].mean(),
                'cv_auc_mean': cv_res['test_roc_auc'].mean(),
                'test_accuracy': acc,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'test_auc': roc_auc,
                'train_time_s': round(end - start, 2)
            })

            if f1 > best_score:
                best_model = pipeline
                best_score = f1

                os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

                joblib.dump(best_model, MODEL_OUTPUT_PATH)
                print(f'Saved best model: {model_name} with F1: {f1:.4f}')


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='cv_f1_mean', ascending=False)

results_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'model_results.csv')
os.makedirs(os.path.dirname(results_path), exist_ok=True)
results_df.to_csv(results_path, index=False)

print(results_df.head(10))