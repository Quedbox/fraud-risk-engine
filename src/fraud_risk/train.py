"""
Единый путь тренировки и оценки baseline-моделей
"""

from fraud_risk.data import load_data, save_data
from fraud_risk.split import split_data
from fraud_risk.features import add_features
from fraud_risk.preprocessing import make_preprocessor
from fraud_risk.evaluate import evaluate_model

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from pathlib import Path

# from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# from xgboost import XGBClassifier

TARGET = 'fraud_bool'
RANDOM_STATE = 42
PROJECT_DIR = Path.cwd()
DATA_PATH = PROJECT_DIR / 'data' / 'raw' / 'Base.csv'

# Исключение 'month' вместе с TARGET может убрать потенциальный leakage связанный с изменением вероятности fraud с течением времени
EXCLUDED_COLS = [TARGET, 'month']


full_raw_data = load_data(DATA_PATH)
featured_full_data = add_features(full_raw_data)
train, val, test = split_data(featured_full_data)

X_train = train.drop(columns=EXCLUDED_COLS)
X_val = val.drop(columns=EXCLUDED_COLS)

y_train = train[TARGET]
y_val = val[TARGET]

numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
categorical_cols = X_train.select_dtypes(include=['str', 'category']).columns.tolist()

preprocessor = make_preprocessor(numeric_cols=numeric_cols, cat_cols=categorical_cols)

model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
y_score = model.predict_proba(X_val)[:,1]
roc_auc, ap_score = evaluate_model(y_val, y_score)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC (AP): {ap_score:.4f}")