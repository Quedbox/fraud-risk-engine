"""
Точка входа в программу - единый запуск процесса от загрузки до оценки модели
"""

import json

from fraud_risk.data import load_data
from fraud_risk.split import split_data
from fraud_risk.features import add_features
from fraud_risk.preprocessing import make_preprocessor
from fraud_risk.train import train_and_eval_model
from fraud_risk.experiment import get_model_params

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
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / 'data' / 'raw' / 'Base.csv'

# Исключение 'month' вместе с TARGET может убрать потенциальный leakage связанный с изменением вероятности fraud с течением времени
EXCLUDED_COLS = [TARGET, 'month']

def main():
    full_raw_data = load_data(DATA_PATH)
    featured_full_data = add_features(full_raw_data)
    train, val, test = split_data(featured_full_data)

    X_train = train.drop(columns=EXCLUDED_COLS)
    X_val = val.drop(columns=EXCLUDED_COLS)

    y_train = train[TARGET]
    y_val = val[TARGET]

    numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['string', 'category','object']).columns.tolist()

    models = {
        'Dummy': DummyClassifier(),
        'LogisticRegression': Pipeline([
            ('preprocessing', make_preprocessor(numeric_cols=numeric_cols, cat_cols=categorical_cols)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        'CatBoost': CatBoostClassifier(iterations=2000,
                                       depth=6,
                                       learning_rate=0.05,
                                       random_seed=RANDOM_STATE,
                                       cat_features=categorical_cols, 
                                       verbose=False, 
                                       eval_metric='PRAUC')
    }

    for model_name, model in models.items():
        fit_kwargs = {}
        fit_config = {}

        if model_name == "CatBoost":
            fit_kwargs = {
                'eval_set': (X_val, y_val),
                'early_stopping_rounds': 1000,
                'use_best_model': True,
            }

            fit_config = {
                "early_stopping_rounds": fit_kwargs["early_stopping_rounds"],
                "use_best_model": fit_kwargs["use_best_model"],
                "eval_set_month": 6,
            }

        config = {
            'model_name': model_name,
            'hyperparameters': get_model_params(model),

            'dataset': 'BAF_Base_v1',
            'feature_version': 'sentinel_v1',

            'preprocessing': ('median_minmax_ohe_v1' if isinstance(model, Pipeline) else 'none'),

            'split': {'train_months': [0,1,2,3,4,5],
                      'validation_month': 6,
                      'test_month': 7},

            'fit': fit_config,
        }

        config_json = json.dumps(config, sort_keys=True, ensure_ascii=False, indent=4)
        print(config_json)
        

    
    # model, metrics = train_and_eval_model(models['LogisticRegression'], X_train, y_train, X_val, y_val)
    # for name, value in metrics.items():
    #     print(f"{name} : {value:.4f}")

if __name__ == '__main__':
    main()