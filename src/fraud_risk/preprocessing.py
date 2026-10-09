"""
Модуль, создающий единый инструмент препроцессинга
"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42

imputer = SimpleImputer(strategy='median')
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# sparse_output оставил по умолчанию True, тк это выходит в экономию оперативки без видимых минусов(Для Log. Reg. проблем нет)
scaler = MinMaxScaler()

def make_preprocessor(numeric_cols:list | tuple, cat_cols:list | tuple) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_cols),
        ('cat', cat_pipeline, cat_cols)
    ])