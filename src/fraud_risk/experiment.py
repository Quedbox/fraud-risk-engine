# Сначала сохранение результатов эксперимента, а потом сохранение самой модели

# Тк это работа про сохранение данных, надо будет импортировать:
from sklearn.pipeline import Pipeline
from pathlib import Path
import pandas as pd
import json
import joblib
from catboost import CatBoost

def save_experiment(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Квадратные скобки нужные, чтобы словарь был спокойно пересобран в DataFrame
    pd.DataFrame([record]).to_csv(
        path, mode='a', header=not path.exists(), index=False
    )

    # mode='a' - режим работы с файлом - append - добавлять построчно, header=not path.exists() - создавать имена колонок только при создании файла

def save_params(params, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)   # Надо до конца разобраться, что значит эта строчка, тк она мелькает ну очень часто

    with path.open('w', encoding='utf-8') as file:   # Буквально: по пути открыть файл для записи в кодировке utf-8 и положить туда в формате JSON параметры модели 
        json.dump(params, file, indent=4, ensure_ascii=False)

def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

def get_model_params(model) -> dict:
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
    else:
        estimator = model

    return estimator.get_params(deep=False)