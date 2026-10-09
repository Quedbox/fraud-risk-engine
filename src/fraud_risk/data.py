import pandas as pd
from pathlib import Path

def load_data(path:str | Path) -> pd.DataFrame:
    """Загружает данные из csv"""
    return pd.read_csv(path)

def save_data(df: pd.DataFrame, path:str | Path) -> None:
    """Сохраняет DataFrame в CSV-файл"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

# Разобрать строчку path.parent.mkdir(parents=True, exist_ok=True)
