import pandas as pd
import numpy as np

MINUS_ONE_COLS = ('prev_address_months_count', 'current_address_months_count', 'credit_risk_score', 'bank_months_count', 'session_length_in_minutes', 'device_distinct_emails_8w')

def minus_one_marker(df:pd.DataFrame) -> pd.DataFrame:
    """ Создает булевую колонку для тех колонок, что содержат информативные -1 значения """

    df = df.copy()
    for col in MINUS_ONE_COLS:
        df[f'{col}_missing'] = (df[col] == -1).astype(int)
    return df

def minus_one_del(df:pd.DataFrame) -> pd.DataFrame:
    """Заменяет значение -1, несущее значение "Пропуск" на np.nan"""
    
    df = df.copy()
    for col in MINUS_ONE_COLS:
        df[col] = df[col].replace(-1, np.nan)
    return df

def add_features(df:pd.DataFrame) -> pd.DataFrame:
    featured_df = df.copy()
    minus_one_df = minus_one_marker(featured_df)
    cleared_df = minus_one_del(minus_one_df)

    return cleared_df