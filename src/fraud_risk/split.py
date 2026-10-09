import pandas as pd

def split_data(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает train, val, test выборки для входного датафрейма"""

    train = df[df['month'] < 6].copy()
    val = df[df['month'] == 6].copy()
    test = df[df['month'] == 7].copy()

    return train, val, test