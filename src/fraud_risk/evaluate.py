"""
Единый путь для оценки показателей классификатора
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


# Логичное обновление перенести часть функционала из train.py в функцию, где это по логике и должно быть(перебор по фпрам)
def evaluate_model(y_true:pd.Series | np.ndarray, y_score: np.ndarray):
    """
    Оценивает предсказания y_pred по roc-auc и pr-auc
    """

    metrics = {
        'roc_auc': float(roc_auc_score(y_true, y_score)),
        'ap' : float(average_precision_score(y_true, y_score))
    }

    fpr_limits = {
        '001': 0.001,
        '01': 0.01,
        '05': 0.05
    }

    for name, limit in fpr_limits.items():
        result = recall_by_thresholds(y_true, y_score, limit)

        metrics[f"recall_fpr_{name}"] = result['recall']
        metrics[f"fpr_{name}"] = result['fpr']
        metrics[f"threshold_{name}"] = result['threshold']

    return metrics

def recall_by_thresholds(y_true: pd.Series | np.ndarray, y_score: np.ndarray, max_fpr=0.05):
    """
    Получая на вход реальные метки и подсказаные в сочетании с ручную или по дефолту указанным значение FPR
    Возвращает Recall, FPR и порог predict_proba, при котором можно получить искомый FPR 
    """

    # Получает на основе вероятностей предсказаний fpr, tpr(recall) и порог классификации, для которого было посчитано это значение
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)

    # Индексы точек, где fpr меньше или равен целевому
    valid_indices = np.where(fpr <= max_fpr)[0]

    # Находим индекс с максимальным Recall
    best_idx = valid_indices[-1]

    # Возращаем найденные значения
    return {'recall': float(tpr[best_idx]),
            'fpr': float(fpr[best_idx]),
            'threshold': float(thresholds[best_idx])}
