"""
Единый путь для оценки показателей классификатора
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

def evaluate_model(y_true:pd.Series | np.ndarray, y_pred: np.ndarray):
    """
    Оценивает предсказания y_pred по roc-auc и pr-auc
    """
    roc_auc = roc_auc_score(y_true, y_pred)
    ap_score = average_precision_score(y_true, y_pred)

    return float(roc_auc), float(ap_score)