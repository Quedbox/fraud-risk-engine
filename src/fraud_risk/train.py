"""
Модуль для единой тренировки моделей
"""

from fraud_risk.evaluate import evaluate_model

def train_and_eval_model(model, X_train, y_train, X_val, y_val, fit_kwargs=None):
    model.fit(X_train, y_train, **(fit_kwargs or {}))
    y_score = model.predict_proba(X_val)[:, 1]
    metrics = evaluate_model(y_true=y_val, y_score=y_score)
    return model, metrics

# Сильно проще реально перенести отдельный процесс в собственную функцию, чем разбрасывать его по разным модулям   