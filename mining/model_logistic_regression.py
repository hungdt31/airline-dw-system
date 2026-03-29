"""
Huấn luyện & đánh giá Logistic Regression (cùng hyperparameter với compare_5_models.py).
Chạy: python mining/model_logistic_regression.py
"""

from __future__ import annotations

import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model_common import (
    DEFAULT_LIMIT,
    evaluate_classifier,
    prepare_train_test,
    print_metrics_terminal,
)


MODEL_NAME = "Logistic Regression"


def run(limit: int = DEFAULT_LIMIT):
    X_train, X_test, y_train, y_test, _, _delay_rate = prepare_train_test(limit=limit)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)

    t0 = time.perf_counter()
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="saga",
        max_iter=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_train)
    train_time_s = time.perf_counter() - t0

    metrics = evaluate_classifier(
        model, scaler, X_test, y_test, train_time_s, MODEL_NAME
    )
    print_metrics_terminal(metrics)
    return metrics


if __name__ == "__main__":
    run()
