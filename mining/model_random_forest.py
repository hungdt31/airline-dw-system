"""
Huấn luyện & đánh giá Random Forest (cùng hyperparameter với compare_5_models.py).
Chạy: python mining/model_random_forest.py
"""

from __future__ import annotations

import time

from sklearn.ensemble import RandomForestClassifier

from model_common import (
    DEFAULT_LIMIT,
    evaluate_classifier,
    prepare_train_test,
    print_metrics_terminal,
)


MODEL_NAME = "Random Forest"


def run(limit: int = DEFAULT_LIMIT):
    X_train, X_test, y_train, y_test, _, _delay_rate = prepare_train_test(limit=limit)

    t0 = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0

    metrics = evaluate_classifier(
        model, None, X_test, y_test, train_time_s, MODEL_NAME
    )
    print_metrics_terminal(metrics)
    return metrics


if __name__ == "__main__":
    run()
