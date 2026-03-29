"""
Huấn luyện & đánh giá Neural Network (MLPClassifier — cùng cấu hình compare_5_models.py).
Chạy: python mining/model_neural_network.py
"""

from __future__ import annotations

import time

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from model_common import (
    DEFAULT_LIMIT,
    evaluate_classifier,
    prepare_train_test,
    print_metrics_terminal,
)


MODEL_NAME = "Neural Network"


def run(limit: int = DEFAULT_LIMIT):
    X_train, X_test, y_train, y_test, _, _delay_rate = prepare_train_test(limit=limit)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)

    t0 = time.perf_counter()
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=512,
        learning_rate="adaptive",
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        verbose=False,
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
