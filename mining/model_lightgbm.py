"""
Huấn luyện & đánh giá LightGBM (cùng hyperparameter với compare_5_models.py).
Cần: pip install lightgbm
Chạy: python mining/model_lightgbm.py
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional

from model_common import (
    DEFAULT_LIMIT,
    evaluate_classifier,
    prepare_train_test,
    print_metrics_terminal,
)

MODEL_NAME = "LightGBM"


def run(limit: int = DEFAULT_LIMIT) -> Optional[Dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ImportError:
        print("Lỗi: chưa cài lightgbm. Chạy: pip install lightgbm", file=sys.stderr)
        return None

    X_train, X_test, y_train, y_test, _, _delay_rate = prepare_train_test(limit=limit)

    t0 = time.perf_counter()
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0

    metrics = evaluate_classifier(
        model, None, X_test, y_test, train_time_s, MODEL_NAME
    )
    print_metrics_terminal(metrics)
    return metrics


if __name__ == "__main__":
    out = run()
    if out is None:
        sys.exit(1)
