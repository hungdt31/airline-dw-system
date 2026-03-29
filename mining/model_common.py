"""
Dữ liệu, tiền xử lý và đánh giá dùng chung cho các script train từng mô hình.
Logic khớp với mining/compare_5_models.py (cùng SQL, feature, split).
"""

from __future__ import annotations

import os
import sqlite3
import warnings
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Đường tới warehouse (cùng compare_5_models)
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "warehouse", "airline_dw.db")

DEFAULT_LIMIT = 500_000
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def load_data(limit: int = DEFAULT_LIMIT) -> Tuple[pd.DataFrame, float]:
    """Nạp mẫu từ DW; trả về (df, tỉ lệ delay)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"""
        SELECT
            f.distance, f.elapsed_time,
            f.is_delayed_15  AS target,
            d.month, d.day_of_week, d.is_weekend, d.quarter,
            CAST(SUBSTR(PRINTF('%04d', CAST(f.scheduled_departure AS INTEGER)),1,2)
                 AS INTEGER) AS dep_hour,
            a.airline_iata,
            ap_o.state AS origin_state,
            ap_d.state AS dest_state
        FROM fact_flights f
        JOIN dim_date    d    ON f.date_key           = d.date_key
        JOIN dim_airline a    ON f.airline_key         = a.airline_key
        JOIN dim_airport ap_o ON f.origin_airport_key = ap_o.airport_key
        JOIN dim_airport ap_d ON f.dest_airport_key   = ap_d.airport_key
        WHERE f.is_cancelled = 0
        LIMIT {limit}
        """,
        conn,
    )
    conn.close()
    delay_rate = float(df["target"].mean())
    return df, delay_rate


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Làm sạch, encode categorical; trả X, y, tên cột feature."""
    df = df.dropna(subset=["target", "month", "day_of_week", "airline_iata"])
    df["distance"] = df["distance"].fillna(df["distance"].median())
    df["elapsed_time"] = df["elapsed_time"].fillna(df["elapsed_time"].median())
    df["dep_hour"] = df["dep_hour"].fillna(12).clip(0, 23)

    le_airline = LabelEncoder()
    le_origin = LabelEncoder()
    le_dest = LabelEncoder()

    df["airline_enc"] = le_airline.fit_transform(df["airline_iata"].astype(str))
    df["origin_enc"] = le_origin.fit_transform(df["origin_state"].fillna("UNK").astype(str))
    df["dest_enc"] = le_dest.fit_transform(df["dest_state"].fillna("UNK").astype(str))

    feature_cols = [
        "month",
        "day_of_week",
        "is_weekend",
        "quarter",
        "dep_hour",
        "distance",
        "elapsed_time",
        "airline_enc",
        "origin_enc",
        "dest_enc",
    ]
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)
    return X, y, feature_cols


def prepare_train_test(
    limit: int = DEFAULT_LIMIT,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[Any, Any, Any, Any, list, float]:
    """
    Load → preprocess → stratified split.
    Trả X_train, X_test, y_train, y_test, feature_cols, delay_rate.
    """
    df, delay_rate = load_data(limit=limit)
    X, y, feature_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols, delay_rate


def evaluate_classifier(
    model: Any,
    scaler: Optional[Any],
    X_test: Any,
    y_test: Any,
    train_time_s: float,
    model_name: str,
) -> Dict[str, Any]:
    """
    Tính các chỉ số giống compare_5_models.evaluate_all (không gồm ROC/CM để gọn).
    F1 dùng average='weighted' như file gốc.
    """
    X = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return {
        "name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "avg_precision": float(average_precision_score(y_test, y_prob)),
        "log_loss": float(log_loss(y_test, y_prob)),
        "brier": float(brier_score_loss(y_test, y_prob)),
        "train_time_s": float(train_time_s),
    }


def print_metrics_terminal(m: Dict[str, Any]) -> None:
    """In bảng chỉ số ra terminal."""
    name = m.get("name", "Model")
    print()
    print("=" * 56)
    print(f"  {name}")
    print("=" * 56)
    print(f"  Accuracy       : {m['accuracy']:.6f}")
    print(f"  AUC-ROC        : {m['auc_roc']:.6f}")
    print(f"  F1-score       : {m['f1']:.6f}")
    print(f"  Avg Precision  : {m['avg_precision']:.6f}")
    print(f"  Log Loss       : {m['log_loss']:.6f}")
    print(f"  Brier Score    : {m['brier']:.6f}")
    print(f"  Train time (s) : {m['train_time_s']:.3f}")
    print("=" * 56)
    print()
