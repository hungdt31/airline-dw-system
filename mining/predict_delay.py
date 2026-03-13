"""
mining/predict_delay.py
=======================
Data Mining — Dự đoán khả năng chuyến bay bị delay >= 15 phút
Thuật toán: Random Forest Classifier
"""

import sqlite3
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
import pickle
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MINING] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "warehouse", "airline_dw.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_delay.pkl")


# ─── 1. Trích xuất features từ DW ─────────────────────────

def load_features(db_path: str = DB_PATH, limit: int = 500_000) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    sql = f"""
        SELECT
            f.departure_delay,
            f.distance,
            f.elapsed_time,
            f.is_delayed_15          AS target,
            f.is_cancelled,
            d.month,
            d.day_of_week,
            d.is_weekend,
            d.quarter,
            CAST(SUBSTR(PRINTF('%04d', CAST(f.scheduled_departure AS INTEGER)),1,2) AS INTEGER)
                                     AS dep_hour,
            a.airline_iata,
            ap_o.state               AS origin_state,
            ap_d.state               AS dest_state
        FROM fact_flights f
        JOIN dim_date    d   ON f.date_key            = d.date_key
        JOIN dim_airline a   ON f.airline_key          = a.airline_key
        JOIN dim_airport ap_o ON f.origin_airport_key = ap_o.airport_key
        JOIN dim_airport ap_d ON f.dest_airport_key   = ap_d.airport_key
        WHERE f.is_cancelled = 0
        LIMIT {limit}
    """
    df = pd.read_sql_query(sql, conn)
    conn.close()
    log.info(f"Đã load {len(df):,} bản ghi cho Data Mining")
    return df


# ─── 2. Tiền xử lý ─────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    df = df.dropna(subset=["target", "month", "day_of_week", "airline_iata"])
    df["departure_delay"] = df["departure_delay"].fillna(0).clip(lower=0)
    df["distance"]        = df["distance"].fillna(df["distance"].median())
    df["elapsed_time"]    = df["elapsed_time"].fillna(df["elapsed_time"].median())
    df["dep_hour"]        = df["dep_hour"].fillna(12).clip(0, 23)

    le_airline = LabelEncoder()
    le_state_o = LabelEncoder()
    le_state_d = LabelEncoder()

    df["airline_enc"]  = le_airline.fit_transform(df["airline_iata"].astype(str))
    df["origin_enc"]   = le_state_o.fit_transform(df["origin_state"].fillna("UNK").astype(str))
    df["dest_enc"]     = le_state_d.fit_transform(df["dest_state"].fillna("UNK").astype(str))

    feature_cols = [
        "month", "day_of_week", "is_weekend", "quarter",
        "dep_hour", "distance", "elapsed_time",
        "airline_enc", "origin_enc", "dest_enc"
    ]

    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)

    encoders = {
        "airline": le_airline,
        "origin_state": le_state_o,
        "dest_state": le_state_d,
        "feature_cols": feature_cols
    }
    return X, y, encoders


# ─── 3. Training ───────────────────────────────────────────

def train(X, y) -> RandomForestClassifier:
    log.info(f"Training Random Forest — {len(X):,} mẫu, tỉ lệ delay: {y.mean():.2%}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred  = clf.predict(X_test)
    y_prob  = clf.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)

    log.info(f"Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}")
    print("\n" + classification_report(y_test, y_pred,
          target_names=["On-time", "Delayed"]))

    # Feature importance
    feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
    feat_imp = feat_imp.sort_values(ascending=False)
    print("Top feature importances:")
    print(feat_imp.head(10).to_string())

    return clf, X_test, y_test


# ─── 4. Save / Load model ──────────────────────────────────

def save_model(clf, encoders, path: str = MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump({"model": clf, "encoders": encoders}, f)
    log.info(f"Model đã lưu: {path}")


def load_model(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["encoders"]


# ─── 5. Predict function ───────────────────────────────────

def predict_flight(
    month: int, day_of_week: int, is_weekend: int, quarter: int,
    dep_hour: int, distance: float, elapsed_time: float,
    airline_iata: str, origin_state: str, dest_state: str,
    model_path: str = MODEL_PATH
) -> dict:
    """
    Dự đoán xác suất delay cho một chuyến bay cụ thể.
    Trả về: {"delayed": bool, "probability": float}
    """
    clf, encoders = load_model(model_path)
    le_airline = encoders["airline"]
    le_origin  = encoders["origin_state"]
    le_dest    = encoders["dest_state"]

    def safe_encode(le, val):
        try:
            return le.transform([val])[0]
        except ValueError:
            return 0   # unknown category

    feature_cols = encoders["feature_cols"]

    # Dùng DataFrame với đúng tên cột như lúc train → tránh UserWarning
    features = pd.DataFrame([{
        "month":        month,
        "day_of_week":  day_of_week,
        "is_weekend":   is_weekend,
        "quarter":      quarter,
        "dep_hour":     dep_hour,
        "distance":     distance,
        "elapsed_time": elapsed_time,
        "airline_enc":  safe_encode(le_airline, airline_iata),
        "origin_enc":   safe_encode(le_origin, origin_state),
        "dest_enc":     safe_encode(le_dest, dest_state),
    }])[feature_cols]   # đảm bảo thứ tự cột khớp với lúc fit

    prob = clf.predict_proba(features)[0][1]
    return {"delayed": prob >= 0.5, "probability": round(float(prob), 4)}


# ─── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Bắt đầu Data Mining pipeline")

    df = load_features()
    X, y, encoders = preprocess(df)
    clf, X_test, y_test = train(X, y)
    save_model(clf, encoders)

    # Demo dự đoán
    result = predict_flight(
        month=6, day_of_week=5, is_weekend=1, quarter=2,
        dep_hour=17, distance=1200, elapsed_time=160,
        airline_iata="AA", origin_state="CA", dest_state="NY"
    )
    print(f"\nDemo dự đoán: {result}")
    log.info("Data Mining hoàn thành!")
