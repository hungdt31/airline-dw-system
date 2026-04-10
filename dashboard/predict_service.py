"""
Tải model delay (XGBoost từ mining/predict_delay.py) một lần — dùng cho tab Dự báo.
"""

import os
import sys

import pandas as pd
import streamlit as st

# Gốc repo để import mining
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mining.predict_delay import MODEL_PATH, load_model  # noqa: E402


def quarter_from_month(month: int) -> int:
    return (month - 1) // 3 + 1


@st.cache_resource
def get_trained_delay_model():
    """
    Trả về (clf, encoders) hoặc (None, None) nếu chưa có model_delay.pkl.
    """
    if not os.path.isfile(MODEL_PATH):
        return None, None
    return load_model(MODEL_PATH)


def predict_delay_row(
    month: int,
    day_of_week: int,
    dep_hour: int,
    distance: float,
    elapsed_time: float,
    airline_iata: str,
    origin_state: str,
    dest_state: str,
) -> dict:
    """
    Dự đoán P(delay >= 15 phút). Cùng logic với mining.predict_delay.predict_flight.
    """
    clf, encoders = get_trained_delay_model()
    if clf is None:
        return {"ok": False, "error": "missing_model"}

    le_airline = encoders["airline"]
    le_origin = encoders["origin_state"]
    le_dest = encoders["dest_state"]
    feature_cols = encoders["feature_cols"]

    def safe_encode(le, val):
        try:
            return le.transform([val])[0]
        except ValueError:
            return 0

    is_weekend = int(day_of_week >= 6)
    quarter = quarter_from_month(month)

    features = pd.DataFrame(
        [
            {
                "month": month,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "quarter": quarter,
                "dep_hour": dep_hour,
                "distance": distance,
                "elapsed_time": elapsed_time,
                "airline_enc": safe_encode(le_airline, airline_iata),
                "origin_enc": safe_encode(le_origin, origin_state),
                "dest_enc": safe_encode(le_dest, dest_state),
            }
        ]
    )[feature_cols]

    prob = float(clf.predict_proba(features)[0][1])
    risk = "🔴 Cao" if prob >= 0.6 else ("🟡 Trung bình" if prob >= 0.4 else "🟢 Thấp")
    return {
        "ok": True,
        "delayed": prob >= 0.5,
        "probability": round(prob, 4),
        "risk_level": risk,
    }
