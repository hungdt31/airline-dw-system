"""
Trang Dự báo chuyến bay — Random Forest (mining/model_delay.pkl).
Bố cục và luồng khớp app.py (hàm page_predict).
"""

import plotly.graph_objects as go
import streamlit as st

from config import STATE_MAP
from db import query
from icons import (
    BOLT,
    BTN_PREDICT,
    CALENDAR_DOW,
    CALENDAR_MONTH,
    ERR,
    FORM_SECTION,
    GEO_DEST,
    GEO_ORIGIN,
    INPUT_DISTANCE,
    INPUT_DURATION,
    INPUT_HOUR,
    OK,
    PREDICT_AIRLINE,
    RESULT_SECTION,
    TARGET,
    TREND_UP,
    WARN,
)
from predict_service import get_trained_delay_model, predict_delay_row

BG = "#FAFAFA"

MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

DOW_LABELS = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]


def render_page_predict() -> None:
    st.markdown("> Nhập thông tin chuyến bay để dự báo khả năng delay")

    clf, encoders = get_trained_delay_model()
    if clf is None:
        st.warning(f"{WARN} Chưa có model. Chạy `python mining/predict_delay.py` để train.")
        return

    valid_airlines = sorted(encoders["airline"].classes_.tolist())
    valid_states = sorted(encoders["origin_state"].classes_.tolist())

    try:
        airlines_df = query("SELECT airline_iata, airline_name FROM dim_airline")
        airline_map = dict(zip(airlines_df["airline_iata"], airlines_df["airline_name"]))
    except Exception:
        airline_map = {}

    st.subheader(f"{FORM_SECTION} Thông tin chuyến bay")
    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox(
            f"{PREDICT_AIRLINE} Hãng hàng không",
            options=valid_airlines,
            format_func=lambda x: f"{x} — {airline_map.get(x, x)}",
        )
        origin_state = st.selectbox(
            f"{GEO_ORIGIN} Bang xuất phát",
            options=valid_states,
            format_func=lambda x: f"{x} — {STATE_MAP.get(x, x)}",
        )
        dest_state = st.selectbox(
            f"{GEO_DEST} Bang đích đến",
            options=valid_states,
            format_func=lambda x: f"{x} — {STATE_MAP.get(x, x)}",
            index=1,
        )

    with col2:
        month = st.selectbox(
            f"{CALENDAR_MONTH} Tháng bay",
            options=list(MONTH_NAMES.keys()),
            format_func=lambda x: MONTH_NAMES[x],
        )
        day_of_week = st.selectbox(
            f"{CALENDAR_DOW} Thứ trong tuần",
            options=[1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: DOW_LABELS[x - 1],
        )
        dep_hour = st.slider(f"{INPUT_HOUR} Giờ khởi hành", 0, 23, 14)

    with col3:
        distance = st.number_input(f"{INPUT_DISTANCE} Khoảng cách (dặm)", 50, 5000, 1000, 50)
        elapsed = st.number_input(f"{INPUT_DURATION} Thời gian bay (phút)", 30, 600, 150, 10)

    if st.button(f"{BTN_PREDICT} Dự báo ngay", type="primary"):
        #  width='stretch': sẽ tự động fill width của container
        if airline not in encoders["airline"].classes_:
            st.error(f"{ERR} Hãng '{airline}' không hợp lệ")
        elif origin_state not in encoders["origin_state"].classes_:
            st.error(f"{ERR} Bang '{origin_state}' không hợp lệ")
        else:
            out = predict_delay_row(
                month=month,
                day_of_week=day_of_week,
                dep_hour=dep_hour,
                distance=float(distance),
                elapsed_time=float(elapsed),
                airline_iata=airline,
                origin_state=origin_state,
                dest_state=dest_state,
            )
            if not out.get("ok"):
                st.error("Không thể dự báo.")
                return

            prob = float(out["probability"])
            delayed = out["delayed"]
            risk = out["risk_level"]

            st.markdown("---")
            st.subheader(f"{RESULT_SECTION} Kết quả dự báo")

            r1, r2, r3 = st.columns(3)
            r1.metric(f"{TARGET} Dự đoán", "CÓ DELAY" if delayed else "ĐÚNG GIỜ")
            r2.metric(f"{TREND_UP} Xác suất delay", f"{prob * 100:.1f}%")
            r3.metric(f"{BOLT} Mức độ rủi ro", risk)

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 32}},
                    title={"text": "Xác suất delay", "font": {"size": 18}},
                    delta={
                        "reference": 50,
                        "increasing": {"color": "red"},
                        "decreasing": {"color": "green"},
                    },
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {"color": "#E8593C" if delayed else "#27AE60"},
                        "steps": [
                            {"range": [0, 40], "color": "#d4edda"},
                            {"range": [40, 60], "color": "#fff3cd"},
                            {"range": [60, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig_gauge, width='stretch')

            if delayed:
                st.error(
                    f"{WARN} Chuyến bay **{airline}** từ **{origin_state}** đến **{dest_state}** "
                    f"lúc **{dep_hour:02d}:00** có **{prob * 100:.1f}%** khả năng bị delay ≥15 phút."
                )
            else:
                st.success(
                    f"{OK} Chuyến bay **{airline}** từ **{origin_state}** đến **{dest_state}** "
                    f"lúc **{dep_hour:02d}:00** được dự báo **đúng giờ** "
                    f"(xác suất delay: {prob * 100:.1f}%)."
                )
