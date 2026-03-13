"""
dashboard/app.py
================
Streamlit Dashboard — 2 cấp phân tích hàng không
Chạy: streamlit run dashboard/app.py
"""

import sqlite3
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "warehouse", "airline_dw.db")

st.set_page_config(
    page_title="Airline Data Warehouse",
    page_icon="✈️",
    layout="wide"
)

# ─── DB helper ─────────────────────────────────────────────

@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data(ttl=300)
def query(sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, get_conn())

# ─── Header ────────────────────────────────────────────────
st.title("Airline Data Warehouse Dashboard")
st.caption("CDIO Project | Dữ liệu: Kaggle US Flight Delays 2015")

# ═══════════════════════════════════════════════════════════
# SIDEBAR — BỘ LỌC
# ═══════════════════════════════════════════════════════════
st.sidebar.header("🔍 Bộ lọc")

# ── 1. Thời gian: chọn chế độ Tháng HOẶC Quý ──────────────
st.sidebar.markdown("**📅 Thời gian**")
time_mode = st.sidebar.radio(
    "Lọc theo",
    options=["Tháng", "Quý"],
    horizontal=True,
    label_visibility="collapsed"
)

time_filter = "1=1"   # default: không lọc

if time_mode == "Tháng":
    months_df    = query("SELECT DISTINCT month, month_name FROM dim_date ORDER BY month")
    month_labels = dict(zip(months_df["month"], months_df["month_name"]))
    sel_months   = st.sidebar.multiselect(
        "Chọn tháng",
        options=months_df["month"].tolist(),
        format_func=lambda x: month_labels.get(x, x),
        default=[]
    )
    if sel_months:
        time_filter = f"d.month IN ({','.join(map(str, sel_months))})"
else:
    sel_quarters = st.sidebar.multiselect(
        "Chọn quý",
        options=[1, 2, 3, 4],
        format_func=lambda x: f"Q{x}",
        default=[]
    )
    if sel_quarters:
        time_filter = f"d.quarter IN ({','.join(map(str, sel_quarters))})"

st.sidebar.markdown("---")

# ── 2. Hãng hàng không ─────────────────────────────────────
airlines_df    = query("SELECT airline_iata, airline_name FROM dim_airline ORDER BY airline_name")
airline_labels = dict(zip(airlines_df["airline_iata"], airlines_df["airline_name"]))
sel_airlines   = st.sidebar.multiselect(
    "🏢 Hãng hàng không",
    options=airlines_df["airline_iata"].tolist(),
    format_func=lambda x: airline_labels.get(x, x),
    default=[]
)

st.sidebar.markdown("---")

# Bảng tên đầy đủ các bang Mỹ
STATE_MAP = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
    "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
    "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
    "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina",
    "ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania",
    "RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee",
    "TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
    "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming","DC":"District of Columbia",
    "PR":"Puerto Rico","VI":"Virgin Islands","GU":"Guam","TT":"Trust Territories",
}

# ── Load toàn bộ airport 1 lần ────────────────────────────
all_airports_df = query("""
    SELECT airport_iata,
           COALESCE(state, '')  AS state,
           COALESCE(city, '')   AS city,
           airport_iata || ' — ' || COALESCE(city, airport_iata) AS label
    FROM dim_airport
    ORDER BY state, airport_iata
""")
# Chỉ lấy các state thực sự có trong dataset
all_states = sorted([s for s in all_airports_df["state"].unique().tolist() if s])
state_options = ["(Tất cả)"] + all_states

def state_label(code):
    return f"{code} — {STATE_MAP.get(code, code)}" if code != "(Tất cả)" else "(Tất cả)"

# ── 3. Điểm đi: chọn State → Airport ──────────────────────
st.sidebar.markdown("**🛫 Điểm đi**")

# Dùng session_state để reset multiselect khi đổi bang
if "prev_origin_state" not in st.session_state:
    st.session_state["prev_origin_state"] = "(Tất cả)"

sel_origin_state = st.sidebar.selectbox(
    "Bang xuất phát",
    options=state_options,
    format_func=state_label,
    key="origin_state"
)
# Khi đổi bang → xóa các sân bay đã chọn để tránh lỗi option không còn tồn tại
if sel_origin_state != st.session_state["prev_origin_state"]:
    st.session_state["prev_origin_state"] = sel_origin_state
    st.session_state["sel_origin_airports"] = []

origin_airport_opts = (
    all_airports_df if sel_origin_state == "(Tất cả)"
    else all_airports_df[all_airports_df["state"] == sel_origin_state]
)
origin_ap_labels = dict(zip(origin_airport_opts["airport_iata"], origin_airport_opts["label"]))
sel_origin = st.sidebar.multiselect(
    "Sân bay đi",
    options=origin_airport_opts["airport_iata"].tolist(),
    format_func=lambda x: origin_ap_labels.get(x, x),
    key="sel_origin_airports"
)

st.sidebar.markdown("---")

# ── 4. Điểm đến: chọn State → Airport ─────────────────────
st.sidebar.markdown("**🛬 Điểm đến**")

if "prev_dest_state" not in st.session_state:
    st.session_state["prev_dest_state"] = "(Tất cả)"

sel_dest_state = st.sidebar.selectbox(
    "Bang đích đến",
    options=state_options,
    format_func=state_label,
    key="dest_state"
)
if sel_dest_state != st.session_state["prev_dest_state"]:
    st.session_state["prev_dest_state"] = sel_dest_state
    st.session_state["sel_dest_airports"] = []

dest_airport_opts = (
    all_airports_df if sel_dest_state == "(Tất cả)"
    else all_airports_df[all_airports_df["state"] == sel_dest_state]
)
dest_ap_labels = dict(zip(dest_airport_opts["airport_iata"], dest_airport_opts["label"]))
sel_dest = st.sidebar.multiselect(
    "Sân bay đến",
    options=dest_airport_opts["airport_iata"].tolist(),
    format_func=lambda x: dest_ap_labels.get(x, x),
    key="sel_dest_airports"
)

st.sidebar.markdown("---")
st.sidebar.caption("Chạy `python pipeline.py` để nạp dữ liệu vào DB")

# ─── Build WHERE clause ────────────────────────────────────
airline_filter = f"a.airline_iata IN ({','.join(repr(x) for x in sel_airlines)})" if sel_airlines else "1=1"

# Ưu tiên: sân bay cụ thể > bang > không lọc
if sel_origin:
    origin_filter = f"ap_o.airport_iata IN ({','.join(repr(x) for x in sel_origin)})"
elif sel_origin_state != "(Tất cả)":
    # Chọn bang nhưng chưa chọn sân bay → lọc toàn bộ sân bay của bang đó
    origin_ap_in_state = origin_airport_opts["airport_iata"].tolist()
    origin_filter = f"ap_o.airport_iata IN ({','.join(repr(x) for x in origin_ap_in_state)})" if origin_ap_in_state else "1=1"
else:
    origin_filter = "1=1"

if sel_dest:
    dest_filter = f"ap_d.airport_iata IN ({','.join(repr(x) for x in sel_dest)})"
elif sel_dest_state != "(Tất cả)":
    dest_ap_in_state = dest_airport_opts["airport_iata"].tolist()
    dest_filter = f"ap_d.airport_iata IN ({','.join(repr(x) for x in dest_ap_in_state)})" if dest_ap_in_state else "1=1"
else:
    dest_filter = "1=1"

# Tất cả query đều JOIN cả 2 bảng airport → dùng ap_o / ap_d nhất quán
def W(*extra):
    """Ghép tất cả active filters."""
    parts = [time_filter, airline_filter, origin_filter, dest_filter] + list(extra)
    return " AND ".join(parts)

# ═══════════════════════════════════════════════════════════
# CẤP 1 — QUẢN LÝ TỔNG QUAN
# ═══════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📊 Cấp 1 — Tổng quan", "🔎 Cấp 2 — Nguyên nhân"])

BASE_JOIN = """
    FROM fact_flights f
    JOIN dim_date    d    ON f.date_key           = d.date_key
    JOIN dim_airline a    ON f.airline_key         = a.airline_key
    JOIN dim_airport ap_o ON f.origin_airport_key = ap_o.airport_key
    JOIN dim_airport ap_d ON f.dest_airport_key   = ap_d.airport_key
"""

with tab1:
    # ── KPI Cards ─────────────────────────────────────────
    kpi = query(f"""
        SELECT
            COUNT(*)                                               AS total_flights,
            ROUND(100.0*(1-1.0*SUM(f.is_delayed_15)/COUNT(*)),2)  AS otp_pct,
            ROUND(100.0*SUM(f.is_cancelled)/COUNT(*),2)            AS cancel_rate,
            ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2)           AS delay_rate
        {BASE_JOIN}
        WHERE {W()}
    """).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✈️ Tổng chuyến bay",   f"{int(kpi.total_flights):,}")
    c2.metric("⏱️ OTP toàn ngành",    f"{kpi.otp_pct}%")
    c3.metric("❌ Tỉ lệ hủy chuyến", f"{kpi.cancel_rate}%")
    c4.metric("⚠️ Tỉ lệ delay ≥15p", f"{kpi.delay_rate}%")

    # Hiển thị bộ lọc đang active
    active = []
    if time_mode == "Tháng" and sel_months:
        active.append(f"Tháng: {', '.join(month_labels[m] for m in sel_months)}")
    elif time_mode == "Quý" and sel_quarters:
        active.append(f"Quý: {', '.join('Q'+str(q) for q in sel_quarters)}")
    if sel_airlines:
        active.append(f"Hãng: {', '.join(airline_labels[x] for x in sel_airlines)}")
    if sel_origin:
        active.append(f"Đi từ: {', '.join(sel_origin)}")
    if sel_dest:
        active.append(f"Đến: {', '.join(sel_dest)}")
    if active:
        st.caption("🔎 Đang lọc: " + " | ".join(active))

    st.divider()

    # ── Row 1: Xu hướng thời gian & Top 5 hãng OTP ────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        if time_mode == "Tháng":
            st.subheader("📈 Số chuyến bay theo tháng")
            df_trend = query(f"""
                SELECT d.month AS period_num, d.month_name AS period_label,
                    COUNT(*) AS total_flights,
                    SUM(f.is_cancelled)  AS cancelled,
                    SUM(f.is_delayed_15) AS delayed
                {BASE_JOIN}
                WHERE {W()}
                GROUP BY d.month, d.month_name ORDER BY d.month
            """)
        else:
            st.subheader("📈 Số chuyến bay theo quý")
            df_trend = query(f"""
                SELECT d.quarter AS period_num,
                    'Q' || d.quarter AS period_label,
                    COUNT(*) AS total_flights,
                    SUM(f.is_cancelled)  AS cancelled,
                    SUM(f.is_delayed_15) AS delayed
                {BASE_JOIN}
                WHERE {W()}
                GROUP BY d.quarter ORDER BY d.quarter
            """)

        if not df_trend.empty:
            fig = go.Figure()
            fig.add_bar(x=df_trend["period_label"], y=df_trend["total_flights"],
                        name="Tổng chuyến", marker_color="#4A90D9")
            fig.add_bar(x=df_trend["period_label"], y=df_trend["delayed"],
                        name="Delay ≥15p",   marker_color="#E8593C")
            fig.add_bar(x=df_trend["period_label"], y=df_trend["cancelled"],
                        name="Hủy",           marker_color="#aaa")
            fig.update_layout(barmode="overlay", height=320,
                              margin=dict(t=10, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu — hãy chạy pipeline.py trước")

    with col_b:
        st.subheader("🏆 Top 5 hãng theo OTP")
        st.write("On-Time Performance — tỉ lệ chuyến bay đúng giờ.")
        df_otp = query(f"""
            SELECT a.airline_name,
                ROUND(100.0*(1-1.0*SUM(f.is_delayed_15)/COUNT(*)),1) AS otp_pct,
                COUNT(*) AS total_flights
            {BASE_JOIN}
            WHERE {W("f.is_cancelled=0")}
            GROUP BY a.airline_name HAVING total_flights > 100
            ORDER BY otp_pct DESC LIMIT 5
        """)
        if not df_otp.empty:
            fig2 = px.bar(df_otp, x="otp_pct", y="airline_name", orientation="h",
                          color="otp_pct", color_continuous_scale="Blues",
                          labels={"otp_pct": "OTP (%)", "airline_name": ""})
            fig2.update_layout(height=320, margin=dict(t=10, b=10),
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Top sân bay delay & Tỉ lệ theo thời gian ───
    col_c, col_d = st.columns([2, 3])

    with col_c:
        st.subheader("🛬 Top 5 sân bay delay cao nhất")
        df_ap = query(f"""
            SELECT ap_o.airport_iata,
                   COALESCE(ap_o.city, ap_o.airport_iata) AS city,
                   ROUND(AVG(f.arrival_delay), 1)          AS avg_delay,
                   COUNT(*)                                AS flights
            {BASE_JOIN}
            WHERE {W("f.is_cancelled=0")}
            GROUP BY ap_o.airport_iata, ap_o.city
            HAVING flights > 500
            ORDER BY avg_delay DESC LIMIT 5
        """)
        if not df_ap.empty:
            fig3 = px.bar(df_ap, x="airport_iata", y="avg_delay",
                          color="avg_delay", color_continuous_scale="OrRd",
                          text="city",
                          labels={"airport_iata": "Sân bay", "avg_delay": "Delay TB (phút)"})
            fig3.update_traces(textposition="outside")
            fig3.update_layout(height=320, margin=dict(t=10, b=10),
                               coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        if time_mode == "Tháng":
            st.subheader("📅 Tỉ lệ delay & hủy theo tháng")
            df_rate = query(f"""
                SELECT d.month AS pnum, d.month_name AS plabel,
                    ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate,
                    ROUND(100.0*SUM(f.is_cancelled)/COUNT(*),2)  AS cancel_rate
                {BASE_JOIN}
                WHERE {W()}
                GROUP BY d.month, d.month_name ORDER BY d.month
            """)
        else:
            st.subheader("📅 Tỉ lệ delay & hủy theo quý")
            df_rate = query(f"""
                SELECT d.quarter AS pnum, 'Q'||d.quarter AS plabel,
                    ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate,
                    ROUND(100.0*SUM(f.is_cancelled)/COUNT(*),2)  AS cancel_rate
                {BASE_JOIN}
                WHERE {W()}
                GROUP BY d.quarter ORDER BY d.quarter
            """)

        if not df_rate.empty:
            fig4 = go.Figure()
            fig4.add_scatter(x=df_rate["plabel"], y=df_rate["delay_rate"],
                             name="Delay rate %", mode="lines+markers",
                             line=dict(color="#E8593C", width=3))
            fig4.add_scatter(x=df_rate["plabel"], y=df_rate["cancel_rate"],
                             name="Cancel rate %", mode="lines+markers",
                             line=dict(color="#888", width=2, dash="dash"))
            fig4.update_layout(height=280, margin=dict(t=10, b=10),
                               legend=dict(orientation="h"))
            st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# CẤP 2 — NGUYÊN NHÂN
# ═══════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔎 Phân tích nguyên nhân delay")

    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown("**Nguyên nhân delay (Pie chart)**")
        df_reason = query(f"""
            SELECT dr.reason_name,
                COUNT(*) AS cnt,
                ROUND(AVG(
                    CASE dr.reason_code
                        WHEN 'AIR_SYSTEM'    THEN f.delay_air_system
                        WHEN 'SECURITY'      THEN f.delay_security
                        WHEN 'AIRLINE'       THEN f.delay_airline
                        WHEN 'LATE_AIRCRAFT' THEN f.delay_late_aircraft
                        WHEN 'WEATHER'       THEN f.delay_weather
                        ELSE 0
                    END), 1) AS avg_min
            {BASE_JOIN}
            JOIN dim_delay_reason dr ON f.delay_reason_key = dr.reason_key
            WHERE dr.reason_code != 'NO_DELAY' AND f.is_cancelled = 0
              AND {W()}
            GROUP BY dr.reason_name ORDER BY cnt DESC
        """)
        if not df_reason.empty:
            fig5 = px.pie(df_reason, values="cnt", names="reason_name",
                          color_discrete_sequence=px.colors.qualitative.Set3)
            fig5.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig5, use_container_width=True)

    with col_f:
        st.markdown("**Delay TB theo nguyên nhân (phút)**")
        if not df_reason.empty:
            fig6 = px.bar(df_reason.sort_values("avg_min", ascending=True),
                          x="avg_min", y="reason_name", orientation="h",
                          color="avg_min", color_continuous_scale="Reds",
                          labels={"avg_min": "Delay TB (phút)", "reason_name": ""})
            fig6.update_layout(height=320, margin=dict(t=10, b=10),
                               coloraxis_showscale=False)
            st.plotly_chart(fig6, use_container_width=True)

    # Xu hướng delay theo mùa / quý
    if time_mode == "Tháng":
        st.subheader("🌤️ Delay trung bình theo tháng & mùa")
        df_season = query(f"""
            SELECT d.month, d.month_name,
                ROUND(AVG(f.arrival_delay),1)                 AS avg_delay,
                ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate,
                CASE
                    WHEN d.month IN (12,1,2) THEN '❄️ Đông'
                    WHEN d.month IN (3,4,5)  THEN '🌸 Xuân'
                    WHEN d.month IN (6,7,8)  THEN '☀️ Hè'
                    ELSE '🍂 Thu'
                END AS season
            {BASE_JOIN}
            WHERE f.is_cancelled=0 AND {W()}
            GROUP BY d.month, d.month_name ORDER BY d.month
        """)
        if not df_season.empty:
            fig7 = px.bar(df_season, x="month_name", y="avg_delay", color="season",
                          labels={"avg_delay": "Delay TB (phút)", "month_name": "Tháng"})
            fig7.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig7, use_container_width=True)
    else:
        st.subheader("🌤️ Delay trung bình theo quý")
        df_qseason = query(f"""
            SELECT 'Q'||d.quarter AS quarter_label,
                ROUND(AVG(f.arrival_delay),1)                 AS avg_delay,
                ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate
            {BASE_JOIN}
            WHERE f.is_cancelled=0 AND {W()}
            GROUP BY d.quarter ORDER BY d.quarter
        """)
        if not df_qseason.empty:
            fig7b = px.bar(df_qseason, x="quarter_label", y="avg_delay",
                           color="avg_delay", color_continuous_scale="Blues",
                           labels={"avg_delay": "Delay TB (phút)", "quarter_label": "Quý"})
            fig7b.update_layout(height=320, margin=dict(t=10, b=10),
                                coloraxis_showscale=False)
            st.plotly_chart(fig7b, use_container_width=True)

    # Delay theo khung giờ bay
    st.subheader("🕐 Delay theo khung giờ bay")
    df_time = query(f"""
        SELECT
            CASE
                WHEN CAST(SUBSTR(PRINTF('%04d', CAST(f.scheduled_departure AS INTEGER)),1,2) AS INTEGER)
                     BETWEEN 0  AND 5  THEN '1. 00-06 Đêm khuya'
                WHEN CAST(SUBSTR(PRINTF('%04d', CAST(f.scheduled_departure AS INTEGER)),1,2) AS INTEGER)
                     BETWEEN 6  AND 11 THEN '2. 06-12 Sáng'
                WHEN CAST(SUBSTR(PRINTF('%04d', CAST(f.scheduled_departure AS INTEGER)),1,2) AS INTEGER)
                     BETWEEN 12 AND 17 THEN '3. 12-18 Chiều'
                ELSE '4. 18-24 Tối'
            END AS time_slot,
            ROUND(AVG(f.departure_delay),1)                  AS avg_dep_delay,
            ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2)    AS delay_rate_pct,
            COUNT(*)                                         AS total
        {BASE_JOIN}
        WHERE f.is_cancelled=0 AND f.scheduled_departure IS NOT NULL
          AND {W()}
        GROUP BY time_slot ORDER BY time_slot
    """)
    if not df_time.empty:
        df_time["time_slot"] = df_time["time_slot"].str[3:]   # bỏ prefix số
        fig8 = px.bar(df_time, x="time_slot", y="avg_dep_delay",
                      color="delay_rate_pct", color_continuous_scale="YlOrRd",
                      labels={"avg_dep_delay": "Delay TB (phút)",
                              "time_slot": "Khung giờ",
                              "delay_rate_pct": "Delay rate %"})
        fig8.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig8, use_container_width=True)
