"""
dashboard/app.py
================
Airline Data Warehouse — Multi-page Streamlit Dashboard
Dữ liệu: Kaggle US Flight Delays 2015
Chạy: streamlit run dashboard/app.py
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Paths ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DB_PATH    = os.path.join(BASE_DIR, ".", "warehouse", "airline_dw.db")
MODEL_PATH = os.path.join(BASE_DIR, ".", "mining", "model_delay.pkl")

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Airline DW Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #1a1a2e; }
  [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
  [data-testid="stSidebar"] .stRadio label { font-size: 14px; padding: 4px 0; }
  div[data-testid="metric-container"] {
    background: #f8faff; border-radius: 10px;
    padding: 12px 16px; border-left: 4px solid #2E75B6;
  }
  .section-title {
    font-size: 20px; font-weight: bold; color: #1F5C99;
    border-left: 4px solid #2E75B6; padding-left: 10px; margin: 16px 0 8px;
  }
</style>
""", unsafe_allow_html=True)

# ── DB helpers ─────────────────────────────────────────────
@st.cache_resource
def get_conn():
    if not os.path.exists(DB_PATH):
        st.error("❌ Chưa có database. Chạy `python pipeline.py` trước.")
        st.stop()
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data(ttl=300)
def query(sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, get_conn())

# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj.get("model"), obj.get("encoders")

# ── State maps ─────────────────────────────────────────────
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
}
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
BG = "#FAFAFA"

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.markdown("## ✈️ Airline DW Analytics")
    st.sidebar.markdown("**HCMUS — CDIO 2026**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "📋 Chọn trang phân tích:",
        [
            "🏠 Tổng quan",
            "📈 Roll-up & Drill-down",
            "🔪 Slice & Dice",
            "🔎 Phân tích nguyên nhân",
            "🧠 Dự báo chuyến bay",
            "🗄️ Database Explorer",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Bộ lọc toàn cục")

    # Thời gian
    st.sidebar.markdown("**📅 Thời gian**")
    time_mode = st.sidebar.radio(
        "Lọc theo", ["Tháng", "Quý"], horizontal=True,
        label_visibility="collapsed", key="global_time_mode"
    )
    time_filter = "1=1"
    if time_mode == "Tháng":
        sel_months = st.sidebar.multiselect(
            "Chọn tháng",
            options=list(MONTH_NAMES.keys()),
            format_func=lambda x: MONTH_NAMES[x],
            default=[]
        )
        if sel_months:
            time_filter = f"d.month IN ({','.join(map(str,sel_months))})"
    else:
        sel_quarters = st.sidebar.multiselect(
            "Chọn quý", [1,2,3,4],
            format_func=lambda x: f"Q{x}", default=[]
        )
        if sel_quarters:
            time_filter = f"d.quarter IN ({','.join(map(str,sel_quarters))})"

    # Hãng hàng không
    airlines_df    = query("SELECT airline_iata, airline_name FROM dim_airline ORDER BY airline_name")
    airline_labels = dict(zip(airlines_df["airline_iata"], airlines_df["airline_name"]))
    sel_airlines   = st.sidebar.multiselect(
        "🏢 Hãng hàng không",
        options=airlines_df["airline_iata"].tolist(),
        format_func=lambda x: airline_labels.get(x, x),
        default=[]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Chạy `python pipeline.py` để nạp dữ liệu vào DB")

    airline_filter = (
        f"a.airline_iata IN ({','.join(repr(x) for x in sel_airlines)})"
        if sel_airlines else "1=1"
    )

    def W(*extra):
        parts = [time_filter, airline_filter] + list(extra)
        return " AND ".join(parts)

    return page, W, time_mode

# ══════════════════════════════════════════════════════════
# PAGE 1: TỔNG QUAN
# ══════════════════════════════════════════════════════════
def page_overview(W):
    st.markdown("## 🏠 Tổng quan — Airline Data Warehouse 2015")
    st.markdown("> Dữ liệu: Kaggle US Flight Delays 2015 | 5.8M chuyến bay | 14 hãng | 322 sân bay")

    BASE = """
        FROM fact_flights f
        JOIN dim_date    d    ON f.date_key           = d.date_key
        JOIN dim_airline a    ON f.airline_key         = a.airline_key
        JOIN dim_airport ap_o ON f.origin_airport_key = ap_o.airport_key
        JOIN dim_airport ap_d ON f.dest_airport_key   = ap_d.airport_key
    """

    kpi = query(f"""
        SELECT COUNT(*) AS total,
               ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate,
               ROUND(100.0*(1-1.0*SUM(f.is_delayed_15)/COUNT(*)),2) AS otp,
               ROUND(100.0*SUM(f.is_cancelled)/COUNT(*),2) AS cancel_rate,
               ROUND(AVG(f.arrival_delay),1) AS avg_delay
        {BASE} WHERE {W()}
    """).iloc[0]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("✈️ Tổng chuyến bay",   f"{int(kpi.total):,}")
    c2.metric("⏱️ OTP",               f"{kpi.otp}%")
    c3.metric("⚠️ Tỉ lệ delay ≥15p", f"{kpi.delay_rate}%")
    c4.metric("❌ Tỉ lệ hủy",         f"{kpi.cancel_rate}%")
    c5.metric("🕐 Delay TB",           f"{kpi.avg_delay} phút")

    st.markdown("---")
    col_a, col_b = st.columns([3,2])

    with col_a:
        st.markdown('<div class="section-title">📈 Số chuyến bay & Tỉ lệ delay theo tháng</div>',
                    unsafe_allow_html=True)
        df_trend = query(f"""
            SELECT d.month, d.month_name,
                   COUNT(*) AS total,
                   SUM(f.is_delayed_15) AS delayed,
                   SUM(f.is_cancelled)  AS cancelled,
                   ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate
            {BASE} WHERE {W()}
            GROUP BY d.month, d.month_name ORDER BY d.month
        """)
        if not df_trend.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(x=df_trend["month_name"], y=df_trend["total"],
                        name="Tổng chuyến", marker_color="#4A90D9",
                        secondary_y=False)
            fig.add_scatter(x=df_trend["month_name"], y=df_trend["delay_rate"],
                            name="Delay rate %", mode="lines+markers",
                            line=dict(color="#E8593C", width=2.5),
                            secondary_y=True)
            fig.update_yaxes(title_text="Số chuyến", secondary_y=False)
            fig.update_yaxes(title_text="Delay rate (%)", secondary_y=True)
            fig.update_layout(height=340, margin=dict(t=10,b=10),
                              legend=dict(orientation="h"),
                              plot_bgcolor=BG, paper_bgcolor=BG)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Chưa có dữ liệu.")

    with col_b:
        st.markdown('<div class="section-title">🏆 Top 5 hãng theo OTP</div>',
                    unsafe_allow_html=True)
        df_otp = query(f"""
            SELECT a.airline_name,
                   ROUND(100.0*(1-1.0*SUM(f.is_delayed_15)/COUNT(*)),1) AS otp,
                   COUNT(*) AS total
            {BASE} WHERE {W("f.is_cancelled=0")}
            GROUP BY a.airline_name HAVING total>100
            ORDER BY otp DESC LIMIT 5
        """)
        if not df_otp.empty:
            fig2 = px.bar(df_otp, x="otp", y="airline_name", orientation="h",
                          color="otp", color_continuous_scale="Blues",
                          labels={"otp":"OTP (%)","airline_name":""})
            fig2.update_layout(height=300, margin=dict(t=10,b=10),
                               coloraxis_showscale=False,
                               plot_bgcolor=BG, paper_bgcolor=BG)
            st.plotly_chart(fig2, width='stretch')

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-title">🛬 Top 5 sân bay delay cao nhất</div>',
                    unsafe_allow_html=True)
        df_ap = query(f"""
            SELECT ap_o.airport_iata,
                   COALESCE(ap_o.city, ap_o.airport_iata) AS city,
                   ROUND(AVG(f.arrival_delay),1) AS avg_delay,
                   COUNT(*) AS flights
            {BASE} WHERE {W("f.is_cancelled=0")}
            GROUP BY ap_o.airport_iata, ap_o.city
            HAVING flights>500 ORDER BY avg_delay DESC LIMIT 5
        """)
        if not df_ap.empty:
            fig3 = px.bar(df_ap, x="airport_iata", y="avg_delay",
                          color="avg_delay", color_continuous_scale="OrRd",
                          text="city",
                          labels={"airport_iata":"Sân bay","avg_delay":"Delay TB (phút)"})
            fig3.update_traces(textposition="outside")
            fig3.update_layout(height=300, coloraxis_showscale=False,
                               plot_bgcolor=BG, paper_bgcolor=BG,
                               margin=dict(t=10,b=10))
            st.plotly_chart(fig3, width='stretch')

    with col_d:
        st.markdown('<div class="section-title">📅 Tỉ lệ delay & hủy theo quý</div>',
                    unsafe_allow_html=True)
        df_qtr = query(f"""
            SELECT d.quarter,
                   ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate,
                   ROUND(100.0*SUM(f.is_cancelled)/COUNT(*),2)  AS cancel_rate
            {BASE} WHERE {W()}
            GROUP BY d.quarter ORDER BY d.quarter
        """)
        if not df_qtr.empty:
            df_qtr["ql"] = "Q" + df_qtr["quarter"].astype(str)
            fig4 = go.Figure()
            fig4.add_scatter(x=df_qtr["ql"], y=df_qtr["delay_rate"],
                             name="Delay %", mode="lines+markers",
                             line=dict(color="#E8593C",width=3))
            fig4.add_scatter(x=df_qtr["ql"], y=df_qtr["cancel_rate"],
                             name="Cancel %", mode="lines+markers",
                             line=dict(color="#888",width=2,dash="dash"))
            fig4.update_layout(height=280, margin=dict(t=10,b=10),
                               legend=dict(orientation="h"),
                               plot_bgcolor=BG, paper_bgcolor=BG)
            st.plotly_chart(fig4, width='stretch')


# ══════════════════════════════════════════════════════════
# PAGE 2: ROLL-UP & DRILL-DOWN
# ══════════════════════════════════════════════════════════
def page_rollup_drilldown(W):
    st.markdown("## 📈 Roll-up & Drill-down (OLAP)")
    st.markdown("> **Roll-up**: Từ chi tiết → tổng hợp (tháng → quý → năm)  \n"
                "> **Drill-down**: Từ tổng hợp → chi tiết (hãng → sân bay → tuyến bay)")

    BASE = """
        FROM fact_flights f
        JOIN dim_date    d    ON f.date_key           = d.date_key
        JOIN dim_airline a    ON f.airline_key         = a.airline_key
        JOIN dim_airport ap_o ON f.origin_airport_key = ap_o.airport_key
        JOIN dim_airport ap_d ON f.dest_airport_key   = ap_d.airport_key
    """

    # ── Roll-up ──────────────────────────────────────────
    st.markdown('<div class="section-title">⬆️ Roll-up: Số chuyến bay & Delay rate</div>',
                unsafe_allow_html=True)
    granularity = st.radio("Mức tổng hợp:", ["Tháng", "Quý"], horizontal=True)

    if granularity == "Tháng":
        df_roll = query(f"""
            SELECT d.month AS pnum, d.month_name AS plabel,
                   COUNT(*) AS total,
                   ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate,
                   ROUND(AVG(f.arrival_delay),1) AS avg_delay
            {BASE} WHERE {W()}
            GROUP BY d.month, d.month_name ORDER BY d.month
        """)
    else:
        df_roll = query(f"""
            SELECT d.quarter AS pnum, 'Q'||d.quarter AS plabel,
                   COUNT(*) AS total,
                   ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate,
                   ROUND(AVG(f.arrival_delay),1) AS avg_delay
            {BASE} WHERE {W()}
            GROUP BY d.quarter ORDER BY d.quarter
        """)

    if not df_roll.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=df_roll["plabel"], y=df_roll["total"],
                    name="Tổng chuyến", marker_color="#4A90D9", secondary_y=False)
        fig.add_scatter(x=df_roll["plabel"], y=df_roll["delay_rate"],
                        name="Delay rate %", mode="lines+markers",
                        line=dict(color="#E8593C",width=2.5), secondary_y=True)
        fig.update_yaxes(title_text="Số chuyến bay", secondary_y=False)
        fig.update_yaxes(title_text="Delay rate (%)", secondary_y=True)
        fig.update_layout(height=380, legend=dict(orientation="h"),
                          plot_bgcolor=BG, paper_bgcolor=BG,
                          title=f"Roll-up theo {granularity}")
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # ── Drill-down: Hãng → Sân bay ───────────────────────
    st.markdown('<div class="section-title">⬇️ Drill-down: Hãng → Sân bay xuất phát</div>',
                unsafe_allow_html=True)

    airlines_df = query("SELECT airline_iata, airline_name FROM dim_airline ORDER BY airline_name")
    airline_sel = st.selectbox(
        "Chọn hãng để Drill-down:",
        options=airlines_df["airline_iata"].tolist(),
        format_func=lambda x: airlines_df.set_index("airline_iata").loc[x,"airline_name"]
    )

    df_drill = query(f"""
        SELECT ap_o.airport_iata AS airport,
               COALESCE(ap_o.city,'') AS city,
               COUNT(*) AS flights,
               ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),1) AS delay_rate,
               ROUND(AVG(f.arrival_delay),1) AS avg_delay,
               SUM(f.is_cancelled) AS cancelled
        {BASE}
        WHERE a.airline_iata = '{airline_sel}' AND {W("f.is_cancelled=0")}
        GROUP BY ap_o.airport_iata, ap_o.city
        HAVING flights > 50
        ORDER BY flights DESC LIMIT 15
    """)

    if not df_drill.empty:
        airline_name = airlines_df.set_index("airline_iata").loc[airline_sel,"airline_name"]
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df_drill, x="flights", y="airport", orientation="h",
                         color="delay_rate", color_continuous_scale="RdYlGn_r",
                         text="city",
                         title=f"Số chuyến theo sân bay — {airline_name}",
                         labels={"flights":"Số chuyến","airport":"Sân bay",
                                 "delay_rate":"Delay rate %"})
            fig.update_layout(yaxis=dict(autorange="reversed"), height=420,
                              plot_bgcolor=BG, paper_bgcolor=BG)
            st.plotly_chart(fig, width='stretch')
        with col2:
            fig2 = px.scatter(df_drill, x="flights", y="avg_delay",
                              size="flights", color="delay_rate",
                              color_continuous_scale="OrRd",
                              hover_data=["airport","city"],
                              title=f"Số chuyến vs Delay TB — {airline_name}",
                              labels={"flights":"Số chuyến","avg_delay":"Delay TB (phút)"})
            fig2.update_layout(height=420, plot_bgcolor=BG, paper_bgcolor=BG)
            st.plotly_chart(fig2, width='stretch')

        st.dataframe(df_drill, width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════
# PAGE 3: SLICE & DICE
# ══════════════════════════════════════════════════════════
def page_slice_dice(W):
    st.markdown("## 🔪 Slice & Dice (Phân tích đa chiều)")
    st.markdown("> **Slice**: Cắt theo 1 chiều — VD: chỉ xem hãng Delta  \n"
                "> **Dice**: Cắt theo nhiều chiều — VD: Delta + California + Tháng 6")

    BASE = """
        FROM fact_flights f
        JOIN dim_date    d    ON f.date_key           = d.date_key
        JOIN dim_airline a    ON f.airline_key         = a.airline_key
        JOIN dim_airport ap_o ON f.origin_airport_key = ap_o.airport_key
        JOIN dim_airport ap_d ON f.dest_airport_key   = ap_d.airport_key
    """

    # ── Bộ lọc Slice & Dice ──────────────────────────────
    st.markdown('<div class="section-title">🎲 Chọn chiều phân tích (Dice)</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    airlines_df = query("SELECT airline_iata, airline_name FROM dim_airline ORDER BY airline_name")
    airports_df = query("""
        SELECT DISTINCT COALESCE(state,'') AS state
        FROM dim_airport WHERE state IS NOT NULL AND state != ''
        ORDER BY state
    """)
    all_states = airports_df["state"].tolist()

    with c1:
        sel_airline = st.selectbox(
            "✂️ Slice — Hãng hàng không:",
            ["(Tất cả)"] + airlines_df["airline_iata"].tolist(),
            format_func=lambda x: "(Tất cả)" if x=="(Tất cả)"
                        else airlines_df.set_index("airline_iata").loc[x,"airline_name"]
        )
    with c2:
        sel_state = st.selectbox(
            "🎲 Dice — Bang xuất phát:",
            ["(Tất cả)"] + all_states,
            format_func=lambda x: f"{x} — {STATE_MAP.get(x,x)}" if x!="(Tất cả)" else x
        )
    with c3:
        sel_month = st.selectbox(
            "🎲 Dice — Tháng:",
            ["(Tất cả)"] + list(MONTH_NAMES.keys()),
            format_func=lambda x: "(Tất cả)" if x=="(Tất cả)" else MONTH_NAMES[x]
        )

    # Build extra filters
    extras = []
    if sel_airline != "(Tất cả)":
        extras.append(f"a.airline_iata = '{sel_airline}'")
    if sel_state != "(Tất cả)":
        extras.append(f"ap_o.state = '{sel_state}'")
    if sel_month != "(Tất cả)":
        extras.append(f"d.month = {sel_month}")

    where = W(*extras) if extras else W()

    # ── KPI Slice ────────────────────────────────────────
    kpi = query(f"""
        SELECT COUNT(*) AS total,
               ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate,
               ROUND(AVG(f.arrival_delay),1) AS avg_delay,
               ROUND(100.0*SUM(f.is_cancelled)/COUNT(*),2) AS cancel_rate
        {BASE} WHERE {where}
    """).iloc[0]

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("✈️ Số chuyến",        f"{int(kpi.total):,}")
    k2.metric("⚠️ Delay rate",       f"{kpi.delay_rate}%")
    k3.metric("🕐 Delay TB",          f"{kpi.avg_delay} phút")
    k4.metric("❌ Hủy chuyến",        f"{kpi.cancel_rate}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📊 Delay rate theo hãng</div>',
                    unsafe_allow_html=True)
        df_by_airline = query(f"""
            SELECT a.airline_name,
                   COUNT(*) AS flights,
                   ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),1) AS delay_rate,
                   ROUND(AVG(f.arrival_delay),1) AS avg_delay
            {BASE} WHERE {where} AND f.is_cancelled=0
            GROUP BY a.airline_name HAVING flights>50
            ORDER BY delay_rate DESC
        """)
        if not df_by_airline.empty:
            fig = px.bar(df_by_airline, x="delay_rate", y="airline_name",
                         orientation="h", color="delay_rate",
                         color_continuous_scale="RdYlGn_r",
                         labels={"delay_rate":"Delay rate (%)","airline_name":""})
            fig.update_layout(height=350, coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"),
                              plot_bgcolor=BG, paper_bgcolor=BG)
            st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-title">📊 Delay rate theo bang đích</div>',
                    unsafe_allow_html=True)
        df_by_state = query(f"""
            SELECT ap_d.state,
                   COUNT(*) AS flights,
                   ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),1) AS delay_rate
            {BASE} WHERE {where} AND f.is_cancelled=0
              AND ap_d.state IS NOT NULL AND ap_d.state != ''
            GROUP BY ap_d.state HAVING flights>200
            ORDER BY delay_rate DESC LIMIT 12
        """)
        if not df_by_state.empty:
            df_by_state["state_label"] = df_by_state["state"].map(
                lambda x: f"{x} ({STATE_MAP.get(x,x)})"
            )
            fig2 = px.bar(df_by_state, x="delay_rate", y="state_label",
                          orientation="h", color="delay_rate",
                          color_continuous_scale="OrRd",
                          labels={"delay_rate":"Delay rate (%)","state_label":"Bang đích"})
            fig2.update_layout(height=350, coloraxis_showscale=False,
                               yaxis=dict(autorange="reversed"),
                               plot_bgcolor=BG, paper_bgcolor=BG)
            st.plotly_chart(fig2, width='stretch')

    # ── Scatter: Số chuyến vs Delay TB theo hãng ─────────
    st.markdown('<div class="section-title">🔵 Scatter: Số chuyến vs Delay TB</div>',
                unsafe_allow_html=True)
    df_scatter = query(f"""
        SELECT a.airline_name,
               COUNT(*) AS flights,
               ROUND(AVG(f.arrival_delay),1) AS avg_delay,
               ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),1) AS delay_rate
        {BASE} WHERE {where} AND f.is_cancelled=0
        GROUP BY a.airline_name HAVING flights>100
    """)
    if not df_scatter.empty:
        fig3 = px.scatter(df_scatter, x="flights", y="avg_delay",
                          size="delay_rate", color="delay_rate",
                          hover_data=["airline_name"],
                          text="airline_name",
                          color_continuous_scale="RdYlGn_r",
                          labels={"flights":"Số chuyến","avg_delay":"Delay TB (phút)",
                                  "delay_rate":"Delay rate %"})
        fig3.update_traces(textposition="top center")
        fig3.update_layout(height=420, plot_bgcolor=BG, paper_bgcolor=BG)
        st.plotly_chart(fig3, width='stretch')


# ══════════════════════════════════════════════════════════
# PAGE 4: PHÂN TÍCH NGUYÊN NHÂN
# ══════════════════════════════════════════════════════════
def page_root_cause(W):
    st.markdown("## 🔎 Phân tích Nguyên nhân Delay")
    st.markdown("> Phân tích sâu nguyên nhân gây delay: Thời tiết, Hãng bay, Hệ thống, ...")

    BASE = """
        FROM fact_flights f
        JOIN dim_date         d    ON f.date_key           = d.date_key
        JOIN dim_airline      a    ON f.airline_key         = a.airline_key
        JOIN dim_airport      ap_o ON f.origin_airport_key = ap_o.airport_key
        JOIN dim_airport      ap_d ON f.dest_airport_key   = ap_d.airport_key
        JOIN dim_delay_reason dr   ON f.delay_reason_key   = dr.reason_key
    """

    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown('<div class="section-title">🥧 Phân bố nguyên nhân delay</div>',
                    unsafe_allow_html=True)
        df_reason = query(f"""
            SELECT dr.reason_name,
                   COUNT(*) AS cnt,
                   ROUND(AVG(CASE dr.reason_code
                       WHEN 'AIR_SYSTEM'    THEN f.delay_air_system
                       WHEN 'SECURITY'      THEN f.delay_security
                       WHEN 'AIRLINE'       THEN f.delay_airline
                       WHEN 'LATE_AIRCRAFT' THEN f.delay_late_aircraft
                       WHEN 'WEATHER'       THEN f.delay_weather
                       ELSE 0 END),1) AS avg_min
            {BASE}
            WHERE dr.reason_code != 'NO_DELAY' AND f.is_cancelled=0
              AND {W()}
            GROUP BY dr.reason_name ORDER BY cnt DESC
        """)
        if not df_reason.empty:
            fig5 = px.pie(df_reason, values="cnt", names="reason_name",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig5.update_layout(height=340, margin=dict(t=10,b=10),
                               paper_bgcolor=BG)
            st.plotly_chart(fig5, width='stretch')

    with col_f:
        st.markdown('<div class="section-title">⏱️ Delay TB theo nguyên nhân (phút)</div>',
                    unsafe_allow_html=True)
        if not df_reason.empty:
            fig6 = px.bar(df_reason.sort_values("avg_min"),
                          x="avg_min", y="reason_name", orientation="h",
                          color="avg_min", color_continuous_scale="Reds",
                          labels={"avg_min":"Delay TB (phút)","reason_name":""})
            fig6.update_layout(height=320, coloraxis_showscale=False,
                               plot_bgcolor=BG, paper_bgcolor=BG,
                               margin=dict(t=10,b=10))
            st.plotly_chart(fig6, width='stretch')

    # ── Xu hướng theo mùa ────────────────────────────────
    st.markdown('<div class="section-title">🌤️ Delay TB theo tháng & mùa</div>',
                unsafe_allow_html=True)
    BASE2 = """
        FROM fact_flights f
        JOIN dim_date    d    ON f.date_key           = d.date_key
        JOIN dim_airline a    ON f.airline_key         = a.airline_key
        JOIN dim_airport ap_o ON f.origin_airport_key = ap_o.airport_key
        JOIN dim_airport ap_d ON f.dest_airport_key   = ap_d.airport_key
    """
    df_season = query(f"""
        SELECT d.month, d.month_name,
               ROUND(AVG(f.arrival_delay),1) AS avg_delay,
               ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate,
               CASE
                   WHEN d.month IN (12,1,2) THEN '❄️ Đông'
                   WHEN d.month IN (3,4,5)  THEN '🌸 Xuân'
                   WHEN d.month IN (6,7,8)  THEN '☀️ Hè'
                   ELSE '🍂 Thu'
               END AS season
        {BASE2} WHERE f.is_cancelled=0 AND {W()}
        GROUP BY d.month, d.month_name ORDER BY d.month
    """)
    if not df_season.empty:
        fig7 = px.bar(df_season, x="month_name", y="avg_delay", color="season",
                      labels={"avg_delay":"Delay TB (phút)","month_name":"Tháng"})
        fig7.update_layout(height=340, plot_bgcolor=BG, paper_bgcolor=BG)
        st.plotly_chart(fig7, width='stretch')

    # ── Delay theo khung giờ ─────────────────────────────
    st.markdown('<div class="section-title">🕐 Delay theo khung giờ bay</div>',
                unsafe_allow_html=True)
    df_time = query(f"""
        SELECT
            CASE
                WHEN CAST(SUBSTR(PRINTF('%04d',CAST(f.scheduled_departure AS INTEGER)),1,2) AS INTEGER)
                     BETWEEN 0  AND 5  THEN '1. 00-06 Đêm khuya'
                WHEN CAST(SUBSTR(PRINTF('%04d',CAST(f.scheduled_departure AS INTEGER)),1,2) AS INTEGER)
                     BETWEEN 6  AND 11 THEN '2. 06-12 Sáng'
                WHEN CAST(SUBSTR(PRINTF('%04d',CAST(f.scheduled_departure AS INTEGER)),1,2) AS INTEGER)
                     BETWEEN 12 AND 17 THEN '3. 12-18 Chiều'
                ELSE '4. 18-24 Tối'
            END AS slot,
            ROUND(AVG(f.departure_delay),1) AS avg_dep,
            ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_pct,
            COUNT(*) AS total
        {BASE2} WHERE f.is_cancelled=0 AND f.scheduled_departure IS NOT NULL
          AND {W()}
        GROUP BY slot ORDER BY slot
    """)
    if not df_time.empty:
        df_time["slot"] = df_time["slot"].str[3:]
        fig8 = px.bar(df_time, x="slot", y="avg_dep",
                      color="delay_pct", color_continuous_scale="YlOrRd",
                      labels={"avg_dep":"Delay TB (phút)","slot":"Khung giờ",
                              "delay_pct":"Delay rate %"})
        fig8.update_layout(height=320, plot_bgcolor=BG, paper_bgcolor=BG)
        st.plotly_chart(fig8, width='stretch')


# ══════════════════════════════════════════════════════════
# PAGE 5: DỰ BÁO CHUYẾN BAY
# ══════════════════════════════════════════════════════════
def page_predict():
    st.markdown("## 🧠 Dự báo Chuyến bay")
    st.markdown("> Nhập thông tin chuyến bay để dự báo khả năng delay")

    clf, encoders = load_model()
    if clf is None:
        st.warning("⚠️ Chưa có model. Chạy `python mining/predict_delay.py` để train.")
        return

    valid_airlines = sorted(encoders["airline"].classes_.tolist())
    valid_states   = sorted(encoders["origin_state"].classes_.tolist())
    feature_cols   = encoders["feature_cols"]

    # Load tên đầy đủ
    try:
        airlines_df = query("SELECT airline_iata, airline_name FROM dim_airline")
        airline_map = dict(zip(airlines_df["airline_iata"], airlines_df["airline_name"]))
    except Exception:
        airline_map = {}

    QUARTER_MAP = {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}

    # ── Input form ───────────────────────────────────────
    st.markdown('<div class="section-title">✏️ Thông tin chuyến bay</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox(
            "🏢 Hãng hàng không",
            options=valid_airlines,
            format_func=lambda x: f"{x} — {airline_map.get(x, x)}"
        )
        origin_state = st.selectbox(
            "🛫 Bang xuất phát",
            options=valid_states,
            format_func=lambda x: f"{x} — {STATE_MAP.get(x, x)}"
        )
        dest_state = st.selectbox(
            "🛬 Bang đích đến",
            options=valid_states,
            format_func=lambda x: f"{x} — {STATE_MAP.get(x, x)}",
            index=1
        )

    with col2:
        month = st.selectbox(
            "📅 Tháng bay",
            options=list(MONTH_NAMES.keys()),
            format_func=lambda x: MONTH_NAMES[x]
        )
        day_of_week = st.selectbox(
            "📆 Thứ trong tuần",
            options=[1,2,3,4,5,6,7],
            format_func=lambda x: ["Thứ 2","Thứ 3","Thứ 4","Thứ 5",
                                    "Thứ 6","Thứ 7","Chủ nhật"][x-1]
        )
        dep_hour = st.slider("🕐 Giờ khởi hành", 0, 23, 14)

    with col3:
        distance    = st.number_input("📏 Khoảng cách (dặm)", 50, 5000, 1000, 50)
        elapsed     = st.number_input("⏱️ Thời gian bay (phút)", 30, 600, 150, 10)

    # ── Predict ──────────────────────────────────────────
    if st.button("🚀 Dự báo ngay", type="primary", width='stretch'):
        def safe_encode(le, val):
            try:    return le.transform([val])[0]
            except: return 0

        is_weekend = int(day_of_week >= 6)
        quarter    = QUARTER_MAP[month]

        # Validate
        valid = True
        if airline not in encoders["airline"].classes_:
            st.error(f"❌ Hãng '{airline}' không hợp lệ")
            valid = False
        if origin_state not in encoders["origin_state"].classes_:
            st.error(f"❌ Bang '{origin_state}' không hợp lệ")
            valid = False

        if valid:
            import pandas as pd_local
            X = pd_local.DataFrame([{
                "month":        month,
                "day_of_week":  day_of_week,
                "is_weekend":   is_weekend,
                "quarter":      quarter,
                "dep_hour":     dep_hour,
                "distance":     float(distance),
                "elapsed_time": float(elapsed),
                "airline_enc":  safe_encode(encoders["airline"], airline),
                "origin_enc":   safe_encode(encoders["origin_state"], origin_state),
                "dest_enc":     safe_encode(encoders["dest_state"], dest_state),
            }])[feature_cols]

            prob = clf.predict_proba(X)[0][1]
            delayed = prob >= 0.5
            risk = "🔴 Cao" if prob >= 0.6 else ("🟡 Trung bình" if prob >= 0.4 else "🟢 Thấp")

            st.markdown("---")
            st.markdown('<div class="section-title">📊 Kết quả dự báo</div>',
                        unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            r1.metric("🎯 Dự đoán",
                      "⚠️ CÓ DELAY" if delayed else "✅ ĐÚNG GIỜ")
            r2.metric("📈 Xác suất delay", f"{prob*100:.1f}%")
            r3.metric("⚡ Mức độ rủi ro", risk)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix":"%","font":{"size":32}},
                title={"text": "Xác suất delay", "font":{"size":18}},
                delta={"reference": 50, "increasing":{"color":"red"},
                       "decreasing":{"color":"green"}},
                gauge={
                    "axis":{"range":[0,100],"ticksuffix":"%"},
                    "bar":{"color": "#E8593C" if delayed else "#27AE60"},
                    "steps":[
                        {"range":[0,40],  "color":"#d4edda"},
                        {"range":[40,60], "color":"#fff3cd"},
                        {"range":[60,100],"color":"#f8d7da"},
                    ],
                    "threshold":{
                        "line":{"color":"black","width":3},
                        "thickness":0.75, "value":50
                    }
                }
            ))
            fig_gauge.update_layout(height=280, paper_bgcolor=BG,
                                    margin=dict(t=40,b=10))
            st.plotly_chart(fig_gauge, width='stretch')

            # Tóm tắt
            if delayed:
                st.error(f"⚠️ Chuyến bay **{airline}** từ **{origin_state}** đến **{dest_state}** "
                         f"lúc **{dep_hour:02d}:00** có **{prob*100:.1f}%** khả năng bị delay ≥15 phút.")
            else:
                st.success(f"✅ Chuyến bay **{airline}** từ **{origin_state}** đến **{dest_state}** "
                           f"lúc **{dep_hour:02d}:00** được dự báo **đúng giờ** "
                           f"(xác suất delay: {prob*100:.1f}%).")


# ══════════════════════════════════════════════════════════
# PAGE 6: DATABASE EXPLORER
# ══════════════════════════════════════════════════════════
def page_db_explorer():
    st.markdown("## 🗄️ Database Explorer")
    st.markdown("> Khám phá trực tiếp các bảng trong Data Warehouse")

    tables = {
        "dim_airline":       "Dimension: Hãng hàng không",
        "dim_airport":       "Dimension: Sân bay",
        "dim_date":          "Dimension: Thời gian",
        "dim_delay_reason":  "Dimension: Nguyên nhân delay",
        "fact_flights":      "Fact: Chuyến bay",
    }

    # ── Thông tin Star Schema ────────────────────────────
    st.markdown('<div class="section-title">⭐ Star Schema — Airline DW</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ```
        ┌──────────────┐
        │  dim_airline │
        └──────┬───────┘
               │
        ┌──────────────┐   ┌────────────────┐
        │  dim_airport │   │   dim_date     │
        └──────┬───────┘   └───────┬────────┘
               │                   │
               └─────┐     ┌───────┘
                     ▼     ▼
              ┌──────────────────┐
              │   fact_flights   │
              └──────────────────┘
                     │     │
               ┌─────┘     └──────────────┐
               ▼                          ▼
        ┌──────────────┐    ┌──────────────────────┐
        │  dim_airport │    │  dim_delay_reason    │
        │  (dest)      │    └──────────────────────┘
        └──────────────┘
        ```
        """)
    with col2:
        # Thống kê nhanh
        for tbl, desc in tables.items():
            try:
                cnt = query(f"SELECT COUNT(*) AS c FROM {tbl}").iloc[0,0]
                st.metric(f"📋 {tbl}", f"{cnt:,} dòng", delta=desc,
                          delta_color="off")
            except Exception:
                pass

    st.markdown("---")

    # ── Xem bảng ────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Xem dữ liệu bảng</div>',
                unsafe_allow_html=True)
    sel_table = st.selectbox(
        "Chọn bảng:",
        options=list(tables.keys()),
        format_func=lambda x: f"{x} — {tables[x]}"
    )

    limit = st.slider("Số dòng hiển thị:", 10, 500, 50)
    try:
        df_show = query(f"SELECT * FROM {sel_table} LIMIT {limit}")
        cnt     = query(f"SELECT COUNT(*) AS c FROM {sel_table}").iloc[0,0]
        st.info(f"Bảng **{sel_table}** có **{cnt:,}** dòng — đang hiển thị {min(limit,cnt)} dòng")
        st.dataframe(df_show, width='stretch', hide_index=True)
    except Exception as e:
        st.error(f"Lỗi: {e}")

    # ── Custom SQL ───────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">💻 Custom SQL Query</div>',
                unsafe_allow_html=True)
    default_sql = f"SELECT * FROM {sel_table} LIMIT 10"
    custom_sql = st.text_area("Nhập SQL query:", value=default_sql, height=100)

    if st.button("▶️ Chạy Query", type="primary"):
        try:
            result = query(custom_sql)
            st.success(f"✅ Kết quả: {len(result):,} dòng")
            st.dataframe(result, width='stretch', hide_index=True)
        except Exception as e:
            st.error(f"❌ Lỗi SQL: {e}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    page, W, time_mode = render_sidebar()

    if page == "🏠 Tổng quan":
        page_overview(W)
    elif page == "📈 Roll-up & Drill-down":
        page_rollup_drilldown(W)
    elif page == "🔪 Slice & Dice":
        page_slice_dice(W)
    elif page == "🔎 Phân tích nguyên nhân":
        page_root_cause(W)
    elif page == "🧠 Dự báo chuyến bay":
        page_predict()
    elif page == "🗄️ Database Explorer":
        page_db_explorer()


if __name__ == "__main__":
    main()
