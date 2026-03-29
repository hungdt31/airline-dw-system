"""
Trang Tổng quan — KPI, xu hướng, OTP, sân bay delay.
"""

from typing import Callable

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config import BASE_JOIN
from icons import (
    CAPTION_FILTERS,
    CHART_AIRPORT,
    CHART_MIX,
    CHART_RANK,
    CHART_TREND,
    METRIC_AVG_DELAY,
    METRIC_CANCEL,
    METRIC_DELAY_RATE,
    METRIC_OTP,
    METRIC_TOTAL_FLIGHTS,
)
from sidebar import FilterContext


def render_page_overview(f: FilterContext, query: Callable[[str], object]) -> None:
    st.markdown(
        """
        ### Về dự án

        - **Mô hình:** Star Schema — 1 Fact Table + 4 Dimensions  
        - **Dataset:** US Flight Delays 2015  
        - **ETL:** Python + SQLite
        """
    )

    kpi = query(f"""
        SELECT
            COUNT(*)                                               AS total_flights,
            ROUND(100.0*(1-1.0*SUM(f.is_delayed_15)/COUNT(*)),2)  AS otp_pct,
            ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2)           AS delay_rate,
            ROUND(100.0*SUM(f.is_cancelled)/COUNT(*),2)            AS cancel_rate,
            ROUND(COALESCE(AVG(f.arrival_delay), 0), 1)            AS avg_delay
        {BASE_JOIN}
        WHERE {f.where()}
    """).iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{METRIC_TOTAL_FLIGHTS} Tổng chuyến bay", f"{int(kpi.total_flights):,}")
    c2.metric(f"{METRIC_OTP} OTP toàn ngành", f"{kpi.otp_pct}%")
    c3.metric(f"{METRIC_DELAY_RATE} Tỉ lệ delay ≥15p", f"{kpi.delay_rate}%")
    c4.metric(f"{METRIC_CANCEL} Tỉ lệ hủy chuyến", f"{kpi.cancel_rate}%")
    c5.metric(f"{METRIC_AVG_DELAY} Delay TB", f"{kpi.avg_delay} phút")

    active = []
    if f.time_mode == "Tháng" and f.sel_months:
        active.append(
            "Tháng: " + ", ".join(f.month_labels[m] for m in f.sel_months)
        )
    elif f.time_mode == "Quý" and f.sel_quarters:
        active.append("Quý: " + ", ".join("Q" + str(q) for q in f.sel_quarters))
    if f.sel_airlines:
        active.append("Hãng: " + ", ".join(f.airline_labels[x] for x in f.sel_airlines))
    if active:
        st.caption(f"{CAPTION_FILTERS} Đang lọc: " + " | ".join(active))

    st.divider()

    col_a, col_b = st.columns([3, 2])

    with col_a:
        if f.time_mode == "Tháng":
            st.subheader(f"{CHART_MIX} Cơ cấu chuyến bay theo tháng")
            df_trend = query(f"""
                SELECT d.month AS period_num, d.month_name AS period_label,
                    COUNT(*) AS total_flights,
                    SUM(f.is_cancelled) AS cancelled,
                    SUM(CASE WHEN f.is_cancelled = 0 AND f.is_delayed_15 = 1 THEN 1 ELSE 0 END)
                        AS delayed_nc,
                    SUM(CASE WHEN f.is_cancelled = 0 AND f.is_delayed_15 = 0 THEN 1 ELSE 0 END)
                        AS on_time
                {BASE_JOIN}
                WHERE {f.where()}
                GROUP BY d.month, d.month_name ORDER BY d.month
            """)
        else:
            st.subheader(f"{CHART_MIX} Cơ cấu chuyến bay theo quý")
            df_trend = query(f"""
                SELECT d.quarter AS period_num,
                    'Q' || d.quarter AS period_label,
                    COUNT(*) AS total_flights,
                    SUM(f.is_cancelled) AS cancelled,
                    SUM(CASE WHEN f.is_cancelled = 0 AND f.is_delayed_15 = 1 THEN 1 ELSE 0 END)
                        AS delayed_nc,
                    SUM(CASE WHEN f.is_cancelled = 0 AND f.is_delayed_15 = 0 THEN 1 ELSE 0 END)
                        AS on_time
                {BASE_JOIN}
                WHERE {f.where()}
                GROUP BY d.quarter ORDER BY d.quarter
            """)

        if not df_trend.empty:
            # Xếp chồng (dưới → trên): Hủy → Delay → Đúng giờ = tổng chuyến
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=df_trend["period_label"],
                    y=df_trend["cancelled"],
                    name="Hủy chuyến",
                    marker_color="#888888",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=df_trend["period_label"],
                    y=df_trend["delayed_nc"],
                    name="Delay ≥15p",
                    marker_color="#E8593C",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=df_trend["period_label"],
                    y=df_trend["on_time"],
                    name="Đúng giờ",
                    marker_color="#4A90D9",
                )
            )
            fig.update_layout(
                barmode="stack",
                height=340,
                margin=dict(t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis_title="Số chuyến",
            )
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Chiều cao mỗi cột = **tổng chuyến**; từ **dưới lên**: hủy → delay ≥15p → đúng giờ."
            )
        else:
            st.info("Chưa có dữ liệu — hãy chạy pipeline.py trước")

    with col_b:
        st.subheader(f"{CHART_RANK} Top 5 hãng theo OTP")
        st.write("On-Time Performance — tỉ lệ chuyến bay đúng giờ.")
        df_otp = query(f"""
            SELECT a.airline_name,
                ROUND(100.0*(1-1.0*SUM(f.is_delayed_15)/COUNT(*)),1) AS otp_pct,
                COUNT(*) AS total_flights
            {BASE_JOIN}
            WHERE {f.where("f.is_cancelled=0")}
            GROUP BY a.airline_name HAVING total_flights > 100
            ORDER BY otp_pct DESC LIMIT 5
        """)
        if not df_otp.empty:
            fig2 = px.bar(
                df_otp,
                x="otp_pct",
                y="airline_name",
                orientation="h",
                color="otp_pct",
                color_continuous_scale="YlGnBu",
                labels={"otp_pct": "OTP (%)", "airline_name": ""},
            )
            fig2.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig2, width='stretch')

    col_c, col_d = st.columns([2, 3])

    with col_c:
        st.subheader(f"{CHART_AIRPORT} Top 5 sân bay delay cao nhất")
        df_ap = query(f"""
            SELECT ap_o.airport_iata,
                   COALESCE(ap_o.city, ap_o.airport_iata) AS city,
                   ROUND(AVG(f.arrival_delay), 1)          AS avg_delay,
                   COUNT(*)                                AS flights
            {BASE_JOIN}
            WHERE {f.where("f.is_cancelled=0")}
            GROUP BY ap_o.airport_iata, ap_o.city
            HAVING flights > 500
            ORDER BY avg_delay DESC LIMIT 5
        """)
        if not df_ap.empty:
            fig3 = px.bar(
                df_ap,
                x="airport_iata",
                y="avg_delay",
                color="avg_delay",
                color_continuous_scale="OrRd",
                text="city",
                labels={"airport_iata": "Sân bay", "avg_delay": "Delay TB (phút)"},
            )
            fig3.update_traces(textposition="outside")
            fig3.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig3, width='stretch')

    with col_d:
        if f.time_mode == "Tháng":
            st.subheader(f"{CHART_TREND} Tỉ lệ delay & hủy theo tháng")
            df_rate = query(f"""
                SELECT d.month AS pnum, d.month_name AS plabel,
                    ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate,
                    ROUND(100.0*SUM(f.is_cancelled)/COUNT(*),2)  AS cancel_rate
                {BASE_JOIN}
                WHERE {f.where()}
                GROUP BY d.month, d.month_name ORDER BY d.month
            """)
        else:
            st.subheader(f"{CHART_TREND} Tỉ lệ delay & hủy theo quý")
            df_rate = query(f"""
                SELECT d.quarter AS pnum, 'Q'||d.quarter AS plabel,
                    ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate,
                    ROUND(100.0*SUM(f.is_cancelled)/COUNT(*),2)  AS cancel_rate
                {BASE_JOIN}
                WHERE {f.where()}
                GROUP BY d.quarter ORDER BY d.quarter
            """)

        if not df_rate.empty:
            fig4 = go.Figure()
            fig4.add_scatter(
                x=df_rate["plabel"],
                y=df_rate["delay_rate"],
                name="Delay rate %",
                mode="lines+markers",
                line=dict(color="#E8593C", width=3),
            )
            fig4.add_scatter(
                x=df_rate["plabel"],
                y=df_rate["cancel_rate"],
                name="Cancel rate %",
                mode="lines+markers",
                line=dict(color="#888", width=2, dash="dash"),
            )
            fig4.update_layout(
                height=280,
                margin=dict(t=10, b=10),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig4, width='stretch')
