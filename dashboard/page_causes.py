"""
Trang Nguyên nhân delay — pie, theo mùa, khung giờ.
"""

from typing import Callable

import plotly.express as px
import streamlit as st

from config import BASE_JOIN
from icons import CAUSES_HOUR, CAUSES_SEASON
from sidebar import FilterContext


def render_page_causes(f: FilterContext, query: Callable[[str], object]) -> None:
    st.markdown(
        "> Phân tích sâu nguyên nhân gây delay: Thời tiết, Hãng bay, Hệ thống, ..."
    )

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
              AND {f.where()}
            GROUP BY dr.reason_name ORDER BY cnt DESC
        """)
        if not df_reason.empty:
            fig5 = px.pie(
                df_reason,
                values="cnt",
                names="reason_name",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig5.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig5, width='stretch')

    with col_f:
        st.markdown("**Delay TB theo nguyên nhân (phút)**")
        if not df_reason.empty:
            fig6 = px.bar(
                df_reason.sort_values("avg_min", ascending=True),
                x="avg_min",
                y="reason_name",
                orientation="h",
                color="avg_min",
                color_continuous_scale="Reds",
                labels={"avg_min": "Delay TB (phút)", "reason_name": ""},
            )
            fig6.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig6, width='stretch')

    if f.time_mode == "Tháng":
        st.subheader(f"{CAUSES_SEASON} Delay trung bình theo tháng & mùa")
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
            WHERE f.is_cancelled=0 AND {f.where()}
            GROUP BY d.month, d.month_name ORDER BY d.month
        """)
        if not df_season.empty:
            fig7 = px.bar(
                df_season,
                x="month_name",
                y="avg_delay",
                color="season",
                labels={"avg_delay": "Delay TB (phút)", "month_name": "Tháng"},
            )
            fig7.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig7, width='stretch')
    else:
        st.subheader(f"{CAUSES_SEASON} Delay trung bình theo quý")
        df_qseason = query(f"""
            SELECT 'Q'||d.quarter AS quarter_label,
                ROUND(AVG(f.arrival_delay),1)                 AS avg_delay,
                ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate
            {BASE_JOIN}
            WHERE f.is_cancelled=0 AND {f.where()}
            GROUP BY d.quarter ORDER BY d.quarter
        """)
        if not df_qseason.empty:
            fig7b = px.bar(
                df_qseason,
                x="quarter_label",
                y="avg_delay",
                color="avg_delay",
                color_continuous_scale="Blues",
                labels={"avg_delay": "Delay TB (phút)", "quarter_label": "Quý"},
            )
            fig7b.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig7b, width='stretch')

    st.subheader(f"{CAUSES_HOUR} Delay theo khung giờ bay")
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
          AND {f.where()}
        GROUP BY time_slot ORDER BY time_slot
    """)
    if not df_time.empty:
        df_time = df_time.copy()
        df_time["time_slot"] = df_time["time_slot"].str[3:]
        fig8 = px.bar(
            df_time,
            x="time_slot",
            y="avg_dep_delay",
            color="delay_rate_pct",
            color_continuous_scale="YlOrRd",
            labels={
                "avg_dep_delay": "Delay TB (phút)",
                "time_slot": "Khung giờ",
                "delay_rate_pct": "Delay rate %",
            },
        )
        fig8.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig8, width='stretch')
