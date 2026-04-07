"""
Trang Nguyên nhân delay — pie, theo mùa, khung giờ.
"""

from typing import Callable, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import BASE_JOIN
from icons import CAUSES_HOUR, CAUSES_SEASON
from sidebar import FilterContext

# Trục tháng cố định (trùng dim_date / SQL) — ghép đủ 12 tháng tránh lệch category khi thiếu tháng
_MONTH_TEMPLATE = pd.DataFrame(
    {
        "month": range(1, 13),
        "month_name": [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
    }
)

_SEASON_COLORS = {
    "❄️ Đông": "#2471A3",
    "🌸 Xuân": "#85C1E2",
    "☀️ Hè": "#E74C3C",
    "🍂 Thu": "#F5B7B1",
}


def _season_label(month: int) -> str:
    if month in (12, 1, 2):
        return "❄️ Đông"
    if month in (3, 4, 5):
        return "🌸 Xuân"
    if month in (6, 7, 8):
        return "☀️ Hè"
    return "🍂 Thu"


def _fig_avg_delay_by_month(df_season: pd.DataFrame) -> go.Figure:
    """
    Một trace Bar duy nhất (tránh px.bar + color tạo nhiều trace → chồng cột cuối trục).
    Luôn 12 nhãn tháng; tháng không có dữ liệu: không vẽ cột (y=None), vẫn giữ khe trên trục X.
    """
    df_m = _MONTH_TEMPLATE.merge(
        df_season[["month", "avg_delay"]].drop_duplicates("month"),
        on="month",
        how="left",
    )
    df_m["season"] = df_m["month"].map(_season_label)
    has = df_m["avg_delay"].notna()
    bar_colors: List[str] = [
        _SEASON_COLORS[s] if h else "rgba(200,200,200,0.35)"
        for s, h in zip(df_m["season"], has)
    ]
    y_vals: List[Optional[float]] = [
        float(v) if pd.notna(v) else None for v in df_m["avg_delay"]
    ]
    customdata: List[str] = [
        f"Delay TB: {v:.1f} phút" if pd.notna(v) else "Không có dữ liệu trong bộ lọc"
        for v in df_m["avg_delay"]
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=df_m["month_name"],
                y=y_vals,
                marker=dict(color=bar_colors, line=dict(width=0)),
                hovertemplate="%{x}<br>%{customdata}<extra></extra>",
                customdata=customdata,
                showlegend=False,
            )
        ]
    )
    seasons_used = set(df_m.loc[has, "season"])
    for season, color in _SEASON_COLORS.items():
        if season in seasons_used:
            fig.add_trace(
                go.Bar(
                    x=[None],
                    y=[None],
                    name=season,
                    marker=dict(color=color),
                    showlegend=True,
                    visible="legendonly",
                )
            )
    month_order = df_m["month_name"].tolist()
    fig.update_layout(
        height=320,
        margin=dict(t=10, b=10),
        bargap=0.1,
        xaxis=dict(
            title="Tháng",
            categoryorder="array",
            categoryarray=month_order,
            type="category",
        ),
        yaxis=dict(title="Delay TB (phút)"),
        legend_title_text="Mùa",
    )
    return fig


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
            df_season = df_season.sort_values("month", ascending=True).reset_index(
                drop=True
            )
            fig7 = _fig_avg_delay_by_month(df_season)
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
