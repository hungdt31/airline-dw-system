"""
Trang Phân tích thời gian — tháng/mùa, quý, khung giờ, ngày trong tuần, heatmap nguyên nhân × tháng.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import BASE_JOIN
from icons import CAUSES_HOUR, CAUSES_SEASON, CALENDAR_DOW
from sidebar import FilterContext

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

_DOW_ORDER_EN = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
_DOW_LABELS_VN = {
    "Monday": "Thứ 2",
    "Tuesday": "Thứ 3",
    "Wednesday": "Thứ 4",
    "Thursday": "Thứ 5",
    "Friday": "Thứ 6",
    "Saturday": "Thứ 7",
    "Sunday": "Chủ nhật",
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


def render_page_time(f: FilterContext, query: Callable[[str], object]) -> None:
    st.markdown(
        "> Phân tích theo **thời gian**: tháng/mùa, quý, **ngày trong tuần**, "
        "khung giờ cất cánh, và phân bố nguyên nhân trễ theo tháng."
    )

    if f.time_mode == "Tháng":
        st.subheader("Phân bố số chuyến: tháng × nguyên nhân")
        df_heat = query(f"""
            SELECT d.month AS month,
                dr.reason_name AS reason_name,
                COUNT(*) AS cnt
            {BASE_JOIN}
            JOIN dim_delay_reason dr ON f.delay_reason_key = dr.reason_key
            WHERE dr.reason_code != 'NO_DELAY' AND f.is_cancelled = 0
              AND {f.where()}
            GROUP BY d.month, dr.reason_name
            ORDER BY d.month, dr.reason_name
        """)
        if not df_heat.empty:
            pivot = df_heat.pivot_table(
                index="reason_name",
                columns="month",
                values="cnt",
                aggfunc="sum",
                fill_value=0,
            )
            pivot = pivot.reindex(columns=range(1, 13), fill_value=0)
            col_labels = [f"T{m}" for m in range(1, 13)]
            fig_hm = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=col_labels,
                    y=pivot.index.astype(str),
                    colorscale="YlOrRd",
                    hovertemplate="%{y}<br>%{x}: %{z} chuyến<extra></extra>",
                )
            )
            fig_hm.update_layout(
                height=max(280, 36 * len(pivot.index)),
                margin=dict(t=10, b=10),
                xaxis_title="Tháng",
                yaxis_title="Nguyên nhân (phân loại chính)",
            )
            st.plotly_chart(fig_hm, width="stretch")
            st.caption(
                "So sánh mùa vụ giữa các loại nguyên nhân (theo phân loại chính trên từng chuyến)."
            )

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
            st.plotly_chart(fig7, width="stretch")
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
            st.plotly_chart(fig7b, width="stretch")

    st.subheader(f"{CALENDAR_DOW} Delay theo ngày trong tuần")
    df_dow = query(f"""
        SELECT d.day_of_week,
            d.day_name,
            ROUND(AVG(f.arrival_delay),1)                  AS avg_arr_delay,
            ROUND(AVG(f.departure_delay),1)                AS avg_dep_delay,
            ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2)  AS delay_rate_pct,
            COUNT(*)                                       AS total_flights
        {BASE_JOIN}
        WHERE f.is_cancelled = 0 AND {f.where()}
        GROUP BY d.day_of_week, d.day_name
        ORDER BY d.day_of_week
    """)
    if not df_dow.empty:
        df_dow = df_dow.copy()
        df_dow["thu"] = df_dow["day_name"].map(_DOW_LABELS_VN).fillna(df_dow["day_name"])
        present_en = set(df_dow["day_name"])
        cat_order = [_DOW_LABELS_VN[d] for d in _DOW_ORDER_EN if d in present_en]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Delay đến TB (phút)**")
            fig_d1 = px.bar(
                df_dow,
                x="thu",
                y="avg_arr_delay",
                category_orders={"thu": cat_order},
                color="avg_arr_delay",
                color_continuous_scale="Reds",
                labels={"avg_arr_delay": "Phút", "thu": ""},
            )
            fig_d1.update_layout(height=300, margin=dict(t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_d1, width="stretch")
        with c2:
            st.markdown("**Tỷ lệ chuyến trễ ≥15 phút (%)**")
            fig_d2 = px.bar(
                df_dow,
                x="thu",
                y="delay_rate_pct",
                category_orders={"thu": cat_order},
                color="delay_rate_pct",
                color_continuous_scale="OrRd",
                labels={"delay_rate_pct": "%", "thu": ""},
            )
            fig_d2.update_layout(height=300, margin=dict(t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_d2, width="stretch")

    wk = query(f"""
        SELECT d.is_weekend,
            ROUND(AVG(f.arrival_delay),1)                  AS avg_arr_delay,
            ROUND(100.0*SUM(f.is_delayed_15)/COUNT(*),2) AS delay_rate_pct,
            COUNT(*) AS n
        {BASE_JOIN}
        WHERE f.is_cancelled = 0 AND {f.where()}
        GROUP BY d.is_weekend
    """)
    if not wk.empty:
        row_by_wk = {int(r["is_weekend"]): r for _, r in wk.iterrows()}
        mc1, mc2 = st.columns(2)
        if 0 in row_by_wk:
            r = row_by_wk[0]
            mc1.metric(
                "Delay TB ngày thường",
                f"{r['avg_arr_delay']:.1f} phút",
                help=(
                    f"Một số duy nhất: AVG(arrival_delay) trên mọi chuyến có is_weekend=0. "
                    f"Trễ ≥15: {r['delay_rate_pct']:.2f}% · {int(r['n']):,} chuyến."
                ),
            )
        if 1 in row_by_wk:
            r = row_by_wk[1]
            mc2.metric(
                "Delay TB cuối tuần (T7 + CN)",
                f"{r['avg_arr_delay']:.1f} phút",
                help=(
                    f"Một số duy nhất: AVG(arrival_delay) trên mọi chuyến có is_weekend=1. "
                    f"Trễ ≥15: {r['delay_rate_pct']:.2f}% · {int(r['n']):,} chuyến."
                ),
            )

    st.subheader(f"{CAUSES_HOUR} Delay theo khung giờ cất cánh (theo lịch)")
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
        st.plotly_chart(fig8, width="stretch")
