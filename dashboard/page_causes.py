"""
Trang Phân tích nguyên nhân delay — pie, delay TB theo nguyên nhân, bảng tóm tắt, tổng phút trễ.
(Phân tích theo tháng/mùa, quý, khung giờ và heatmap → trang Phân tích thời gian.)
"""

from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from config import BASE_JOIN
from sidebar import FilterContext


def render_page_causes(f: FilterContext, query: Callable[[str], object]) -> None:
    st.markdown(
        "> Phân tích sâu **nguyên nhân** gây delay: Thời tiết, Hãng bay, Hệ thống, … "
        "(xu hướng theo **thời gian** xem trang *Phân tích thời gian*)."
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
                    END), 1) AS avg_min,
                ROUND(SUM(
                    CASE dr.reason_code
                        WHEN 'AIR_SYSTEM'    THEN f.delay_air_system
                        WHEN 'SECURITY'      THEN f.delay_security
                        WHEN 'AIRLINE'       THEN f.delay_airline
                        WHEN 'LATE_AIRCRAFT' THEN f.delay_late_aircraft
                        WHEN 'WEATHER'       THEN f.delay_weather
                        ELSE 0
                    END), 0) AS total_min
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

    if not df_reason.empty:
        total_ops = int(
            query(
                f"""
            SELECT COUNT(*) AS n
            {BASE_JOIN}
            WHERE f.is_cancelled = 0 AND {f.where()}
            """
            )["n"].iloc[0]
        )
        tbl = df_reason.copy()
        tbl["pct_trong_nn"] = (100.0 * tbl["cnt"] / tbl["cnt"].sum()).round(2)
        tbl["pct_tren_tong_chuyen"] = (100.0 * tbl["cnt"] / max(total_ops, 1)).round(2)
        tbl = tbl.rename(
            columns={
                "reason_name": "Nguyên nhân",
                "cnt": "Số chuyến",
                "avg_min": "Delay TB (phút)",
                "total_min": "Tổng phút trễ (theo cột thành phần)",
                "pct_trong_nn": "% trong nhóm có NN",
                "pct_tren_tong_chuyen": "% trên tổng chuyến (đã lọc)",
            }
        )
        st.markdown("**Bảng tóm tắt nguyên nhân (theo phân loại chính trên từng chuyến)**")
        st.caption(
            "Mỗi chuyến chỉ gán một nguyên nhân chính (ETL). "
            "“Tổng phút trễ” = cộng đúng cột thành phần khớp nguyên nhân đó."
        )
        st.dataframe(
            tbl[
                [
                    "Nguyên nhân",
                    "Số chuyến",
                    "% trên tổng chuyến (đã lọc)",
                    "% trong nhóm có NN",
                    "Delay TB (phút)",
                    "Tổng phút trễ (theo cột thành phần)",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

        st.markdown("**Đóng góp tổng phút trễ theo nguyên nhân**")
        fig_total = px.bar(
            tbl.sort_values("Tổng phút trễ (theo cột thành phần)", ascending=True),
            x="Tổng phút trễ (theo cột thành phần)",
            y="Nguyên nhân",
            orientation="h",
            color="Tổng phút trễ (theo cột thành phần)",
            color_continuous_scale="OrRd",
        )
        fig_total.update_layout(
            height=max(260, 40 * len(tbl)),
            margin=dict(t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_total, width="stretch")
