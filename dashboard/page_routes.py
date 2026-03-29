"""
Phân tích tuyến bay (sân bay đi → đến): lọc địa lý trên trang; hãng + thời gian từ sidebar.
Mỗi hãng được chọn (multiselect) hiển thị top tuyến riêng.
"""

from typing import Callable, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from config import BASE_JOIN, STATE_MAP
from icons import GEO_DEST, GEO_ORIGIN, PREDICT_AIRLINE
from sidebar import FilterContext


def _state_label(code: str) -> str:
    return f"{code} — {STATE_MAP.get(code, code)}" if code != "(Tất cả)" else "(Tất cả)"


def _render_route_geo_filters(query: Callable[[str], pd.DataFrame]) -> Tuple[str, str]:
    """
    UI điểm đi / điểm đến trên trang; trả về (origin_sql_clause, dest_sql_clause).
    """
    all_airports_df = query("""
        SELECT airport_iata,
               COALESCE(state, '')  AS state,
               COALESCE(city, '')   AS city,
               airport_iata || ' — ' || COALESCE(city, airport_iata) AS label
        FROM dim_airport
        ORDER BY state, airport_iata
    """)
    all_states = sorted([s for s in all_airports_df["state"].unique().tolist() if s])
    state_options = ["(Tất cả)"] + all_states

    st.session_state.setdefault("routes_prev_o_state", "(Tất cả)")
    st.session_state.setdefault("routes_prev_d_state", "(Tất cả)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{GEO_ORIGIN} Điểm đi**")
        o_state = st.selectbox(
            "Bang xuất phát",
            options=state_options,
            format_func=_state_label,
            key="routes_origin_state",
        )
        if o_state != st.session_state.get("routes_prev_o_state", "(Tất cả)"):
            st.session_state["routes_prev_o_state"] = o_state
            st.session_state["routes_sel_origin_ap"] = []

        o_opts = (
            all_airports_df
            if o_state == "(Tất cả)"
            else all_airports_df[all_airports_df["state"] == o_state]
        )
        o_labels = dict(zip(o_opts["airport_iata"], o_opts["label"]))
        sel_o = st.multiselect(
            "Sân bay đi",
            options=o_opts["airport_iata"].tolist(),
            format_func=lambda x: o_labels.get(x, x),
            key="routes_sel_origin_ap",
        )

    with c2:
        st.markdown(f"**{GEO_DEST} Điểm đến**")
        d_state = st.selectbox(
            "Bang đích đến",
            options=state_options,
            format_func=_state_label,
            key="routes_dest_state",
        )
        if d_state != st.session_state.get("routes_prev_d_state", "(Tất cả)"):
            st.session_state["routes_prev_d_state"] = d_state
            st.session_state["routes_sel_dest_ap"] = []

        d_opts = (
            all_airports_df
            if d_state == "(Tất cả)"
            else all_airports_df[all_airports_df["state"] == d_state]
        )
        d_labels = dict(zip(d_opts["airport_iata"], d_opts["label"]))
        sel_d = st.multiselect(
            "Sân bay đến",
            options=d_opts["airport_iata"].tolist(),
            format_func=lambda x: d_labels.get(x, x),
            key="routes_sel_dest_ap",
        )

    if sel_o:
        origin_f = f"ap_o.airport_iata IN ({','.join(repr(x) for x in sel_o)})"
    elif o_state != "(Tất cả)":
        ap_list = o_opts["airport_iata"].tolist()
        origin_f = (
            f"ap_o.airport_iata IN ({','.join(repr(x) for x in ap_list)})" if ap_list else "1=1"
        )
    else:
        origin_f = "1=1"

    if sel_d:
        dest_f = f"ap_d.airport_iata IN ({','.join(repr(x) for x in sel_d)})"
    elif d_state != "(Tất cả)":
        ap_list = d_opts["airport_iata"].tolist()
        dest_f = (
            f"ap_d.airport_iata IN ({','.join(repr(x) for x in ap_list)})" if ap_list else "1=1"
        )
    else:
        dest_f = "1=1"

    return origin_f, dest_f


def render_page_routes(f: FilterContext, query: Callable[[str], object]) -> None:
    st.markdown(
        """
        > **Thời gian** và **hãng** (có thể chọn nhiều) lấy từ **sidebar**.  
        > **Điểm đi / điểm đến** lọc tại trang này. Mỗi hãng được chọn có một khối **top tuyến** riêng.
        """
    )

    origin_f, dest_f = _render_route_geo_filters(query)
    st.divider()

    if not f.sel_airlines:
        st.warning("Chọn **ít nhất một hãng** ở sidebar để xem phân tích tuyến theo từng hãng.")
        return

    col_top_n, col_min_fl = st.columns(2)
    with col_top_n:
        top_n = st.slider("Số tuyến tối đa mỗi hãng", 5, 40, 15)
    with col_min_fl:
        min_flights = st.number_input(
            "Tối thiểu chuyến / tuyến (HAVING)", 5, 500, 20, 5
        )

    for airline_iata in f.sel_airlines:
        label = f.airline_labels.get(airline_iata, airline_iata)
        st.subheader(f"{PREDICT_AIRLINE} {label} — `{airline_iata}`")

        wf = " AND ".join(
            [
                f.time_filter,
                f"a.airline_iata = {repr(airline_iata)}",
                origin_f,
                dest_f,
                "f.is_cancelled = 0",
            ]
        )

        df = query(f"""
            SELECT ap_o.airport_iata || ' → ' || ap_d.airport_iata AS route,
                   COUNT(*) AS flights,
                   ROUND(AVG(f.arrival_delay), 1) AS avg_arr_delay,
                   ROUND(100.0 * SUM(f.is_delayed_15) / COUNT(*), 2) AS delay_pct
            {BASE_JOIN}
            WHERE {wf}
            GROUP BY ap_o.airport_iata, ap_d.airport_iata
            HAVING flights >= {int(min_flights)}
            ORDER BY flights DESC
            LIMIT {int(top_n)}
        """)

        if not isinstance(df, pd.DataFrame) or df.empty:
            st.caption("Không có tuyến đủ điều kiện với bộ lọc hiện tại.")
            continue

        fig = px.bar(
            df.sort_values("flights", ascending=True),
            x="flights",
            y="route",
            orientation="h",
            color="avg_arr_delay",
            color_continuous_scale="Reds",
            labels={
                "flights": "Số chuyến",
                "route": "Tuyến",
                "avg_arr_delay": "Delay đến TB (phút)",
            },
            title=None,
        )
        fig.update_layout(height=max(280, len(df) * 26), margin=dict(t=10, l=10))
        st.plotly_chart(fig, width='stretch')

        with st.expander("Bảng số liệu"):
            st.dataframe(df, width='stretch', hide_index=True)
