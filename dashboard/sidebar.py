"""
Sidebar: điều hướng multi-page + bộ lọc OLAP (FilterContext).

Lọc địa lý (sân bay đi/đến) nằm trên trang **Phân tích tuyến đường**, không lặp ở sidebar.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import pandas as pd
import streamlit as st

from icons import (
    BRAND,
    NAV_CAUSES,
    NAV_EXPLORER,
    NAV_HOME,
    NAV_PREDICT,
    NAV_ROUTES,
    NAV_TIME,
    SECTION_AIRLINE,
    SECTION_CALENDAR,
    SECTION_FILTERS,
)

# (page_id, nhãn radio) — icon đồng bộ với PAGE_MAIN_TITLES
APP_PAGES: List[Tuple[str, str]] = [
    ("home", f"{NAV_HOME} Tổng quan"),
    ("routes", f"{NAV_ROUTES} Phân tích tuyến đường"),
    ("time", f"{NAV_TIME} Phân tích thời gian"),
    ("causes", f"{NAV_CAUSES} Phân tích nguyên nhân"),
    ("predict", f"{NAV_PREDICT} Dự báo chuyến bay"),
    ("explorer", f"{NAV_EXPLORER} Database Explorer"),
]

PAGE_MAIN_TITLES: Dict[str, str] = {
    "home": f"{NAV_HOME} Tổng quan — Airline Data Warehouse 2015",
    "routes": f"{NAV_ROUTES} Phân tích tuyến đường",
    "time": f"{NAV_TIME} Phân tích thời gian",
    "causes": f"{NAV_CAUSES} Phân tích nguyên nhân",
    "predict": f"{NAV_PREDICT} Dự báo chuyến bay",
    "explorer": f"{NAV_EXPLORER} Database Explorer",
}


def inject_dashboard_styles() -> None:
    st.markdown(
        """
<style>
  [data-testid="stSidebar"] .stRadio label {
    font-size: 14px;
    padding: 4px 0;
  }
  div[data-testid="metric-container"] {
    background-color: var(--st-secondary-background-color);
    border-radius: 10px;
    padding: 12px 16px;
    border-left: 4px solid var(--st-primary-color);
  }
  .section-title {
    font-size: 20px;
    font-weight: bold;
    color: var(--st-text-color);
    border-left: 4px solid var(--st-primary-color);
    padding-left: 10px;
    margin: 16px 0 8px;
  }
</style>
""",
        unsafe_allow_html=True,
    )


@dataclass
class FilterContext:
    """Trạng thái bộ lọc global: thời gian + hãng (không gồm sân bay — xem trang tuyến)."""

    time_mode: str
    time_filter: str
    month_labels: dict
    sel_months: List[int]
    sel_quarters: List[int]
    airline_labels: dict
    sel_airlines: List[str]
    _airline_filter: str

    def where(self, *extra: str) -> str:
        parts = [self.time_filter, self._airline_filter] + list(extra)
        return " AND ".join(parts)


def render_sidebar_filters(query: Callable[[str], pd.DataFrame]) -> FilterContext:
    st.sidebar.markdown(f"**{SECTION_CALENDAR} Thời gian**")
    time_mode = st.sidebar.radio(
        "Lọc theo",
        options=["Tháng", "Quý"],
        horizontal=True,
        label_visibility="collapsed",
    )

    time_filter = "1=1"

    if time_mode == "Tháng":
        months_df = query("SELECT DISTINCT month, month_name FROM dim_date ORDER BY month")
        month_labels = dict(zip(months_df["month"], months_df["month_name"]))
        sel_months = st.sidebar.multiselect(
            "Chọn tháng",
            options=months_df["month"].tolist(),
            format_func=lambda x: month_labels.get(x, x),
            default=[],
        )
        if sel_months:
            time_filter = f"d.month IN ({','.join(map(str, sel_months))})"
        sel_quarters = []
    else:
        month_labels = {}
        sel_months = []
        sel_quarters = st.sidebar.multiselect(
            "Chọn quý",
            options=[1, 2, 3, 4],
            format_func=lambda x: f"Q{x}",
            default=[],
        )
        if sel_quarters:
            time_filter = f"d.quarter IN ({','.join(map(str, sel_quarters))})"

    airlines_df = query("SELECT airline_iata, airline_name FROM dim_airline ORDER BY airline_name")
    airline_labels = dict(zip(airlines_df["airline_iata"], airlines_df["airline_name"]))
    sel_airlines = st.sidebar.multiselect(
        f"{SECTION_AIRLINE} Hãng hàng không",
        options=airlines_df["airline_iata"].tolist(),
        format_func=lambda x: airline_labels.get(x, x),
        default=[],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Q1: Tháng 1 - Tháng 3 | Q2: Tháng 4 - Tháng 6 | Q3: Tháng 7 - Tháng 9 | Q4: Tháng 10 - Tháng 12"
    )
    st.sidebar.caption("Chạy `python pipeline.py` để nạp dữ liệu vào DB")

    airline_filter = (
        f"a.airline_iata IN ({','.join(repr(x) for x in sel_airlines)})"
        if sel_airlines
        else "1=1"
    )

    return FilterContext(
        time_mode=time_mode,
        time_filter=time_filter,
        month_labels=month_labels,
        sel_months=sel_months,
        sel_quarters=sel_quarters,
        airline_labels=airline_labels,
        sel_airlines=sel_airlines,
        _airline_filter=airline_filter,
    )


def render_app_sidebar(query: Callable[[str], pd.DataFrame]) -> Tuple[str, FilterContext]:
    st.sidebar.markdown(f"# {BRAND} Airline Data Warehouse")
    st.sidebar.caption(
        "Data Mining Assignment · Kaggle US Flight Delays 2015 · Star Schema · SQLite"
    )
    st.sidebar.markdown("---")

    labels = [lab for _, lab in APP_PAGES]
    choice = st.sidebar.radio(
        "Chọn trang phân tích:",
        options=labels,
        label_visibility="visible",
    )
    page_id = next(pid for pid, lab in APP_PAGES if lab == choice)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {SECTION_FILTERS} Bộ lọc toàn cục")
    st.sidebar.caption(
        "Áp dụng: Tổng quan, Phân tích tuyến, Phân tích thời gian, Nguyên nhân. "
        "Không áp dụng: Dự báo, Database Explorer."
    )

    filters = render_sidebar_filters(query)
    return page_id, filters
