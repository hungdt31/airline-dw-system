"""
Database Explorer: Star Schema, xem bảng, SQL tùy chỉnh (chỉ SELECT, có kiểm tra từ khóa).
"""

import random
import sqlite3
from typing import Tuple

import pandas as pd
import streamlit as st

from config import DB_PATH
from db import query
from icons import ERR, OK, SCHEMA_STAR, SQL_CONSOLE, TABLE_PREVIEW, TABLE_ROW

TABLES = {
    "dim_airline": "Dimension: Hãng hàng không",
    "dim_airport": "Dimension: Sân bay",
    "dim_date": "Dimension: Thời gian",
    "dim_delay_reason": "Dimension: Nguyên nhân delay",
    "fact_flights": "Fact: Chuyến bay",
}

ALLOWED_TABLES = list(TABLES.keys())

_SESSION_SQL_KEY = "explorer_custom_sql"
_SESSION_TABLE_KEY = "_explorer_sql_sync_table"


def _random_select_query() -> str:
    """Câu SELECT mẫu an toàn (không ORDER BY RANDOM() trên fact lớn)."""
    templates = [
        "SELECT * FROM dim_airline LIMIT 20",
        "SELECT * FROM dim_airport LIMIT 25",
        "SELECT month, month_name, quarter FROM dim_date ORDER BY month LIMIT 12",
        "SELECT * FROM dim_delay_reason LIMIT 10",
        "SELECT COUNT(*) AS total_flights FROM fact_flights",
        "SELECT is_delayed_15, COUNT(*) AS n FROM fact_flights GROUP BY is_delayed_15 LIMIT 10",
        f"SELECT * FROM {random.choice(ALLOWED_TABLES)} LIMIT {random.randint(10, 40)}",
    ]
    return random.choice(templates)


def _validate_readonly_sql(raw: str) -> Tuple[bool, str]:
    """Trả về (ok, lỗi)."""
    upper = raw.upper().strip()
    if not upper.startswith("SELECT"):
        return False, "Chỉ cho phép câu lệnh bắt đầu bằng SELECT."
    forbidden = (
        " ATTACH ",
        " DETACH ",
        " PRAGMA ",
        " VACUUM ",
        " DROP ",
        " DELETE ",
        " INSERT ",
        " UPDATE ",
        " REPLACE ",
    )
    if any(bad in f" {upper} " for bad in forbidden):
        return False, "Câu lệnh chứa từ khóa không được phép (chỉ truy vấn đọc)."
    return True, ""


def render_page_explorer() -> None:
    st.markdown("> Khám phá trực tiếp các bảng trong Data Warehouse")

    st.subheader(f"{SCHEMA_STAR} Star Schema — Airline DW")
    col_schema, col_stats = st.columns(2)
    with col_schema:
        st.markdown(
            """
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
"""
        )
    with col_stats:
        for tbl, desc in TABLES.items():
            try:
                cnt = int(query(f"SELECT COUNT(*) AS c FROM {tbl}").iloc[0, 0])
                st.metric(f"{TABLE_ROW} {tbl}", f"{cnt:,} dòng", delta=desc, delta_color="off")
            except Exception:
                pass

    st.markdown("---")
    st.subheader(f"{TABLE_PREVIEW} Xem dữ liệu bảng")

    sel_table = st.selectbox(
        "Chọn bảng:",
        options=ALLOWED_TABLES,
        format_func=lambda x: f"{x} — {TABLES[x]}",
    )

    limit = st.slider("Số dòng hiển thị:", 10, 500, 50)
    try:
        cnt = int(query(f"SELECT COUNT(*) AS c FROM {sel_table}").iloc[0, 0])
        shown = min(limit, cnt)
        df_show = query(f"SELECT * FROM {sel_table} LIMIT {int(limit)}")
        st.info(
            f"Bảng **{sel_table}** có **{cnt:,}** dòng — đang hiển thị **{shown}** dòng"
        )
        st.dataframe(df_show, width='stretch', hide_index=True)
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return

    st.markdown("---")
    st.subheader(f"{SQL_CONSOLE} Custom SQL Query")

    default_sql = f"SELECT * FROM {sel_table} LIMIT 10"
    if _SESSION_SQL_KEY not in st.session_state:
        st.session_state[_SESSION_SQL_KEY] = default_sql
    if st.session_state.get(_SESSION_TABLE_KEY) != sel_table:
        st.session_state[_SESSION_TABLE_KEY] = sel_table
        st.session_state[_SESSION_SQL_KEY] = default_sql

    # Random / Clear phải cập nhật session_state TRƯỚC khi tạo text_area (cùng key)
    b_rand, b_clear = st.columns(2)
    with b_rand:
        click_rand = st.button(
            "Random Query",
            width='stretch',
            key="explorer_sql_random",
        )
    with b_clear:
        click_clear = st.button(
            "Clear",
            width='stretch',
            key="explorer_sql_clear",
        )
    if click_rand:
        st.session_state[_SESSION_SQL_KEY] = _random_select_query()
    if click_clear:
        st.session_state[_SESSION_SQL_KEY] = ""

    st.text_area(
        "Nhập SQL query:",
        key=_SESSION_SQL_KEY,
        height=100,
        label_visibility="visible",
    )

    click_exec = st.button(
        "Execute",
        type="primary",
        width='stretch',
        key="explorer_sql_execute",
    )
    if click_exec:
        raw = (st.session_state.get(_SESSION_SQL_KEY) or "").strip()
        ok, err_msg = _validate_readonly_sql(raw)
        if not ok:
            st.error(err_msg)
        else:
            try:
                result = query(raw)
                st.success(f"{OK} Kết quả: {len(result):,} dòng")
                st.dataframe(result, width='stretch', hide_index=True)
            except (sqlite3.Error, ValueError, pd.io.sql.DatabaseError) as e:
                st.error(f"{ERR} Lỗi SQL: {e}")

    st.caption(f"Đường dẫn DB: `{DB_PATH}`")
