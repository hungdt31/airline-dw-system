"""
Kết nối SQLite + truy vấn có cache (Streamlit).
"""

import os
import sqlite3

import pandas as pd
import streamlit as st

from config import DB_PATH
from icons import ERR


@st.cache_resource
def get_conn():
    if not os.path.exists(DB_PATH):
        st.error(f"{ERR} Chưa có database. Chạy `python pipeline.py` trước.")
        st.stop()
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data(ttl=300)
def query(sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, get_conn())
