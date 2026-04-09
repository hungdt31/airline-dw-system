"""
dashboard/app.py
================
Multi-page Streamlit dashboard (radio sidebar).

Chạy từ thư mục gốc repo:
    streamlit run dashboard/app.py
"""

import os
import sys

_DASH_DIR = os.path.dirname(os.path.abspath(__file__))
if _DASH_DIR not in sys.path:
    sys.path.insert(0, _DASH_DIR)

import streamlit as st

from icons import FAVICON
from db import query
from page_explorer import render_page_explorer
from page_routes import render_page_routes
from sidebar import PAGE_MAIN_TITLES, inject_dashboard_styles, render_app_sidebar
from page_causes import render_page_causes
from page_overview import render_page_overview
from page_predict import render_page_predict
from page_time import render_page_time

st.set_page_config(
    page_title="Airline DW Analytics",
    page_icon=FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_dashboard_styles()

page_id, filters = render_app_sidebar(query)

st.title(PAGE_MAIN_TITLES[page_id])

if page_id == "home":
    render_page_overview(filters, query)
elif page_id == "time":
    render_page_time(filters, query)
elif page_id == "causes":
    render_page_causes(filters, query)
elif page_id == "routes":
    render_page_routes(filters, query)
elif page_id == "predict":
    render_page_predict()
elif page_id == "explorer":
    render_page_explorer()
