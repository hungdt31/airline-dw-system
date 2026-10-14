"""
Tải DB + model từ HF dataset (DATA_REPO) nếu chưa có — dùng khi deploy cloud.
Chạy local sau `python pipeline.py` / `python mining/predict_delay.py` thì file đã có → bỏ qua.
"""

import os

import streamlit as st

from config import DATA_REPO, DB_PATH, MODEL_PATH
from icons import ERR


@st.cache_resource(show_spinner=False)
def ensure_remote_data() -> None:
    missing = [
        (name, path)
        for name, path in [("airline_dw.db", DB_PATH), ("model_delay.pkl", MODEL_PATH)]
        if not os.path.exists(path)
    ]
    if not missing:
        return

    from huggingface_hub import hf_hub_download

    with st.spinner(f"Đang tải dữ liệu từ Hugging Face ({DATA_REPO}) — lần đầu mất khoảng 1 phút..."):
        for name, path in missing:
            try:
                hf_hub_download(
                    DATA_REPO, name, repo_type="dataset", local_dir=os.path.dirname(path)
                )
            except Exception as e:  # mất mạng / repo sai → để db.get_conn báo lỗi thiếu DB
                st.error(f"{ERR} Không tải được `{name}` từ `{DATA_REPO}`: {e}")
