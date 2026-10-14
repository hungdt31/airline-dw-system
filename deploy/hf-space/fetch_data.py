"""
Tải warehouse DB + model từ HF dataset về đúng đường dẫn dashboard đang đọc:
    warehouse/airline_dw.db   (dashboard/config.py → DB_PATH)
    mining/model_delay.pkl    (mining/predict_delay.py → MODEL_PATH)
"""

import os

from huggingface_hub import hf_hub_download

REPO = os.environ.get("DATA_REPO", "kkoi31/airline-dw-data")

for name, dest in [("airline_dw.db", "warehouse"), ("model_delay.pkl", "mining")]:
    path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=dest)
    print(f"{name} → {path} ({os.path.getsize(path):,} bytes)")
