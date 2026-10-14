FROM python:3.12-slim

# xgboost cần OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces chạy container với uid 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Tải DB + model lúc build (dataset public) → ăn vào image, restart không phải tải lại
COPY --chown=user fetch_data.py .
RUN python fetch_data.py

COPY --chown=user dashboard/ dashboard/
COPY --chown=user mining/predict_delay.py mining/

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
