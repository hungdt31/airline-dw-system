# ✈️ Airline Data Warehouse System

Hệ thống Data Warehouse cho lĩnh vực hàng không — CDIO Project

## 📁 Cấu trúc thư mục

```
airline_dw/
├── data/                   # Đặt file CSV từ Kaggle vào đây
│   ├── airports.csv
│   ├── airlines.csv
│   └── flights.csv
├── etl/
│   ├── extract.py          # Đọc & validate dữ liệu nguồn
│   ├── transform.py        # Làm sạch, chuẩn hóa, tính KPI
│   └── load.py             # Nạp vào Data Warehouse
├── warehouse/
│   └── schema.sql          # DDL tạo Star Schema
├── queries/
│   └── olap_queries.sql    # Toàn bộ câu truy vấn OLAP
├── dashboard/
│   └── app.py              # Streamlit Dashboard
├── mining/
│   └── predict_delay.py    # Data Mining — dự đoán delay
├── notebooks/
│   └── exploration.ipynb   # EDA notebook
├── pipeline.py             # Chạy toàn bộ ETL pipeline
└── requirements.txt
```

## 🚀 Cài đặt & Chạy

```bash
# 0. Tạo môi trường ảo
python -m venv venv

# 1. Cài thư viện
pip install -r requirements.txt

# 2. Đặt file CSV vào thư mục data/

# 3. Chạy ETL pipeline
python pipeline.py

# 4. Mở Dashboard
streamlit run dashboard/app.py

# 5. Chạy Data Mining
python mining/predict_delay.py
```

## 📊 Dataset
Kaggle: https://www.kaggle.com/datasets/usdot/flight-delays?select=airlines.csv
