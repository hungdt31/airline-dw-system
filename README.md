# Airline Data Warehouse

Hệ thống Data Warehouse (star schema) cho dữ liệu chuyến bay — ETL streaming, truy vấn OLAP, dashboard Streamlit, pipeline khai thác dữ liệu và tài liệu báo cáo/slide.

## Cấu trúc thư mục

```
airline_dw/
├── data/                      # Dữ liệu nguồn (CSV, lookup) + output EDA
│   ├── flights.csv            # (tải từ Kaggle, đặt vào đây)
│   ├── airlines.csv
│   ├── airports.csv
│   ├── L_AIRPORT_ID.xls       # Map FAA ID → IATA (ETL dùng nếu có)
│   └── eda_output/              # Báo cáo/hình EDA (sau khi chạy script)
├── etl/
│   ├── extract.py               # Đọc streaming chunk flights
│   ├── transform.py             # Chuẩn hóa, KPI
│   ├── load.py                  # Nạp SQLite DW
│   └── airport_id_lookup.py
├── warehouse/
│   ├── airline_dw.db            # SQLite sau ETL (tạo khi chạy pipeline)
│   └── schema.sql               # DDL star schema
├── queries/
│   └── olap_queries.sql         # Truy vấn OLAP mẫu
├── dashboard/                   # Streamlit đa trang (OLAP + dự đoán)
│   ├── app.py
│   ├── db.py
│   └── page_*.py
├── mining/                      # ML: huấn luyện, so sánh model, báo cáo HTML
│   ├── predict_delay.py         # Huấn luyện XGBoost → model_delay.pkl
│   ├── compare_5_models.py      # So 5 thuật toán → five_models_report.html
│   ├── compare_models.py, compare_all_models.py, compare_lr_rf.py
│   ├── run_models.py, predict_cli.py, analyze_imbalance.py
│   └── model_*.py
├── scripts/                     # Tiện ích: EDA trước ETL, export, kiểm tra
│   ├── eda_before_etl.py
│   ├── export_flights_by_month.py
│   └── ...
├── slide/                       # Beamer — thuyết trình
├── report/                      # LaTeX — báo cáo dài
├── pipeline.py                  # ETL toàn bộ (ghi log pipeline.log)
├── app.py                       # Bản sao entry gốc (khuyến nghị dùng dashboard/app.py)
├── requirements.txt
└── README.md
```

## Chuẩn bị dữ liệu

1. Tải bộ [US Flight Delays (Kaggle)](https://www.kaggle.com/datasets/usdot/flight-delays?select=airlines.csv).
2. Đặt `flights.csv`, `airlines.csv`, `airports.csv` vào thư mục `data/`.
3. (Khuyến nghị) Thêm `L_AIRPORT_ID.xls` vào `data/` nếu pipeline cần map mã sân bay theo FAA ID.

## Cài đặt

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

pip install -r requirements.txt
```

**Khai thác dữ liệu (XGBoost / so sánh 5 model):** các script trong `mining/` dùng XGBoost; `compare_5_models.py` còn cần LightGBM.

```bash
pip install xgboost lightgbm
```

(Nếu đọc file `.xls` gặp lỗi, có thể cần thêm `xlrd` tùy phiên bản pandas.)

## Chạy ETL

Tạo hoặc ghi đè `warehouse/airline_dw.db`:

```bash
python pipeline.py
```

Log chi tiết: `pipeline.log`.

## Dashboard (Streamlit)

```bash
streamlit run dashboard/app.py
```

Ứng dụng đọc `warehouse/airline_dw.db` và (nếu có) `mining/model_delay.pkl` cho trang dự đoán.

## Khai thác dữ liệu (mining)

| Mục đích | Lệnh | Ghi chú |
|----------|------|---------|
| Huấn luyện XGBoost, lưu model | `python mining/predict_delay.py` | Cần DW đã ETL |
| So sánh 5 thuật toán + HTML | `python mining/compare_5_models.py` | Cần `xgboost`, `lightgbm` |
| Các thử nghiệm / so sánh khác | `python mining/compare_models.py` (và các file tương ứng) | Xem docstring từng file |

Output tiêu biểu: `mining/five_models_report.html`, `mining/model_delay.pkl`.

## EDA và script phụ

- EDA trước ETL (HTML/figures): `python scripts/eda_before_etl.py` (output thường nằm trong `data/eda_output/`).
- Các script khác trong `scripts/`: xem mô tả trong từng file.

## Tài liệu (LaTeX)

- Slide: biên dịch `slide/main.tex` (Beamer).
- Báo cáo: `report/main.tex`; yêu cầu môi trường LaTeX theo `report/REQUIRE.md` nếu có.

## Dataset

[Kaggle — Flight Delays](https://www.kaggle.com/datasets/usdot/flight-delays?select=airlines.csv)
