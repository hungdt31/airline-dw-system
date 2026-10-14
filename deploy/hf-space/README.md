---
title: Airline DW Analytics
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
short_description: OLAP dashboard + delay prediction for US Flight Delays 2015
---

# Airline Data Warehouse — Dashboard

Dashboard Streamlit cho Data Warehouse (star schema, SQLite) xây từ bộ
[Kaggle US Flight Delays 2015](https://www.kaggle.com/datasets/usdot/flight-delays):
phân tích OLAP theo thời gian, tuyến đường, nguyên nhân delay và dự báo delay ≥15 phút bằng XGBoost.

- Dữ liệu (DB + model): [kkoi31/airline-dw-data](https://huggingface.co/datasets/kkoi31/airline-dw-data)
- Mã nguồn: [hungdt31/airline-dw-system](https://github.com/hungdt31/airline-dw-system)
