"""
Xem thống kê outlier các cột số trong fact_flights
Chạy: python scripts/check_outliers.py
"""
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "warehouse", "airline_dw.db")

# Cột cần kiểm tra
COLS = [
    "arrival_delay",
    "departure_delay",
    "elapsed_time",
    "distance",
    "air_time",
    "delay_air_system",
    "delay_airline",
    "delay_late_aircraft",
    "delay_weather",
    "delay_security",
]

print("Loading 500,000 mẫu ngẫu nhiên...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(f"""
    SELECT {", ".join(COLS)}
    FROM fact_flights
    WHERE is_cancelled = 0
    ORDER BY RANDOM()
    LIMIT 500000
""", conn)
conn.close()
print(f"Loaded: {len(df):,} rows\n")

rows = []
for col in COLS:
    s = df[col].dropna()
    if len(s) == 0:
        continue
    Q1  = s.quantile(0.25)
    Q3  = s.quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - 1.5 * IQR
    hi  = Q3 + 1.5 * IQR
    n_hi  = int((s > hi).sum())
    n_lo  = int((s < lo).sum())
    n_neg = int((s < 0).sum())
    rows.append({
        "Cột"          : col,
        "Min"          : round(s.min(), 1),
        "Q1"           : round(Q1, 1),
        "Median"       : round(s.median(), 1),
        "Mean"         : round(s.mean(), 1),
        "Q3"           : round(Q3, 1),
        "Max"          : round(s.max(), 1),
        "Upper fence"  : round(hi, 1),
        "Lower fence"  : round(lo, 1),
        "Outlier trên" : f"{n_hi:,} ({100*n_hi/len(s):.2f}%)",
        "Outlier dưới" : f"{n_lo:,} ({100*n_lo/len(s):.2f}%)",
        "Giá trị âm"   : f"{n_neg:,} ({100*n_neg/len(s):.2f}%)",
        "Tỷ lệ thiếu"  : f"{100*df[col].isna().mean():.2f}%",
    })

result = pd.DataFrame(rows).set_index("Cột")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.1f}")
print(result.to_string())