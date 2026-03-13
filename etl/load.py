"""
etl/load.py
===========
Nạp dữ liệu đã transform vào SQLite Data Warehouse.
"""

import sqlite3
import pandas as pd
import os
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "warehouse", "airline_dw.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "warehouse", "schema.sql")


def get_connection(db_path: str = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_schema(conn: sqlite3.Connection):
    """Tạo toàn bộ schema từ file SQL."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        sql = f.read()
    conn.executescript(sql)
    conn.commit()
    log.info("Schema khởi tạo thành công")


def load_dim_airlines(conn: sqlite3.Connection, df: pd.DataFrame) -> dict:
    df.to_sql("dim_airline", conn, if_exists="append", index=False)
    conn.commit()
    rows = conn.execute("SELECT airline_iata, airline_key FROM dim_airline").fetchall()
    log.info(f"dim_airline: nạp {len(df)} bản ghi")
    return {row[0]: row[1] for row in rows}


def load_dim_airports(conn: sqlite3.Connection, df: pd.DataFrame) -> dict:
    df.to_sql("dim_airport", conn, if_exists="append", index=False)
    conn.commit()
    rows = conn.execute("SELECT airport_iata, airport_key FROM dim_airport").fetchall()
    log.info(f"dim_airport: nạp {len(df)} bản ghi")
    return {row[0]: row[1] for row in rows}


def load_dim_dates(conn: sqlite3.Connection, df: pd.DataFrame) -> dict:
    df.to_sql("dim_date", conn, if_exists="append", index=False)
    conn.commit()
    rows = conn.execute("SELECT date_key, date_key FROM dim_date").fetchall()
    log.info(f"dim_date: nạp {len(df)} bản ghi")
    # Map from string key (YYYYMMDD) to integer date_key
    rows2 = conn.execute("SELECT CAST(date_key AS TEXT), date_key FROM dim_date").fetchall()
    return {row[0]: row[1] for row in rows2}


def get_delay_reason_map(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT reason_code, reason_key FROM dim_delay_reason").fetchall()
    return {row[0]: row[1] for row in rows}


def load_fact_flights(conn: sqlite3.Connection, df: pd.DataFrame, chunksize: int = 50_000):
    """Nạp fact_flights theo từng batch."""
    total = len(df)
    loaded = 0
    for i in range(0, total, chunksize):
        chunk = df.iloc[i:i+chunksize]
        chunk.to_sql("fact_flights", conn, if_exists="append", index=False)
        loaded += len(chunk)
        log.info(f"  Đã nạp {loaded:,}/{total:,} chuyến bay...")
    conn.commit()
    log.info(f"fact_flights: nạp xong {total:,} bản ghi")


def get_row_count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def print_summary(conn: sqlite3.Connection):
    tables = ["dim_airline", "dim_airport", "dim_date", "dim_delay_reason", "fact_flights"]
    print("\n" + "="*45)
    print("  DATA WAREHOUSE — TÓM TẮT")
    print("="*45)
    for t in tables:
        try:
            n = get_row_count(conn, t)
            print(f"  {t:<25} {n:>10,} bản ghi")
        except Exception:
            pass
    print("="*45 + "\n")
