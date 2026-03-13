"""
etl/extract.py  —  Memory-efficient streaming reader
=====================================================
KHÔNG load toàn bộ flights.csv vào RAM.
Trả về generator chunk — pipeline xử lý và xả từng chunk.
"""

import pandas as pd
import os
import logging

log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# dtype tường minh: float32 thay float64, int8/16 thay int64 → tiết kiệm ~60% RAM/chunk
FLIGHTS_DTYPE = {
    "YEAR":                "int16",
    "MONTH":               "int8",
    "DAY":                 "int8",
    "AIRLINE":             "str",
    "FLIGHT_NUMBER":       "str",
    "TAIL_NUMBER":         "str",
    "ORIGIN_AIRPORT":      "str",
    "DESTINATION_AIRPORT": "str",
    "SCHEDULED_DEPARTURE": "float32",
    "DEPARTURE_TIME":      "float32",
    "DEPARTURE_DELAY":     "float32",
    "SCHEDULED_ARRIVAL":   "float32",
    "ARRIVAL_TIME":        "float32",
    "ARRIVAL_DELAY":       "float32",
    "ELAPSED_TIME":        "float32",
    "AIR_TIME":            "float32",
    "DISTANCE":            "float32",
    "AIR_SYSTEM_DELAY":    "float32",
    "SECURITY_DELAY":      "float32",
    "AIRLINE_DELAY":       "float32",
    "LATE_AIRCRAFT_DELAY": "float32",
    "WEATHER_DELAY":       "float32",
    "CANCELLED":           "float32",   # float vì có NaN trước khi fillna
    "DIVERTED":            "float32",
    "CANCELLATION_REASON": "str",
}

USECOLS = list(FLIGHTS_DTYPE.keys())


def load_airlines(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "airlines.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    log.info(f"airlines: {len(df)} hãng")
    return df


def load_airports(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "airports.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    log.info(f"airports: {len(df)} sân bay")
    return df


def stream_flights(path: str = None, chunksize: int = 100_000):
    """
    Generator — yield từng chunk 100k dòng đã lọc sơ bộ.
    Caller xử lý xong → chunk bị GC giải phóng → RAM luôn thấp.
    """
    path = path or os.path.join(DATA_DIR, "flights.csv")

    # Lấy header để biết cột thực sự có trong file
    header = pd.read_csv(path, nrows=0)
    header.columns = [c.strip().upper() for c in header.columns]
    usecols = [c for c in USECOLS if c in header.columns]
    dtype   = {k: v for k, v in FLIGHTS_DTYPE.items() if k in header.columns}

    log.info(f"Stream {path} | chunk={chunksize:,}")

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
        na_values=["", "NA", "N/A"],
        keep_default_na=True,
        low_memory=False,
    ):
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        chunk = chunk.dropna(subset=["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
        chunk = chunk[chunk["MONTH"].between(1, 12)]
        chunk = chunk[chunk["DAY"].between(1, 31)]
        if len(chunk) > 0:
            yield chunk


def scan_dim_date(path: str = None, chunksize: int = 200_000) -> pd.DataFrame:
    """Quét file 1 lần chỉ để thu thập các ngày duy nhất → build dim_date."""
    path = path or os.path.join(DATA_DIR, "flights.csv")
    header = pd.read_csv(path, nrows=0)
    header.columns = [c.strip().upper() for c in header.columns]
    date_cols = [c for c in ["YEAR", "MONTH", "DAY"] if c in header.columns]
    dtype_d   = {"YEAR": "int16", "MONTH": "int8", "DAY": "int8"}

    unique_dates = set()
    for chunk in pd.read_csv(path, usecols=date_cols, dtype=dtype_d,
                             chunksize=chunksize, low_memory=False):
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        chunk = chunk.dropna()
        chunk = chunk[chunk["MONTH"].between(1, 12)]
        chunk = chunk[chunk["DAY"].between(1, 31)]
        for row in chunk.drop_duplicates().itertuples(index=False):
            unique_dates.add((int(row.YEAR), int(row.MONTH), int(row.DAY)))

    df = pd.DataFrame(list(unique_dates), columns=["YEAR", "MONTH", "DAY"])
    log.info(f"scan_dim_date: {len(df)} ngày duy nhất")
    return df
