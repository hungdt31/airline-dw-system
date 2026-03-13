"""
etl/transform.py  —  Vectorized transforms (không dùng df.apply row-by-row)
============================================================================
Mọi phép tính đều dùng numpy/pandas vectorized → nhanh gấp 50-100x so với apply().
"""

import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DIMENSION TRANSFORMS  (chạy 1 lần, data nhỏ)
# ─────────────────────────────────────────────────────────────

def transform_airlines(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["IATA_CODE", "AIRLINE"]].copy()
    df.columns = ["airline_iata", "airline_name"]
    df = df.drop_duplicates("airline_iata").dropna()
    df["airline_name"] = df["airline_name"].str.strip()
    log.info(f"dim_airline: {len(df)} bản ghi")
    return df


def transform_airports(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["IATA_CODE", "AIRPORT", "CITY", "STATE", "COUNTRY", "LATITUDE", "LONGITUDE"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    rename = {
        "IATA_CODE": "airport_iata", "AIRPORT": "airport_name",
        "CITY": "city", "STATE": "state",
        "COUNTRY": "country", "LATITUDE": "latitude", "LONGITUDE": "longitude"
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df.drop_duplicates("airport_iata").dropna(subset=["airport_iata"])
    if "country" not in df.columns:
        df["country"] = "USA"
    log.info(f"dim_airport: {len(df)} bản ghi")
    return df


def build_dim_date(df_dates: pd.DataFrame) -> pd.DataFrame:
    """
    df_dates: DataFrame có cột YEAR, MONTH, DAY (từ scan_dim_date).
    Dùng pd.to_datetime vectorized thay vì loop iterrows.
    """
    df = df_dates[["YEAR", "MONTH", "DAY"]].drop_duplicates().dropna().astype(int)
    dt = pd.to_datetime(df.rename(columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}))

    result = pd.DataFrame({
        "date_key":     dt.dt.strftime("%Y%m%d").astype(int),
        "full_date":    dt.dt.strftime("%Y-%m-%d"),
        "year":         dt.dt.year,
        "quarter":      dt.dt.quarter,
        "month":        dt.dt.month,
        "month_name":   dt.dt.strftime("%B"),
        "week_of_year": dt.dt.isocalendar().week.astype(int),
        "day_of_month": dt.dt.day,
        "day_of_week":  dt.dt.dayofweek + 1,
        "day_name":     dt.dt.strftime("%A"),
        "is_weekend":   (dt.dt.dayofweek >= 5).astype(int),
    }).drop_duplicates("date_key").reset_index(drop=True)

    log.info(f"dim_date: {len(result)} ngày")
    return result


# ─────────────────────────────────────────────────────────────
# FACT TRANSFORM  (chạy per-chunk, PHẢI nhanh)
# ─────────────────────────────────────────────────────────────

def transform_chunk(
    chunk: pd.DataFrame,
    airline_map: dict,
    airport_map: dict,
    date_map: dict,
    delay_reason_map: dict,
) -> pd.DataFrame:
    """
    Transform 1 chunk flights thô → fact_flights rows.
    Toàn bộ vectorized — KHÔNG dùng apply/iterrows.
    """
    df = chunk.copy()

    # ── Điền NaN delay bằng 0 ──────────────────────────────
    delay_cols = ["DEPARTURE_DELAY", "ARRIVAL_DELAY",
                  "AIR_SYSTEM_DELAY", "SECURITY_DELAY",
                  "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]
    for c in delay_cols:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # ── Flags vectorized ────────────────────────────────────
    df["CANCELLED"]     = pd.to_numeric(df.get("CANCELLED", 0), errors="coerce").fillna(0).astype(np.int8)
    df["DIVERTED"]      = pd.to_numeric(df.get("DIVERTED",  0), errors="coerce").fillna(0).astype(np.int8)
    df["IS_DELAYED_15"] = (df["ARRIVAL_DELAY"] >= 15).astype(np.int8)

    # ── Date key (vectorized string concat) ─────────────────
    year_s  = df["YEAR"].astype(int).astype(str).str.zfill(4)
    month_s = df["MONTH"].astype(int).astype(str).str.zfill(2)
    day_s   = df["DAY"].astype(int).astype(str).str.zfill(2)
    df["date_key_raw"] = year_s + month_s + day_s

    # ── Delay reason vectorized (np.select, không apply) ────
    conditions = [
        df["WEATHER_DELAY"]       > 0,
        df["AIR_SYSTEM_DELAY"]    > 0,
        df["LATE_AIRCRAFT_DELAY"] > 0,
        df["AIRLINE_DELAY"]       > 0,
        df["SECURITY_DELAY"]      > 0,
    ]
    choices = [
        delay_reason_map.get("WEATHER"),
        delay_reason_map.get("AIR_SYSTEM"),
        delay_reason_map.get("LATE_AIRCRAFT"),
        delay_reason_map.get("AIRLINE"),
        delay_reason_map.get("SECURITY"),
    ]
    df["delay_reason_key"] = np.select(
        conditions, choices, default=delay_reason_map.get("NO_DELAY")
    )

    # ── Map sang surrogate keys ──────────────────────────────
    fact = pd.DataFrame({
        "airline_key":         df["AIRLINE"].map(airline_map),
        "origin_airport_key":  df["ORIGIN_AIRPORT"].map(airport_map),
        "dest_airport_key":    df["DESTINATION_AIRPORT"].map(airport_map),
        "date_key":            df["date_key_raw"].map(date_map),
        "delay_reason_key":    df["delay_reason_key"],
        "flight_number":       df.get("FLIGHT_NUMBER"),
        "tail_number":         df.get("TAIL_NUMBER"),
        "scheduled_departure": df.get("SCHEDULED_DEPARTURE"),
        "actual_departure":    df.get("DEPARTURE_TIME"),
        "departure_delay":     df["DEPARTURE_DELAY"],
        "scheduled_arrival":   df.get("SCHEDULED_ARRIVAL"),
        "actual_arrival":      df.get("ARRIVAL_TIME"),
        "arrival_delay":       df["ARRIVAL_DELAY"],
        "elapsed_time":        df.get("ELAPSED_TIME"),
        "air_time":            df.get("AIR_TIME"),
        "distance":            df.get("DISTANCE"),
        "delay_air_system":    df["AIR_SYSTEM_DELAY"],
        "delay_security":      df["SECURITY_DELAY"],
        "delay_airline":       df["AIRLINE_DELAY"],
        "delay_late_aircraft": df["LATE_AIRCRAFT_DELAY"],
        "delay_weather":       df["WEATHER_DELAY"],
        "is_cancelled":        df["CANCELLED"],
        "cancellation_reason": df.get("CANCELLATION_REASON"),
        "is_diverted":         df["DIVERTED"],
        "is_delayed_15":       df["IS_DELAYED_15"],
    })

    # Bỏ dòng không map được foreign key
    fact = fact.dropna(subset=["airline_key", "origin_airport_key",
                                "dest_airport_key", "date_key"])
    return fact
