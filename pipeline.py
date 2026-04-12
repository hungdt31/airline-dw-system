"""
pipeline.py  —  True streaming ETL
====================================
Kiến trúc: đọc chunk → transform → INSERT → xả RAM → chunk tiếp theo.
RAM sử dụng tối đa ≈ 1 chunk (~100k dòng × ~30 cột × 4 bytes) ≈ 120 MB.
Tổng thời gian ước tính với 5.8M dòng: 3–8 phút tùy ổ đĩa.
"""

import logging
import sys
import os
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from etl.extract import load_airlines, load_airports, stream_flights, scan_dim_date
from etl.transform import (
    transform_airlines, transform_airports,
    build_dim_date, transform_chunk,
)
from etl.load import (
    get_connection, init_schema,
    load_dim_airlines, load_dim_airports, load_dim_dates,
    get_delay_reason_map, load_fact_flights, print_summary,
)
from etl.airport_id_lookup import build_faa_id_to_iata


# ── Tuning ────────────────────────────────────────────────────────
CHUNK_SIZE   = 100_000   # dòng mỗi lần đọc (~120 MB RAM/chunk)
INSERT_BATCH = 50_000    # dòng mỗi lần INSERT vào SQLite
# Giảm CHUNK_SIZE xuống 50_000 nếu máy vẫn lag
# ─────────────────────────────────────────────────────────────────


def run_pipeline():
    t0 = time.time()
    log.info("=" * 55)
    log.info("  AIRLINE DATA WAREHOUSE — ETL PIPELINE (STREAMING)")
    log.info("=" * 55)

    # ── [1] Dimensions nhỏ — load bình thường ────────────────
    log.info("\n[1/4] Đọc & transform dimension tables")
    dim_airline = transform_airlines(load_airlines())
    dim_airport = transform_airports(load_airports())

    # Quét file 1 lần nhanh chỉ để lấy ngày duy nhất
    log.info("  Scanning dates từ flights.csv ...")
    raw_dates = scan_dim_date(chunksize=300_000)
    dim_date  = build_dim_date(raw_dates)
    del raw_dates

    # ── [2] Init schema ──────────────────────────────────────
    log.info("\n[2/4] Khởi tạo schema Data Warehouse")
    conn = get_connection()
    init_schema(conn)

    # ── [3] Load dimensions + lấy lookup maps ────────────────
    log.info("\n[3/4] Nạp Dimension tables")
    airline_map      = load_dim_airlines(conn, dim_airline)
    airport_map      = load_dim_airports(conn, dim_airport)
    date_map         = load_dim_dates(conn, dim_date)
    delay_reason_map = get_delay_reason_map(conn)

    del dim_airline, dim_airport, dim_date   # giải phóng RAM

    # Map Airport ID (số) → IATA nhờ L_AIRPORT_ID.xls (CSV)
    l_airport_file = _REPO_ROOT / "data" / "L_AIRPORT_ID.xls"
    airports_csv = _REPO_ROOT / "data" / "airports.csv"
    faa_id_to_iata = build_faa_id_to_iata(l_airport_file, airports_csv)

    # ── [4] Stream flights: read chunk → transform → insert ──
    log.info(f"\n[4/4] Stream + transform + load fact_flights (chunk={CHUNK_SIZE:,})")

    total_in  = 0
    total_out = 0
    chunk_no  = 0

    for chunk in stream_flights(chunksize=CHUNK_SIZE):
        chunk_no  += 1
        total_in  += len(chunk)
        t_chunk    = time.time()

        fact = transform_chunk(
            chunk,
            airline_map,
            airport_map,
            date_map,
            delay_reason_map,
            faa_id_to_iata=faa_id_to_iata,
        )
        del chunk   # xả ngay, không cần nữa

        load_fact_flights(conn, fact, chunksize=INSERT_BATCH)
        total_out += len(fact)
        del fact

        elapsed = time.time() - t_chunk
        log.info(
            f"  Chunk {chunk_no:>3} | {total_in:>9,} dòng đọc "
            f"| {total_out:>9,} dòng inserted "
            f"| {elapsed:.1f}s"
        )

    # ── Done ─────────────────────────────────────────────────
    print_summary(conn)
    conn.close()

    total_time = time.time() - t0
    log.info(f"\nHoàn thành trong {total_time/60:.1f} phút")
    log.info(f"DB: warehouse/airline_dw.db")


if __name__ == "__main__":
    run_pipeline()
