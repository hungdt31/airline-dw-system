"""
scripts/eda_before_etl.py  —  EDA trước ETL (hỗ trợ quyết định làm sạch dữ liệu)
==================================================================================
Chạy từ thư mục gốc repo:
    python scripts/eda_before_etl.py
    python scripts/eda_before_etl.py --nrows 500000
    python scripts/eda_before_etl.py --nrows 0          # toàn bộ file (chậm)

Kết quả (tất cả trong data/eda_output/):
    eda_before_etl_report.html   — báo cáo tổng hợp HTML
    eda_data_dictionary.csv      — bảng mô tả cột tổng hợp (all files)
    eda_data_dictionary_airlines.csv
    eda_data_dictionary_airports.csv
    eda_data_dictionary_flights_sample.csv
    eda_data_dictionary_l_airport_id.csv
    eda_missing_all_cols.csv     — missing rate từng cột
    eda_delay_null_vs_ontime.csv — kiểm chứng all-null delay components vs đúng giờ
    figures/eda_delay_null_vs_ontime.png
    eda_numeric_stats.csv        — thống kê mô tả + outlier delay
    eda_categorical_stats.csv    — cardinality + top values
    eda_join_quality.csv         — join check flights↔airlines/airports
    figures/eda_missing_bar.png
    figures/eda_delay_dist.png
    figures/eda_distance_dist.png
    figures/eda_dep_hour_dist.png
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# Cho phép chạy trực tiếp hoặc qua python -m
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = _REPO_ROOT / "data"
OUT_DIR    = DATA_DIR / "eda_output"
FIG_DIR    = OUT_DIR / "figures"

CSV_FLIGHTS  = DATA_DIR / "flights.csv"
CSV_AIRLINES = DATA_DIR / "airlines.csv"
CSV_AIRPORTS = DATA_DIR / "airports.csv"
XLS_AIRPORT_ID = DATA_DIR / "L_AIRPORT_ID.xls"

# Mô tả cột (dịch tiếng Việt) cho Data Dictionary
FLIGHTS_COLUMN_DESCRIPTIONS_VI = {
    "YEAR": "Năm của chuyến bay.",
    "MONTH": "Tháng của chuyến bay.",
    "DAY": "Ngày trong tháng của chuyến bay.",
    "DAY_OF_WEEK": "Thứ trong tuần của chuyến bay.",
    "AIRLINE": "Mã định danh hãng hàng không.",
    "FLIGHT_NUMBER": "Mã định danh chuyến bay.",
    "TAIL_NUMBER": "Mã định danh tàu bay.",
    "ORIGIN_AIRPORT": "Sân bay khởi hành.",
    "DESTINATION_AIRPORT": "Sân bay đích.",
    "SCHEDULED_DEPARTURE": "Giờ khởi hành theo lịch.",
    "DEPARTURE_TIME": "Giờ khởi hành thực tế (xấp xỉ WHEELS_OFF - TAXI_OUT).",
    "DEPARTURE_DELAY": "Tổng số phút trễ khởi hành.",
    "TAXI_OUT": "Thời gian lăn bánh từ cổng ra đường băng trước khi cất cánh.",
    "WHEELS_OFF": "Thời điểm bánh máy bay rời mặt đất.",
    "SCHEDULED_TIME": "Tổng thời gian bay theo lịch.",
    "ELAPSED_TIME": "Tổng thời gian thực tế của chuyến bay (AIR_TIME + TAXI_IN + TAXI_OUT).",
    "AIR_TIME": "Thời gian trên không (từ WHEELS_OFF đến WHEELS_ON).",
    "DISTANCE": "Khoảng cách giữa hai sân bay.",
    "WHEELS_ON": "Thời điểm bánh máy bay chạm đất.",
    "TAXI_IN": "Thời gian lăn từ đường băng vào cổng tại sân bay đích.",
    "SCHEDULED_ARRIVAL": "Giờ đến theo lịch.",
    "ARRIVAL_TIME": "Giờ đến thực tế (xấp xỉ WHEELS_ON + TAXI_IN).",
    "ARRIVAL_DELAY": "Độ trễ đến (ARRIVAL_TIME - SCHEDULED_ARRIVAL).",
    "DIVERTED": "Chuyến bay hạ cánh lệch sân bay đích theo lịch.",
    "CANCELLED": "Chuyến bay bị hủy (1 = hủy).",
    "CANCELLATION_REASON": "Lý do hủy chuyến: A - Hãng; B - Thời tiết; C - Hệ thống không lưu; D - An ninh.",
    "AIR_SYSTEM_DELAY": "Số phút trễ do hệ thống không lưu.",
    "SECURITY_DELAY": "Số phút trễ do an ninh.",
    "AIRLINE_DELAY": "Số phút trễ do hãng hàng không.",
    "LATE_AIRCRAFT_DELAY": "Số phút trễ do máy bay đến muộn từ chuyến trước.",
    "WEATHER_DELAY": "Số phút trễ do thời tiết.",
}

AIRLINES_COLUMN_DESCRIPTIONS_VI = {
    "IATA_CODE": "Mã định danh hãng hàng không (IATA).",
    "AIRLINE": "Tên hãng hàng không.",
}

AIRPORTS_COLUMN_DESCRIPTIONS_VI = {
    "IATA_CODE": "Mã định danh địa điểm/sân bay (IATA).",
    "AIRPORT": "Tên sân bay.",
    "CITY": "Thành phố của sân bay.",
    "STATE": "Bang/Tiểu bang của sân bay.",
    "COUNTRY": "Tên quốc gia của sân bay.",
    "LATITUDE": "Vĩ độ của sân bay.",
    "LONGITUDE": "Kinh độ của sân bay.",
}

L_AIRPORT_ID_COLUMN_DESCRIPTIONS_VI = {
    "CODE": "Mã định danh sân bay (Airport ID).",
    "DESCRIPTION": "Mô tả tên sân bay tương ứng với mã.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Cột cần phân tích
# ─────────────────────────────────────────────────────────────────────────────
DELAY_COLS = [
    "DEPARTURE_DELAY", "ARRIVAL_DELAY",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY",
    "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
]
COMPONENT_DELAY_COLS = [
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY",
    "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
]
NUMERIC_COLS = [
    "DEPARTURE_DELAY", "ARRIVAL_DELAY", "ELAPSED_TIME", "AIR_TIME",
    "DISTANCE", "SCHEDULED_DEPARTURE", "DEPARTURE_TIME",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
]
CATEGORICAL_COLS = [
    "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "CANCELLATION_REASON", "MONTH", "DAY_OF_WEEK",
]
# Cột cần đọc từ flights.csv (loại bớt cột không cần thiết cho EDA)
USECOLS_EDA = [
    "YEAR", "MONTH", "DAY", "DAY_OF_WEEK",
    "AIRLINE", "FLIGHT_NUMBER",
    "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "DEPARTURE_DELAY",
    "SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "ARRIVAL_DELAY",
    "ELAPSED_TIME", "AIR_TIME", "DISTANCE",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
    "CANCELLED", "DIVERTED", "CANCELLATION_REASON",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(n: float | int, decimals: int = 2) -> str:
    """Định dạng số có dấu phân cách hàng nghìn."""
    if pd.isna(n):
        return "—"
    if isinstance(n, float):
        return f"{n:,.{decimals}f}"
    return f"{int(n):,}"


def _section(title: str) -> str:
    bar = "=" * 70
    return f"\n{bar}\n  {title}\n{bar}"


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Đọc dữ liệu (streaming chunk cho flights.csv)
# ─────────────────────────────────────────────────────────────────────────────

def read_small_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc toàn bộ airlines.csv và airports.csv vào RAM."""
    df_al = pd.read_csv(CSV_AIRLINES)
    df_al.columns = [c.strip().upper() for c in df_al.columns]

    df_ap = pd.read_csv(CSV_AIRPORTS)
    df_ap.columns = [c.strip().upper() for c in df_ap.columns]

    print(f"  airlines.csv : {len(df_al):,} hãng")
    print(f"  airports.csv : {len(df_ap):,} sân bay")
    return df_al, df_ap


def read_airport_id_xls() -> pd.DataFrame:
    """Đọc file tra cứu L_AIRPORT_ID.xls (nếu có)."""
    if not XLS_AIRPORT_ID.is_file():
        print("  L_AIRPORT_ID.xls: không tìm thấy, bỏ qua mô tả cột file này")
        return pd.DataFrame()
    read_errors: list[str] = []
    for engine in [None, "xlrd", "openpyxl"]:
        try:
            kwargs = {} if engine is None else {"engine": engine}
            df_apid = pd.read_excel(XLS_AIRPORT_ID, **kwargs)
            df_apid.columns = [c.strip().upper() for c in df_apid.columns]
            print(f"  L_AIRPORT_ID.xls: {len(df_apid):,} dòng")
            return df_apid
        except Exception as exc:
            engine_name = "auto" if engine is None else engine
            read_errors.append(f"{engine_name}: {exc}")

    print("  L_AIRPORT_ID.xls: không đọc được bằng pandas.read_excel, bỏ qua file này")
    print("    chi tiết:", " | ".join(read_errors))
    return pd.DataFrame()


def load_flights_sample(nrows: int | None, chunksize: int = 200_000) -> pd.DataFrame:
    """
    Đọc tối đa nrows dòng từ flights.csv theo chunk (tiết kiệm RAM).
    Trả về một DataFrame duy nhất để phân tích.
    """
    header = pd.read_csv(CSV_FLIGHTS, nrows=0)
    header.columns = [c.strip().upper() for c in header.columns]
    usecols = [c for c in USECOLS_EDA if c in header.columns]

    chunks: list[pd.DataFrame] = []
    total_read = 0

    for chunk in pd.read_csv(
        CSV_FLIGHTS,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
        na_values=["", "NA", "N/A"],
    ):
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        chunks.append(chunk)
        total_read += len(chunk)
        print(f"    đã đọc {total_read:,} dòng …", end="\r")
        if nrows and total_read >= nrows:
            break

    df = pd.concat(chunks, ignore_index=True)
    if nrows:
        df = df.iloc[:nrows]
    print(f"\n  flights.csv  : {len(df):,} dòng đã nạp vào bộ nhớ")
    return df


def build_column_dictionary(
    df: pd.DataFrame,
    description_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Tạo bảng mô tả cột cho một nguồn dữ liệu:
    - tên cột, dtype
    - số lượng / tỷ lệ missing
    - số giá trị khác nhau
    - ví dụ giá trị (tối đa 3)
    """
    if df.empty:
        return pd.DataFrame()

    n_rows = len(df)
    records: list[dict] = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        sample_values = non_null.astype(str).unique()[:3]
        sample_text = ", ".join(sample_values) if len(sample_values) else "—"
        n_missing = int(series.isna().sum())
        col_upper = str(col).upper()
        description = (description_map or {}).get(col_upper, "Chưa có mô tả cho cột này.")
        records.append({
            "column": col,
            "description": description,
            "dtype": str(series.dtype),
            "n_rows_profiled": n_rows,
            "n_missing": n_missing,
            "pct_missing": round(100.0 * n_missing / n_rows, 3) if n_rows else 0.0,
            "n_unique_non_null": int(non_null.nunique()),
            "sample_values": sample_text,
        })
    return pd.DataFrame(records)[[
        "column",
        "description",
        "dtype",
        "n_rows_profiled",
        "n_missing",
        "pct_missing",
        "n_unique_non_null",
        "sample_values",
    ]]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Phân tích missing values
# ─────────────────────────────────────────────────────────────────────────────

def analyze_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Trả về DataFrame tóm tắt missing rate toàn bộ cột."""
    n = len(df)
    records = []
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        pct = 100.0 * n_miss / n if n else 0
        dtype_str = str(df[col].dtype)
        # Gợi ý xử lý ETL
        if col in DELAY_COLS:
            action = "fillna(0) — NaN delay = không có trễ loại này"
        elif n_miss == 0:
            action = "OK — không cần xử lý"
        elif pct < 5:
            action = "fillna / dropna — tỷ lệ thấp"
        elif pct > 50:
            action = "CÂN NHẮC DROP cột — quá nhiều missing"
        else:
            action = "Xem xét fillna hoặc drop dòng"
        records.append({
            "column":      col,
            "dtype":       dtype_str,
            "n_missing":   n_miss,
            "pct_missing": round(pct, 2),
            "etl_action":  action,
        })
    return pd.DataFrame(records).sort_values("pct_missing", ascending=False)


def analyze_delay_null_vs_ontime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra giả thuyết:
    Nếu toàn bộ cột delay thành phần đều NaN thì chuyến có đúng giờ hay không.

    Ghi chú:
    - "Đúng giờ" trong hàng không thường dùng ngưỡng ARRIVAL_DELAY <= 15 phút (OTP).
    - Đồng thời xuất thêm nhãn "đúng giờ nghiêm ngặt" ARRIVAL_DELAY <= 0 để tham khảo.
    - Loại chuyến hủy/chuyển hướng khỏi mẫu đánh giá đúng giờ.
    """
    required_cols = set(COMPONENT_DELAY_COLS + ["ARRIVAL_DELAY", "CANCELLED", "DIVERTED"])
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    # Tối ưu RAM: chỉ tạo các Series cần thiết, tránh copy toàn DataFrame.
    cancelled = pd.to_numeric(df["CANCELLED"], errors="coerce").fillna(0)
    diverted = pd.to_numeric(df["DIVERTED"], errors="coerce").fillna(0)
    arrival_delay = pd.to_numeric(df["ARRIVAL_DELAY"], errors="coerce")

    # Chỉ xét chuyến không hủy/chuyển hướng để đánh giá "đúng giờ".
    valid_mask = (cancelled == 0) & (diverted == 0)
    n_total = int(valid_mask.sum())
    if n_total == 0:
        return pd.DataFrame()

    # Hai điều kiện nhóm cần kiểm tra.
    # all-null: tất cả delay component đều NaN
    component_null_count = df[COMPONENT_DELAY_COLS].isna().sum(axis=1)
    group_all_component_null = component_null_count.eq(len(COMPONENT_DELAY_COLS)) & valid_mask

    # zero-or-null: mỗi cột thuộc {0, NaN}
    component_zero_or_null = df[COMPONENT_DELAY_COLS].fillna(0).eq(0).all(axis=1)
    group_all_component_zero_or_null = component_zero_or_null & valid_mask

    # Nhãn đúng giờ.
    on_time_arrival_15 = arrival_delay.le(15) & valid_mask
    on_time_arrival_0 = arrival_delay.le(0) & valid_mask

    records: list[dict] = []
    for group_mask, group_label in [
        (group_all_component_null, "All component delay = NaN"),
        (group_all_component_zero_or_null, "All component delay in {0, NaN}"),
    ]:
        n_group = int(group_mask.sum())
        if n_group == 0:
            continue

        n_on_time_15 = int((on_time_arrival_15 & group_mask).sum())
        n_on_time_0 = int((on_time_arrival_0 & group_mask).sum())
        n_delayed_15 = int(n_group - n_on_time_15)
        n_delayed_0 = int(n_group - n_on_time_0)

        records.append({
            "group": group_label,
            "n_group": n_group,
            "pct_of_valid_flights": round(100.0 * n_group / n_total, 3),
            "n_on_time_arrival_15": n_on_time_15,
            "pct_on_time_arrival_15_in_group": round(100.0 * n_on_time_15 / n_group, 3),
            "n_delayed_arrival_gt15": n_delayed_15,
            "pct_delayed_arrival_gt15_in_group": round(100.0 * n_delayed_15 / n_group, 3),
            "n_on_time_arrival_0": n_on_time_0,
            "pct_on_time_arrival_0_in_group": round(100.0 * n_on_time_0 / n_group, 3),
            "n_delayed_arrival_gt0": n_delayed_0,
            "pct_delayed_arrival_gt0_in_group": round(100.0 * n_delayed_0 / n_group, 3),
        })

    return pd.DataFrame(records)


def plot_delay_null_vs_ontime(df_check: pd.DataFrame, out: Path) -> None:
    """Vẽ biểu đồ % đúng giờ (<=15p, <=0p) theo từng nhóm kiểm chứng."""
    if df_check.empty:
        return

    labels = df_check["group"].tolist()
    otp15 = df_check["pct_on_time_arrival_15_in_group"].tolist()
    otp0 = df_check["pct_on_time_arrival_0_in_group"].tolist()

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars_15 = ax.bar(x - width / 2, otp15, width, label="On-time (ARRIVAL_DELAY <= 15)", color="#1f77b4")
    bars_0 = ax.bar(x + width / 2, otp0, width, label="On-time strict (ARRIVAL_DELAY <= 0)", color="#ff7f0e")

    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_title("Kiểm chứng: Delay-component trống có phải chuyến đúng giờ?")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    for bars in [bars_15, bars_0]:
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_missing_bar(df_miss: pd.DataFrame, out: Path) -> None:
    """Bar chart top cột có missing > 0."""
    miss = df_miss[df_miss["pct_missing"] > 0].sort_values("pct_missing", ascending=True)
    if miss.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(miss))))
    colors = ["#d62728" if p > 50 else "#ff7f0e" if p > 10 else "#1f77b4"
              for p in miss["pct_missing"]]
    ax.barh(miss["column"], miss["pct_missing"], color=colors)
    ax.set_xlabel("% Missing")
    ax.set_title("Tỷ lệ missing từng cột (chỉ hiện cột có missing > 0)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.axvline(5, color="orange", linestyle="--", linewidth=0.8, label="5%")
    ax.axvline(50, color="red", linestyle="--", linewidth=0.8, label="50%")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Phân tích outlier và thống kê mô tả cột số
# ─────────────────────────────────────────────────────────────────────────────

def analyze_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thống kê mô tả + phát hiện outlier đặc thù hàng không:
    - delay âm (đến sớm)  → hợp lệ, GIỮ nguyên
    - delay > 1000 phút   → hiếm, nhưng thật; với mô hình: nhãn nhị phân trung hoà
    - distance, elapsed_time cực nhỏ/âm → cần kiểm tra
    """
    cols_present = [c for c in NUMERIC_COLS if c in df.columns]
    records = []
    for col in cols_present:
        s = pd.to_numeric(df[col], errors="coerce")
        n_valid = int(s.notna().sum())
        if n_valid == 0:
            continue
        v = s.dropna()
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        lo_fence = q1 - 1.5 * iqr
        hi_fence = q3 + 1.5 * iqr

        n_negative  = int((v < 0).sum())
        n_gt1000    = int((v > 1000).sum()) if col in DELAY_COLS else 0
        n_iqr_out   = int(((v < lo_fence) | (v > hi_fence)).sum())

        # Quyết định ETL
        if col in DELAY_COLS:
            decision = "fillna(0); giữ âm (đến sớm); giữ >1000 (thật); nhãn nhị phân trung hoà"
        elif col == "DISTANCE":
            decision = "giữ nguyên; outlier xa theo IQR là chuyến dài thật"
        elif col == "ELAPSED_TIME":
            decision = "fillna(median); loại <0 nếu có"
        else:
            decision = "kiểm tra thêm nếu cần"

        records.append({
            "column":        col,
            "n_valid":       n_valid,
            "mean":          round(float(v.mean()), 2),
            "std":           round(float(v.std()), 2),
            "min":           round(float(v.min()), 2),
            "p25":           round(float(q1), 2),
            "median":        round(float(v.median()), 2),
            "p75":           round(float(q3), 2),
            "max":           round(float(v.max()), 2),
            "n_negative":    n_negative,
            "pct_negative":  round(100.0 * n_negative / n_valid, 2),
            "n_gt1000min":   n_gt1000,
            "pct_gt1000min": round(100.0 * n_gt1000 / n_valid, 2),
            "n_iqr_outlier": n_iqr_out,
            "pct_iqr_out":   round(100.0 * n_iqr_out / n_valid, 2),
            "etl_decision":  decision,
        })
    return pd.DataFrame(records)


def plot_delay_distribution(df: pd.DataFrame, out: Path) -> None:
    """Histogram DEPARTURE_DELAY và ARRIVAL_DELAY cùng trục (clip ±300 phút)."""
    cols = [c for c in ["DEPARTURE_DELAY", "ARRIVAL_DELAY"] if c in df.columns]
    if not cols:
        return
    n_cols = len(cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        clipped = s.clip(-80, 300)
        ax.hist(clipped, bins=80, density=True, color="steelblue",
                alpha=0.85, edgecolor="white", linewidth=0.2)
        ax.axvline(0,   color="green",  linestyle="--", linewidth=1, label="0 phút")
        ax.axvline(15,  color="orange", linestyle="--", linewidth=1, label="15 phút")
        ax.axvline(1000, color="red", linestyle=":", linewidth=0.8, label=">1000 clip")
        n_neg = int((s < 0).sum())
        n_gt1k = int((s > 1000).sum())
        ax.set_title(
            f"{col}\nâm={_fmt(n_neg)} ({100*n_neg/len(s):.1f}%)  "
            f">1000={_fmt(n_gt1k)} ({100*n_gt1k/len(s):.2f}%)",
            fontsize=9,
        )
        ax.set_xlabel("Phút (clip ±300 để đọc hình dạng)")
        ax.legend(fontsize=7)
    fig.suptitle("Phân phối delay (trước khi làm sạch ETL)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distance_distribution(df: pd.DataFrame, out: Path) -> None:
    """Histogram DISTANCE và ELAPSED_TIME."""
    cols = [c for c in ["DISTANCE", "ELAPSED_TIME"] if c in df.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(s, bins=80, density=True, color="#2ca02c",
                alpha=0.85, edgecolor="white", linewidth=0.2)
        ax.set_title(
            f"{col}  |  mean={s.mean():.0f}  std={s.std():.0f}"
            f"\nmin={s.min():.0f}  max={s.max():.0f}",
            fontsize=9,
        )
        ax.set_xlabel("Dặm" if col == "DISTANCE" else "Phút")
    fig.suptitle("Phân phối DISTANCE và ELAPSED_TIME", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dep_hour_distribution(df: pd.DataFrame, out: Path) -> None:
    """Histogram DEP_HOUR (trích từ SCHEDULED_DEPARTURE dạng HHMM)."""
    if "SCHEDULED_DEPARTURE" not in df.columns:
        return
    s = pd.to_numeric(df["SCHEDULED_DEPARTURE"], errors="coerce").dropna()
    dep_hour = (s // 100).clip(0, 23).astype(int)
    fig, ax = plt.subplots(figsize=(9, 4))
    counts = dep_hour.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#9467bd", alpha=0.85, edgecolor="white")
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Giờ khởi hành (0–23)")
    ax.set_ylabel("Số chuyến")
    ax.set_title("Phân phối theo giờ khởi hành (dep_hour)\n→ feature quan trọng cho mô hình dự báo")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Phân tích cột categorical
# ─────────────────────────────────────────────────────────────────────────────

def analyze_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Cardinality, top-5 giá trị, tỷ lệ null cho cột phân loại."""
    cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    records = []
    for col in cols_present:
        s = df[col].astype(str).replace("nan", pd.NA)
        n_null    = int(s.isna().sum())
        n_unique  = int(s.dropna().nunique())
        top5      = s.dropna().value_counts().head(5)
        top5_str  = ", ".join(f"{v}({c:,})" for v, c in top5.items())
        records.append({
            "column":        col,
            "n_unique":      n_unique,
            "n_null":        n_null,
            "pct_null":      round(100.0 * n_null / len(df), 2),
            "top5_values":   top5_str,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Kiểm tra chất lượng JOIN flights↔airlines/airports
# ─────────────────────────────────────────────────────────────────────────────

def analyze_join_quality(
    df: pd.DataFrame,
    df_airlines: pd.DataFrame,
    df_airports: pd.DataFrame,
) -> pd.DataFrame:
    """
    Kiểm tra tỷ lệ flights.AIRLINE / ORIGIN_AIRPORT / DEST_AIRPORT
    không map được sang airlines.csv / airports.csv.
    Giá trị không map được → sẽ bị dropna trong ETL transform_chunk.
    """
    # Lấy set IATA codes hợp lệ
    al_iata_col = "IATA_CODE" if "IATA_CODE" in df_airlines.columns else df_airlines.columns[0]
    ap_iata_col = "IATA_CODE" if "IATA_CODE" in df_airports.columns else df_airports.columns[0]
    valid_airlines  = set(df_airlines[al_iata_col].dropna().str.strip())
    valid_airports  = set(df_airports[ap_iata_col].dropna().str.strip())

    records = []
    checks = [
        ("AIRLINE",              valid_airlines,  "flights.AIRLINE → dim_airline"),
        ("ORIGIN_AIRPORT",       valid_airports,  "flights.ORIGIN_AIRPORT → dim_airport"),
        ("DESTINATION_AIRPORT",  valid_airports,  "flights.DEST_AIRPORT → dim_airport"),
    ]
    for col, valid_set, description in checks:
        if col not in df.columns:
            continue
        s = df[col].dropna().str.strip()
        n_total   = len(s)
        n_matched = int(s.isin(valid_set).sum())
        n_lost    = n_total - n_matched
        # Top unmatchable values
        unmatched_vals = s[~s.isin(valid_set)].value_counts().head(5)
        top_unmatched  = ", ".join(f"{v}({c})" for v, c in unmatched_vals.items()) or "—"
        records.append({
            "check":           description,
            "n_rows_checked":  n_total,
            "n_matched":       n_matched,
            "n_unmatched":     n_lost,
            "pct_lost_in_etl": round(100.0 * n_lost / n_total, 3) if n_total else 0,
            "top_unmatched":   top_unmatched,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Phân tích cancelled / diverted
# ─────────────────────────────────────────────────────────────────────────────

def analyze_cancelled_diverted(df: pd.DataFrame) -> dict:
    n = len(df)
    result = {}
    for col in ["CANCELLED", "DIVERTED"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0)
            n1 = int((s == 1).sum())
            result[col] = {"count": n1, "pct": round(100.0 * n1 / n, 2)}
    # Phân tích CANCELLATION_REASON
    if "CANCELLATION_REASON" in df.columns:
        cr = df["CANCELLATION_REASON"].dropna()
        result["CANCELLATION_REASON"] = dict(cr.value_counts().head(10))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tổng hợp quyết định ETL
# ─────────────────────────────────────────────────────────────────────────────

def print_etl_decisions(
    df: pd.DataFrame,
    df_miss: pd.DataFrame,
    df_num: pd.DataFrame,
    df_join: pd.DataFrame,
    cancelled_info: dict,
) -> list[str]:
    """In và trả về danh sách quyết định ETL dựa trên kết quả EDA."""
    decisions = []

    # --- Missing ---
    high_miss = df_miss[df_miss["pct_missing"] > 50]
    if not high_miss.empty:
        for _, row in high_miss.iterrows():
            decisions.append(
                f"⚠  Cột '{row['column']}': {row['pct_missing']:.1f}% missing — "
                f"ETL: {row['etl_action']}"
            )
    delay_miss = df_miss[df_miss["column"].isin(DELAY_COLS) & (df_miss["pct_missing"] > 0)]
    if not delay_miss.empty:
        avg_miss = delay_miss["pct_missing"].mean()
        decisions.append(
            f"✔  Cột delay thành phần: missing trung bình {avg_miss:.1f}% "
            f"→ ETL dùng fillna(0) (NaN = không có trễ loại đó)"
        )

    # --- Outlier delay ---
    if not df_num.empty and "DEPARTURE_DELAY" in df_num["column"].values:
        row = df_num[df_num["column"] == "DEPARTURE_DELAY"].iloc[0]
        decisions.append(
            f"✔  DEPARTURE_DELAY âm: {row['pct_negative']:.1f}% "
            f"→ GIỮ (đến sớm hơn lịch là hợp lệ)"
        )
        if row["n_gt1000min"] > 0:
            decisions.append(
                f"✔  DEPARTURE_DELAY > 1000 phút: {row['n_gt1000min']:,} bản ghi "
                f"({row['pct_gt1000min']:.3f}%) → GIỮ (hoãn qua ngày do sự cố thật); "
                f"nhãn nhị phân is_delayed_15 trung hoà giá trị cực đoan"
            )

    # --- Cần normalize không? ---
    decisions.append(
        "ℹ  Kho dữ liệu (DW): KHÔNG normalize — lưu giá trị thô nguyên bản "
        "(normalize chỉ áp dụng trong pipeline Mining cho LR/MLP)"
    )

    # --- JOIN quality ---
    if not df_join.empty:
        max_lost = df_join["pct_lost_in_etl"].max()
        if max_lost < 1:
            decisions.append(
                f"✔  JOIN quality: tối đa {max_lost:.2f}% flights không map được "
                f"→ dropna trong ETL, chấp nhận được"
            )
        else:
            decisions.append(
                f"⚠  JOIN quality: {max_lost:.1f}% flights không map được "
                f"→ kiểm tra lại airlines.csv/airports.csv cho đủ"
            )

    # --- Cancelled/Diverted ---
    if "CANCELLED" in cancelled_info:
        info = cancelled_info["CANCELLED"]
        decisions.append(
            f"✔  Chuyến bị hủy (CANCELLED=1): {info['count']:,} ({info['pct']:.2f}%) "
            f"→ ETL vẫn nạp vào fact_flights; cột is_delayed_15=0 cho chuyến hủy"
        )

    return decisions


# ─────────────────────────────────────────────────────────────────────────────
# 8. Xuất báo cáo HTML
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_html(df: pd.DataFrame, max_rows: int = 200) -> str:
    return df.head(max_rows).to_html(
        index=False,
        classes="tbl",
        border=0,
        float_format=lambda x: f"{x:,.4f}",
        na_rep="—",
    )


def save_html_report(
    n_flights: int,
    dictionary_tables: dict[str, pd.DataFrame],
    df_miss: pd.DataFrame,
    df_delay_null_check: pd.DataFrame,
    df_num: pd.DataFrame,
    df_cat: pd.DataFrame,
    df_join: pd.DataFrame,
    cancelled_info: dict,
    decisions: list[str],
    out: Path,
) -> None:
    def _decision_html(decisions: list[str]) -> str:
        rows = ""
        for d in decisions:
            color = "#d62728" if d.startswith("⚠") else "#2ca02c" if d.startswith("✔") else "#1f77b4"
            rows += f'<li style="margin:4px 0; color:{color}">{d}</li>\n'
        return f"<ul>{rows}</ul>"

    def _dictionary_sections_html(tables: dict[str, pd.DataFrame]) -> str:
        blocks: list[str] = []
        for source_name, table in tables.items():
            if table.empty:
                blocks.append(
                    f"<h3>{source_name}</h3>"
                    "<p style='font-size:13px;color:#777'>Không có dữ liệu mô tả cho file này.</p>"
                )
                continue
            blocks.append(f"<h3>{source_name}</h3>{_df_to_html(table, max_rows=500)}")
        return "\n".join(blocks)

    html = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
      <meta charset="UTF-8">
      <title>EDA Trước ETL — Airline DW</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 30px; background: #f9f9f9; color: #222; }}
        h1   {{ color: #1e3a5f; }}
        h2   {{ color: #1e3a5f; border-bottom: 2px solid #1e3a5f; padding-bottom: 4px; margin-top: 40px; }}
        h3   {{ color: #2c6e9f; }}
        .tbl {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; font-size: 13px; }}
        .tbl th {{ background: #1e3a5f; color: white; padding: 6px 10px; text-align: left; }}
        .tbl td {{ border: 1px solid #ddd; padding: 5px 10px; }}
        .tbl tr:nth-child(even) {{ background: #eef2f7; }}
        .img-box {{ display: inline-block; margin: 8px; }}
        img {{ max-width: 700px; border: 1px solid #ccc; border-radius: 4px; }}
        .kpi {{ display: inline-block; background: #1e3a5f; color: white;
                padding: 10px 24px; border-radius: 6px; margin: 6px;
                font-size: 15px; font-weight: bold; }}
        .note {{ background: #fffde7; border-left: 4px solid #f9a825;
                 padding: 10px 14px; margin: 12px 0; font-size: 13px; }}
      </style>
    </head>
    <body>
    <h1>EDA Trước ETL — Airline DW (US Flights 2015)</h1>
    <div>
      <span class="kpi">Tổng dòng phân tích: {n_flights:,}</span>
      <span class="kpi">Tổng cột: {len(df_miss)}</span>
      <span class="kpi">Cột có missing: {int((df_miss["pct_missing"] > 0).sum())}</span>
    </div>

    <h2>1. Data Dictionary các file nguồn</h2>
    <div class="note">
      Bảng mô tả cột của 4 nguồn: <code>airlines.csv</code>, <code>airports.csv</code>,
      <code>flights.csv</code> (trên tập mẫu đang chạy), và <code>L_AIRPORT_ID.xls</code> (nếu đọc được).
    </div>
    {_dictionary_sections_html(dictionary_tables)}

    <h2>2. Tỷ lệ Missing Từng Cột</h2>
    <div class="note">
      Cột delay thành phần (AIR_SYSTEM, WEATHER…) có ~76% missing — hiện tượng đặc thù:
      chỉ ghi nhận khi chuyến thật sự delay. ETL quy ước fillna(0).
    </div>
    <div class="img-box">
      <img src="figures/eda_missing_bar.png" alt="Missing bar">
    </div>
    {_df_to_html(df_miss)}

    <h2>3. Thống Kê Mô Tả và Outlier Cột Số</h2>
    <div class="note">
      Delay âm = chuyến đến sớm hơn lịch (bình thường, GIỮ nguyên).
      Delay >1000 phút = hoãn qua ngày (hiếm nhưng thật, GIỮ; nhãn nhị phân trung hoà).
    </div>
    <div class="img-box"><img src="figures/eda_delay_dist.png" alt="Delay dist"></div>
    <div class="img-box"><img src="figures/eda_distance_dist.png" alt="Distance dist"></div>
    <div class="img-box"><img src="figures/eda_dep_hour_dist.png" alt="Dep hour dist"></div>
    {_df_to_html(df_num)}

    <h2>4. Kiểm chứng giả thuyết "Delay-component trống = đúng giờ?"</h2>
    <div class="note">
      Nhóm kiểm chứng loại chuyến hủy/chuyển hướng (CANCELLED=0, DIVERTED=0).<br>
      Theo chuẩn OTP, chuyến đúng giờ nếu <code>ARRIVAL_DELAY &lt;= 15</code>.
      Bảng dưới đồng thời hiển thị thêm chuẩn nghiêm ngặt <code>ARRIVAL_DELAY &lt;= 0</code>.
    </div>
    <div class="img-box">
      <img src="figures/eda_delay_null_vs_ontime.png" alt="Delay null vs ontime">
    </div>
    {_df_to_html(df_delay_null_check)}

    <h2>5. Cột Categorical — Cardinality và Giá Trị Phổ Biến</h2>
    {_df_to_html(df_cat)}

    <h2>6. Chất Lượng JOIN (flights ↔ airlines / airports)</h2>
    <div class="note">
      Hàng không map được FK → sẽ bị dropna trong ETL transform_chunk.
      Tỷ lệ mất &lt;1% là chấp nhận được.
    </div>
    {_df_to_html(df_join)}

    <h2>7. Phân Tích Chuyến Hủy / Chuyển Hướng</h2>
    <ul>
    {"".join(f"<li><b>{k}</b>: {v}</li>" for k, v in cancelled_info.items())}
    </ul>

    <h2>8. Tổng Hợp Quyết Định ETL</h2>
    <div class="note">
      Dựa trên kết quả EDA — các quyết định đã được hiện thực trong
      <code>etl/transform.py</code> và <code>etl/extract.py</code>.
    </div>
    {_decision_html(decisions)}

    <p style="color:#999; font-size:11px; margin-top:40px">
      Sinh tự động bởi scripts/eda_before_etl.py
    </p>
    </body>
    </html>
    """).strip()

    out.write_text(html, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EDA trước ETL — phân tích missing, outlier, join quality, quyết định làm sạch."
    )
    p.add_argument(
        "--nrows", type=int, default=300_000,
        help="Số dòng tối đa đọc từ flights.csv (mặc định 300,000). Dùng 0 = toàn bộ."
    )
    return p.parse_args()


def main() -> int:
    # Fix encoding trên Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    args = parse_args()
    nrows = None if args.nrows == 0 else args.nrows

    _ensure_dirs()

    # ── Kiểm tra file đầu vào ────────────────────────────────
    for f in [CSV_FLIGHTS, CSV_AIRLINES, CSV_AIRPORTS]:
        if not f.is_file():
            print(f"[LỖI] Không tìm thấy: {f}", file=sys.stderr)
            return 1

    print(_section("BƯỚC 1 — Đọc dữ liệu"))
    df_airlines, df_airports = read_small_files()
    df_airport_id = read_airport_id_xls()
    df = load_flights_sample(nrows)

    print(_section("BƯỚC 1B — Tạo bảng mô tả cột dữ liệu nguồn"))
    dictionary_tables = {
        "airlines.csv": build_column_dictionary(df_airlines, AIRLINES_COLUMN_DESCRIPTIONS_VI),
        "airports.csv": build_column_dictionary(df_airports, AIRPORTS_COLUMN_DESCRIPTIONS_VI),
        "flights.csv (sample)": build_column_dictionary(df, FLIGHTS_COLUMN_DESCRIPTIONS_VI),
        "L_AIRPORT_ID.xls": build_column_dictionary(df_airport_id, L_AIRPORT_ID_COLUMN_DESCRIPTIONS_VI) if not df_airport_id.empty else pd.DataFrame(),
    }
    df_dictionary = pd.concat(
        [table for table in dictionary_tables.values() if not table.empty],
        ignore_index=True
    )
    df_dictionary.to_csv(OUT_DIR / "eda_data_dictionary.csv", index=False)
    dictionary_output_files = {
        "airlines.csv": OUT_DIR / "eda_data_dictionary_airlines.csv",
        "airports.csv": OUT_DIR / "eda_data_dictionary_airports.csv",
        "flights.csv (sample)": OUT_DIR / "eda_data_dictionary_flights_sample.csv",
        "L_AIRPORT_ID.xls": OUT_DIR / "eda_data_dictionary_l_airport_id.csv",
    }
    for source_name, table in dictionary_tables.items():
        if not table.empty:
            table.to_csv(dictionary_output_files[source_name], index=False)

    print(df_dictionary[["column", "description", "dtype", "pct_missing"]].head(20).to_string(index=False))
    print("  → Đã lưu: eda_data_dictionary.csv + các file dictionary riêng theo từng nguồn")

    # ── Missing values ────────────────────────────────────────
    print(_section("BƯỚC 2 — Phân tích Missing Values"))
    df_miss = analyze_missing(df)
    df_miss.to_csv(OUT_DIR / "eda_missing_all_cols.csv", index=False)
    # In top 10 cột nhiều missing nhất
    top_miss = df_miss[df_miss["pct_missing"] > 0].head(10)
    print(top_miss[["column", "pct_missing", "etl_action"]].to_string(index=False))
    plot_missing_bar(df_miss, FIG_DIR / "eda_missing_bar.png")
    print(f"  → Đã lưu: eda_missing_all_cols.csv, eda_missing_bar.png")

    # ── Kiểm chứng all-null delay components vs đúng giờ ───────
    print(_section("BƯỚC 2B — Kiểm chứng cột delay trống có phải đúng giờ?"))
    df_delay_null_check = analyze_delay_null_vs_ontime(df)
    if df_delay_null_check.empty:
        print("  Không đủ cột để kiểm tra (cần ARRIVAL_DELAY, CANCELLED, DIVERTED và 5 cột delay thành phần).")
    else:
        out_delay_check = OUT_DIR / "eda_delay_null_vs_ontime.csv"
        df_delay_null_check.to_csv(out_delay_check, index=False)
        plot_delay_null_vs_ontime(df_delay_null_check, FIG_DIR / "eda_delay_null_vs_ontime.png")
        print(df_delay_null_check.to_string(index=False))
        print("  → Đã lưu: eda_delay_null_vs_ontime.csv, eda_delay_null_vs_ontime.png")
        print("  Gợi ý đọc kết quả: nếu pct_on_time_arrival_15_in_group gần 100% thì giả thuyết đúng mạnh.")

    # ── Numeric stats + outlier ───────────────────────────────
    print(_section("BƯỚC 3 — Thống kê số và Outlier Delay"))
    df_num = analyze_numeric(df)
    df_num.to_csv(OUT_DIR / "eda_numeric_stats.csv", index=False)
    # In cột delay quan trọng
    delay_rows = df_num[df_num["column"].isin(["DEPARTURE_DELAY", "ARRIVAL_DELAY"])]
    if not delay_rows.empty:
        for _, r in delay_rows.iterrows():
            print(f"  {r['column']:25s} | mean={r['mean']:7.1f} | "
                  f"min={r['min']:8.1f} | max={r['max']:8.1f} | "
                  f"âm={r['pct_negative']:.1f}% | >1000min={r['n_gt1000min']:,}")
    plot_delay_distribution(df, FIG_DIR / "eda_delay_dist.png")
    plot_distance_distribution(df, FIG_DIR / "eda_distance_dist.png")
    plot_dep_hour_distribution(df, FIG_DIR / "eda_dep_hour_dist.png")
    print(f"  → Đã lưu: eda_numeric_stats.csv + 3 biểu đồ phân phối")

    # ── Categorical ───────────────────────────────────────────
    print(_section("BƯỚC 4 — Phân tích Cột Categorical"))
    df_cat = analyze_categorical(df)
    df_cat.to_csv(OUT_DIR / "eda_categorical_stats.csv", index=False)
    print(df_cat[["column", "n_unique", "pct_null", "top5_values"]].to_string(index=False))
    print(f"  → Đã lưu: eda_categorical_stats.csv")

    # ── JOIN quality ──────────────────────────────────────────
    print(_section("BƯỚC 5 — Kiểm tra chất lượng JOIN"))
    df_join = analyze_join_quality(df, df_airlines, df_airports)
    df_join.to_csv(OUT_DIR / "eda_join_quality.csv", index=False)
    print(df_join[["check", "n_unmatched", "pct_lost_in_etl", "top_unmatched"]].to_string(index=False))
    print(f"  → Đã lưu: eda_join_quality.csv")

    # ── Cancelled / Diverted ──────────────────────────────────
    print(_section("BƯỚC 6 — Chuyến Hủy / Chuyển Hướng"))
    cancelled_info = analyze_cancelled_diverted(df)
    for k, v in cancelled_info.items():
        if isinstance(v, dict) and "count" in v:
            print(f"  {k}: {v['count']:,} ({v['pct']}%)")
        else:
            print(f"  {k}: {v}")

    # ── Quyết định ETL ────────────────────────────────────────
    print(_section("BƯỚC 7 — Tổng Hợp Quyết Định ETL"))
    decisions = print_etl_decisions(df, df_miss, df_num, df_join, cancelled_info)
    for d in decisions:
        print(f"  {d}")

    # ── HTML report ───────────────────────────────────────────
    print(_section("BƯỚC 8 — Xuất báo cáo HTML"))
    html_out = OUT_DIR / "eda_before_etl_report.html"
    save_html_report(
        n_flights=len(df),
        dictionary_tables=dictionary_tables,
        df_miss=df_miss,
        df_delay_null_check=df_delay_null_check,
        df_num=df_num,
        df_cat=df_cat,
        df_join=df_join,
        cancelled_info=cancelled_info,
        decisions=decisions,
        out=html_out,
    )
    print(f"  → Đã lưu: {html_out}")
    print("\n✅  EDA hoàn tất. Mở file trên trình duyệt để xem báo cáo đầy đủ.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
