"""
EDA phục vụ báo cáo preprocessing: outlier (IQR), heatmap tương quan,
phân phối trước/sau khi điền NaN delay như etl/transform.transform_chunk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Thư mục chứa script = data/
_DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DATA_DIR.parent

DEFAULT_FLIGHTS_CSV = _DATA_DIR / "flights.csv"
OUTPUT_ROOT = _DATA_DIR / "eda_output"
FIGURE_DIR = OUTPUT_ROOT / "figures"

# Khớp delay_cols trong etl/transform.py — transform_chunk
DELAY_FILL_COLS = [
    "DEPARTURE_DELAY",
    "ARRIVAL_DELAY",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
]

# Cột số thường có trong flights Kaggle (subset cho outlier / corr)
NUMERIC_CANDIDATES = [
    "SCHEDULED_DEPARTURE",
    "DEPARTURE_TIME",
    "DEPARTURE_DELAY",
    "SCHEDULED_ARRIVAL",
    "ARRIVAL_TIME",
    "ARRIVAL_DELAY",
    "ELAPSED_TIME",
    "AIR_TIME",
    "DISTANCE",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
    "CANCELLED",
    "DIVERTED",
]

# Histogram delay: cắt đuôi để dễ so sánh hình dạng (ghi chú trên hình)
DELAY_PLOT_CLIP = (-80.0, 300.0)

sns.set_theme(style="whitegrid", context="notebook")


def ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def resolve_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in NUMERIC_CANDIDATES if c in df.columns]


def load_flights_sample(
    csv_path: Path,
    nrows: Optional[int],
    extra_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Đọc mẫu CSV; chuẩn hóa tên cột IN HOA như pipeline ETL."""
    need = set(NUMERIC_CANDIDATES)
    if extra_cols:
        need.update(extra_cols)
    # Chỉ đọc cột tồn tại trong file (peek header)
    header = pd.read_csv(csv_path, nrows=0)
    header.columns = [c.strip().upper() for c in header.columns]
    usecols = [c for c in sorted(need) if c in header.columns]
    if not usecols:
        raise ValueError(f"Không tìm thấy cột số mong đợi trong {csv_path}")

    df = pd.read_csv(csv_path, usecols=usecols, nrows=nrows, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def apply_delay_cleaning_like_etl(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror: pd.to_numeric(..., errors='coerce').fillna(0.0) cho các cột delay."""
    out = df.copy()
    for c in DELAY_FILL_COLS:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def build_outlier_summary_table(
    df: pd.DataFrame,
    cols: list[str],
    iqr_whisker: float = 1.5,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Bảng thống kê: missing, mô tả, số điểm ngoài rào IQR (Tukey),
    và tỷ lệ |z| > z_threshold trên dữ liệu hợp lệ (robust cho báo cáo).
    """
    rows: list[dict] = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        n = int(len(s))
        n_missing = int(s.isna().sum())
        valid = s.dropna()
        n_valid = int(len(valid))
        rec: dict = {
            "column": col,
            "n_rows": n,
            "n_missing": n_missing,
            "pct_missing": round(100.0 * n_missing / n, 4) if n else 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q1": np.nan,
            "median": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "lower_fence": np.nan,
            "upper_fence": np.nan,
            "n_below_fence": 0,
            "n_above_fence": 0,
            "pct_iqr_outliers": 0.0,
            "n_abs_z_gt_threshold": 0,
            "pct_abs_z_gt_threshold": 0.0,
        }
        if n_valid == 0:
            rows.append(rec)
            continue

        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1
        lo = q1 - iqr_whisker * iqr
        hi = q3 + iqr_whisker * iqr
        below = int((valid < lo).sum())
        above = int((valid > hi).sum())
        outlier_n = below + above

        mu = float(valid.mean())
        sigma = float(valid.std(ddof=0))
        if sigma > 0 and np.isfinite(sigma):
            z = (valid - mu) / sigma
            n_z = int((z.abs() > z_threshold).sum())
            rec["n_abs_z_gt_threshold"] = n_z
            rec["pct_abs_z_gt_threshold"] = round(100.0 * n_z / n_valid, 4)
        else:
            rec["n_abs_z_gt_threshold"] = 0
            rec["pct_abs_z_gt_threshold"] = 0.0

        rec["mean"] = round(mu, 6)
        rec["std"] = round(sigma, 6) if np.isfinite(sigma) else np.nan
        rec["min"] = float(valid.min())
        rec["max"] = float(valid.max())
        rec["q1"] = q1
        rec["median"] = float(valid.median())
        rec["q3"] = q3
        rec["iqr"] = iqr
        rec["lower_fence"] = lo
        rec["upper_fence"] = hi
        rec["n_below_fence"] = below
        rec["n_above_fence"] = above
        rec["pct_iqr_outliers"] = round(100.0 * outlier_n / n_valid, 4)

        rows.append(rec)

    return pd.DataFrame(rows)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list[str],
    outfile: Path,
    title: str = "Ma trận tương quan (Pearson)",
    annot_max_cols: int = 14,
) -> None:
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(min_periods=min(50, max(10, len(sub) // 100)))
    plt.figure(figsize=(max(8, 0.6 * len(cols)), max(7, 0.55 * len(cols))))
    annot = len(cols) <= annot_max_cols
    sns.heatmap(
        corr,
        annot=annot,
        fmt=".2f" if annot else "",
        cmap="RdBu_r",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.75},
    )
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close()


def _hist_one_axis(
    ax: plt.Axes,
    values: pd.Series,
    title_line: str,
    xlabel: str,
    clip_lo: float,
    clip_hi: float,
    note: str | None = None,
) -> None:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) == 0:
        ax.text(0.5, 0.5, "Không có dữ liệu hợp lệ", ha="center", va="center")
        ax.set_title(title_line)
        return
    plot_v = v.clip(clip_lo, clip_hi)
    ax.hist(
        plot_v,
        bins=72,
        density=True,
        alpha=0.88,
        color="steelblue",
        edgecolor="white",
        linewidth=0.25,
    )
    ax.set_title(title_line)
    ax.set_xlabel(xlabel)
    if note:
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#333333",
        )


def plot_before_after_delay_distributions(
    df_raw: pd.DataFrame,
    columns: list[str],
    prefix: str = "delay",
) -> list[Path]:
    """
    Với mỗi cột delay: 2 panel Trước (raw, có NaN) / Sau (fill 0 như ETL).
    Trả về danh sách đường dẫn file PNG.
    """
    df_after = apply_delay_cleaning_like_etl(df_raw)
    saved: list[Path] = []
    for col in columns:
        if col not in df_raw.columns:
            continue
        before = df_raw[col]
        after = df_after[col]
        n_miss = int(pd.to_numeric(before, errors="coerce").isna().sum())
        n_tot = len(before)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
        lo, hi = DELAY_PLOT_CLIP
        _hist_one_axis(
            axes[0],
            before,
            f"{col}\nTrước làm sạch",
            "phút",
            lo,
            hi,
            note=f"n={n_tot}, thiếu={n_miss} ({100*n_miss/n_tot:.2f}%)" if n_tot else None,
        )
        _hist_one_axis(
            axes[1],
            after,
            f"{col}\nSau (NaN delay → 0, giống ETL)",
            "phút",
            lo,
            hi,
            note=f"Histogram clip [{lo}, {hi}] để đọc hình dạng",
        )
        fig.suptitle(
            "So sánh phân phối (cùng thang trục Y — mật độ)",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        out = FIGURE_DIR / f"{prefix}_before_after_{col.lower()}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
    return saved

