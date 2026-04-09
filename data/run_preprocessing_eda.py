"""
Chạy EDA preprocessing trên flights.csv (mẫu nrows) — phục vụ Chương 4 / slide.

Từ thư mục gốc repo:
    python -m data.run_preprocessing_eda
    python -m data.run_preprocessing_eda --nrows 500000 --only-operated
    python -m data.run_preprocessing_eda --delay-plots all

Kết quả:
    data/eda_output/outliers_summary.csv
    data/eda_output/figures/correlation_heatmap.png
    data/eda_output/figures/delay_before_after_*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phép chạy trực tiếp: python data/run_preprocessing_eda.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.eda_lib import (  # noqa: E402
    DEFAULT_FLIGHTS_CSV,
    DELAY_FILL_COLS,
    FIGURE_DIR,
    OUTPUT_ROOT,
    apply_delay_cleaning_like_etl,
    build_outlier_summary_table,
    ensure_output_dirs,
    load_flights_sample,
    plot_before_after_delay_distributions,
    plot_correlation_heatmap,
    resolve_numeric_columns,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EDA: outlier (IQR/Z), heatmap tương quan, phân phối trước/sau làm sạch delay.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_FLIGHTS_CSV,
        help=f"Đường dẫn flights.csv (mặc định: {DEFAULT_FLIGHTS_CSV})",
    )
    p.add_argument(
        "--nrows",
        type=int,
        default=300_000,
        help="Số dòng đọc tối đa. Dùng 0 để đọc toàn bộ file (tốn RAM).",
    )
    p.add_argument(
        "--only-operated",
        action="store_true",
        help="Lọc CANCELLED==0 nếu có cột CANCELLED (phân tích chuyến không hủy).",
    )
    p.add_argument("--skip-outliers", action="store_true")
    p.add_argument("--skip-corr", action="store_true")
    p.add_argument("--skip-dist", action="store_true")
    p.add_argument(
        "--delay-plots",
        choices=("main", "all"),
        default="main",
        help="main: chỉ DEPARTURE/ARRIVAL delay; all: mọi cột delay như ETL.",
    )
    p.add_argument("--iqr-k", type=float, default=1.5, help="Hệ số rào IQR Tukey.")
    p.add_argument("--z", type=float, default=3.0, help="Ngưỡng |z| cho outlier kiểu Z.")
    return p.parse_args()


def main() -> int:
    # Tránh UnicodeEncodeError khi in tiếng Việt trên console Windows (cp125x).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    args = parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"Lỗi: không tìm thấy file CSV: {csv_path}", file=sys.stderr)
        return 1

    nrows = None if args.nrows == 0 else args.nrows
    extra = ["CANCELLED"] if args.only_operated else None

    print(f"Đọc {csv_path} (nrows={nrows}) …")
    df = load_flights_sample(csv_path, nrows=nrows, extra_cols=extra)

    if args.only_operated and "CANCELLED" in df.columns:
        c = pd.to_numeric(df["CANCELLED"], errors="coerce").fillna(1)
        df = df.loc[c == 0].copy()
        print(f"Sau lọc CANCELLED==0: {len(df)} dòng")

    ensure_output_dirs()
    num_cols = resolve_numeric_columns(df)
    print(f"Cột số dùng cho EDA: {len(num_cols)}")

    if not args.skip_outliers:
        out_tbl = build_outlier_summary_table(df, num_cols, iqr_whisker=args.iqr_k, z_threshold=args.z)
        out_path = OUTPUT_ROOT / "outliers_summary.csv"
        out_tbl.to_csv(out_path, index=False)
        print(f"Đã ghi: {out_path}")

    if not args.skip_corr and len(num_cols) >= 2:
        # Bỏ CANCELLED/DIVERTED khỏi heatmap nếu chỉ còn hằng số sau lọc
        corr_cols = [c for c in num_cols if c not in ("CANCELLED", "DIVERTED") or df[c].nunique(dropna=True) > 1]
        if len(corr_cols) >= 2:
            heat_path = FIGURE_DIR / "correlation_heatmap.png"
            plot_correlation_heatmap(
                df,
                corr_cols,
                heat_path,
                title="Ma trận tương quan Pearson (cột số flights)",
            )
            print(f"Đã ghi: {heat_path}")

    if not args.skip_dist:
        if args.delay_plots == "main":
            wanted = ("DEPARTURE_DELAY", "ARRIVAL_DELAY")
        else:
            wanted = tuple(DELAY_FILL_COLS)
        delay_plot_cols = [c for c in wanted if c in df.columns]
        paths = plot_before_after_delay_distributions(df, delay_plot_cols, prefix="delay")
        for p in paths:
            print(f"Đã ghi: {p}")

    # Mẫu sau bước điền NaN delay → 0 (giống transform_chunk), tối đa 50k dòng
    cleaned = apply_delay_cleaning_like_etl(df)
    sample_csv = OUTPUT_ROOT / "sample_after_delay_clean.csv"
    cleaned.head(50_000).to_csv(sample_csv, index=False)
    print(f"Đã ghi: {sample_csv}")

    print("Hoàn tất.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
