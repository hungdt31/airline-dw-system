"""
scripts/export_flights_by_month.py
==================================
Lọc các dòng trong data/flights.csv theo tháng (mặc định tháng 10) và ghi ra file CSV mới.
Đọc theo chunk để phù hợp file lớn (~5,8M dòng).
Mặc định chỉ ghi **20 dòng** đầu tiên khớp điều kiện (file nhẹ để mở thử); dùng --limit 0 để xuất hết.

Chạy từ thư mục gốc repo:
    python scripts/export_flights_by_month.py
    python scripts/export_flights_by_month.py --limit 50 --out data/sample_oct.csv
    python scripts/export_flights_by_month.py --limit 0 --out data/flights_october_full.csv
    python scripts/export_flights_by_month.py -m 10 --year 2015

Tham số:
    --input   Đường dẫn flights.csv (mặc định: data/flights.csv)
    --out     File CSV đầu ra (mặc định: data/flights_october.csv)
    --month   Tháng 1–12 (mặc định: 10)
    --year    Nếu có, thêm điều kiện YEAR (ví dụ 2015); mặc định chỉ lọc theo MONTH
    --limit   Tối đa số dòng ghi ra (mặc định 20); 0 = không giới hạn
    --chunksize  Số dòng mỗi lần đọc (mặc định 200_000)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = _REPO_ROOT / "data" / "flights.csv"
DEFAULT_OUT = _REPO_ROOT / "data" / "flights_october.csv"


def export_flights_by_month(
    input_csv: Path,
    output_csv: Path,
    month: int,
    year: int | None,
    chunksize: int,
    limit: int,
) -> int:
    """
    Lọc flights.csv theo MONTH (và tùy chọn YEAR), ghi incremental ra output_csv.
    Nếu limit > 0: dừng sau khi đã ghi đủ `limit` dòng (đọc chunk cho đến khi đủ).
    Nếu limit == 0: ghi toàn bộ dòng khớp điều kiện.
    Trả về số dòng đã ghi.
    """
    if not input_csv.is_file():
        raise FileNotFoundError(f"Không thấy file: {input_csv}")
    if not (1 <= month <= 12):
        raise ValueError("month phải trong khoảng 1–12")
    if limit < 0:
        raise ValueError("limit phải >= 0 (0 = không giới hạn)")

    header_df = pd.read_csv(input_csv, nrows=0)
    header_df.columns = [c.strip().upper() for c in header_df.columns]
    cols = list(header_df.columns)
    if "MONTH" not in cols:
        raise ValueError("flights.csv không có cột MONTH")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header_written = False
    n_out = 0

    for chunk in pd.read_csv(
        input_csv,
        chunksize=chunksize,
        low_memory=False,
        na_values=["", "NA", "N/A"],
    ):
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        mask = chunk["MONTH"] == month
        if year is not None:
            if "YEAR" not in chunk.columns:
                raise ValueError("Đã truyền --year nhưng CSV không có cột YEAR")
            mask &= chunk["YEAR"] == year
        sub = chunk.loc[mask]
        if sub.empty:
            continue

        if limit > 0:
            remaining = limit - n_out
            if remaining <= 0:
                break
            if len(sub) > remaining:
                sub = sub.iloc[:remaining].copy()

        sub.to_csv(
            output_csv,
            mode="a" if header_written else "w",
            header=not header_written,
            index=False,
        )
        header_written = True
        n_out += len(sub)

        if limit > 0 and n_out >= limit:
            break

    if not header_written:
        # Không có dòng nào khớp — tạo file rỗng với đúng header nguồn
        pd.DataFrame(columns=cols).to_csv(output_csv, index=False)

    return n_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lọc chuyến bay theo tháng từ data/flights.csv (streaming chunk)."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Đường dẫn flights.csv (mặc định: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="File CSV đầu ra (mặc định: data/flights_october.csv)",
    )
    p.add_argument(
        "-m",
        "--month",
        type=int,
        default=10,
        help="Tháng cần lọc (1–12), mặc định 10",
    )
    p.add_argument(
        "--year",
        type=int,
        default=None,
        help="Nếu set, chỉ giữ các dòng có YEAR khớp (ví dụ 2015)",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Số dòng đọc mỗi chunk",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Tối đa số dòng ghi ra (mặc định 20); 0 = xuất toàn bộ dòng khớp tháng/năm",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        n = export_flights_by_month(
            input_csv=args.input.resolve(),
            output_csv=args.out.resolve(),
            month=args.month,
            year=args.year,
            chunksize=args.chunksize,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        return 1

    yr = f", YEAR={args.year}" if args.year is not None else ""
    cap = f", tối đa {args.limit} dòng" if args.limit > 0 else ", không giới hạn"
    print(f"Đã ghi {n:,} dòng (MONTH={args.month}{yr}{cap}) → {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
