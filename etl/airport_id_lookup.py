"""
etl/airport_id_lookup.py
========================
Ánh xạ BTS Airport ID (số trong flights.csv) → mã IATA 3 chữ cái nhờ L_AIRPORT_ID
+ khớp tên sân bay với cột AIRPORT trong airports.csv.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def normalize_airport_label(text: str) -> str:
    """
    Chuẩn hóa tên sân bay để so khớp giữa Description (BTS) và AIRPORT (Kaggle).
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text).lower().replace("/", " ").replace("-", " ")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Bỏ hậu tố thường gặp (lặp cho đến khi không còn khớp)
    suffixes = (
        " international airport",
        " regional airport",
        " municipal airport",
        " international",
        " airport",
        " field",
        " airpark",
        " airstrip",
        " seaplane base",
    )
    changed = True
    while changed and t:
        changed = False
        for suf in suffixes:
            if t.endswith(suf):
                t = t[: -len(suf)].strip()
                changed = True
    return t


def build_faa_id_to_iata(
    l_airport_path: Path,
    airports_csv_path: Path,
) -> dict[int, str]:
    """
    Đọc L_AIRPORT_ID (CSV/“.xls” giả) và airports.csv, trả về dict[int, str].
    """
    if not l_airport_path.is_file():
        log.warning("Không có %s — bỏ qua map Airport ID → IATA", l_airport_path)
        return {}

    ap = pd.read_csv(airports_csv_path)
    ap.columns = [c.strip().upper() for c in ap.columns]
    if "IATA_CODE" not in ap.columns or "AIRPORT" not in ap.columns:
        raise ValueError("airports.csv cần có IATA_CODE và AIRPORT")

    ap["norm"] = ap["AIRPORT"].astype(str).map(normalize_airport_label)
    norm_to_iata: dict[str, str] = {}
    for _, row in ap.iterrows():
        n = row["norm"]
        if n and n not in norm_to_iata:
            norm_to_iata[n] = row["IATA_CODE"]

    l_df = pd.read_csv(l_airport_path, dtype=str, low_memory=False)
    l_df.columns = [c.strip().upper() for c in l_df.columns]
    code_col = "CODE" if "CODE" in l_df.columns else None
    desc_col = "DESCRIPTION" if "DESCRIPTION" in l_df.columns else None
    if code_col is None or desc_col is None:
        raise ValueError("L_AIRPORT_ID cần có cột Code và Description")

    l_df[code_col] = l_df[code_col].astype(str).str.strip().str.strip('"')
    l_df["TAIL"] = (
        l_df[desc_col]
        .astype(str)
        .str.split(": ", n=1)
        .str[-1]
        .str.strip()
    )
    l_df["norm_tail"] = l_df["TAIL"].map(normalize_airport_label)

    mapping: dict[int, str] = {}
    for _, row in l_df.iterrows():
        try:
            aid = int(row[code_col])
        except ValueError:
            continue
        nt = row["norm_tail"]
        if not nt:
            continue
        if nt in norm_to_iata:
            mapping[aid] = norm_to_iata[nt]
            continue
        # Một lần fallback: tiền tố / chứa (ít dương tính hơn contains hai chiều)
        hits = ap[ap["norm"].str.startswith(nt, na=False)]
        if len(hits) == 1:
            mapping[aid] = hits.iloc[0]["IATA_CODE"]
        elif len(hits) > 1:
            hits2 = hits[hits["norm"] == nt]
            if len(hits2) == 1:
                mapping[aid] = hits2.iloc[0]["IATA_CODE"]

    n_l = len(l_df)
    log.info(
        "Airport ID → IATA: map được %s / %s mã (L_AIRPORT_ID)",
        f"{len(mapping):,}",
        f"{n_l:,}",
    )
    return mapping


def resolve_airport_codes(series: pd.Series, id_to_iata: dict[int, str]) -> pd.Series:
    """
    Cột ORIGIN/DEST: giữ IATA 3 chữ; nếu là số và có trong id_to_iata thì thay bằng IATA.
    """
    if not id_to_iata:
        return series.astype(str).str.strip()

    s = series.astype(str).str.strip()
    ids = pd.to_numeric(s, errors="coerce")
    int_ids = ids.dropna().astype("int64")
    mapped_partial = int_ids.map(id_to_iata)
    full = mapped_partial.reindex(s.index)
    out = s.copy()
    hit = full.notna()
    out.loc[hit] = full[hit].astype(str)
    return out
