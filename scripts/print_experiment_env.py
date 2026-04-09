#!/usr/bin/env python3
"""
In thông tin môi trường phục vụ bảng "Cấu hình môi trường thực nghiệm" trong báo cáo.

Chạy từ thư mục gốc repo:
    python scripts/print_experiment_env.py

Không bắt buộc cài thêm gói; trên Windows dùng PowerShell một lần để lấy tên CPU / RAM chính xác.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import sqlite3
import subprocess
import sys
from typing import Any


def _run_powershell_json(script: str) -> dict[str, Any] | None:
    """Thực thi đoạn PowerShell trả về JSON (chỉ Windows)."""
    if sys.platform != "win32":
        return None
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        return json.loads(proc.stdout)
    except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return None


def hardware_summary() -> dict[str, str]:
    """CPU, RAM (Windows qua CIM); các OS khác: thông tin hạn chế hơn."""
    out: dict[str, str] = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor() or "(không đọc được — xem Cài đặt Windows)",
        "ram_gb": "?",
        "disk_c": "?",
    }

    if sys.platform == "win32":
        ps = """
        $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
        $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
        @{ Cpu = $cpu; RamGB = $ram } | ConvertTo-Json -Compress
        """
        data = _run_powershell_json(ps.strip())
        if data:
            out["cpu"] = str(data.get("Cpu", out["cpu"])).strip()
            out["ram_gb"] = str(data.get("RamGB", "?"))

    # Ổ chứa workspace (ổ của thư mục làm việc hiện tại)
    root = disk_root_for_usage()
    try:
        du = shutil.disk_usage(root)
        out["disk_c"] = f"{root}: tổng {du.total / (1024**3):.1f} GB, còn trống {du.free / (1024**3):.1f} GB"
    except OSError:
        pass

    if out["ram_gb"] == "?" and sys.platform != "win32":
        # Linux: thử đọc /proc/meminfo (KB)
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        out["ram_gb"] = f"{kb / (1024**2):.2f}"
                        break
        except OSError:
            pass

    out["python"] = sys.version.split()[0]
    return out


def disk_root_for_usage() -> str:
    """Gốc ổ đĩa để gọi shutil.disk_usage (Windows: ổ của cwd)."""
    import os

    if sys.platform == "win32":
        drive, _ = os.path.splitdrive(os.getcwd())
        return (drive + "\\") if drive else "C:\\"
    return "/"


def package_version(dist_name: str) -> str:
    """Phiên bản đã cài theo tên distribution trên PyPI (có thể khác tên import)."""
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "(chưa cài trong môi trường hiện tại)"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    hw = hardware_summary()

    print("=== Phần cứng / hệ điều hành (điền vào cột Thông số) ===\n")
    print(f"Hệ điều hành   : {hw['os']}")
    print(f"CPU            : {hw['cpu']}")
    print(f"RAM            : {hw['ram_gb']} GB" if hw["ram_gb"] != "?" else "RAM            : ?")
    print(f"Lưu trữ (tham chiếu): {hw['disk_c']}")
    print()

    print("=== Phần mềm ===\n")
    print(f"Python (runtime): {hw['python']}")
    print(f"sqlite3 (stdlib): thư viện SQLite {sqlite3.sqlite_version}")

    # Mapping: tên hiển thị báo cáo -> tên distribution pip
    libs = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("tensorflow", "tensorflow"),
        ("keras (nếu tách gói)", "keras"),
        ("plotly", "plotly"),
        ("streamlit (dashboard)", "streamlit"),
    ]

    for label, dist in libs:
        print(f"{label:28} : {package_version(dist)}")

    print("\n--- Gợi ý dòng LaTeX (chỉnh sửa nếu cần) ---\n")
    ram_line = f"{hw['ram_gb']}~GB" if hw["ram_gb"] != "?" else "[dung lượng]~GB"
    cpu_tex = hw["cpu"].replace("&", r"\&")
    print(f"CPU            & {cpu_tex} \\\\")
    print(f"RAM            & {ram_line} \\\\")
    print("Lưu trữ        & SSD, khoảng [X]~GB (ổ hệ thống) \\\\")


if __name__ == "__main__":
    main()
