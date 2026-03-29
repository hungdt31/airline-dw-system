"""
Menu chạy từng mô hình hoặc chạy cả 5; in Accuracy, AUC-ROC, F1, Avg Precision,
Log Loss, Brier Score, Train time (s) ra terminal.

Chạy từ thư mục gốc repo:
    python mining/run_models.py
    python mining/run_models.py --all
    python mining/run_models.py --model lr
    python mining/run_models.py --limit 100000

Mã --model: lr | rf | nn | xgb | lgb
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

# Đảm bảo import model_* từ thư mục mining
_MINING_DIR = os.path.dirname(os.path.abspath(__file__))
if _MINING_DIR not in sys.path:
    sys.path.insert(0, _MINING_DIR)

from model_common import DEFAULT_LIMIT  # noqa: E402

# (mã CLI, mô tả, tên module, hàm run)
MODEL_REGISTRY: List[Tuple[str, str, str]] = [
    ("lr", "Logistic Regression", "model_logistic_regression"),
    ("rf", "Random Forest", "model_random_forest"),
    ("nn", "Neural Network (MLP)", "model_neural_network"),
    ("xgb", "XGBoost", "model_xgboost"),
    ("lgb", "LightGBM", "model_lightgbm"),
]


def _load_run(module_name: str) -> Callable[..., dict]:
    mod = importlib.import_module(module_name)
    return mod.run


def run_one(key: str, limit: int) -> Optional[dict]:
    key = key.lower().strip()
    for code, _label, modname in MODEL_REGISTRY:
        if code == key:
            run_fn = _load_run(modname)
            result = run_fn(limit=limit)
            if result is None:
                sys.exit(1)
            return result
    print(f"Không biết mã mô hình: {key!r}. Dùng: lr, rf, nn, xgb, lgb", file=sys.stderr)
    return None


def run_all(limit: int) -> List[dict]:
    results: List[dict] = []
    for code, label, modname in MODEL_REGISTRY:
        print(f"\n>>> Đang chạy: [{code}] {label} ...")
        try:
            run_fn = _load_run(modname)
            m = run_fn(limit=limit)
            if m is not None:
                results.append(m)
        except Exception as ex:  # noqa: BLE001
            print(f"    [LỖI] {label}: {ex}", file=sys.stderr)
    return results


def _print_summary_table(rows: List[dict]) -> None:
    if not rows:
        return
    headers = [
        "Model",
        "Accuracy",
        "AUC-ROC",
        "F1",
        "AvgPrec",
        "LogLoss",
        "Brier",
        "Train(s)",
    ]
    print("\n" + "=" * 100)
    print("  TÓM TẮT (chạy nhiều mô hình)")
    print("=" * 100)
    fmt = (
        f"  {{:<22}} {{:>10.4f}} {{:>8.4f}} {{:>8.4f}} {{:>8.4f}} "
        f"{{:>9.4f}} {{:>8.4f}} {{:>9.2f}}"
    )
    print("  " + " ".join(f"{h:>10}" if i else f"{h:<22}" for i, h in enumerate(headers)))
    print("  " + "-" * 96)
    for m in rows:
        print(
            fmt.format(
                m["name"][:22],
                m["accuracy"],
                m["auc_roc"],
                m["f1"],
                m["avg_precision"],
                m["log_loss"],
                m["brier"],
                m["train_time_s"],
            )
        )
    print("=" * 100 + "\n")


def interactive_menu(limit: int) -> None:
    while True:
        print(
            """
╔══════════════════════════════════════════════════════════╗
║  CHẠY MÔ HÌNH DELAY (cùng pipeline với compare_5_models) ║
╠══════════════════════════════════════════════════════════╣
║  1. Logistic Regression                                  ║
║  2. Random Forest                                        ║
║  3. Neural Network (MLP)                                 ║
║  4. XGBoost   (cần pip install xgboost)                  ║
║  5. LightGBM  (cần pip install lightgbm)                 ║
║  6. Chạy cả 5 mô hình lần lượt                           ║
║  0. Thoát                                                ║
╚══════════════════════════════════════════════════════════╝
"""
        )
        choice = input("Chọn [0-6]: ").strip()
        if choice == "0":
            print("Thoát.")
            return
        if choice == "6":
            rows = run_all(limit=limit)
            _print_summary_table(rows)
            continue
        mapping = {"1": "lr", "2": "rf", "3": "nn", "4": "xgb", "5": "lgb"}
        key = mapping.get(choice)
        if not key:
            print("Lựa chọn không hợp lệ.\n")
            continue
        run_one(key, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy mô hình delay — menu hoặc CLI.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Chạy cả 5 mô hình, in bảng tóm tắt cuối cùng.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="CODE",
        help="Một mô hình: lr | rf | nn | xgb | lgb",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Số dòng tối đa load từ DB (mặc định {DEFAULT_LIMIT}).",
    )
    args = parser.parse_args()

    if args.all:
        rows = run_all(limit=args.limit)
        _print_summary_table(rows)
        return

    if args.model:
        run_one(args.model, limit=args.limit)
        return

    interactive_menu(limit=args.limit)


if __name__ == "__main__":
    main()
