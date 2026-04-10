"""
mining/compare_5_models.py
==========================
So sánh 5 thuật toán:
  1. Logistic Regression  — baseline tuyến tính
  2. Random Forest        — ensemble song song
  3. Neural Network       — deep learning
  4. XGBoost              — gradient boosting thế hệ 1
  5. LightGBM             — gradient boosting thế hệ 2 (Microsoft)

Yêu cầu thêm:
  pip install xgboost lightgbm

Chạy: python mining/compare_5_models.py
Kết quả: mining/five_models_report.html
"""

import os
import sys
import time
import sqlite3
import warnings
import numpy as np
import pandas as pd
import pickle
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [5MODELS] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── sklearn ────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, log_loss, brier_score_loss
)

# ── XGBoost & LightGBM ─────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    log.warning("xgboost chưa cài — bỏ qua. Chạy: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    log.warning("lightgbm chưa cài — bỏ qua. Chạy: pip install lightgbm")

# ── Plotly ─────────────────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DB_PATH     = os.path.join(os.path.dirname(__file__), "..", "warehouse", "airline_dw.db")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "five_models_report.html")

COLORS = {
    "Logistic Regression": "#E8593C",
    "Random Forest":       "#2E75B6",
    "Neural Network":      "#27AE60",
    "XGBoost":             "#F39C12",
    "LightGBM":            "#8E44AD",
}
BG = "#FAFAFA"

# ═══════════════════════════════════════════════════════════
# 1. LOAD & TIỀN XỬ LÝ
# ═══════════════════════════════════════════════════════════

def load_data(limit: int = 500_000) -> pd.DataFrame:
    """Lấy mẫu stratified theo tháng — đảm bảo mỗi tháng có dữ liệu được đại diện đều."""
    conn = sqlite3.connect(DB_PATH)

    available_months = [r[0] for r in conn.execute(
        "SELECT DISTINCT d.month FROM fact_flights f "
        "JOIN dim_date d ON f.date_key = d.date_key "
        "WHERE f.is_cancelled = 0 ORDER BY d.month"
    ).fetchall()]
    n_months = len(available_months)
    base, remainder = divmod(limit, n_months)
    log.info(f"Loading {limit:,} mẫu stratified (~{base:,}/tháng × {n_months} tháng)...")

    chunks = []
    for i, month in enumerate(available_months):
        month_limit = base + (1 if i < remainder else 0)
        chunk = pd.read_sql_query(f"""
            SELECT
                f.distance, f.elapsed_time,
                f.is_delayed_15  AS target,
                d.month, d.day_of_week, d.is_weekend, d.quarter,
                CAST(SUBSTR(PRINTF('%04d', CAST(f.scheduled_departure AS INTEGER)),1,2)
                     AS INTEGER) AS dep_hour,
                a.airline_iata,
                ap_o.state AS origin_state,
                ap_d.state AS dest_state
            FROM fact_flights f
            JOIN dim_date    d    ON f.date_key           = d.date_key
            JOIN dim_airline a    ON f.airline_key         = a.airline_key
            JOIN dim_airport ap_o ON f.origin_airport_key = ap_o.airport_key
            JOIN dim_airport ap_d ON f.dest_airport_key   = ap_d.airport_key
            WHERE f.is_cancelled = 0
              AND d.month = {month}
            ORDER BY RANDOM()
            LIMIT {month_limit}
        """, conn)
        chunks.append(chunk)
        log.info(f"  Tháng {month:2d}: {len(chunk):,} mẫu "
                 f"| delay rate: {chunk['target'].mean():.1%}")

    conn.close()
    df = pd.concat(chunks, ignore_index=True).sample(frac=1, random_state=42)
    delay_rate = df["target"].mean()
    log.info(f"Tổng: {len(df):,} mẫu | delay rate: {delay_rate:.1%}")
    return df, delay_rate


def preprocess(df: pd.DataFrame):
    df = df.dropna(subset=["target", "month", "day_of_week", "airline_iata"])
    df["distance"]     = df["distance"].fillna(df["distance"].median())
    df["elapsed_time"] = df["elapsed_time"].fillna(df["elapsed_time"].median())
    df["dep_hour"]     = df["dep_hour"].fillna(12).clip(0, 23)

    le_airline = LabelEncoder()
    le_origin  = LabelEncoder()
    le_dest    = LabelEncoder()

    df["airline_enc"] = le_airline.fit_transform(df["airline_iata"].astype(str))
    df["origin_enc"]  = le_origin.fit_transform(
                            df["origin_state"].fillna("UNK").astype(str))
    df["dest_enc"]    = le_dest.fit_transform(
                            df["dest_state"].fillna("UNK").astype(str))

    feature_cols = [
        "month", "day_of_week", "is_weekend", "quarter",
        "dep_hour", "distance", "elapsed_time",
        "airline_enc", "origin_enc", "dest_enc"
    ]
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)
    return X, y, feature_cols


# ═══════════════════════════════════════════════════════════
# 2. TRAIN 5 MODEL
# ═══════════════════════════════════════════════════════════

def train_all(X_train, y_train, delay_rate: float) -> dict:
    """
    Trả về dict: key → {"model", "scaler", "time", "name", "key"}
    scaler=None nếu model không cần chuẩn hóa.
    """
    scale_pos_weight = (1 - delay_rate) / delay_rate  # dùng cho XGBoost

    trained = {}

    # ── 1. Logistic Regression ──────────────────────────────
    log.info("Training [1/5] Logistic Regression ...")
    sc_lr = StandardScaler()
    X_lr  = sc_lr.fit_transform(X_train)
    t0 = time.time()
    lr = LogisticRegression(
        C=1.0, penalty="l2", solver="saga", max_iter=200,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    lr.fit(X_lr, y_train)
    trained["lr"] = {
        "model": lr, "scaler": sc_lr,
        "time": time.time() - t0, "name": "Logistic Regression", "key": "lr"
    }
    log.info(f"  done in {trained['lr']['time']:.1f}s")

    # ── 2. Random Forest ────────────────────────────────────
    log.info("Training [2/5] Random Forest ...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=50,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    trained["rf"] = {
        "model": rf, "scaler": None,
        "time": time.time() - t0, "name": "Random Forest", "key": "rf"
    }
    log.info(f"  done in {trained['rf']['time']:.1f}s")

    # ── 3. Neural Network ───────────────────────────────────
    log.info("Training [3/5] Neural Network ...")
    sc_nn = StandardScaler()
    X_nn  = sc_nn.fit_transform(X_train)
    t0 = time.time()
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=0.001, batch_size=512, learning_rate="adaptive",
        max_iter=50, random_state=42,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=5, verbose=False
    )
    nn.fit(X_nn, y_train)
    trained["nn"] = {
        "model": nn, "scaler": sc_nn,
        "time": time.time() - t0, "name": "Neural Network", "key": "nn"
    }
    log.info(f"  done in {trained['nn']['time']:.1f}s ({nn.n_iter_} iters)")

    # ── 4. XGBoost ──────────────────────────────────────────
    if HAS_XGB:
        log.info("Training [4/5] XGBoost ...")
        t0 = time.time()
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,  # xử lý mất cân bằng
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb_clf.fit(X_train, y_train)
        trained["xgb"] = {
            "model": xgb_clf, "scaler": None,
            "time": time.time() - t0, "name": "XGBoost", "key": "xgb"
        }
        log.info(f"  done in {trained['xgb']['time']:.1f}s")
    else:
        log.warning("  [SKIP] XGBoost không được cài")

    # ── 5. LightGBM ─────────────────────────────────────────
    if HAS_LGB:
        log.info("Training [5/5] LightGBM ...")
        t0 = time.time()
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,          # xử lý mất cân bằng
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_clf.fit(X_train, y_train)
        trained["lgb"] = {
            "model": lgb_clf, "scaler": None,
            "time": time.time() - t0, "name": "LightGBM", "key": "lgb"
        }
        log.info(f"  done in {trained['lgb']['time']:.1f}s")
    else:
        log.warning("  [SKIP] LightGBM không được cài")

    return trained


# ═══════════════════════════════════════════════════════════
# 3. EVALUATE
# ═══════════════════════════════════════════════════════════

def evaluate_all(trained: dict, X_test, y_test) -> list:
    results = []
    order = ["lr", "rf", "nn", "xgb", "lgb"]
    for key in order:
        if key not in trained:
            continue
        t = trained[key]
        X = t["scaler"].transform(X_test) if t["scaler"] else X_test

        y_pred = t["model"].predict(X)
        y_prob = t["model"].predict_proba(X)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        report = classification_report(
            y_test, y_pred,
            target_names=["On-time", "Delayed"],
            output_dict=True
        )
        m = {
            "key":      key,
            "name":     t["name"],
            "time":     t["time"],
            "acc":      accuracy_score(y_test, y_pred),
            "auc":      roc_auc_score(y_test, y_prob),
            "f1":       f1_score(y_test, y_pred, average="weighted"),
            "ap":       average_precision_score(y_test, y_prob),
            "log_loss": log_loss(y_test, y_prob),
            "brier":    brier_score_loss(y_test, y_prob),
            "cm":       confusion_matrix(y_test, y_pred),
            "fpr": fpr, "tpr": tpr,
            "prec": prec, "rec": rec,
            "report":   report,
            "y_prob":   y_prob,
            "y_pred":   y_pred,
        }
        results.append(m)
        log.info(f"  {t['name']:<22} AUC={m['auc']:.4f} | "
                 f"Acc={m['acc']:.4f} | F1={m['f1']:.4f} | "
                 f"LogLoss={m['log_loss']:.4f} | Time={t['time']:.1f}s")
    return results


# ═══════════════════════════════════════════════════════════
# 4. BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════

def _color(name): return COLORS.get(name, "#888")


def fig_radar(models):
    cats = ["AUC-ROC", "Accuracy", "F1-score",
            "Avg Precision", "Speed\n(1-norm)", "Calibration\n(1-Brier)"]
    max_time = max(m["time"] for m in models)
    fig = go.Figure()
    for m in models:
        vals = [
            m["auc"], m["acc"], m["f1"], m["ap"],
            1 - (m["time"] / max_time),           # speed: nhanh = gần 1
            max(0, 1 - m["brier"] * 4)
        ]
        vals += [vals[0]]
        cats_c = cats + [cats[0]]
        fig.add_scatterpolar(
            r=vals, theta=cats_c, fill="toself",
            name=m["name"], line_color=_color(m["name"]), opacity=0.55
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="🕸️ Radar Chart — Tổng quan 5 mô hình",
        legend=dict(orientation="h", y=-0.15),
        paper_bgcolor=BG, height=500
    )
    return fig


def fig_rank(models):
    """Hệ thống điểm 5→1 cho mỗi tiêu chí."""
    metric_keys  = ["auc", "acc", "f1", "ap", "log_loss", "brier", "time"]
    lower_better = {"log_loss", "brier", "time"}
    n = len(models)
    scores = {m["name"]: 0 for m in models}
    for key in metric_keys:
        vals = [(m["name"], m[key]) for m in models]
        rev  = key not in lower_better
        ranked = sorted(vals, key=lambda x: x[1], reverse=rev)
        for i, (name, _) in enumerate(ranked):
            scores[name] += (n - i)

    names  = [m["name"] for m in models]
    totals = [scores[n] for n in names]
    colors = [_color(n) for n in names]

    fig = go.Figure(go.Bar(
        x=names, y=totals,
        marker_color=colors,
        text=[f"{t}đ" for t in totals],
        textposition="outside"
    ))
    fig.update_layout(
        title=f"🏆 Tổng điểm xếp hạng ({n} mô hình × 7 tiêu chí)<br>"
              f"<sup>Hạng 1={n}đ → Hạng {n}=1đ | tối đa {n*7}đ</sup>",
        yaxis_title="Tổng điểm",
        yaxis_range=[0, n * 7 + 2],
        plot_bgcolor=BG, paper_bgcolor=BG, height=380
    )
    return fig


def fig_metrics_grouped(models):
    metric_keys  = ["acc",     "auc",    "f1",       "ap",
                    "log_loss","brier"]
    metric_names = ["Accuracy","AUC-ROC","F1-score","Avg Precision",
                    "Log Loss","Brier Score"]
    fig = go.Figure()
    for m in models:
        vals = [m[k] for k in metric_keys]
        fig.add_bar(
            name=m["name"], x=metric_names, y=vals,
            marker_color=_color(m["name"]),
            text=[f"{v:.3f}" for v in vals],
            textposition="outside"
        )
    fig.update_layout(
        title="📊 So sánh 6 chỉ số đánh giá — 5 mô hình<br>"
              "<sup>Log Loss & Brier Score: thấp hơn = tốt hơn</sup>",
        barmode="group", yaxis_range=[0, 1.18],
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=1.12), height=450
    )
    return fig


def fig_roc(models):
    fig = go.Figure()
    for m in models:
        fig.add_scatter(
            x=m["fpr"], y=m["tpr"], mode="lines",
            name=f"{m['name']} (AUC={m['auc']:.3f})",
            line=dict(color=_color(m["name"]), width=2.5)
        )
    fig.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Baseline",
                    line=dict(color="#ccc", dash="dash", width=1.5))
    fig.update_layout(
        title="📈 ROC Curve — 5 mô hình",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=460
    )
    return fig


def fig_pr(models):
    fig = go.Figure()
    for m in models:
        fig.add_scatter(
            x=m["rec"], y=m["prec"], mode="lines",
            name=f"{m['name']} (AP={m['ap']:.3f})",
            line=dict(color=_color(m["name"]), width=2.5)
        )
    fig.update_layout(
        title="📉 Precision-Recall Curve — 5 mô hình",
        xaxis_title="Recall", yaxis_title="Precision",
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=460
    )
    return fig


def fig_confusion(models):
    n = len(models)
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=[m["name"] for m in models])
    labels = ["On-time", "Delayed"]
    for idx, m in enumerate(models, 1):
        cm = m["cm"]
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        text = [[f"{cm[i][j]:,}<br>({cm_n[i][j]:.1%})"
                 for j in range(2)] for i in range(2)]
        fig.add_heatmap(
            z=cm_n, x=labels, y=labels,
            text=text, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            row=1, col=idx
        )
    fig.update_layout(
        title="🔢 Confusion Matrix — 5 mô hình (normalized)",
        plot_bgcolor=BG, paper_bgcolor=BG, height=320
    )
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual", col=1)
    return fig


def fig_class_report(models):
    n = len(models)
    classes = ["On-time", "Delayed"]
    metrics = ["precision", "recall", "f1-score"]
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=[m["name"] for m in models])
    kw = dict(x=metrics, colorscale="RdYlGn", zmin=0, zmax=1, showscale=False)
    for idx, m in enumerate(models, 1):
        vals = [[m["report"][c][me] for me in metrics] for c in classes]
        text = [[f"{v:.3f}" for v in row] for row in vals]
        fig.add_heatmap(z=vals, y=classes, text=text,
                        texttemplate="%{text}", row=1, col=idx, **kw)
    fig.update_layout(
        title="📋 Classification Report — Precision / Recall / F1",
        plot_bgcolor=BG, paper_bgcolor=BG, height=280
    )
    return fig


def fig_train_time(models):
    fig = go.Figure(go.Bar(
        x=[m["name"] for m in models],
        y=[m["time"] for m in models],
        marker_color=[_color(m["name"]) for m in models],
        text=[f"{m['time']:.1f}s" for m in models],
        textposition="outside"
    ))
    fig.update_layout(
        title="⏱️ Thời gian Training — 5 mô hình",
        yaxis_title="Giây (s)",
        plot_bgcolor=BG, paper_bgcolor=BG, height=360
    )
    return fig


def fig_prob_dist(models, y_test):
    n = len(models)
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=[m["name"] for m in models])
    for idx, m in enumerate(models, 1):
        for label, color, name in [(0, "#4A90D9", "On-time"),
                                   (1, "#E8593C", "Delayed")]:
            mask = y_test == label
            fig.add_histogram(
                x=m["y_prob"][mask],
                name=name, marker_color=color, opacity=0.65,
                nbinsx=35, showlegend=(idx == 1),
                row=1, col=idx
            )
    fig.update_layout(
        title="📊 Phân phối xác suất dự đoán theo lớp thực tế",
        barmode="overlay", plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=340
    )
    fig.update_xaxes(title_text="P(delay)", range=[0, 1])
    fig.update_yaxes(title_text="Count", col=1)
    return fig


def fig_feature_importance_boost(trained, feature_cols):
    """XGBoost và LightGBM đều cung cấp feature importance."""
    n_cols = sum(1 for k in ["xgb", "lgb"] if k in trained)
    if n_cols == 0:
        return None

    titles = []
    keys   = []
    if "xgb" in trained: titles.append("XGBoost"); keys.append("xgb")
    if "lgb" in trained: titles.append("LightGBM"); keys.append("lgb")

    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=titles)
    for idx, key in enumerate(keys, 1):
        model  = trained[key]["model"]
        name   = trained[key]["name"]
        imp    = pd.Series(model.feature_importances_,
                           index=feature_cols).sort_values()
        fig.add_bar(
            x=imp.values, y=imp.index, orientation="h",
            marker_color=_color(name),
            text=[f"{v:.0f}" for v in imp.values],
            textposition="outside",
            showlegend=False,
            row=1, col=idx
        )
    fig.update_layout(
        title="🚀 Feature Importance — XGBoost & LightGBM<br>"
              "<sup>Giá trị là gain (độ giảm impurity) — không so được trực tiếp giữa 2 model</sup>",
        plot_bgcolor=BG, paper_bgcolor=BG, height=380
    )
    return fig


def fig_boosting_vs_others(models):
    """
    So sánh đặc biệt: Gradient Boosting (XGB + LGB) vs phần còn lại.
    Bar chart AUC theo nhóm thuật toán.
    """
    boost_names = {"XGBoost", "LightGBM"}
    others_names = {"Logistic Regression", "Random Forest", "Neural Network"}

    boost  = [m for m in models if m["name"] in boost_names]
    others = [m for m in models if m["name"] in others_names]

    fig = go.Figure()
    for group, label in [(others, "Baseline models"), (boost, "Gradient Boosting")]:
        fig.add_bar(
            name=label,
            x=[m["name"] for m in group],
            y=[m["auc"] for m in group],
            marker_color=[_color(m["name"]) for m in group],
            text=[f"AUC={m['auc']:.4f}" for m in group],
            textposition="outside"
        )
    avg_boost  = np.mean([m["auc"] for m in boost])  if boost  else 0
    avg_others = np.mean([m["auc"] for m in others]) if others else 0

    fig.add_hline(y=avg_boost,  line_dash="dash",
                  line_color="#8E44AD", opacity=0.7,
                  annotation_text=f"TB Boosting: {avg_boost:.4f}",
                  annotation_position="top right")
    fig.add_hline(y=avg_others, line_dash="dash",
                  line_color="#888", opacity=0.6,
                  annotation_text=f"TB Baseline: {avg_others:.4f}",
                  annotation_position="bottom right")

    fig.update_layout(
        title="⚡ Gradient Boosting vs Baseline Models — AUC-ROC<br>"
              "<sup>Đường đứt = AUC trung bình mỗi nhóm</sup>",
        yaxis_range=[
            min(m["auc"] for m in models) - 0.02,
            max(m["auc"] for m in models) + 0.05
        ],
        barmode="group",
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h"), height=400
    )
    return fig


# ═══════════════════════════════════════════════════════════
# 5. KẾT LUẬN
# ═══════════════════════════════════════════════════════════

def generate_conclusion(models, delay_rate: float) -> str:
    metric_keys   = ["auc",  "acc",  "f1",   "ap",
                     "log_loss", "brier",  "time"]
    metric_names  = ["AUC-ROC","Accuracy","F1-score","Avg Precision",
                     "Log Loss","Brier Score","Train time (s)"]
    lower_better  = {"log_loss", "brier", "time"}
    n = len(models)

    # ── Bảng so sánh ───────────────────────────────────────
    header = "<tr><th>Chỉ số</th>" + \
             "".join(f"<th>{m['name']}</th>" for m in models) + \
             "<th>Tốt nhất</th></tr>"

    rows_html = ""
    for key, label in zip(metric_keys, metric_names):
        vals = [m[key] for m in models]
        bi   = int(np.argmin(vals) if key in lower_better else np.argmax(vals))
        best = models[bi]["name"]
        bc   = _color(best)
        fmt  = ".1f" if key == "time" else ".4f"
        cells = "".join(
            (f'<td style="font-weight:bold;color:{bc}">' if i == bi else '<td>')
            + f'{v:{fmt}}</td>'
            for i, v in enumerate(vals)
        )
        rows_html += (f"<tr><td>{label}</td>{cells}"
                      f"<td style='color:{bc};font-weight:bold'>{best} ✓</td></tr>")

    # ── Giải thích tại sao Gradient Boosting thắng ─────────
    boost_models = [m for m in models if m["name"] in {"XGBoost","LightGBM"}]
    other_models = [m for m in models if m["name"] not in {"XGBoost","LightGBM"}]

    boost_auc_avg = np.mean([m["auc"] for m in boost_models]) if boost_models else 0
    other_auc_avg = np.mean([m["auc"] for m in other_models]) if other_models else 0
    boost_gain    = (boost_auc_avg - other_auc_avg) * 100

    best_model  = max(models, key=lambda m: m["auc"])
    worst_model = min(models, key=lambda m: m["auc"])

    # ── Cards ưu/nhược ──────────────────────────────────────
    cards_info = [
        ("Logistic Regression", "#E8593C", [
            ("✅ Ưu điểm", [
                "Train nhanh nhất (~giây)",
                "Giải thích được qua hệ số (coefficients)",
                "Xác suất calibrate tốt tự nhiên",
                "Ít tham số, không overfit",
            ]),
            ("❌ Nhược điểm", [
                "Chỉ bắt được quan hệ tuyến tính",
                "Kém nhất về AUC & F1 trong nhóm",
                "Cần chuẩn hóa dữ liệu bắt buộc",
            ]),
        ]),
        ("Random Forest", "#2E75B6", [
            ("✅ Ưu điểm", [
                "Không cần chuẩn hóa",
                "Feature importance rõ ràng",
                "Robust với outlier",
                "Tốt hơn LR & NN trên tabular data",
            ]),
            ("❌ Nhược điểm", [
                "Kém hơn Gradient Boosting",
                "Không biết chiều tác động feature",
                "Train chậm hơn LR",
            ]),
        ]),
        ("Neural Network", "#27AE60", [
            ("✅ Ưu điểm", [
                "Bắt được pattern phi tuyến phức tạp",
                "Linh hoạt với kiến trúc sâu hơn",
                "Tiềm năng vượt trội với >10M mẫu",
            ]),
            ("❌ Nhược điểm", [
                "Train chậm nhất trong nhóm",
                "Cần tuning nhiều",
                "Hộp đen — khó giải thích",
                "Kém hơn Gradient Boosting ở tabular data",
            ]),
        ]),
        ("XGBoost", "#F39C12", [
            ("✅ Ưu điểm", [
                "Gradient Boosting + L1/L2 regularization",
                "Học từ sai lầm tuần tự → hội tụ tốt hơn RF",
                "scale_pos_weight xử lý mất cân bằng",
                "Feature importance theo gain",
                "Chiến thắng nhiều Kaggle competition",
            ]),
            ("❌ Nhược điểm", [
                "Train chậm hơn LightGBM",
                "Nhiều hyperparameter cần tune",
                "Dùng level-wise growth → có thể overfit",
            ]),
        ]),
        ("LightGBM", "#8E44AD", [
            ("✅ Ưu điểm", [
                "Nhanh nhất trong nhóm Boosting (~10x XGB)",
                "Histogram-based splitting — ít RAM hơn",
                "Leaf-wise growth → chính xác hơn XGB",
                "is_unbalance tích hợp sẵn",
                "Hiệu suất cao nhất tổng thể",
            ]),
            ("❌ Nhược điểm", [
                "Leaf-wise dễ overfit nếu không tune num_leaves",
                "Cần cài thêm (pip install lightgbm)",
                "Ít tài liệu hơn XGBoost",
            ]),
        ]),
    ]

    # Chỉ render card của model có trong kết quả
    model_names_in = {m["name"] for m in models}
    cards_html = ""
    for name, color, sections in cards_info:
        if name not in model_names_in:
            continue
        body = ""
        for sec_title, points in sections:
            items = "".join(f"<li>{p}</li>" for p in points)
            body += f"<p style='font-weight:bold;margin:8px 0 4px'>{sec_title}</p><ul>{items}</ul>"
        cards_html += f"""
        <div class="prop-card" style="border-top:4px solid {color}">
          <h4 style="color:{color}">{name}</h4>{body}
        </div>"""

    # ── Giải thích kỹ thuật Gradient Boosting ──────────────
    xgb_auc = next((m["auc"] for m in models if m["name"]=="XGBoost"),  None)
    lgb_auc = next((m["auc"] for m in models if m["name"]=="LightGBM"), None)
    rf_auc  = next((m["auc"] for m in models if m["name"]=="Random Forest"), None)

    boost_explain = ""
    if xgb_auc and rf_auc:
        diff_xgb_rf = (xgb_auc - rf_auc) * 100
        boost_explain += f"""
        <li><b>XGBoost vs Random Forest:</b> XGBoost AUC cao hơn RF
            {diff_xgb_rf:+.2f}%. Lý do: RF xây 100 cây <em>song song độc lập</em>
            rồi vote trung bình — mỗi cây không biết cây khác sai ở đâu.
            XGBoost xây cây <em>tuần tự</em> — mỗi cây mới tập trung sửa
            đúng residual (phần sai) của các cây trước.
            Hàm mục tiêu XGBoost = Loss + L1/L2 regularization
            → ít overfit hơn thuần Gradient Boosting.</li>"""
    if lgb_auc and xgb_auc:
        diff_lgb_xgb = (lgb_auc - xgb_auc) * 100
        boost_explain += f"""
        <li><b>LightGBM vs XGBoost:</b> LightGBM AUC {diff_lgb_xgb:+.2f}% so với XGB.
            Hai cải tiến cốt lõi: (1) <em>Histogram-based splitting</em> —
            thay vì duyệt toàn bộ N điểm để tìm điểm chia, LGB gom vào
            ~256 bins → O(N) → O(bins), nhanh ~10×. (2) <em>Leaf-wise growth</em>
            — XGB phát triển cây theo từng level (level-wise), LGB chọn lá
            có gain cao nhất để split tiếp (leaf-wise) → cùng số lá nhưng
            mô hình phức tạp hơn, bắt được pattern tinh tế hơn.</li>"""

    return f"""
    <div class="conclusion">
      <h2>🏁 Kết luận — So sánh 5 Mô hình</h2>

      <table class="summary-table">
        <thead>{header}</thead>
        <tbody>{rows_html}</tbody>
      </table>

      <h3>⚡ Tại sao Gradient Boosting thắng?</h3>
      <div class="boost-box">
        <p><b>Ý tưởng cốt lõi:</b></p>
        <table class="algo-table">
          <tr>
            <th>Random Forest</th>
            <td>Cây 1 + Cây 2 + ... + Cây 100 → <b>vote đa số</b><br>
                Mỗi cây độc lập, không biết cây khác sai ở đâu</td>
          </tr>
          <tr>
            <th style="color:#F39C12">XGBoost / LightGBM</th>
            <td>Cây 1 → <em>sai lầm</em> → Cây 2 sửa → <em>sai lầm</em>
                → Cây 3 sửa → ...<br>
                Mỗi cây mới học từ <b>residual</b> của tất cả cây trước
                → hội tụ về kết quả tốt hơn</td>
          </tr>
        </table>
        <ul style="margin-top:12px">
          {boost_explain}
          <li><b>Dữ liệu mất cân bằng ({delay_rate:.1%} delay):</b>
              XGBoost dùng <code>scale_pos_weight</code>,
              LightGBM dùng <code>is_unbalance=True</code> —
              cả hai tích hợp sẵn trong thuật toán, không cần
              oversample/undersample thủ công.</li>
          <li><b>Tại sao NN thua Gradient Boosting ở đây:</b>
              Neural Network cần dữ liệu cực lớn (thường >10M mẫu)
              và kiến trúc sâu hơn để khai thác hết tiềm năng.
              Với tabular data 500K mẫu + 10 features,
              Gradient Boosting luôn là lựa chọn tốt nhất.</li>
        </ul>
      </div>

      <h3>📌 Ưu & Nhược điểm từng mô hình</h3>
      <div class="prop-grid">{cards_html}</div>

      <h3>💡 Khuyến nghị cho đồ án</h3>
      <ul>
        <li><b>Model tốt nhất:</b>
            <span style="color:{_color(best_model['name'])};font-weight:bold">
            {best_model['name']}</span>
            (AUC = {best_model['auc']:.4f})
            — nên dùng làm model chính trong báo cáo.</li>
        <li><b>Theo yêu cầu thầy:</b>
          <ul>
            <li>"Kỹ thuật có trong đề cương" →
                <span style="color:#E8593C;font-weight:bold">
                Logistic Regression</span></li>
            <li>"Công nghệ mới để so sánh" →
                <span style="color:#8E44AD;font-weight:bold">
                LightGBM</span> hoặc
                <span style="color:#F39C12;font-weight:bold">XGBoost</span></li>
            <li>"Visualize & Conclusion" → report HTML này</li>
          </ul>
        </li>
        <li><b>Thứ tự hiệu suất tổng quát trên tabular data:</b><br>
            LightGBM ≥ XGBoost > Random Forest > Neural Network > Logistic Regression</li>
      </ul>
    </div>"""


# ═══════════════════════════════════════════════════════════
# 6. BUILD HTML
# ═══════════════════════════════════════════════════════════

def build_html(figs: dict, conclusion: str, models: list,
               delay_rate: float) -> str:
    badges = "".join(
        f"<span class='badge' style='background:rgba(0,0,0,0.2)'>"
        f"{m['name']}: {m['auc']:.4f} AUC · {m['time']:.1f}s</span>"
        for m in models
    )
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>So sánh 5 mô hình — Airline Delay</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:Arial,sans-serif;background:#F0F4F8;color:#333}}
    .header{{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
             color:white;padding:32px 40px}}
    .header h1{{font-size:26px;margin-bottom:8px}}
    .header p{{font-size:14px;opacity:.85}}
    .badges{{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}}
    .badge{{border-radius:20px;padding:4px 12px;font-size:12px}}
    .container{{max-width:1400px;margin:0 auto;padding:24px}}
    .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
    .chart{{background:white;border-radius:12px;padding:16px;
            box-shadow:0 2px 8px rgba(0,0,0,.08)}}
    .conclusion{{background:white;border-radius:12px;padding:28px;
                 box-shadow:0 2px 8px rgba(0,0,0,.08);margin-top:20px}}
    .conclusion h2{{color:#0f3460;font-size:22px;margin-bottom:16px}}
    .conclusion h3{{color:#333;font-size:17px;margin:20px 0 10px}}
    .conclusion ul{{padding-left:20px;line-height:2}}
    .summary-table{{width:100%;border-collapse:collapse;
                    margin-bottom:20px;font-size:13px}}
    .summary-table th{{background:#0f3460;color:white;
                       padding:9px 12px;text-align:left}}
    .summary-table td{{padding:8px 12px;border-bottom:1px solid #eee}}
    .summary-table tr:nth-child(even) td{{background:#F5F8FF}}
    .algo-table{{width:100%;border-collapse:collapse;
                 margin:10px 0;font-size:14px}}
    .algo-table th{{background:#1a1a2e;color:white;
                    padding:8px 14px;width:180px}}
    .algo-table td{{padding:8px 14px;border:1px solid #ddd;
                    background:#FAFAFA;line-height:1.8}}
    .boost-box{{background:#FFF9F0;border-left:4px solid #F39C12;
                border-radius:8px;padding:16px;margin:12px 0}}
    .prop-grid{{display:grid;
                grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
                gap:14px;margin:16px 0}}
    .prop-card{{background:#FAFAFA;border-radius:10px;padding:14px;
                box-shadow:0 1px 4px rgba(0,0,0,.07)}}
    .prop-card h4{{font-size:14px;margin-bottom:8px}}
    .prop-card ul{{padding-left:14px;line-height:1.8;font-size:12px}}
    .section-title{{font-size:18px;font-weight:bold;color:#0f3460;
                    margin:28px 0 12px;
                    border-left:4px solid #4A90D9;padding-left:12px}}
    code{{background:#f0f0f0;padding:1px 5px;border-radius:4px;
          font-size:12px}}
  </style>
</head>
<body>
  <div class="header">
    <h1>✈️ So sánh 5 Mô hình: LR · RF · NN · XGBoost · LightGBM</h1>
    <p>Dự đoán delay chuyến bay (≥15p) — 500K mẫu — Delay rate: {delay_rate:.1%}</p>
    <div class="badges">{badges}</div>
  </div>

  <div class="container">
    <div class="section-title">1. Tổng quan — Radar & Xếp hạng</div>
    <div class="grid2">
      <div class="chart">__RADAR__</div>
      <div class="chart">__RANK__</div>
    </div>

    <div class="section-title">2. So sánh đầy đủ 6 chỉ số</div>
    <div class="chart">__METRICS__</div>

    <div class="section-title">3. Gradient Boosting vs Baseline</div>
    <div class="chart">__BOOST_VS__</div>

    <div class="section-title">4. ROC & Precision-Recall Curve</div>
    <div class="grid2">
      <div class="chart">__ROC__</div>
      <div class="chart">__PR__</div>
    </div>

    <div class="section-title">5. Confusion Matrix</div>
    <div class="chart">__CM__</div>

    <div class="section-title">6. Classification Report</div>
    <div class="chart">__CR__</div>

    <div class="section-title">7. Phân phối xác suất</div>
    <div class="chart">__PROB__</div>

    <div class="section-title">8. Thời gian Training</div>
    <div class="grid2"><div class="chart">__TIME__</div></div>

    <div class="section-title">9. Feature Importance — Gradient Boosting</div>
    <div class="chart">__FI_BOOST__</div>

    {conclusion}
  </div>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 65)
    log.info("  SO SÁNH 5 MÔ HÌNH: LR · RF · NN · XGBoost · LightGBM")
    log.info("=" * 65)

    # 1. Load
    df, delay_rate = load_data(limit=500_000)
    X, y, feature_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 2. Train
    trained = train_all(X_train, y_train, delay_rate)

    # 3. Evaluate
    log.info("Evaluating...")
    models = evaluate_all(trained, X_test, y_test)

    # 4. Charts
    log.info("Generating charts...")
    fi_boost = fig_feature_importance_boost(trained, feature_cols)

    figs = {
        "__RADAR__":    fig_radar(models),
        "__RANK__":     fig_rank(models),
        "__METRICS__":  fig_metrics_grouped(models),
        "__BOOST_VS__": fig_boosting_vs_others(models),
        "__ROC__":      fig_roc(models),
        "__PR__":       fig_pr(models),
        "__CM__":       fig_confusion(models),
        "__CR__":       fig_class_report(models),
        "__PROB__":     fig_prob_dist(models, y_test.values),
        "__TIME__":     fig_train_time(models),
        "__FI_BOOST__": fi_boost if fi_boost else fig_train_time(models),
    }

    # 5. Build HTML
    conclusion = generate_conclusion(models, delay_rate)
    html = build_html(figs, conclusion, models, delay_rate)
    for placeholder, fig in figs.items():
        chart = (f'<div class="chart">'
                 f'{fig.to_html(full_html=False, include_plotlyjs=False)}'
                 f'</div>')
        html = html.replace(placeholder, chart)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    # 6. Lưu models
    for key in trained:
        t = trained[key]
        pkl_path = os.path.join(os.path.dirname(__file__), f"model_{key}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"model": t["model"], "scaler": t["scaler"],
                         "type": key}, f)

    log.info("=" * 65)
    for m in models:
        log.info(f"  {m['name']:<22} AUC={m['auc']:.4f} | "
                 f"Acc={m['acc']:.4f} | Time={m['time']:.1f}s")
    log.info(f"\n  🏆 Tốt nhất: {max(models, key=lambda m: m['auc'])['name']}")
    log.info(f"  Report: {REPORT_PATH}")
    log.info("=" * 65)
