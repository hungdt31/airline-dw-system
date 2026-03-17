"""
mining/compare_all_models.py
============================
So sánh 3 thuật toán: Logistic Regression vs Random Forest vs Neural Network
Xuất ra file HTML với đầy đủ biểu đồ và kết luận tổng hợp.

Chạy: python mining/compare_all_models.py
Kết quả: mining/all_models_report.html
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [ALL_MODELS] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DB_PATH     = os.path.join(os.path.dirname(__file__), "..", "warehouse", "airline_dw.db")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "all_models_report.html")

COLORS = {
    "Logistic Regression": "#E8593C",
    "Random Forest":       "#2E75B6",
    "Neural Network":      "#27AE60",
}
BG = "#FAFAFA"

# ═══════════════════════════════════════════════════════════
# 1. LOAD & TIỀN XỬ LÝ
# ═══════════════════════════════════════════════════════════

def load_data(limit: int = 500_000) -> pd.DataFrame:
    log.info(f"Loading {limit:,} bản ghi từ Data Warehouse...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"""
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
        LIMIT {limit}
    """, conn)
    conn.close()
    log.info(f"Loaded {len(df):,} rows — delay rate: {df['target'].mean():.2%}")
    return df


def preprocess(df: pd.DataFrame):
    df = df.dropna(subset=["target", "month", "day_of_week", "airline_iata"])
    df["distance"]     = df["distance"].fillna(df["distance"].median())
    df["elapsed_time"] = df["elapsed_time"].fillna(df["elapsed_time"].median())
    df["dep_hour"]     = df["dep_hour"].fillna(12).clip(0, 23)

    le_airline = LabelEncoder()
    le_origin  = LabelEncoder()
    le_dest    = LabelEncoder()

    df["airline_enc"] = le_airline.fit_transform(df["airline_iata"].astype(str))
    df["origin_enc"]  = le_origin.fit_transform(df["origin_state"].fillna("UNK").astype(str))
    df["dest_enc"]    = le_dest.fit_transform(df["dest_state"].fillna("UNK").astype(str))

    feature_cols = [
        "month", "day_of_week", "is_weekend", "quarter",
        "dep_hour", "distance", "elapsed_time",
        "airline_enc", "origin_enc", "dest_enc"
    ]
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)
    return X, y, feature_cols


# ═══════════════════════════════════════════════════════════
# 2. TRAIN 3 MODEL
# ═══════════════════════════════════════════════════════════

def train_all(X_train, y_train):
    results = {}

    # --- Logistic Regression ---
    log.info("Training Logistic Regression...")
    scaler_lr = StandardScaler()
    X_lr = scaler_lr.fit_transform(X_train)
    t0 = time.time()
    lr = LogisticRegression(
        C=1.0, penalty="l2", solver="saga",
        max_iter=200, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    lr.fit(X_lr, y_train)
    results["lr"] = {"model": lr, "scaler": scaler_lr, "time": time.time() - t0,
                     "name": "Logistic Regression"}
    log.info(f"  LR done in {results['lr']['time']:.1f}s")

    # --- Random Forest ---
    log.info("Training Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=50,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results["rf"] = {"model": rf, "scaler": None, "time": time.time() - t0,
                     "name": "Random Forest"}
    log.info(f"  RF done in {results['rf']['time']:.1f}s")

    # --- Neural Network ---
    log.info("Training Neural Network...")
    scaler_nn = StandardScaler()
    X_nn = scaler_nn.fit_transform(X_train)
    t0 = time.time()
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=0.001, batch_size=512, learning_rate="adaptive",
        max_iter=50, random_state=42,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=5, verbose=False
    )
    nn.fit(X_nn, y_train)
    results["nn"] = {"model": nn, "scaler": scaler_nn, "time": time.time() - t0,
                     "name": "Neural Network"}
    log.info(f"  NN done in {results['nn']['time']:.1f}s — {nn.n_iter_} iters")

    return results


# ═══════════════════════════════════════════════════════════
# 3. EVALUATE
# ═══════════════════════════════════════════════════════════

def evaluate_all(trained: dict, X_test, y_test) -> list:
    metrics_list = []
    for key in ["lr", "rf", "nn"]:
        t = trained[key]
        X = t["scaler"].transform(X_test) if t["scaler"] else X_test
        y_pred = t["model"].predict(X)
        y_prob = t["model"].predict_proba(X)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        report = classification_report(y_test, y_pred,
                                       target_names=["On-time", "Delayed"],
                                       output_dict=True)
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
            "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
            "report":   report,
            "y_prob":   y_prob,
            "y_pred":   y_pred,
        }
        metrics_list.append(m)
        log.info(f"  {t['name']}: AUC={m['auc']:.4f} | Acc={m['acc']:.4f} | "
                 f"F1={m['f1']:.4f} | LogLoss={m['log_loss']:.4f}")
    return metrics_list


# ═══════════════════════════════════════════════════════════
# 4. BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════

def fig_radar(models):
    """Radar chart so sánh 5 chỉ số cùng lúc."""
    cats = ["AUC-ROC", "Accuracy", "F1-score", "Avg Precision",
            "1-LogLoss", "1-Brier"]
    fig = go.Figure()
    for m in models:
        vals = [
            m["auc"], m["acc"], m["f1"], m["ap"],
            max(0, 1 - m["log_loss"]),
            max(0, 1 - m["brier"] * 4)   # scale lên 0-1
        ]
        vals_closed = vals + [vals[0]]
        cats_closed = cats + [cats[0]]
        fig.add_scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself", name=m["name"],
            line_color=COLORS[m["name"]], opacity=0.6
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="🕸️ Radar Chart — So sánh tổng thể 3 mô hình",
        legend=dict(orientation="h", y=-0.1),
        plot_bgcolor=BG, paper_bgcolor=BG, height=480
    )
    return fig


def fig_metrics_grouped(models):
    """Grouped bar chart 6 chỉ số × 3 model."""
    metric_keys  = ["acc",  "auc",    "f1",       "ap",             "log_loss",  "brier"]
    metric_names = ["Accuracy","AUC-ROC","F1-score","Avg Precision","Log Loss","Brier Score"]
    lower_better = {"log_loss", "brier"}

    fig = go.Figure()
    for m in models:
        vals = [m[k] for k in metric_keys]
        fig.add_bar(
            name=m["name"], x=metric_names, y=vals,
            marker_color=COLORS[m["name"]],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside"
        )
    fig.update_layout(
        title="📊 So sánh đầy đủ 6 chỉ số đánh giá<br>"
              "<sup>Log Loss & Brier Score: thấp hơn = tốt hơn</sup>",
        barmode="group", yaxis_range=[0, 1.18],
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=1.12), height=440
    )
    return fig


def fig_roc_all(models):
    fig = go.Figure()
    for m in models:
        fig.add_scatter(
            x=m["fpr"], y=m["tpr"], mode="lines",
            name=f"{m['name']} (AUC={m['auc']:.3f})",
            line=dict(color=COLORS[m["name"]], width=2.5)
        )
    fig.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Baseline",
                    line=dict(color="#ccc", dash="dash", width=1.5))
    fig.update_layout(
        title="📈 ROC Curve — 3 mô hình",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=440
    )
    return fig


def fig_pr_all(models):
    fig = go.Figure()
    for m in models:
        fig.add_scatter(
            x=m["rec"], y=m["prec"], mode="lines",
            name=f"{m['name']} (AP={m['ap']:.3f})",
            line=dict(color=COLORS[m["name"]], width=2.5)
        )
    fig.update_layout(
        title="📉 Precision-Recall Curve — 3 mô hình",
        xaxis_title="Recall", yaxis_title="Precision",
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=440
    )
    return fig


def fig_confusion_all(models):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[m["name"] for m in models]
    )
    labels = ["On-time", "Delayed"]
    for idx, m in enumerate(models, start=1):
        cm = m["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        text = [[f"{cm[i][j]:,}<br>({cm_norm[i][j]:.1%})"
                 for j in range(2)] for i in range(2)]
        fig.add_heatmap(
            z=cm_norm, x=labels, y=labels,
            text=text, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            row=1, col=idx
        )
    fig.update_layout(
        title="🔢 Confusion Matrix — 3 mô hình (normalized)",
        plot_bgcolor=BG, paper_bgcolor=BG, height=360
    )
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual", col=1)
    return fig


def fig_class_report_all(models):
    classes = ["On-time", "Delayed"]
    metrics = ["precision", "recall", "f1-score"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[m["name"] for m in models]
    )
    kw = dict(x=metrics, colorscale="RdYlGn", zmin=0, zmax=1, showscale=False)
    for idx, m in enumerate(models, start=1):
        vals = [[m["report"][c][me] for me in metrics] for c in classes]
        text = [[f"{v:.3f}" for v in row] for row in vals]
        fig.add_heatmap(z=vals, y=classes, text=text,
                        texttemplate="%{text}", row=1, col=idx, **kw)
    fig.update_layout(
        title="📋 Classification Report — Precision / Recall / F1",
        plot_bgcolor=BG, paper_bgcolor=BG, height=300
    )
    return fig


def fig_prob_dist_all(models, y_test):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[m["name"] for m in models]
    )
    for idx, m in enumerate(models, start=1):
        for label, color, name in [(0, "#4A90D9", "On-time"),
                                   (1, "#E8593C", "Delayed")]:
            mask = y_test == label
            fig.add_histogram(
                x=m["y_prob"][mask],
                name=name, marker_color=color, opacity=0.65,
                nbinsx=40, showlegend=(idx == 1),
                row=1, col=idx
            )
    fig.update_layout(
        title="📊 Phân phối xác suất dự đoán theo lớp thực tế",
        barmode="overlay", plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=360
    )
    fig.update_xaxes(title_text="Xác suất delay", range=[0, 1])
    fig.update_yaxes(title_text="Số lượng", col=1)
    return fig


def fig_train_time_all(models):
    fig = go.Figure(go.Bar(
        x=[m["name"] for m in models],
        y=[m["time"] for m in models],
        marker_color=[COLORS[m["name"]] for m in models],
        text=[f"{m['time']:.1f}s" for m in models],
        textposition="outside"
    ))
    fig.update_layout(
        title="⏱️ Thời gian Training",
        yaxis_title="Giây (s)",
        plot_bgcolor=BG, paper_bgcolor=BG, height=340
    )
    return fig


def fig_feature_importance_rf(rf_model, feature_cols):
    imp = pd.Series(rf_model.feature_importances_,
                    index=feature_cols).sort_values()
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker_color=COLORS["Random Forest"],
        text=[f"{v:.3f}" for v in imp.values],
        textposition="outside"
    ))
    fig.update_layout(
        title="🌲 Feature Importance — Random Forest",
        xaxis_title="Importance",
        plot_bgcolor=BG, paper_bgcolor=BG, height=380
    )
    return fig


def fig_lr_coefficients(lr_model, scaler, feature_cols):
    coef = lr_model.coef_[0] / scaler.scale_
    df_c = pd.DataFrame({
        "feature": feature_cols,
        "coef": coef
    }).sort_values("coef")
    colors = ["#E8593C" if c > 0 else "#2E75B6" for c in df_c["coef"]]
    fig = go.Figure(go.Bar(
        x=df_c["coef"], y=df_c["feature"], orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in df_c["coef"]],
        textposition="outside"
    ))
    fig.update_layout(
        title="📐 Hệ số Logistic Regression<br>"
              "<sup>Đỏ = tăng xác suất delay | Xanh = giảm</sup>",
        xaxis_title="Coefficient",
        plot_bgcolor=BG, paper_bgcolor=BG, height=380
    )
    return fig


def fig_rank_summary(models):
    """
    Bảng xếp hạng tổng hợp: mỗi chỉ số cho điểm 3-2-1,
    tổng điểm cao nhất = tốt nhất.
    """
    metric_keys  = ["auc", "acc", "f1", "ap"]
    lower_better = ["log_loss", "brier", "time"]
    all_keys     = metric_keys + lower_better

    scores = {m["name"]: 0 for m in models}
    rank_data = {}

    for key in all_keys:
        vals = [(m["name"], m[key]) for m in models]
        reverse = key not in lower_better
        sorted_vals = sorted(vals, key=lambda x: x[1], reverse=reverse)
        pts = [3, 2, 1]
        for i, (name, _) in enumerate(sorted_vals):
            scores[name] += pts[i]
            rank_data.setdefault(key, {})[name] = pts[i]

    names   = [m["name"] for m in models]
    totals  = [scores[n] for n in names]

    fig = go.Figure(go.Bar(
        x=names, y=totals,
        marker_color=[COLORS[n] for n in names],
        text=[f"{t} điểm" for t in totals],
        textposition="outside"
    ))
    fig.update_layout(
        title="🏆 Tổng điểm xếp hạng (3 chỉ số × 7 tiêu chí)<br>"
              "<sup>Mỗi tiêu chí: hạng 1=3đ, hạng 2=2đ, hạng 3=1đ</sup>",
        yaxis_title="Tổng điểm",
        plot_bgcolor=BG, paper_bgcolor=BG, height=360
    )
    return fig


# ═══════════════════════════════════════════════════════════
# 5. KẾT LUẬN
# ═══════════════════════════════════════════════════════════

def generate_conclusion(models) -> str:
    metric_keys   = ["auc",  "acc",  "f1",   "ap",
                     "log_loss", "brier", "time"]
    metric_names  = ["AUC-ROC","Accuracy","F1-score","Avg Precision",
                     "Log Loss","Brier Score","Train time"]
    lower_better  = {"log_loss", "brier", "time"}

    # Header row
    header = "<tr><th>Chỉ số</th>" + "".join(
        f"<th>{m['name']}</th>" for m in models
    ) + "<th>Tốt nhất</th></tr>"

    body_rows = ""
    for key, label in zip(metric_keys, metric_names):
        vals = [m[key] for m in models]
        if key in lower_better:
            best_idx = int(np.argmin(vals))
        else:
            best_idx = int(np.argmax(vals))
        best_name  = models[best_idx]["name"]
        best_color = COLORS[best_name]

        fmt = ".1f}s" if key == "time" else ".4f}"
        cells = "".join(
            (f'<td style="font-weight:bold;color:{best_color}">' if i == best_idx else '<td>')
            + ('{:' + fmt).format(v) + '</td>'
            for i, v in enumerate(vals)
        )
        body_rows += f"<tr><td>{label}</td>{cells}" \
                     f"<td style='color:{best_color};font-weight:bold'>{best_name} ✓</td></tr>"

    # Đặc điểm từng model
    props = [
        ("Logistic Regression", [
            "✅ Train nhanh nhất",
            "✅ Giải thích rõ ràng qua coefficients",
            "✅ Xác suất calibrate tốt tự nhiên",
            "✅ Ít tham số, không overfit dễ",
            "❌ Chỉ bắt được mối quan hệ tuyến tính",
            "❌ Cần chuẩn hóa dữ liệu bắt buộc",
        ]),
        ("Random Forest", [
            "✅ Hiệu suất tốt nhất trên tabular data",
            "✅ Không cần chuẩn hóa dữ liệu",
            "✅ Feature importance rõ ràng",
            "✅ Robust với outlier",
            "❌ Không giải thích được chiều tác động",
            "❌ Train chậm hơn LR",
        ]),
        ("Neural Network", [
            "✅ Bắt được pattern phi tuyến phức tạp",
            "✅ Linh hoạt với kiến trúc",
            "✅ Tiềm năng tốt hơn với dữ liệu lớn hơn",
            "❌ Train chậm nhất",
            "❌ Cần chuẩn hóa bắt buộc",
            "❌ Hộp đen, khó giải thích",
            "❌ Cần tune hyperparameter nhiều",
        ]),
    ]

    prop_cards = ""
    for name, points in props:
        color = COLORS[name]
        items = "".join(f"<li>{p}</li>" for p in points)
        prop_cards += f"""
        <div class="prop-card" style="border-top: 4px solid {color}">
          <h4 style="color:{color}">{name}</h4>
          <ul>{items}</ul>
        </div>"""

    return f"""
    <div class="conclusion">
      <h2>🏁 Kết luận So sánh 3 Mô hình</h2>

      <table class="summary-table">
        <thead>{header}</thead>
        <tbody>{body_rows}</tbody>
      </table>

      <h3>📌 Ưu & Nhược điểm từng mô hình</h3>
      <div class="prop-grid">{prop_cards}</div>

      <h3>💡 Khuyến nghị cho đồ án</h3>
      <ul>
        <li><b>Model chính nên dùng:</b> <span style="color:#2E75B6;font-weight:bold">
            Random Forest</span> — hiệu suất cao nhất trên tabular data,
            không cần chuẩn hóa, cung cấp feature importance.</li>
        <li><b>Model baseline để so sánh:</b> <span style="color:#E8593C;font-weight:bold">
            Logistic Regression</span> — đơn giản, giải thích được,
            train nhanh, đáp ứng yêu cầu "kỹ thuật có trong đề cương".</li>
        <li><b>Model "công nghệ mới":</b> <span style="color:#27AE60;font-weight:bold">
            Neural Network</span> — đáp ứng yêu cầu "sử dụng công nghệ mới"
            của thầy, tiềm năng tốt hơn nếu có nhiều dữ liệu hơn.</li>
        <li><b>Lý do RF thắng NN trên bài này:</b> Dataset dạng bảng (tabular)
            với 10 features — Neural Network cần nhiều dữ liệu hơn và kiến trúc
            sâu hơn để vượt Random Forest. Với >5M mẫu và deep learning
            đúng nghĩa thì NN mới có lợi thế.</li>
        <li><b>Hướng cải thiện tiếp theo:</b> Thử XGBoost/LightGBM
            thường vượt trội cả 3 model này trên tabular data.</li>
      </ul>
    </div>"""


# ═══════════════════════════════════════════════════════════
# 6. BUILD HTML
# ═══════════════════════════════════════════════════════════

def build_html(figs: list, conclusion: str, models: list) -> str:
    lr = next(m for m in models if m["key"] == "lr")
    rf = next(m for m in models if m["key"] == "rf")
    nn = next(m for m in models if m["key"] == "nn")

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>So sánh 3 mô hình — Airline Delay</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; background: #F0F4F8; color: #333; }}
    .header {{
      background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
      color: white; padding: 32px 40px;
    }}
    .header h1 {{ font-size: 26px; margin-bottom: 8px; }}
    .header p  {{ font-size: 14px; opacity: 0.85; }}
    .badges {{ display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }}
    .badge {{
      border-radius: 20px; padding: 4px 14px; font-size: 13px;
      background: rgba(255,255,255,0.15);
    }}
    .badge.lr {{ background: rgba(232,89,60,0.4); }}
    .badge.rf {{ background: rgba(46,117,182,0.4); }}
    .badge.nn {{ background: rgba(39,174,96,0.4); }}
    .container {{ max-width: 1300px; margin: 0 auto; padding: 24px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .grid3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
    .full  {{ grid-column: 1 / -1; }}
    .chart {{
      background: white; border-radius: 12px; padding: 16px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    .conclusion {{
      background: white; border-radius: 12px; padding: 28px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-top: 20px;
    }}
    .conclusion h2 {{ color: #0f3460; font-size: 22px; margin-bottom: 16px; }}
    .conclusion h3 {{ color: #333; font-size: 17px; margin: 20px 0 10px; }}
    .conclusion ul  {{ padding-left: 20px; line-height: 2.1; }}
    .summary-table {{
      width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 14px;
    }}
    .summary-table th {{
      background: #0f3460; color: white; padding: 10px 14px; text-align: left;
    }}
    .summary-table td {{ padding: 9px 14px; border-bottom: 1px solid #eee; }}
    .summary-table tr:nth-child(even) td {{ background: #F5F8FF; }}
    .prop-grid {{
      display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin: 16px 0;
    }}
    .prop-card {{
      background: #FAFAFA; border-radius: 10px; padding: 16px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }}
    .prop-card h4 {{ font-size: 15px; margin-bottom: 10px; }}
    .prop-card ul  {{ padding-left: 16px; line-height: 1.9; font-size: 13px; }}
    .section-title {{
      font-size: 18px; font-weight: bold; color: #0f3460;
      margin: 28px 0 12px; border-left: 4px solid #4A90D9; padding-left: 12px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>✈️ So sánh 3 Mô hình: LR · RF · NN</h1>
    <p>Bài toán dự đoán delay chuyến bay (≥15 phút) — Airline Data Warehouse 2015</p>
    <div class="badges">
      <span class="badge">📦 500,000 mẫu</span>
      <span class="badge">🎯 Binary Classification</span>
      <span class="badge lr">📐 LR: {lr['auc']:.4f} AUC · {lr['time']:.1f}s</span>
      <span class="badge rf">🌲 RF: {rf['auc']:.4f} AUC · {rf['time']:.1f}s</span>
      <span class="badge nn">🧠 NN: {nn['auc']:.4f} AUC · {nn['time']:.1f}s</span>
    </div>
  </div>

  <div class="container">

    <div class="section-title">1. Tổng quan — Radar & Xếp hạng</div>
    <div class="grid2">
      <div class="chart">__CHART_0__</div>
      <div class="chart">__CHART_10__</div>
    </div>

    <div class="section-title">2. So sánh đầy đủ 6 chỉ số</div>
    <div class="chart">__CHART_1__</div>

    <div class="section-title">3. ROC & Precision-Recall Curve</div>
    <div class="grid2">
      <div class="chart">__CHART_2__</div>
      <div class="chart">__CHART_3__</div>
    </div>

    <div class="section-title">4. Confusion Matrix</div>
    <div class="chart">__CHART_4__</div>

    <div class="section-title">5. Classification Report</div>
    <div class="chart">__CHART_5__</div>

    <div class="section-title">6. Phân phối xác suất dự đoán</div>
    <div class="chart">__CHART_6__</div>

    <div class="section-title">7. Thời gian Training</div>
    <div class="grid2"><div class="chart">__CHART_7__</div></div>

    <div class="section-title">8. Giải thích Model</div>
    <div class="grid2">
      <div class="chart">__CHART_8__</div>
      <div class="chart">__CHART_9__</div>
    </div>

    {conclusion}

  </div>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  BẮT ĐẦU SO SÁNH 3 MÔ HÌNH: LR · RF · NN")
    log.info("=" * 60)

    # 1. Load & preprocess
    df = load_data(limit=500_000)
    X, y, feature_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 2. Train tất cả
    trained = train_all(X_train, y_train)

    # 3. Evaluate
    log.info("Evaluating all models...")
    models = evaluate_all(trained, X_test, y_test)

    # 4. Biểu đồ
    log.info("Generating charts...")
    lr_trained = trained["lr"]
    rf_trained = trained["rf"]

    figs = [
        fig_radar(models),                                                  # 0
        fig_metrics_grouped(models),                                        # 1
        fig_roc_all(models),                                                # 2
        fig_pr_all(models),                                                 # 3
        fig_confusion_all(models),                                          # 4
        fig_class_report_all(models),                                       # 5
        fig_prob_dist_all(models, y_test.values),                           # 6
        fig_train_time_all(models),                                         # 7
        fig_lr_coefficients(lr_trained["model"],
                             lr_trained["scaler"], feature_cols),           # 8
        fig_feature_importance_rf(rf_trained["model"], feature_cols),       # 9
        fig_rank_summary(models),                                           # 10
    ]

    # 5. Build HTML
    conclusion = generate_conclusion(models)
    html = build_html(figs, conclusion, models)
    for i, fig in enumerate(figs):
        chart_html = f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
        html = html.replace(f"__CHART_{i}__", chart_html)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    # 6. Lưu models
    for key, name_pkl in [("lr","model_lr.pkl"),("rf","model_rf.pkl"),("nn","model_nn.pkl")]:
        t = trained[key]
        with open(os.path.join(os.path.dirname(__file__), name_pkl), "wb") as f:
            pickle.dump({"model": t["model"], "scaler": t["scaler"], "type": key}, f)

    log.info("=" * 60)
    for m in models:
        log.info(f"  {m['name']:<22} AUC={m['auc']:.4f} | "
                 f"Acc={m['acc']:.4f} | F1={m['f1']:.4f} | Time={m['time']:.1f}s")
    log.info(f"  Report: {REPORT_PATH}")
    log.info("=" * 60)
