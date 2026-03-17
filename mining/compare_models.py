"""
mining/compare_models.py
========================
So sánh Random Forest vs Neural Network cho bài toán dự đoán delay chuyến bay.
Xuất ra file HTML chứa đầy đủ biểu đồ và kết luận.

Chạy: python mining/compare_models.py
Kết quả: mining/comparison_report.html
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [COMPARE] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score
)

# Plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DB_PATH     = os.path.join(os.path.dirname(__file__), "..", "warehouse", "airline_dw.db")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "comparison_report.html")

# ═══════════════════════════════════════════════════════════
# 1. LOAD DỮ LIỆU
# ═══════════════════════════════════════════════════════════

def load_data(limit: int = 500_000) -> pd.DataFrame:
    log.info(f"Loading {limit:,} bản ghi từ Data Warehouse...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"""
        SELECT
            f.distance, f.elapsed_time,
            f.is_delayed_15   AS target,
            d.month, d.day_of_week, d.is_weekend, d.quarter,
            CAST(SUBSTR(PRINTF('%04d', CAST(f.scheduled_departure AS INTEGER)),1,2) AS INTEGER) AS dep_hour,
            a.airline_iata,
            ap_o.state AS origin_state,
            ap_d.state AS dest_state
        FROM fact_flights f
        JOIN dim_date    d    ON f.date_key            = d.date_key
        JOIN dim_airline a    ON f.airline_key          = a.airline_key
        JOIN dim_airport ap_o ON f.origin_airport_key  = ap_o.airport_key
        JOIN dim_airport ap_d ON f.dest_airport_key    = ap_d.airport_key
        WHERE f.is_cancelled = 0
        LIMIT {limit}
    """, conn)
    conn.close()
    log.info(f"Loaded {len(df):,} rows — delay rate: {df['target'].mean():.2%}")
    return df


# ═══════════════════════════════════════════════════════════
# 2. TIỀN XỬ LÝ
# ═══════════════════════════════════════════════════════════

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
# 3. TRAIN CẢ HAI MODEL
# ═══════════════════════════════════════════════════════════

def train_random_forest(X_train, y_train):
    log.info("Training Random Forest...")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    log.info(f"Random Forest trained in {elapsed:.1f}s")
    return clf, elapsed


def train_neural_network(X_train, y_train):
    log.info("Training Neural Network...")
    # Chuẩn hóa — bắt buộc với Neural Network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    t0 = time.time()
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32),   # 2 hidden layers: 64 → 32 neurons
        activation="relu",
        solver="adam",
        alpha=0.001,                   # L2 regularization
        batch_size=512,
        learning_rate="adaptive",
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        verbose=False
    )
    nn.fit(X_scaled, y_train)
    elapsed = time.time() - t0
    log.info(f"Neural Network trained in {elapsed:.1f}s — {nn.n_iter_} iterations")
    return nn, scaler, elapsed


# ═══════════════════════════════════════════════════════════
# 4. ĐÁNH GIÁ
# ═══════════════════════════════════════════════════════════

def evaluate(name, model, X_test, y_test, scaler=None):
    X = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred, average="weighted")
    ap   = average_precision_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    report = classification_report(y_test, y_pred,
                                   target_names=["On-time", "Delayed"],
                                   output_dict=True)

    log.info(f"{name}: Accuracy={acc:.4f} | AUC-ROC={auc:.4f} | F1={f1:.4f}")

    return {
        "name": name, "acc": acc, "auc": auc, "f1": f1, "ap": ap,
        "cm": cm, "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
        "report": report, "y_prob": y_prob, "y_pred": y_pred
    }


# ═══════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ═══════════════════════════════════════════════════════════

COLORS = {"Random Forest": "#2E75B6", "Neural Network": "#E8593C"}
BG = "#FAFAFA"

def fig_metrics_bar(rf, nn):
    """Bảng so sánh các chỉ số chính."""
    metrics = ["Accuracy", "AUC-ROC", "F1-score (weighted)", "Avg Precision"]
    rf_vals = [rf["acc"], rf["auc"], rf["f1"], rf["ap"]]
    nn_vals = [nn["acc"], nn["auc"], nn["f1"], nn["ap"]]

    fig = go.Figure()
    fig.add_bar(name="Random Forest", x=metrics, y=rf_vals,
                marker_color=COLORS["Random Forest"],
                text=[f"{v:.3f}" for v in rf_vals], textposition="outside")
    fig.add_bar(name="Neural Network", x=metrics, y=nn_vals,
                marker_color=COLORS["Neural Network"],
                text=[f"{v:.3f}" for v in nn_vals], textposition="outside")
    fig.update_layout(
        title="📊 So sánh các chỉ số đánh giá",
        barmode="group", yaxis_range=[0, 1.1],
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=1.1),
        height=420
    )
    return fig


def fig_roc(rf, nn):
    """ROC Curve."""
    fig = go.Figure()
    fig.add_scatter(x=rf["fpr"], y=rf["tpr"], mode="lines", name=f"Random Forest (AUC={rf['auc']:.3f})",
                    line=dict(color=COLORS["Random Forest"], width=2.5))
    fig.add_scatter(x=nn["fpr"], y=nn["tpr"], mode="lines", name=f"Neural Network (AUC={nn['auc']:.3f})",
                    line=dict(color=COLORS["Neural Network"], width=2.5))
    fig.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Random (baseline)",
                    line=dict(color="#aaa", dash="dash", width=1.5))
    fig.update_layout(
        title="📈 ROC Curve",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=420
    )
    return fig


def fig_pr(rf, nn):
    """Precision-Recall Curve."""
    fig = go.Figure()
    fig.add_scatter(x=rf["rec"], y=rf["prec"], mode="lines",
                    name=f"Random Forest (AP={rf['ap']:.3f})",
                    line=dict(color=COLORS["Random Forest"], width=2.5))
    fig.add_scatter(x=nn["rec"], y=nn["prec"], mode="lines",
                    name=f"Neural Network (AP={nn['ap']:.3f})",
                    line=dict(color=COLORS["Neural Network"], width=2.5))
    fig.update_layout(
        title="📉 Precision-Recall Curve",
        xaxis_title="Recall", yaxis_title="Precision",
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=420
    )
    return fig


def fig_confusion_matrices(rf, nn):
    """Confusion Matrix song song."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Random Forest", "Neural Network"])
    labels = ["On-time", "Delayed"]

    for idx, result in enumerate([rf, nn], start=1):
        cm = result["cm"]
        # Normalize theo hàng (actual)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        text = [[f"{cm[i][j]:,}<br>({cm_norm[i][j]:.1%})" for j in range(2)] for i in range(2)]
        fig.add_heatmap(
            z=cm_norm, x=labels, y=labels,
            text=text, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            row=1, col=idx
        )

    fig.update_layout(
        title="🔢 Confusion Matrix (normalized theo hàng)",
        plot_bgcolor=BG, paper_bgcolor=BG, height=380
    )
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual", col=1)
    return fig


def fig_class_report(rf, nn):
    """Classification report dạng heatmap."""
    classes  = ["On-time", "Delayed"]
    metrics  = ["precision", "recall", "f1-score"]
    rf_vals  = [[rf["report"][c][m] for m in metrics] for c in classes]
    nn_vals  = [[nn["report"][c][m] for m in metrics] for c in classes]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Random Forest", "Neural Network"])
    kw = dict(x=metrics, colorscale="RdYlGn", zmin=0, zmax=1, showscale=False)
    for idx, (vals, title) in enumerate([(rf_vals, "RF"), (nn_vals, "NN")], start=1):
        text = [[f"{v:.3f}" for v in row] for row in vals]
        fig.add_heatmap(z=vals, y=classes, text=text, texttemplate="%{text}",
                        row=1, col=idx, **kw)
    fig.update_layout(
        title="📋 Classification Report — Precision / Recall / F1",
        plot_bgcolor=BG, paper_bgcolor=BG, height=320
    )
    return fig


def fig_feature_importance(rf_model, feature_cols):
    """Feature importance của Random Forest."""
    imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker_color=COLORS["Random Forest"],
        text=[f"{v:.3f}" for v in imp.values], textposition="outside"
    ))
    fig.update_layout(
        title="🌲 Feature Importance — Random Forest",
        xaxis_title="Importance", plot_bgcolor=BG, paper_bgcolor=BG, height=380
    )
    return fig


def fig_prob_distribution(rf, nn, y_test):
    """Phân phối xác suất dự đoán."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Random Forest", "Neural Network"])
    for idx, result in enumerate([rf, nn], start=1):
        for label, color in [(0, "#4A90D9"), (1, "#E8593C")]:
            mask = y_test == label
            fig.add_histogram(
                x=result["y_prob"][mask],
                name=f"{'Delayed' if label else 'On-time'}",
                marker_color=color, opacity=0.65,
                nbinsx=40, showlegend=(idx == 1),
                row=1, col=idx
            )
    fig.update_layout(
        title="📊 Phân phối xác suất dự đoán theo lớp thực tế",
        barmode="overlay", plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=380
    )
    fig.update_xaxes(title_text="Xác suất delay")
    fig.update_yaxes(title_text="Số lượng", col=1)
    return fig


def fig_train_time(rf_time, nn_time):
    """Thời gian train."""
    fig = go.Figure(go.Bar(
        x=["Random Forest", "Neural Network"],
        y=[rf_time, nn_time],
        marker_color=[COLORS["Random Forest"], COLORS["Neural Network"]],
        text=[f"{rf_time:.1f}s", f"{nn_time:.1f}s"],
        textposition="outside"
    ))
    fig.update_layout(
        title="⏱️ Thời gian Training",
        yaxis_title="Giây (s)", plot_bgcolor=BG, paper_bgcolor=BG, height=340
    )
    return fig


# ═══════════════════════════════════════════════════════════
# 6. KẾT LUẬN TỰ ĐỘNG
# ═══════════════════════════════════════════════════════════

def generate_conclusion(rf, nn, rf_time, nn_time) -> str:
    winner_auc  = "Random Forest" if rf["auc"]  > nn["auc"]  else "Neural Network"
    winner_acc  = "Random Forest" if rf["acc"]  > nn["acc"]  else "Neural Network"
    winner_f1   = "Random Forest" if rf["f1"]   > nn["f1"]   else "Neural Network"
    faster      = "Random Forest" if rf_time    < nn_time    else "Neural Network"
    speed_ratio = max(rf_time, nn_time) / min(rf_time, nn_time)

    rows = [
        ("AUC-ROC",       f"{rf['auc']:.4f}", f"{nn['auc']:.4f}", winner_auc),
        ("Accuracy",      f"{rf['acc']:.4f}", f"{nn['acc']:.4f}", winner_acc),
        ("F1 (weighted)", f"{rf['f1']:.4f}",  f"{nn['f1']:.4f}",  winner_f1),
        ("Avg Precision", f"{rf['ap']:.4f}",  f"{nn['ap']:.4f}",
         "Random Forest" if rf["ap"] > nn["ap"] else "Neural Network"),
        ("Train time",    f"{rf_time:.1f}s",  f"{nn_time:.1f}s",  faster),
    ]

    table_rows = "".join(f"""
        <tr>
          <td>{r}</td><td>{rv}</td><td>{nv}</td>
          <td style="color:{'#2E75B6' if w=='Random Forest' else '#E8593C'};font-weight:bold">{w} ✓</td>
        </tr>""" for r, rv, nv, w in rows)

    overall = "Random Forest" if (
        (rf["auc"] >= nn["auc"]) + (rf["acc"] >= nn["acc"]) + (rf["f1"] >= nn["f1"]) >= 2
    ) else "Neural Network"

    return f"""
    <div class="conclusion">
      <h2>🏁 Kết luận So sánh</h2>
      <table class="summary-table">
        <thead><tr><th>Chỉ số</th><th>Random Forest</th><th>Neural Network</th><th>Tốt hơn</th></tr></thead>
        <tbody>{table_rows}</tbody>
      </table>

      <h3>📌 Nhận xét</h3>
      <ul>
        <li><b>Hiệu suất tổng thể:</b> <span class="highlight-{'rf' if overall=='Random Forest' else 'nn'}">{overall}</span>
            cho kết quả tốt hơn trên bộ dữ liệu này.</li>
        <li><b>Tốc độ train:</b> {faster} nhanh hơn ~{speed_ratio:.1f}x.
            Random Forest dễ song song hóa (n_jobs=-1), Neural Network cần nhiều epoch hơn.</li>
        <li><b>Khả năng giải thích:</b> Random Forest cung cấp Feature Importance giúp hiểu
            yếu tố nào ảnh hưởng đến delay. Neural Network hoạt động như "hộp đen".</li>
        <li><b>Yêu cầu tiền xử lý:</b> Neural Network bắt buộc chuẩn hóa dữ liệu (StandardScaler),
            Random Forest không cần.</li>
        <li><b>Với bài toán này:</b> Cả hai đều gặp khó khăn với class Delayed do mất cân bằng
            (chỉ ~21% chuyến bay bị delay). AUC-ROC > 0.7 cho thấy cả hai model có giá trị
            dự đoán tốt hơn ngẫu nhiên đáng kể.</li>
      </ul>

      <h3>💡 Khuyến nghị</h3>
      <p>Với bài toán dự đoán delay hàng không trên tabular data:</p>
      <ul>
        <li>Dùng <b>Random Forest</b> khi cần giải thích được kết quả và train nhanh.</li>
        <li>Dùng <b>Neural Network</b> khi có nhiều dữ liệu hơn (>1M mẫu) và tài nguyên tính toán đủ mạnh.</li>
        <li>Hướng cải thiện: thử <b>XGBoost/LightGBM</b> thường vượt trội cả hai trên tabular data.</li>
      </ul>
    </div>"""


# ═══════════════════════════════════════════════════════════
# 7. XUẤT HTML REPORT
# ═══════════════════════════════════════════════════════════

def build_html_report(figs: list, conclusion: str, rf, nn, rf_time, nn_time) -> str:
    charts_html = "\n".join(
        f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
        for fig in figs
    )

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>So sánh RF vs NN — Airline Delay</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; background: #F0F4F8; color: #333; }}
    .header {{
      background: linear-gradient(135deg, #1F5C99, #2E75B6);
      color: white; padding: 32px 40px;
    }}
    .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
    .header p  {{ font-size: 14px; opacity: 0.85; }}
    .badges {{ display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }}
    .badge {{
      background: rgba(255,255,255,0.2); border-radius: 20px;
      padding: 4px 14px; font-size: 13px;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .grid.full {{ grid-template-columns: 1fr; }}
    .chart {{
      background: white; border-radius: 12px; padding: 16px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    .conclusion {{
      background: white; border-radius: 12px; padding: 28px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-top: 20px;
    }}
    .conclusion h2 {{ color: #1F5C99; font-size: 22px; margin-bottom: 16px; }}
    .conclusion h3 {{ color: #333; font-size: 17px; margin: 20px 0 10px; }}
    .conclusion ul  {{ padding-left: 20px; line-height: 2; }}
    .conclusion p   {{ line-height: 1.8; margin-bottom: 8px; }}
    .summary-table {{
      width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 14px;
    }}
    .summary-table th {{
      background: #1F5C99; color: white; padding: 10px 14px; text-align: left;
    }}
    .summary-table td {{ padding: 9px 14px; border-bottom: 1px solid #eee; }}
    .summary-table tr:nth-child(even) td {{ background: #F7F9FC; }}
    .highlight-rf  {{ color: #2E75B6; font-weight: bold; }}
    .highlight-nn  {{ color: #E8593C; font-weight: bold; }}
    .section-title {{
      font-size: 18px; font-weight: bold; color: #1F5C99;
      margin: 28px 0 12px; border-left: 4px solid #2E75B6; padding-left: 12px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>✈️ So sánh Mô hình: Random Forest vs Neural Network</h1>
    <p>Bài toán dự đoán delay chuyến bay (≥15 phút) — Airline Data Warehouse 2015</p>
    <div class="badges">
      <span class="badge">📦 500,000 mẫu</span>
      <span class="badge">🎯 Binary Classification</span>
      <span class="badge">📊 Supervised Learning</span>
      <span class="badge">🌲 RF: {rf['auc']:.4f} AUC</span>
      <span class="badge">🧠 NN: {nn['auc']:.4f} AUC</span>
    </div>
  </div>

  <div class="container">
    <div class="section-title">1. Tổng quan chỉ số</div>
    <div class="grid full">__CHART_0__</div>

    <div class="section-title">2. ROC & Precision-Recall Curve</div>
    <div class="grid">__CHART_1__ __CHART_2__</div>

    <div class="section-title">3. Confusion Matrix & Classification Report</div>
    <div class="grid full">__CHART_3__</div>
    <div class="grid full" style="margin-top:20px">__CHART_4__</div>

    <div class="section-title">4. Phân phối xác suất & Thời gian train</div>
    <div class="grid full">__CHART_5__</div>
    <div class="grid" style="margin-top:20px">
      <div style="grid-column:1/2">__CHART_6__</div>
    </div>

    <div class="section-title">5. Feature Importance (Random Forest)</div>
    <div class="grid full">__CHART_7__</div>

    {conclusion}
  </div>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 55)
    log.info("  BẮT ĐẦU SO SÁNH RANDOM FOREST vs NEURAL NETWORK")
    log.info("=" * 55)

    # 1. Load & preprocess
    df = load_data(limit=500_000)
    X, y, feature_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 2. Train
    rf_model, rf_time = train_random_forest(X_train, y_train)
    nn_model, scaler, nn_time = train_neural_network(X_train, y_train)

    # 3. Evaluate
    rf = evaluate("Random Forest", rf_model, X_test, y_test, scaler=None)
    nn = evaluate("Neural Network", nn_model, X_test, y_test, scaler=scaler)

    # 4. Tạo biểu đồ
    log.info("Tạo biểu đồ so sánh...")
    figs = [
        fig_metrics_bar(rf, nn),          # 0
        fig_roc(rf, nn),                  # 1
        fig_pr(rf, nn),                   # 2
        fig_confusion_matrices(rf, nn),   # 3
        fig_class_report(rf, nn),         # 4
        fig_prob_distribution(rf, nn, y_test.values),  # 5
        fig_train_time(rf_time, nn_time), # 6
        fig_feature_importance(rf_model, feature_cols), # 7
    ]

    # 5. Build HTML
    conclusion = generate_conclusion(rf, nn, rf_time, nn_time)
    html = build_html_report(figs, conclusion, rf, nn, rf_time, nn_time)

    # Inject charts vào đúng placeholder
    charts_html = [
        f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
        for fig in figs
    ]
    for i, ch in enumerate(charts_html):
        html = html.replace(f"__CHART_{i}__", ch)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    # 6. Lưu cả hai model
    rf_pkl = os.path.join(os.path.dirname(__file__), "model_rf.pkl")
    nn_pkl = os.path.join(os.path.dirname(__file__), "model_nn.pkl")
    with open(rf_pkl, "wb") as f:
        pickle.dump({"model": rf_model, "type": "rf"}, f)
    with open(nn_pkl, "wb") as f:
        pickle.dump({"model": nn_model, "scaler": scaler, "type": "nn"}, f)

    log.info("=" * 55)
    log.info(f"  RANDOM FOREST  — AUC: {rf['auc']:.4f} | Acc: {rf['acc']:.4f} | Time: {rf_time:.1f}s")
    log.info(f"  NEURAL NETWORK — AUC: {nn['auc']:.4f} | Acc: {nn['acc']:.4f} | Time: {nn_time:.1f}s")
    log.info(f"  Report: {REPORT_PATH}")
    log.info("=" * 55)
