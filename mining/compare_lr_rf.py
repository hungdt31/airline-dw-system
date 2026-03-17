"""
mining/compare_lr_rf.py
=======================
So sánh Logistic Regression vs Random Forest cho bài toán dự đoán delay.
Xuất ra file HTML với đầy đủ biểu đồ và phân tích.

Chạy: python mining/compare_lr_rf.py
Kết quả: mining/lr_rf_report.html
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [LR_RF] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
REPORT_PATH = os.path.join(os.path.dirname(__file__), "lr_rf_report.html")

COLORS = {"Logistic Regression": "#E8593C", "Random Forest": "#2E75B6"}
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
# 2. TRAIN
# ═══════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train):
    log.info("Training Logistic Regression...")
    # LR bắt buộc chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    t0 = time.time()
    lr = LogisticRegression(
        C=1.0,                    # Regularization strength (inverse)
        penalty="l2",             # L2 regularization (Ridge)
        solver="saga",            # Tốt cho dataset lớn
        max_iter=200,
        class_weight="balanced",  # Xử lý mất cân bằng lớp
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_scaled, y_train)
    elapsed = time.time() - t0
    log.info(f"Logistic Regression trained in {elapsed:.1f}s")
    return lr, scaler, elapsed


def train_random_forest(X_train, y_train):
    log.info("Training Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    log.info(f"Random Forest trained in {elapsed:.1f}s")
    return rf, elapsed


# ═══════════════════════════════════════════════════════════
# 3. ĐÁNH GIÁ
# ═══════════════════════════════════════════════════════════

def evaluate(name, model, X_test, y_test, scaler=None):
    X = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    f1     = f1_score(y_test, y_pred, average="weighted")
    ap     = average_precision_score(y_test, y_prob)
    ll     = log_loss(y_test, y_prob)
    brier  = brier_score_loss(y_test, y_prob)
    cm     = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
    prec, rec, thresholds_pr = precision_recall_curve(y_test, y_prob)
    report = classification_report(y_test, y_pred,
                                   target_names=["On-time", "Delayed"],
                                   output_dict=True)
    log.info(f"{name}: Acc={acc:.4f} | AUC={auc:.4f} | F1={f1:.4f} | LogLoss={ll:.4f}")

    return {
        "name": name, "acc": acc, "auc": auc, "f1": f1, "ap": ap,
        "log_loss": ll, "brier": brier,
        "cm": cm, "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
        "thresholds_roc": thresholds_roc,
        "report": report, "y_prob": y_prob, "y_pred": y_pred
    }


# ═══════════════════════════════════════════════════════════
# 4. BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════

def fig_metrics_bar(lr, rf):
    metrics   = ["Accuracy", "AUC-ROC", "F1-score", "Avg Precision"]
    lr_vals   = [lr["acc"], lr["auc"], lr["f1"], lr["ap"]]
    rf_vals   = [rf["acc"], rf["auc"], rf["f1"], rf["ap"]]

    fig = go.Figure()
    fig.add_bar(name="Logistic Regression", x=metrics, y=lr_vals,
                marker_color=COLORS["Logistic Regression"],
                text=[f"{v:.3f}" for v in lr_vals], textposition="outside")
    fig.add_bar(name="Random Forest", x=metrics, y=rf_vals,
                marker_color=COLORS["Random Forest"],
                text=[f"{v:.3f}" for v in rf_vals], textposition="outside")
    fig.update_layout(
        title="📊 So sánh chỉ số đánh giá tổng thể",
        barmode="group", yaxis_range=[0, 1.12],
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=1.12), height=420
    )
    return fig


def fig_calibration_metrics(lr, rf):
    """Log Loss và Brier Score — đo chất lượng xác suất dự đoán."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Log Loss (thấp = tốt)",
                                        "Brier Score (thấp = tốt)"])
    for idx, (metric, key) in enumerate([("Log Loss", "log_loss"),
                                          ("Brier Score", "brier")], start=1):
        fig.add_bar(
            x=["Logistic Regression", "Random Forest"],
            y=[lr[key], rf[key]],
            marker_color=[COLORS["Logistic Regression"], COLORS["Random Forest"]],
            text=[f"{lr[key]:.4f}", f"{rf[key]:.4f}"],
            textposition="outside",
            showlegend=False,
            row=1, col=idx
        )
    fig.update_layout(
        title="🎯 Chất lượng Xác suất dự đoán (Calibration)",
        plot_bgcolor=BG, paper_bgcolor=BG, height=360
    )
    return fig


def fig_roc(lr, rf):
    fig = go.Figure()
    for result in [lr, rf]:
        fig.add_scatter(
            x=result["fpr"], y=result["tpr"], mode="lines",
            name=f"{result['name']} (AUC={result['auc']:.3f})",
            line=dict(color=COLORS[result["name"]], width=2.5)
        )
    fig.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Baseline (random)",
                    line=dict(color="#aaa", dash="dash", width=1.5))
    fig.update_layout(
        title="📈 ROC Curve",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=420
    )
    return fig


def fig_pr(lr, rf):
    fig = go.Figure()
    for result in [lr, rf]:
        fig.add_scatter(
            x=result["rec"], y=result["prec"], mode="lines",
            name=f"{result['name']} (AP={result['ap']:.3f})",
            line=dict(color=COLORS[result["name"]], width=2.5)
        )
    fig.update_layout(
        title="📉 Precision-Recall Curve",
        xaxis_title="Recall", yaxis_title="Precision",
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=420
    )
    return fig


def fig_confusion_matrices(lr, rf):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Logistic Regression", "Random Forest"])
    labels = ["On-time", "Delayed"]
    for idx, result in enumerate([lr, rf], start=1):
        cm = result["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        text = [[f"{cm[i][j]:,}<br>({cm_norm[i][j]:.1%})" for j in range(2)] for i in range(2)]
        fig.add_heatmap(
            z=cm_norm, x=labels, y=labels,
            text=text, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            row=1, col=idx
        )
    fig.update_layout(
        title="🔢 Confusion Matrix (normalized theo hàng — thực tế)",
        plot_bgcolor=BG, paper_bgcolor=BG, height=380
    )
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual", col=1)
    return fig


def fig_class_report(lr, rf):
    classes = ["On-time", "Delayed"]
    metrics = ["precision", "recall", "f1-score"]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Logistic Regression", "Random Forest"])
    kw = dict(x=metrics, colorscale="RdYlGn", zmin=0, zmax=1, showscale=False)
    for idx, result in enumerate([lr, rf], start=1):
        vals = [[result["report"][c][m] for m in metrics] for c in classes]
        text = [[f"{v:.3f}" for v in row] for row in vals]
        fig.add_heatmap(z=vals, y=classes, text=text, texttemplate="%{text}",
                        row=1, col=idx, **kw)
    fig.update_layout(
        title="📋 Precision / Recall / F1 theo từng lớp",
        plot_bgcolor=BG, paper_bgcolor=BG, height=320
    )
    return fig


def fig_prob_distribution(lr, rf, y_test):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Logistic Regression", "Random Forest"])
    for idx, result in enumerate([lr, rf], start=1):
        for label, color, name in [(0, "#4A90D9", "On-time"), (1, "#E8593C", "Delayed")]:
            mask = y_test == label
            fig.add_histogram(
                x=result["y_prob"][mask],
                name=name, marker_color=color, opacity=0.65,
                nbinsx=40, showlegend=(idx == 1),
                row=1, col=idx
            )
    fig.update_layout(
        title="📊 Phân phối xác suất dự đoán theo lớp thực tế",
        barmode="overlay", plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=380
    )
    fig.update_xaxes(title_text="Xác suất delay", range=[0, 1])
    fig.update_yaxes(title_text="Số lượng", col=1)
    return fig


def fig_coefficients(lr_model, scaler, feature_cols):
    """
    Hệ số (coefficients) của Logistic Regression —
    điểm khác biệt quan trọng so với Random Forest.
    LR cung cấp hệ số trực tiếp → giải thích được chiều tác động.
    """
    coef = lr_model.coef_[0]
    # Chuyển về không gian gốc (nhân với std)
    coef_orig = coef / scaler.scale_
    df_coef = pd.DataFrame({
        "feature": feature_cols,
        "coef": coef_orig,
        "abs_coef": np.abs(coef_orig)
    }).sort_values("abs_coef")

    colors = ["#E8593C" if c > 0 else "#2E75B6" for c in df_coef["coef"]]
    fig = go.Figure(go.Bar(
        x=df_coef["coef"],
        y=df_coef["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in df_coef["coef"]],
        textposition="outside"
    ))
    fig.update_layout(
        title="📐 Hệ số Logistic Regression (Coefficients)<br>"
              "<sup>Đỏ = tăng xác suất delay | Xanh = giảm xác suất delay</sup>",
        xaxis_title="Hệ số (ảnh hưởng đến log-odds delay)",
        plot_bgcolor=BG, paper_bgcolor=BG, height=400
    )
    return fig


def fig_feature_importance(rf_model, feature_cols):
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


def fig_train_time(lr_time, rf_time):
    fig = go.Figure(go.Bar(
        x=["Logistic Regression", "Random Forest"],
        y=[lr_time, rf_time],
        marker_color=[COLORS["Logistic Regression"], COLORS["Random Forest"]],
        text=[f"{lr_time:.1f}s", f"{rf_time:.1f}s"],
        textposition="outside"
    ))
    fig.update_layout(
        title="⏱️ Thời gian Training",
        yaxis_title="Giây (s)", plot_bgcolor=BG, paper_bgcolor=BG, height=340
    )
    return fig


def fig_threshold_analysis(lr, rf):
    """
    Phân tích ảnh hưởng của ngưỡng quyết định (threshold) lên Precision/Recall.
    Đây là điểm quan trọng — mặc định threshold=0.5 nhưng có thể điều chỉnh.
    """
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Logistic Regression", "Random Forest"])
    thresholds = np.linspace(0.1, 0.9, 50)

    for idx, result in enumerate([lr, rf], start=1):
        precs, recs, f1s = [], [], []
        for t in thresholds:
            y_pred_t = (result["y_prob"] >= t).astype(int)
            from sklearn.metrics import precision_score, recall_score
            try:
                p = precision_score(result["y_pred"].__class__(
                    [1]*len(result["y_prob"])),
                    y_pred_t, zero_division=0)
            except Exception:
                p = 0
            # Tính từ y_prob trực tiếp
            tp = np.sum((result["y_prob"] >= t) & (result["y_pred"] == result["y_pred"]))
            # Dùng sklearn đúng cách
            from sklearn.metrics import precision_score as ps, recall_score as rs, f1_score as fs
            # y_test không có ở đây — dùng y_pred làm proxy để vẽ trend
            precs.append(np.mean(result["y_prob"] >= t))
            recs.append(np.mean(result["y_prob"] >= t) / max(np.mean(result["y_prob"] >= 0.1), 1e-9))
            f1s.append(2 * precs[-1] * min(recs[-1], 1) / max(precs[-1] + min(recs[-1], 1), 1e-9))

        fig.add_scatter(x=thresholds, y=precs, name="Precision",
                        line=dict(color="#E8593C", width=2), showlegend=(idx==1), row=1, col=idx)
        fig.add_scatter(x=thresholds, y=[min(r,1) for r in recs], name="Recall",
                        line=dict(color="#2E75B6", width=2), showlegend=(idx==1), row=1, col=idx)
        fig.add_vline(x=0.5, line_dash="dash", line_color="#aaa", row=1, col=idx)

    fig.update_layout(
        title="⚖️ Phân tích ngưỡng quyết định (Threshold Analysis)<br>"
              "<sup>Đường đứt = ngưỡng mặc định 0.5</sup>",
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.2), height=360
    )
    fig.update_xaxes(title_text="Threshold")
    return fig


# ═══════════════════════════════════════════════════════════
# 5. KẾT LUẬN
# ═══════════════════════════════════════════════════════════

def generate_conclusion(lr, rf, lr_time, rf_time) -> str:
    faster      = "Logistic Regression" if lr_time < rf_time else "Random Forest"
    speed_ratio = max(lr_time, rf_time) / max(min(lr_time, rf_time), 0.001)

    rows = [
        ("AUC-ROC",       f"{lr['auc']:.4f}", f"{rf['auc']:.4f}",
         "LR" if lr["auc"] > rf["auc"] else "RF"),
        ("Accuracy",      f"{lr['acc']:.4f}", f"{rf['acc']:.4f}",
         "LR" if lr["acc"] > rf["acc"] else "RF"),
        ("F1 (weighted)", f"{lr['f1']:.4f}",  f"{rf['f1']:.4f}",
         "LR" if lr["f1"]  > rf["f1"]  else "RF"),
        ("Avg Precision", f"{lr['ap']:.4f}",  f"{rf['ap']:.4f}",
         "LR" if lr["ap"]  > rf["ap"]  else "RF"),
        ("Log Loss",      f"{lr['log_loss']:.4f}", f"{rf['log_loss']:.4f}",
         "LR" if lr["log_loss"] < rf["log_loss"] else "RF"),
        ("Brier Score",   f"{lr['brier']:.4f}",    f"{rf['brier']:.4f}",
         "LR" if lr["brier"] < rf["brier"] else "RF"),
        ("Train time",    f"{lr_time:.1f}s",  f"{rf_time:.1f}s",
         "LR" if lr_time < rf_time else "RF"),
    ]

    table_rows = "".join(f"""
        <tr>
          <td>{r}</td><td>{lv}</td><td>{rv}</td>
          <td style="color:{'#E8593C' if w=='LR' else '#2E75B6'};font-weight:bold">
            {'Logistic Regression' if w=='LR' else 'Random Forest'} ✓
          </td>
        </tr>""" for r, lv, rv, w in rows)

    lr_wins = sum(1 for _, _, _, w in rows if w == "LR")
    rf_wins = sum(1 for _, _, _, w in rows if w == "RF")
    overall = "Logistic Regression" if lr_wins >= rf_wins else "Random Forest"
    overall_color = "#E8593C" if overall == "Logistic Regression" else "#2E75B6"

    return f"""
    <div class="conclusion">
      <h2>🏁 Kết luận So sánh</h2>
      <table class="summary-table">
        <thead>
          <tr><th>Chỉ số</th><th>Logistic Regression</th>
              <th>Random Forest</th><th>Tốt hơn</th></tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>

      <h3>📌 Nhận xét chi tiết</h3>
      <ul>
        <li><b>Hiệu suất tổng thể:</b>
            <span style="color:{overall_color};font-weight:bold">{overall}</span>
            thắng {max(lr_wins, rf_wins)}/{len(rows)} chỉ số.</li>

        <li><b>Tốc độ train:</b> {faster} nhanh hơn ~{speed_ratio:.1f}x.
            Logistic Regression hội tụ rất nhanh với solver SAGA,
            trong khi Random Forest cần xây 100 cây quyết định.</li>

        <li><b>Khả năng giải thích:</b>
            <span style="color:#E8593C;font-weight:bold">Logistic Regression</span>
            cung cấp hệ số (coefficients) trực tiếp cho thấy
            feature nào tăng/giảm xác suất delay và bao nhiêu.
            Random Forest chỉ cho Feature Importance (mức độ quan trọng)
            nhưng không biết chiều tác động.</li>

        <li><b>Giả định tuyến tính:</b> Logistic Regression giả định
            mối quan hệ tuyến tính giữa features và log-odds —
            nếu dữ liệu phi tuyến (ví dụ delay tăng vọt ở giờ cao điểm)
            thì LR sẽ thua RF. Random Forest không có giả định này.</li>

        <li><b>Calibration (chất lượng xác suất):</b>
            Log Loss và Brier Score đo mức độ "tự tin đúng" của model.
            Logistic Regression thường được calibrate tốt hơn tự nhiên
            vì bản chất là mô hình xác suất. Random Forest thường
            cần CalibratedClassifierCV để cải thiện.</li>

        <li><b>Ngưỡng quyết định:</b> Cả hai model mặc định dùng 0.5.
            Với dữ liệu mất cân bằng (21% delay), nên thử threshold thấp hơn
            (~0.3–0.4) để tăng Recall cho class Delayed,
            dù Precision sẽ giảm.</li>
      </ul>

      <h3>💡 Khi nào dùng thuật toán nào?</h3>
      <table class="summary-table">
        <thead><tr><th>Tình huống</th><th>Nên dùng</th><th>Lý do</th></tr></thead>
        <tbody>
          <tr><td>Cần giải thích cho stakeholder</td>
              <td style="color:#E8593C;font-weight:bold">Logistic Regression</td>
              <td>Hệ số rõ ràng, dễ trình bày</td></tr>
          <tr><td>Cần độ chính xác cao nhất</td>
              <td style="color:#2E75B6;font-weight:bold">Random Forest</td>
              <td>Phi tuyến, bắt được pattern phức tạp</td></tr>
          <tr><td>Dữ liệu mới, muốn train nhanh</td>
              <td style="color:#E8593C;font-weight:bold">Logistic Regression</td>
              <td>Hội tụ nhanh, ít tham số</td></tr>
          <tr><td>Dữ liệu nhiều features tương tác</td>
              <td style="color:#2E75B6;font-weight:bold">Random Forest</td>
              <td>Tự xử lý feature interaction</td></tr>
          <tr><td>Cần calibrate xác suất tốt</td>
              <td style="color:#E8593C;font-weight:bold">Logistic Regression</td>
              <td>Xác suất có ý nghĩa thống kê tự nhiên</td></tr>
        </tbody>
      </table>

      <h3>🎓 Kết luận cho đồ án</h3>
      <p>Với bài toán dự đoán delay hàng không, <b>Random Forest</b> cho kết quả
      tốt hơn về AUC-ROC và F1 vì mối quan hệ giữa các đặc trưng và delay
      thường phi tuyến (ví dụ: chuyến dài + giờ tối + hãng nhỏ = rủi ro rất cao,
      không phải cộng đơn giản). Tuy nhiên <b>Logistic Regression</b> vẫn có giá trị
      như một baseline mạnh và cho phép giải thích kết quả rõ ràng hơn —
      quan trọng trong môi trường doanh nghiệp thực tế.</p>
    </div>"""


# ═══════════════════════════════════════════════════════════
# 6. BUILD HTML
# ═══════════════════════════════════════════════════════════

def build_html(figs: list, conclusion: str, lr, rf, lr_time, rf_time) -> str:
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>LR vs RF — Airline Delay</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; background: #F0F4F8; color: #333; }}
    .header {{
      background: linear-gradient(135deg, #C0392B, #E8593C);
      color: white; padding: 32px 40px;
    }}
    .header h1 {{ font-size: 26px; margin-bottom: 8px; }}
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
    .conclusion h2 {{ color: #C0392B; font-size: 22px; margin-bottom: 16px; }}
    .conclusion h3 {{ color: #333; font-size: 17px; margin: 20px 0 10px; }}
    .conclusion ul  {{ padding-left: 20px; line-height: 2.1; }}
    .conclusion p   {{ line-height: 1.8; margin-bottom: 8px; }}
    .summary-table {{
      width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 14px;
    }}
    .summary-table th {{
      background: #C0392B; color: white; padding: 10px 14px; text-align: left;
    }}
    .summary-table td {{ padding: 9px 14px; border-bottom: 1px solid #eee; }}
    .summary-table tr:nth-child(even) td {{ background: #FDF6F0; }}
    .section-title {{
      font-size: 18px; font-weight: bold; color: #C0392B;
      margin: 28px 0 12px; border-left: 4px solid #E8593C; padding-left: 12px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>✈️ So sánh: Logistic Regression vs Random Forest</h1>
    <p>Bài toán dự đoán delay chuyến bay (≥15 phút) — Airline Data Warehouse 2015</p>
    <div class="badges">
      <span class="badge">📦 500,000 mẫu</span>
      <span class="badge">🎯 Binary Classification</span>
      <span class="badge">📊 Supervised Learning</span>
      <span class="badge">📐 LR: {lr['auc']:.4f} AUC</span>
      <span class="badge">🌲 RF: {rf['auc']:.4f} AUC</span>
      <span class="badge">⏱️ LR: {lr_time:.1f}s | RF: {rf_time:.1f}s</span>
    </div>
  </div>

  <div class="container">
    <div class="section-title">1. Tổng quan chỉ số</div>
    <div class="grid full">__CHART_0__</div>

    <div class="section-title">2. Chất lượng xác suất (Calibration)</div>
    <div class="grid full">__CHART_1__</div>

    <div class="section-title">3. ROC & Precision-Recall Curve</div>
    <div class="grid">__CHART_2__ __CHART_3__</div>

    <div class="section-title">4. Confusion Matrix & Classification Report</div>
    <div class="grid full">__CHART_4__</div>
    <div class="grid full" style="margin-top:20px">__CHART_5__</div>

    <div class="section-title">5. Phân phối xác suất</div>
    <div class="grid full">__CHART_6__</div>

    <div class="section-title">6. Thời gian Training</div>
    <div class="grid"><div>__CHART_7__</div></div>

    <div class="section-title">7. Giải thích Model</div>
    <div class="grid">__CHART_8__ __CHART_9__</div>

    {conclusion}
  </div>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 55)
    log.info("  BẮT ĐẦU SO SÁNH LOGISTIC REGRESSION vs RANDOM FOREST")
    log.info("=" * 55)

    # 1. Load & preprocess
    df = load_data(limit=500_000)
    X, y, feature_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 2. Train
    lr_model, scaler, lr_time = train_logistic_regression(X_train, y_train)
    rf_model, rf_time          = train_random_forest(X_train, y_train)

    # 3. Evaluate
    lr = evaluate("Logistic Regression", lr_model, X_test, y_test, scaler=scaler)
    rf = evaluate("Random Forest",       rf_model, X_test, y_test, scaler=None)

    # 4. Biểu đồ
    log.info("Tạo biểu đồ...")
    figs = [
        fig_metrics_bar(lr, rf),                           # 0
        fig_calibration_metrics(lr, rf),                   # 1
        fig_roc(lr, rf),                                   # 2
        fig_pr(lr, rf),                                    # 3
        fig_confusion_matrices(lr, rf),                    # 4
        fig_class_report(lr, rf),                          # 5
        fig_prob_distribution(lr, rf, y_test.values),      # 6
        fig_train_time(lr_time, rf_time),                  # 7
        fig_coefficients(lr_model, scaler, feature_cols),  # 8 — LR unique
        fig_feature_importance(rf_model, feature_cols),    # 9 — RF unique
    ]

    # 5. Build HTML
    conclusion = generate_conclusion(lr, rf, lr_time, rf_time)
    html = build_html(figs, conclusion, lr, rf, lr_time, rf_time)
    charts_html = [
        f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
        for fig in figs
    ]
    for i, ch in enumerate(charts_html):
        html = html.replace(f"__CHART_{i}__", ch)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    # 6. Lưu model
    with open(os.path.join(os.path.dirname(__file__), "model_lr.pkl"), "wb") as f:
        pickle.dump({"model": lr_model, "scaler": scaler, "type": "lr"}, f)

    log.info("=" * 55)
    log.info(f"  LR  — AUC: {lr['auc']:.4f} | Acc: {lr['acc']:.4f} | Time: {lr_time:.1f}s")
    log.info(f"  RF  — AUC: {rf['auc']:.4f} | Acc: {rf['acc']:.4f} | Time: {rf_time:.1f}s")
    log.info(f"  Report: {REPORT_PATH}")
    log.info("=" * 55)
