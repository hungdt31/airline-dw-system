"""
mining/analyze_imbalance.py
===========================
Chứng minh sự mất cân bằng lớp trong dữ liệu hàng không.
Xuất ra: mining/imbalance_report.html

Chạy: python mining/analyze_imbalance.py
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH     = os.path.join(os.path.dirname(__file__), "..", "warehouse", "airline_dw.db")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "imbalance_report.html")
BG = "#FAFAFA"

# ═══════════════════════════════════════════════════════════
# 1. LOAD DỮ LIỆU
# ═══════════════════════════════════════════════════════════
print("Loading data...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("""
    SELECT
        f.arrival_delay,
        f.departure_delay,
        f.is_delayed_15,
        f.is_cancelled,
        d.month,
        d.day_of_week,
        a.airline_iata
    FROM fact_flights f
    JOIN dim_date    d ON f.date_key    = d.date_key
    JOIN dim_airline a ON f.airline_key = a.airline_key
    WHERE f.is_cancelled = 0
    LIMIT 500000
""", conn)
conn.close()
print(f"Loaded {len(df):,} rows")

# ═══════════════════════════════════════════════════════════
# 2. THỐNG KÊ MẤT CÂN BẰNG
# ═══════════════════════════════════════════════════════════
n_total   = len(df)
n_delayed = int(df["is_delayed_15"].sum())
n_ontime  = n_total - n_delayed
pct_delay = n_delayed / n_total * 100
pct_ontime = n_ontime / n_total * 100
imbalance_ratio = n_ontime / n_delayed

print(f"\n{'='*50}")
print(f"  PHÂN TÍCH MẤT CÂN BẰNG LỚP")
print(f"{'='*50}")
print(f"  Tổng mẫu         : {n_total:,}")
print(f"  On-time (0)      : {n_ontime:,} ({pct_ontime:.2f}%)")
print(f"  Delayed (1)      : {n_delayed:,} ({pct_delay:.2f}%)")
print(f"  Imbalance ratio  : {imbalance_ratio:.2f}:1")
print(f"  scale_pos_weight : {imbalance_ratio:.2f}  (dùng cho XGBoost)")
print(f"{'='*50}")

# Model trivial (đoán tất cả là On-time)
trivial_acc = pct_ontime
print(f"\n  Model ngốc (đoán tất cả On-time):")
print(f"    Accuracy = {trivial_acc:.2f}%  ← trông cao nhưng vô nghĩa")
print(f"    Recall   = 0.00%             ← không phát hiện được delay nào")
print(f"    AUC-ROC  = 0.500             ← tương đương đoán ngẫu nhiên")

# Thống kê arrival_delay
delay_stats = df["arrival_delay"].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
print(f"\n  Thống kê arrival_delay (phút):")
print(f"    Min    = {delay_stats['min']:.1f}")
print(f"    Q1     = {delay_stats['25%']:.1f}")
print(f"    Median = {delay_stats['50%']:.1f}")
print(f"    Mean   = {delay_stats['mean']:.1f}")
print(f"    Q3     = {delay_stats['75%']:.1f}")
print(f"    P90    = {delay_stats['90%']:.1f}")
print(f"    P95    = {delay_stats['95%']:.1f}")
print(f"    P99    = {delay_stats['99%']:.1f}")
print(f"    Max    = {delay_stats['max']:.1f}")

# Outlier theo IQR
Q1  = df["arrival_delay"].quantile(0.25)
Q3  = df["arrival_delay"].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR
lower_fence = Q1 - 1.5 * IQR
n_outlier_upper = int((df["arrival_delay"] > upper_fence).sum())
n_outlier_lower = int((df["arrival_delay"] < lower_fence).sum())
print(f"\n  Outlier Detection (IQR method):")
print(f"    Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f}")
print(f"    Lower fence = {lower_fence:.1f}")
print(f"    Upper fence = {upper_fence:.1f}")
print(f"    Outlier trên (delay cực lớn) : {n_outlier_upper:,} ({n_outlier_upper/n_total*100:.2f}%)")
print(f"    Outlier dưới (đến rất sớm)  : {n_outlier_lower:,} ({n_outlier_lower/n_total*100:.2f}%)")

# ═══════════════════════════════════════════════════════════
# 3. TẠO BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════

# ── Chart 1: Pie chart nhãn ────────────────────────────────
fig_pie = go.Figure(go.Pie(
    labels=["On-time (0)", "Delayed ≥15p (1)"],
    values=[n_ontime, n_delayed],
    marker_colors=["#2E75B6", "#E8593C"],
    textinfo="label+percent+value",
    textfont_size=14,
    hole=0.35,
))
fig_pie.update_layout(
    title=dict(text=f"Phân phối nhãn — Imbalance ratio {imbalance_ratio:.2f}:1",
               font_size=16),
    paper_bgcolor=BG, height=380
)

# ── Chart 2: Bar chart so sánh trivial model vs chỉ số ──────
metrics = ["Accuracy", "Recall (Delayed)", "AUC-ROC"]
trivial  = [pct_ontime/100, 0.0, 0.5]
meaning  = ["Cao nhưng misleading", "Không phát hiện gì", "Đoán ngẫu nhiên"]

fig_trivial = go.Figure()
fig_trivial.add_bar(
    x=metrics, y=trivial,
    marker_color=["#F39C12", "#E8593C", "#888"],
    text=[f"{v:.2f}" for v in trivial],
    textposition="outside",
    name="Model ngốc (đoán tất cả On-time)"
)
fig_trivial.add_hline(y=0.5, line_dash="dash", line_color="#2E75B6",
                      # annotation_text="Ngưỡng tốt hơn ngẫu nhiên (0.5)",
                      # annotation_position="top right"
                    )
fig_trivial.update_layout(
    title="⚠️ Ảnh hưởng của dữ liệu mất cân bằng",
    yaxis_range=[0, 1.1],
    yaxis_tickformat=".0%",
    yaxis_title="Giá trị",
    plot_bgcolor=BG, paper_bgcolor=BG, height=360
)

# ── Chart 3: Histogram arrival_delay ────────────────────────
# Giới hạn hiển thị [-60, 200] để thấy rõ phân phối
df_plot = df[(df["arrival_delay"] >= -60) & (df["arrival_delay"] <= 200)].copy()

fig_hist = go.Figure()
fig_hist.add_histogram(
    x=df_plot["arrival_delay"],
    nbinsx=80,
    marker_color="#4A90D9",
    opacity=0.8,
    name="arrival_delay"
)
fig_hist.add_vline(x=0,  line_dash="dash", line_color="#27AE60", line_width=2,
                   annotation_text="0 phút", annotation_position="top right")
fig_hist.add_vline(x=15, line_dash="dash", line_color="#E8593C", line_width=2,
                   annotation_text="Ngưỡng 15p (is_delayed_15=1)",
                   annotation_position="top right")
fig_hist.add_vrect(x0=15, x1=200, fillcolor="#E8593C", opacity=0.06,
                   annotation_text=f"Delayed: {pct_delay:.1f}%",
                   annotation_position="top right")
fig_hist.add_vrect(x0=-60, x1=15, fillcolor="#2E75B6", opacity=0.06,
                   annotation_text=f"On-time: {pct_ontime:.1f}%",
                   annotation_position="top left")
fig_hist.update_layout(
    title="Phân phối arrival_delay — Lệch phải (Right-skewed), đuôi dài",
    xaxis_title="Arrival Delay (phút)",
    yaxis_title="Số lượng chuyến bay",
    plot_bgcolor=BG, paper_bgcolor=BG, height=400
)

# ── Chart 4: IQR Outlier boxplot ────────────────────────────
df_box = df["arrival_delay"].clip(-200, 500)
fig_box = go.Figure()
fig_box.add_box(
    y=df_box,
    name="arrival_delay",
    marker_color="#4A90D9",
    boxpoints="suspectedoutliers",
    jitter=0.3,
    pointpos=-1.8,
)
fig_box.add_hline(y=upper_fence, line_dash="dot", line_color="#E8593C",
                  annotation_text=f"Upper fence = {upper_fence:.1f}",
                  annotation_position="right")
fig_box.add_hline(y=lower_fence, line_dash="dot", line_color="#E8593C",
                  annotation_text=f"Lower fence = {lower_fence:.1f}",
                  annotation_position="right")
fig_box.update_layout(
    title=f"Boxplot + IQR Outlier Detection<br>"
          f"<sup>Upper outliers: {n_outlier_upper:,} ({n_outlier_upper/n_total*100:.2f}%) | "
          f"Lower outliers: {n_outlier_lower:,} ({n_outlier_lower/n_total*100:.2f}%)</sup>",
    yaxis_title="Arrival Delay (phút)",
    plot_bgcolor=BG, paper_bgcolor=BG, height=420
)

# ── Chart 5: Imbalance theo hãng bay ────────────────────────
airline_balance = df.groupby("airline_iata").agg(
    total=("is_delayed_15", "count"),
    delayed=("is_delayed_15", "sum")
).reset_index()
airline_balance["delay_rate"] = airline_balance["delayed"] / airline_balance["total"] * 100
airline_balance = airline_balance.sort_values("delay_rate", ascending=False)

fig_airline = px.bar(
    airline_balance, x="airline_iata", y="delay_rate",
    color="delay_rate", color_continuous_scale="RdYlGn_r",
    text=[f"{v:.1f}%" for v in airline_balance["delay_rate"]],
    labels={"airline_iata": "Hãng bay", "delay_rate": "Tỉ lệ delay (%)"},
    title="Tỉ lệ delay theo từng hãng — Mất cân bằng KHÔNG đồng đều"
)
fig_airline.add_hline(y=pct_delay, line_dash="dash", line_color="#333",
                      annotation_text=f"TB toàn bộ: {pct_delay:.1f}%",
                      annotation_position="top right")
fig_airline.update_traces(textposition="outside")
fig_airline.update_layout(
    coloraxis_showscale=False,
    plot_bgcolor=BG, paper_bgcolor=BG, height=380
)

# ── Chart 6: Imbalance theo tháng ────────────────────────────
month_balance = df.groupby("month").agg(
    total=("is_delayed_15", "count"),
    delayed=("is_delayed_15", "sum")
).reset_index()
month_balance["delay_rate"] = month_balance["delayed"] / month_balance["total"] * 100
month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
month_balance["month_name"] = month_balance["month"].map(month_names)

fig_month = px.bar(
    month_balance, x="month_name", y="delay_rate",
    color="delay_rate", color_continuous_scale="RdYlGn_r",
    text=[f"{v:.1f}%" for v in month_balance["delay_rate"]],
    labels={"month_name": "Tháng", "delay_rate": "Tỉ lệ delay (%)"},
    title="Tỉ lệ delay theo tháng — Mất cân bằng thay đổi theo mùa"
)
fig_month.add_hline(y=pct_delay, line_dash="dash", line_color="#333",
                    annotation_text=f"TB: {pct_delay:.1f}%",
                    annotation_position="top right")
fig_month.update_traces(textposition="outside")
fig_month.update_layout(
    coloraxis_showscale=False,
    plot_bgcolor=BG, paper_bgcolor=BG, height=360
)

# ═══════════════════════════════════════════════════════════
# 4. BUILD HTML REPORT
# ═══════════════════════════════════════════════════════════
def to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)

html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Phân tích Mất cân bằng lớp — Airline Delay</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; background: #F0F4F8; color: #333; }}
    .header {{
      background: linear-gradient(135deg, #1F5C99, #2E75B6);
      color: white; padding: 28px 40px;
    }}
    .header h1 {{ font-size: 24px; margin-bottom: 6px; }}
    .header p  {{ font-size: 14px; opacity: .85; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .full  {{ grid-column: 1/-1; }}
    .card {{
      background: white; border-radius: 12px; padding: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,.08);
    }}
    .kpi-grid {{
      display: grid; grid-template-columns: repeat(4,1fr); gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi {{
      background: white; border-radius: 10px; padding: 16px 20px;
      border-left: 4px solid #2E75B6;
      box-shadow: 0 2px 6px rgba(0,0,0,.07);
    }}
    .kpi .label {{ font-size: 12px; color: #888; margin-bottom: 4px; }}
    .kpi .value {{ font-size: 26px; font-weight: bold; color: #1F5C99; }}
    .kpi .sub   {{ font-size: 12px; color: #aaa; margin-top: 2px; }}
    .kpi.red  {{ border-left-color: #E8593C; }}
    .kpi.red .value {{ color: #E8593C; }}
    .kpi.green {{ border-left-color: #27AE60; }}
    .kpi.green .value {{ color: #27AE60; }}
    .kpi.gray  {{ border-left-color: #888; }}
    .kpi.gray .value  {{ color: #555; }}
    .section-title {{
      font-size: 17px; font-weight: bold; color: #1F5C99;
      border-left: 4px solid #2E75B6; padding-left: 10px;
      margin: 24px 0 12px;
    }}
    .conclusion {{
      background: white; border-radius: 12px; padding: 24px;
      box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-top: 20px;
    }}
    .conclusion h2 {{ color: #1F5C99; font-size: 20px; margin-bottom: 12px; }}
    .conclusion h3 {{ color: #333; font-size: 16px; margin: 16px 0 8px; }}
    .conclusion ul {{ padding-left: 20px; line-height: 2; }}
    table.metrics {{
      width: 100%; border-collapse: collapse; font-size: 14px;
      margin: 12px 0;
    }}
    table.metrics th {{
      background: #1F5C99; color: white; padding: 9px 14px; text-align: left;
    }}
    table.metrics td {{ padding: 8px 14px; border-bottom: 1px solid #eee; }}
    table.metrics tr:nth-child(even) td {{ background: #F5F8FF; }}
    code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
  </style>
</head>
<body>
<div class="header">
  <h1>⚖️ Phân tích Mất cân bằng lớp — Airline Delay 2015</h1>
  <p>Chứng minh tính mất cân bằng và lý do lựa chọn chỉ số đánh giá phù hợp</p>
</div>

<div class="container">

  <!-- KPI Cards -->
  <div class="kpi-grid" style="margin-top:20px">
    <div class="kpi green">
      <div class="label">On-time (Lớp 0)</div>
      <div class="value">{pct_ontime:.2f}%</div>
      <div class="sub">{n_ontime:,} chuyến bay</div>
    </div>
    <div class="kpi red">
      <div class="label">Delayed ≥15p (Lớp 1)</div>
      <div class="value">{pct_delay:.2f}%</div>
      <div class="sub">{n_delayed:,} chuyến bay</div>
    </div>
    <div class="kpi gray">
      <div class="label">Imbalance Ratio</div>
      <div class="value">{imbalance_ratio:.2f}:1</div>
      <div class="sub">On-time / Delayed</div>
    </div>
    <div class="kpi red">
      <div class="label">Model ngốc — Accuracy</div>
      <div class="value">{trivial_acc:.1f}%</div>
      <div class="sub">Recall = 0% | AUC = 0.50</div>
    </div>
  </div>

  <!-- Biểu đồ 1+2 -->
  <div class="section-title">1. Phân phối nhãn & Vấn đề model ngốc</div>
  <div class="grid2">
    <div class="card">{to_html(fig_pie)}</div>
    <div class="card">{to_html(fig_trivial)}</div>
  </div>

  <!-- Biểu đồ 3 -->
  <div class="section-title">2. Phân phối arrival_delay — Right-skewed</div>
  <div class="card full">{to_html(fig_hist)}</div>

  <!-- Biểu đồ 4 -->
  <div class="section-title">3. Outlier Detection — IQR Method</div>
  <div class="card full">{to_html(fig_box)}</div>

  <!-- Bảng thống kê IQR -->
  <div class="card" style="margin-top:16px">
    <table class="metrics">
      <thead><tr><th>Chỉ số</th><th>Giá trị (phút)</th><th>Ý nghĩa</th></tr></thead>
      <tbody>
        <tr><td>Q1 (25th percentile)</td><td>{Q1:.1f}</td><td>25% chuyến bay đến sớm hơn giá trị này</td></tr>
        <tr><td>Median (Q2)</td><td>{df["arrival_delay"].median():.1f}</td><td>Điểm giữa — phần lớn đúng giờ</td></tr>
        <tr><td>Mean</td><td>{df["arrival_delay"].mean():.1f}</td><td>Trung bình bị kéo lên bởi outlier delay lớn</td></tr>
        <tr><td>Q3 (75th percentile)</td><td>{Q3:.1f}</td><td>75% chuyến bay đến trước giá trị này</td></tr>
        <tr><td>IQR</td><td>{IQR:.1f}</td><td>Khoảng trải giữa = Q3 − Q1</td></tr>
        <tr><td>Upper fence (Q3 + 1.5×IQR)</td><td>{upper_fence:.1f}</td><td>Ngưỡng phát hiện outlier trên</td></tr>
        <tr><td>Lower fence (Q1 − 1.5×IQR)</td><td>{lower_fence:.1f}</td><td>Ngưỡng phát hiện outlier dưới</td></tr>
        <tr><td>Outlier trên (delay cực lớn)</td><td>{n_outlier_upper:,} ({n_outlier_upper/n_total*100:.2f}%)</td><td>Chuyến bay trễ bất thường</td></tr>
        <tr><td>Outlier dưới (đến rất sớm)</td><td>{n_outlier_lower:,} ({n_outlier_lower/n_total*100:.2f}%)</td><td>Chuyến bay đến sớm bất thường</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Biểu đồ 5+6 -->
  <div class="section-title">4. Mất cân bằng không đồng đều theo Hãng & Tháng</div>
  <div class="grid2">
    <div class="card">{to_html(fig_airline)}</div>
    <div class="card">{to_html(fig_month)}</div>
  </div>

  <!-- Kết luận -->
  <div class="conclusion">
    <h2>📋 Kết luận — Lý do chọn chỉ số đánh giá</h2>

    <h3>1. Xác nhận mất cân bằng</h3>
    <ul>
      <li>Tỉ lệ <b>On-time: {pct_ontime:.2f}%</b> vs <b>Delayed: {pct_delay:.2f}%</b>
          → Imbalance ratio <b>{imbalance_ratio:.2f}:1</b></li>
      <li>Phân phối <code>arrival_delay</code> lệch phải mạnh (right-skewed):
          median = {df["arrival_delay"].median():.1f}p,
          mean = {df["arrival_delay"].mean():.1f}p,
          max = {df["arrival_delay"].max():.0f}p</li>
      <li>Outlier trên: {n_outlier_upper:,} bản ghi ({n_outlier_upper/n_total*100:.2f}%)
          có arrival_delay > {upper_fence:.0f} phút</li>
    </ul>

    <h3>2. Vấn đề với Accuracy</h3>
    <ul>
      <li>Model đoán tất cả "On-time" đạt Accuracy = <b>{trivial_acc:.1f}%</b>
          — cao hơn nhiều model thực sự!</li>
      <li>Nhưng Recall = 0%: không phát hiện được 1 chuyến delay nào</li>
      <li>→ Accuracy không phải chỉ số phù hợp cho dữ liệu mất cân bằng</li>
    </ul>

    <h3>3. Thứ tự ưu tiên chỉ số</h3>
    <table class="metrics">
      <thead><tr><th>Ưu tiên</th><th>Chỉ số</th><th>Lý do</th></tr></thead>
      <tbody>
        <tr><td><b>1</b></td><td>AUC-ROC</td>
            <td>Không phụ thuộc ngưỡng, không bị ảnh hưởng mất cân bằng</td></tr>
        <tr><td><b>2</b></td><td>Average Precision</td>
            <td>Nhạy hơn AUC-ROC với lớp minority (delay)</td></tr>
        <tr><td><b>3</b></td><td>F1-score (Delayed)</td>
            <td>Cân bằng Precision và Recall cho lớp quan tâm</td></tr>
        <tr><td><b>4</b></td><td>Recall (Delayed)</td>
            <td>Bỏ sót delay (FN) nguy hiểm hơn báo nhầm (FP)</td></tr>
        <tr><td><b>5</b></td><td>Log Loss / Brier</td>
            <td>Đánh giá chất lượng xác suất dự đoán</td></tr>
        <tr><td>6</td><td>Accuracy</td>
            <td>Ít có giá trị — chỉ dùng tham khảo</td></tr>
      </tbody>
    </table>

    <h3>4. Chiến lược xử lý</h3>
    <ul>
      <li><code>class_weight='balanced'</code> (LR, RF, MLP):
          tăng trọng số lớp delay lên ×{imbalance_ratio:.2f}</li>
      <li><code>scale_pos_weight={imbalance_ratio:.2f}</code> (XGBoost):
          tương đương</li>
      <li><code>is_unbalance=True</code> (LightGBM): tự động</li>
      <li>Đánh giá bằng AUC-ROC và AP thay vì Accuracy để tránh
          misleading</li>
    </ul>
  </div>

</div>
</body>
</html>"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅ Report saved: {REPORT_PATH}")
print(f"   Mở file trong trình duyệt để xem biểu đồ đầy đủ.")
