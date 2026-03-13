"""
mining/predict_cli.py
=====================
Dự đoán delay cho một chuyến bay cụ thể — KHÔNG cần train lại.
Yêu cầu: đã có file model_delay.pkl (chạy predict_delay.py 1 lần trước).

Cách dùng:
    python mining/predict_cli.py
    python mining/predict_cli.py --airline DL --origin CA --dest NY --hour 17 --month 6
"""

import argparse
import os
import sys
import pickle
import sqlite3
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_delay.pkl")
DB_PATH    = os.path.join(os.path.dirname(__file__), "..", "warehouse", "airline_dw.db")

MONTH_NAMES = {
    1:"January",2:"February",3:"March",4:"April",
    5:"May",6:"June",7:"July",8:"August",
    9:"September",10:"October",11:"November",12:"December"
}
DAY_NAMES = {1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",
             5:"Friday",6:"Saturday",7:"Sunday"}
QUARTER_MAP = {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}


# ─── Load model ────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ Chưa có model! Hãy chạy trước:")
        print("   python mining/predict_delay.py")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["encoders"]


# ─── Load tên đầy đủ từ DB ─────────────────────────────────

def load_labels_from_db() -> dict:
    """
    Lấy tên đầy đủ của hãng và bang từ Data Warehouse.
    Trả về:
        airline_map: { "DL": "Delta Air Lines" }
        state_map:   { "CA": "California" }   (lấy từ city nếu không có state name)
    """
    if not os.path.exists(DB_PATH):
        return {}, {}

    conn = sqlite3.connect(DB_PATH)

    # Tên hãng hàng không
    try:
        df_a = pd.read_sql("SELECT airline_iata, airline_name FROM dim_airline", conn)
        airline_map = dict(zip(df_a["airline_iata"], df_a["airline_name"]))
    except Exception:
        airline_map = {}

    # Tên bang: lấy state → dùng state code vì dataset Mỹ
    # Dùng bảng US states chuẩn (không phụ thuộc DB)
    state_map = {
        "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
        "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
        "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
        "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
        "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
        "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
        "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina",
        "ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania",
        "RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee",
        "TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
        "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming","DC":"District of Columbia",
        "PR":"Puerto Rico","VI":"Virgin Islands","GU":"Guam","TT":"Trust Territories",
    }

    conn.close()
    return airline_map, state_map


# ─── Validation ────────────────────────────────────────────

def get_valid_values(encoders):
    return {
        "airlines":      sorted(encoders["airline"].classes_.tolist()),
        "origin_states": sorted(encoders["origin_state"].classes_.tolist()),
        "dest_states":   sorted(encoders["dest_state"].classes_.tolist()),
    }


def validate_and_encode(le, val, field_name, valid_list, label_map=None):
    """Encode giá trị — báo lỗi kèm tên đầy đủ nếu không hợp lệ."""
    if val not in le.classes_:
        print(f"\n❌ '{val}' không hợp lệ cho [{field_name}].")
        if label_map:
            labeled = [f"{code} ({label_map.get(code, code)})" for code in valid_list]
            print(f"   Các giá trị hợp lệ:\n   " + ", ".join(labeled))
        else:
            print(f"   Các giá trị hợp lệ: {', '.join(valid_list)}")
        sys.exit(1)
    return le.transform([val])[0]


# ─── Predict ───────────────────────────────────────────────

def predict(airline: str, origin_state: str, dest_state: str,
            month: int, day_of_week: int, dep_hour: int,
            distance: float, elapsed_time: float) -> dict:

    clf, encoders = load_model()
    feature_cols  = encoders["feature_cols"]
    valid         = get_valid_values(encoders)
    airline_map, state_map = load_labels_from_db()

    is_weekend = int(day_of_week >= 6)
    quarter    = QUARTER_MAP[month]

    features = pd.DataFrame([{
        "month":        month,
        "day_of_week":  day_of_week,
        "is_weekend":   is_weekend,
        "quarter":      quarter,
        "dep_hour":     dep_hour,
        "distance":     distance,
        "elapsed_time": elapsed_time,
        "airline_enc":  validate_and_encode(encoders["airline"],      airline,      "hãng hàng không", valid["airlines"],      airline_map),
        "origin_enc":   validate_and_encode(encoders["origin_state"], origin_state, "bang xuất phát",  valid["origin_states"], state_map),
        "dest_enc":     validate_and_encode(encoders["dest_state"],   dest_state,   "bang đích đến",   valid["dest_states"],   state_map),
    }])[feature_cols]

    prob = clf.predict_proba(features)[0][1]
    return {
        "delayed":      prob >= 0.5,
        "probability":  round(float(prob), 4),
        "risk_level":   "🔴 Cao" if prob >= 0.6 else ("🟡 Trung bình" if prob >= 0.4 else "🟢 Thấp"),
        "airline_name": airline_map.get(airline, airline),
        "origin_name":  state_map.get(origin_state, origin_state),
        "dest_name":    state_map.get(dest_state, dest_state),
    }


# ─── Interactive prompt ────────────────────────────────────

def prompt_input() -> dict:
    _, encoders = load_model()
    valid = get_valid_values(encoders)
    airline_map, state_map = load_labels_from_db()

    print("\n" + "="*55)
    print("  ✈️  DỰ ĐOÁN DELAY CHUYẾN BAY")
    print("="*55)

    # Hãng hàng không — in dạng "DL (Delta Air Lines)"
    print("\nCác hãng hàng không:")
    for code in valid["airlines"]:
        name = airline_map.get(code, "")
        print(f"  {code:<4} {name}")
    airline = input("Nhập mã hãng: ").strip().upper()

    # Bang xuất phát — in dạng "CA (California)"
    print("\nCác bang xuất phát:")
    items = [f"{c} ({state_map.get(c, c)})" for c in valid["origin_states"]]
    # In 4 cột
    col_w = max(len(x) for x in items) + 2
    for i, item in enumerate(items):
        print(f"  {item:<{col_w}}", end="\n" if (i+1) % 4 == 0 else "")
    print()
    origin = input("Nhập mã bang xuất phát: ").strip().upper()

    # Bang đích — in dạng "NY (New York)"
    print("\nCác bang đích đến:")
    items_d = [f"{c} ({state_map.get(c, c)})" for c in valid["dest_states"]]
    for i, item in enumerate(items_d):
        print(f"  {item:<{col_w}}", end="\n" if (i+1) % 4 == 0 else "")
    print()
    dest = input("Nhập mã bang đích: ").strip().upper()

    print("\nTháng bay:")
    for k, v in MONTH_NAMES.items():
        print(f"  {k:>2}. {v}")
    month = int(input("Chọn tháng (1-12): ").strip())

    print("\nThứ trong tuần:")
    for k, v in DAY_NAMES.items():
        print(f"  {k}. {v}")
    day_of_week = int(input("Chọn thứ (1=Mon ... 7=Sun): ").strip())

    dep_hour = int(input("\nGiờ khởi hành dự kiến (0-23): ").strip())
    distance = float(input("Khoảng cách chuyến bay (dặm, VD: 1200): ").strip())
    elapsed  = float(input("Thời gian bay dự kiến (phút, VD: 160): ").strip())

    return dict(airline=airline, origin_state=origin, dest_state=dest,
                month=month, day_of_week=day_of_week, dep_hour=dep_hour,
                distance=distance, elapsed_time=elapsed)


# ─── Print result ──────────────────────────────────────────

def print_result(params: dict, result: dict):
    print("\n" + "="*55)
    print("  KẾT QUẢ DỰ ĐOÁN")
    print("="*55)
    print(f"  Hãng        : {params['airline']} — {result['airline_name']}")
    print(f"  Điểm đi     : {params['origin_state']} — {result['origin_name']}")
    print(f"  Điểm đến    : {params['dest_state']} — {result['dest_name']}")
    print(f"  Tháng       : {MONTH_NAMES.get(params['month'], params['month'])}")
    print(f"  Thứ         : {DAY_NAMES.get(params['day_of_week'], params['day_of_week'])}")
    print(f"  Giờ bay     : {params['dep_hour']:02d}:00")
    print(f"  Khoảng cách : {params['distance']:.0f} dặm")
    print("-"*55)
    status = "⚠️  CÓ THỂ BỊ DELAY" if result["delayed"] else "✅ KHÔNG CÓ DELAY"
    print(f"  Dự đoán     : {status}")
    print(f"  Xác suất    : {result['probability']*100:.1f}%")
    print(f"  Mức độ rủi ro: {result['risk_level']}")
    print("="*55 + "\n")


# ─── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dự đoán delay chuyến bay")
    parser.add_argument("--airline",  type=str,   help="Mã hãng (VD: DL)")
    parser.add_argument("--origin",   type=str,   help="Mã bang xuất phát (VD: CA)")
    parser.add_argument("--dest",     type=str,   help="Mã bang đích (VD: NY)")
    parser.add_argument("--month",    type=int,   help="Tháng bay (1-12)")
    parser.add_argument("--dow",      type=int,   help="Thứ (1=Mon..7=Sun)")
    parser.add_argument("--hour",     type=int,   help="Giờ khởi hành (0-23)")
    parser.add_argument("--distance", type=float, help="Khoảng cách (dặm)")
    parser.add_argument("--elapsed",  type=float, help="Thời gian bay (phút)")
    args = parser.parse_args()

    if all([args.airline, args.origin, args.dest, args.month,
            args.dow, args.hour, args.distance, args.elapsed]):
        params = dict(
            airline=args.airline.upper(), origin_state=args.origin.upper(),
            dest_state=args.dest.upper(), month=args.month,
            day_of_week=args.dow, dep_hour=args.hour,
            distance=args.distance, elapsed_time=args.elapsed
        )
    else:
        params = prompt_input()

    result = predict(**params)
    print_result(params, result)


if __name__ == "__main__":
    main()
