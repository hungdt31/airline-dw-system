"""
Cấu hình chung: đường dẫn DB, bản đồ bang Mỹ, SQL fragment JOIN OLAP.
"""

import os

# Thư mục dashboard/ và gốc repo
DASH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DASH_DIR)

DB_PATH = os.path.join(ROOT_DIR, "warehouse", "airline_dw.db")

BASE_JOIN = """
    FROM fact_flights f
    JOIN dim_date    d    ON f.date_key           = d.date_key
    JOIN dim_airline a    ON f.airline_key         = a.airline_key
    JOIN dim_airport ap_o ON f.origin_airport_key = ap_o.airport_key
    JOIN dim_airport ap_d ON f.dest_airport_key   = ap_d.airport_key
"""

# Slice & Dice / phân tích theo nguyên nhân (dim_delay_reason)
BASE_JOIN_REASON = """
    FROM fact_flights f
    JOIN dim_date          d    ON f.date_key           = d.date_key
    JOIN dim_airline       a    ON f.airline_key       = a.airline_key
    JOIN dim_airport       ap_o ON f.origin_airport_key = ap_o.airport_key
    JOIN dim_airport       ap_d ON f.dest_airport_key   = ap_d.airport_key
    JOIN dim_delay_reason  dr   ON f.delay_reason_key   = dr.reason_key
"""

# Bang Mỹ — dùng sidebar + tab dự đoán (hiển thị tên đầy đủ)
STATE_MAP = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico", "VI": "Virgin Islands", "GU": "Guam", "TT": "Trust Territories",
}
