-- ============================================================
--  DATA WAREHOUSE — AIRLINE STAR SCHEMA
--  CDIO Project
-- ============================================================

-- Xóa bảng cũ nếu tồn tại (thứ tự: fact trước, dim sau)
DROP TABLE IF EXISTS fact_flights;
DROP TABLE IF EXISTS dim_airline;
DROP TABLE IF EXISTS dim_airport;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_delay_reason;

-- ------------------------------------------------------------
-- DIM_AIRLINE
-- ------------------------------------------------------------
CREATE TABLE dim_airline (
    airline_key     INTEGER PRIMARY KEY AUTOINCREMENT,
    airline_iata    TEXT    NOT NULL UNIQUE,   -- mã IATA (VD: AA, DL)
    airline_name    TEXT    NOT NULL
);

-- ------------------------------------------------------------
-- DIM_AIRPORT
-- ------------------------------------------------------------
CREATE TABLE dim_airport (
    airport_key     INTEGER PRIMARY KEY AUTOINCREMENT,
    airport_iata    TEXT    NOT NULL UNIQUE,
    airport_name    TEXT,
    city            TEXT,
    state           TEXT,
    country         TEXT    DEFAULT 'USA',
    latitude        REAL,
    longitude       REAL
);

-- ------------------------------------------------------------
-- DIM_DATE
-- ------------------------------------------------------------
CREATE TABLE dim_date (
    date_key        INTEGER PRIMARY KEY,       -- YYYYMMDD
    full_date       TEXT    NOT NULL,          -- YYYY-MM-DD
    year            INTEGER NOT NULL,
    quarter         INTEGER NOT NULL,          -- 1-4
    month           INTEGER NOT NULL,          -- 1-12
    month_name      TEXT    NOT NULL,
    week_of_year    INTEGER,
    day_of_month    INTEGER NOT NULL,
    day_of_week     INTEGER NOT NULL,          -- 1=Mon, 7=Sun
    day_name        TEXT    NOT NULL,
    is_weekend      INTEGER DEFAULT 0          -- 0/1
);

-- ------------------------------------------------------------
-- DIM_DELAY_REASON
-- ------------------------------------------------------------
CREATE TABLE dim_delay_reason (
    reason_key      INTEGER PRIMARY KEY AUTOINCREMENT,
    reason_code     TEXT    NOT NULL UNIQUE,
    reason_name     TEXT    NOT NULL,
    reason_category TEXT                       -- 'Operational','Weather','Carrier'
);

INSERT INTO dim_delay_reason (reason_code, reason_name, reason_category) VALUES
    ('AIR_SYSTEM',    'Hệ thống kiểm soát không lưu',   'Operational'),
    ('SECURITY',      'An ninh',                          'Operational'),
    ('AIRLINE',       'Hãng hàng không',                  'Carrier'),
    ('LATE_AIRCRAFT', 'Máy bay đến muộn',                 'Carrier'),
    ('WEATHER',       'Thời tiết',                        'Weather'),
    ('NO_DELAY',      'Không delay',                      'None');

-- ------------------------------------------------------------
-- FACT_FLIGHTS
-- ------------------------------------------------------------
CREATE TABLE fact_flights (
    flight_key              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Foreign keys
    airline_key             INTEGER REFERENCES dim_airline(airline_key),
    origin_airport_key      INTEGER REFERENCES dim_airport(airport_key),
    dest_airport_key        INTEGER REFERENCES dim_airport(airport_key),
    date_key                INTEGER REFERENCES dim_date(date_key),
    delay_reason_key        INTEGER REFERENCES dim_delay_reason(reason_key),

    -- Degenerate dimensions
    flight_number           TEXT,
    tail_number             TEXT,

    -- Measures — thời gian (phút)
    scheduled_departure     TEXT,
    actual_departure        TEXT,
    departure_delay         REAL    DEFAULT 0,
    scheduled_arrival       TEXT,
    actual_arrival          TEXT,
    arrival_delay           REAL    DEFAULT 0,
    elapsed_time            REAL,
    air_time                REAL,
    distance                REAL,

    -- Delay breakdown (phút)
    delay_air_system        REAL    DEFAULT 0,
    delay_security          REAL    DEFAULT 0,
    delay_airline           REAL    DEFAULT 0,
    delay_late_aircraft     REAL    DEFAULT 0,
    delay_weather           REAL    DEFAULT 0,

    -- Flags
    is_cancelled            INTEGER DEFAULT 0,   -- 0/1
    cancellation_reason     TEXT,                -- A/B/C/D
    is_diverted             INTEGER DEFAULT 0,
    is_delayed_15           INTEGER DEFAULT 0    -- delay >= 15 phút
);

-- ------------------------------------------------------------
-- INDEXES để tăng tốc truy vấn OLAP
-- ------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_fact_airline   ON fact_flights(airline_key);
CREATE INDEX IF NOT EXISTS idx_fact_origin    ON fact_flights(origin_airport_key);
CREATE INDEX IF NOT EXISTS idx_fact_dest      ON fact_flights(dest_airport_key);
CREATE INDEX IF NOT EXISTS idx_fact_date      ON fact_flights(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_cancelled ON fact_flights(is_cancelled);
CREATE INDEX IF NOT EXISTS idx_fact_delayed   ON fact_flights(is_delayed_15);
