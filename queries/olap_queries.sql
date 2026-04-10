-- ============================================================
--  OLAP QUERIES — AIRLINE DATA WAREHOUSE
--  CDIO Project
-- ============================================================


-- ------------------------------------------------------------
-- 1. TỔNG SỐ CHUYẾN BAY THEO THÁNG / QUÝ / NĂM
-- ------------------------------------------------------------

-- Theo tháng
SELECT
    d.year,
    d.month,
    d.month_name,
    COUNT(*)                                        AS total_flights,
    SUM(f.is_cancelled)                             AS cancelled,
    SUM(f.is_delayed_15)                            AS delayed_15,
    ROUND(100.0 * SUM(f.is_cancelled) / COUNT(*), 2) AS cancel_rate_pct,
    ROUND(100.0 * SUM(f.is_delayed_15) / COUNT(*), 2) AS delay_rate_pct
FROM fact_flights f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;


-- Theo quý
SELECT
    d.year,
    d.quarter,
    COUNT(*)                                         AS total_flights,
    SUM(f.is_cancelled)                              AS cancelled,
    ROUND(100.0 * SUM(f.is_cancelled) / COUNT(*), 2) AS cancel_rate_pct
FROM fact_flights f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.year, d.quarter
ORDER BY d.year, d.quarter;


-- Theo năm
SELECT
    d.year,
    COUNT(*)                                         AS total_flights,
    SUM(f.is_cancelled)                              AS cancelled,
    SUM(f.is_delayed_15)                             AS delayed_15
FROM fact_flights f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.year
ORDER BY d.year;


-- ------------------------------------------------------------
-- 2. TOP 5 HÃNG CÓ NHIỀU CHUYẾN BAY NHẤT
-- ------------------------------------------------------------
SELECT
    a.airline_name,
    COUNT(*)                                          AS total_flights,
    SUM(f.is_cancelled)                               AS cancelled,
    ROUND(100.0 * SUM(f.is_cancelled) / COUNT(*), 2)  AS cancel_rate_pct
FROM fact_flights f
JOIN dim_airline a ON f.airline_key = a.airline_key
GROUP BY a.airline_name
ORDER BY total_flights DESC
LIMIT 5;


-- ------------------------------------------------------------
-- 3. TỈ LỆ CHUYẾN BAY ĐÚNG GIỜ (OTP) THEO HÃNG
--    OTP = 1 - tỉ lệ chuyến bay delay >= 15 phút
-- ------------------------------------------------------------
SELECT
    a.airline_name,
    COUNT(*)                                                   AS total_flights,
    SUM(f.is_delayed_15)                                       AS delayed,
    ROUND(100.0 * (1 - 1.0*SUM(f.is_delayed_15)/COUNT(*)), 2) AS otp_pct,
    ROUND(AVG(f.arrival_delay), 1)                             AS avg_arrival_delay_min
FROM fact_flights f
JOIN dim_airline a ON f.airline_key = a.airline_key
WHERE f.is_cancelled = 0
GROUP BY a.airline_name
ORDER BY otp_pct DESC;


-- ------------------------------------------------------------
-- 4. TỈ LỆ HỦY CHUYẾN THEO NGUYÊN NHÂN
-- ------------------------------------------------------------
SELECT
    f.cancellation_reason,
    CASE f.cancellation_reason
        WHEN 'A' THEN 'Carrier'
        WHEN 'B' THEN 'Weather'
        WHEN 'C' THEN 'National Air System'
        WHEN 'D' THEN 'Security'
        ELSE 'Unknown'
    END AS reason_label,
    COUNT(*)                                                   AS cancelled_flights,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2)         AS pct_of_total
FROM fact_flights f
WHERE f.is_cancelled = 1
GROUP BY f.cancellation_reason
ORDER BY cancelled_flights DESC;


-- ------------------------------------------------------------
-- 5. TRUNG BÌNH THỜI GIAN DELAY THEO SÂN BAY ĐI / ĐẾN
-- ------------------------------------------------------------

-- Sân bay đi
SELECT
    ap.airport_iata,
    ap.airport_name,
    ap.city,
    COUNT(*)                        AS total_flights,
    ROUND(AVG(f.departure_delay),1) AS avg_departure_delay,
    ROUND(AVG(f.arrival_delay),1)   AS avg_arrival_delay
FROM fact_flights f
JOIN dim_airport ap ON f.origin_airport_key = ap.airport_key
WHERE f.is_cancelled = 0
GROUP BY ap.airport_iata, ap.airport_name, ap.city
HAVING total_flights > 100
ORDER BY avg_departure_delay DESC
LIMIT 10;

-- Sân bay đến
SELECT
    ap.airport_iata,
    ap.airport_name,
    ap.city,
    COUNT(*)                       AS total_flights,
    ROUND(AVG(f.arrival_delay), 1) AS avg_arrival_delay
FROM fact_flights f
JOIN dim_airport ap ON f.dest_airport_key = ap.airport_key
WHERE f.is_cancelled = 0
GROUP BY ap.airport_iata, ap.airport_name, ap.city
HAVING total_flights > 100
ORDER BY avg_arrival_delay DESC
LIMIT 10;


-- ------------------------------------------------------------
-- 6. NGUYÊN NHÂN DELAY VÀ THỜI GIAN DELAY TRUNG BÌNH
-- ------------------------------------------------------------
SELECT
    dr.reason_name,
    dr.reason_category,
    COUNT(*)                              AS affected_flights,
    ROUND(AVG(
        CASE dr.reason_code
            WHEN 'AIR_SYSTEM'    THEN f.delay_air_system
            WHEN 'SECURITY'      THEN f.delay_security
            WHEN 'AIRLINE'       THEN f.delay_airline
            WHEN 'LATE_AIRCRAFT' THEN f.delay_late_aircraft
            WHEN 'WEATHER'       THEN f.delay_weather
            ELSE 0
        END
    ), 1) AS avg_delay_min
FROM fact_flights f
JOIN dim_delay_reason dr ON f.delay_reason_key = dr.reason_key
WHERE dr.reason_code != 'NO_DELAY'
  AND f.is_cancelled = 0
GROUP BY dr.reason_name, dr.reason_category
ORDER BY avg_delay_min DESC;


-- ------------------------------------------------------------
-- 7. XU HƯỚNG DELAY THEO THÁNG / MÙA
-- ------------------------------------------------------------
SELECT
    d.month_name,
    d.month,
    COUNT(*)                                          AS total_flights,
    ROUND(AVG(f.arrival_delay), 1)                   AS avg_delay_min,
    ROUND(100.0 * SUM(f.is_delayed_15)/COUNT(*), 2)  AS delay_rate_pct,
    CASE
        WHEN d.month IN (12,1,2) THEN 'Đông'
        WHEN d.month IN (3,4,5)  THEN 'Xuân'
        WHEN d.month IN (6,7,8)  THEN 'Hè'
        ELSE 'Thu'
    END AS season
FROM fact_flights f
JOIN dim_date d ON f.date_key = d.date_key
WHERE f.is_cancelled = 0
GROUP BY d.month, d.month_name
ORDER BY d.month;


-- ------------------------------------------------------------
-- 8. DELAY THEO KHUNG GIỜ BAY (giờ khởi hành)
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN CAST(SUBSTR(PRINTF('%04d', CAST(f.scheduled_departure AS INTEGER)), 1, 2) AS INTEGER) BETWEEN 0  AND 5  THEN '00-06 Đêm khuya'
        WHEN CAST(SUBSTR(PRINTF('%04d', CAST(f.scheduled_departure AS INTEGER)), 1, 2) AS INTEGER) BETWEEN 6  AND 11 THEN '06-12 Sáng'
        WHEN CAST(SUBSTR(PRINTF('%04d', CAST(f.scheduled_departure AS INTEGER)), 1, 2) AS INTEGER) BETWEEN 12 AND 17 THEN '12-18 Chiều'
        ELSE '18-24 Tối'
    END AS time_slot,
    COUNT(*)                                          AS total_flights,
    ROUND(AVG(f.departure_delay), 1)                 AS avg_dep_delay,
    ROUND(100.0 * SUM(f.is_delayed_15)/COUNT(*), 2)  AS delay_rate_pct
FROM fact_flights f
WHERE f.is_cancelled = 0
  AND f.scheduled_departure IS NOT NULL
GROUP BY time_slot
ORDER BY time_slot;


-- ------------------------------------------------------------
-- 9. TOP 5 SÂN BAY ĐÓNG GÓP NHIỀU NHẤT VÀO DELAY / HỦY
-- ------------------------------------------------------------
SELECT
    ap.airport_iata,
    ap.airport_name,
    ap.city,
    COUNT(*)                                           AS total_flights,
    SUM(f.is_cancelled)                                AS total_cancelled,
    SUM(f.is_delayed_15)                               AS total_delayed,
    ROUND(100.0*(SUM(f.is_cancelled)+SUM(f.is_delayed_15))/COUNT(*),2) AS disruption_rate_pct
FROM fact_flights f
JOIN dim_airport ap ON f.origin_airport_key = ap.airport_key
GROUP BY ap.airport_iata, ap.airport_name, ap.city
HAVING total_flights > 1000
ORDER BY disruption_rate_pct DESC
LIMIT 5;
