# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

University (HCMUT) data-mining / data-warehouse project on the Kaggle **US Flight Delays 2015** dataset (~5.8M flights). Components: a streaming ETL into a SQLite star schema, a Streamlit OLAP dashboard, ML models that predict `arrival_delay >= 15 min`, and LaTeX report/slides. Code comments, docstrings, log messages and UI text are in **Vietnamese** — keep new text consistent with that.

There is no test suite, linter config, or build system. Every script is run from the **repo root**.

## Commands

```bash
pip install -r requirements.txt
pip install xgboost lightgbm          # not in requirements.txt but needed by mining/ and the dashboard predict page
# xlrd may be needed to read data/L_AIRPORT_ID.xls depending on pandas version

python pipeline.py                    # full ETL → warehouse/airline_dw.db (overwrites; logs to pipeline.log)
streamlit run dashboard/app.py        # dashboard (needs the DB; predict page also needs mining/model_delay.pkl)

python mining/predict_delay.py        # train production XGBoost → mining/model_delay.pkl
python mining/predict_cli.py --airline DL --origin CA --dest NY --hour 17 --month 6
python mining/run_models.py --model xgb --limit 100000   # one model: lr | rf | nn | xgb | lgb; or --all
python mining/compare_5_models.py     # 5-model comparison → mining/five_models_report.html

python scripts/eda_before_etl.py --nrows 500000          # raw-data EDA → data/eda_output/ (--nrows 0 = whole file)
python -m data.run_preprocessing_eda --nrows 500000      # preprocessing EDA figures for report ch.4
```

Raw inputs (`data/flights.csv`, `airlines.csv`, `airports.csv`) are downloaded from Kaggle and are git-ignored, as are `*.db`, `*.pkl`, `*.csv`, `*.json`, `*.log`, and LaTeX build outputs. Only `data/L_AIRPORT_ID.xls` is committed.

## Architecture

### Data flow

`data/*.csv` → `etl/` (via `pipeline.py`) → `warehouse/airline_dw.db` → read by `dashboard/`, `mining/`, `queries/olap_queries.sql`. `mining/predict_delay.py` writes `mining/model_delay.pkl`, which `dashboard/predict_service.py` loads.

### ETL (`pipeline.py` + `etl/`)

Designed to keep RAM ~one chunk: `extract.stream_flights` yields `CHUNK_SIZE` (100k) row chunks with explicit narrow dtypes (`FLIGHTS_DTYPE`); each chunk goes through `transform.transform_chunk` and is inserted with `load.load_fact_flights`, then freed. Dimensions are loaded first and return natural-key → surrogate-key dicts (`airline_map`, `airport_map`, `date_map` keyed by `"YYYYMMDD"` string, `delay_reason_map`) that `transform_chunk` uses to map FKs. `dim_date` is built by a separate fast pre-scan of `flights.csv` (`scan_dim_date`).

- `transform.py` must stay **vectorized** (no `apply`/`iterrows`) — this is a stated design constraint.
- `init_schema` runs `warehouse/schema.sql`, which `DROP`s and recreates all tables, so every pipeline run rebuilds the DB. `dim_delay_reason` rows are seeded in the schema SQL, not by Python.
- Some months of `flights.csv` use numeric BTS airport IDs instead of IATA codes. `etl/airport_id_lookup.py` builds an ID → IATA map from `L_AIRPORT_ID.xls` by fuzzy-matching normalized airport names against `airports.csv`; `transform_chunk` applies it before FK mapping. Rows whose FKs can't be mapped are dropped.
- `delay_reason_key` is a single dominant reason chosen by `np.select` priority order: WEATHER > AIR_SYSTEM > LATE_AIRCRAFT > AIRLINE > SECURITY > NO_DELAY.
- `is_delayed_15` (= `ARRIVAL_DELAY >= 15`) is the ML target. Null delay columns are filled with 0.

### Dashboard (`dashboard/`)

`dashboard/app.py` is the entry point; it adds `dashboard/` to `sys.path`, so modules import each other by bare name (`from config import ...`, `from db import query`). `sidebar.render_app_sidebar` returns `(page_id, FilterContext)`; `app.py` dispatches to `page_*.render_page_*(filters, query)`. To add a page, register it in `APP_PAGES` and `PAGE_MAIN_TITLES` in `sidebar.py` and add a branch in `app.py`.

- Pages build raw SQL strings on top of the shared `BASE_JOIN` / `BASE_JOIN_REASON` fragments in `config.py` (aliases `f`, `d`, `a`, `ap_o`, `ap_d`, `dr`) and run them through `db.query` (cached for 5 minutes).
- Icons and labels are centralized in `icons.py`.
- Deployment (Streamlit Community Cloud, entrypoint `dashboard/app.py`): the DB and model are not in git, so `remote_data.ensure_remote_data()` downloads them from the public HF dataset `kkoi31/airline-dw-data` (`DATA_REPO` in `config.py`) when they're missing. Cloud deps are in `dashboard/requirements.txt` and `packages.txt` (`libgomp1` for xgboost). The root `requirements.txt` pins versions that don't install on Python ≥3.13. `deploy/hf-space/` holds a tested Docker Space setup, but HF Docker Spaces require a PRO plan.
- The root-level `app.py` is an older single-file version of the dashboard. Prefer `dashboard/app.py`; don't edit both unless asked.

### Mining (`mining/`)

Features (shared by all models): `month, day_of_week, is_weekend, quarter, dep_hour, distance, elapsed_time` + label-encoded `airline_iata`, origin `state`, dest `state`. Non-cancelled flights only; `dep_hour` is taken from `scheduled_departure` (HHMM).

- `predict_delay.py` is the **production model**: XGBoost with `scale_pos_weight` for class imbalance, a month-stratified 500k sample, and a pickle of `{"model": clf, "encoders": {"airline", "origin_state", "dest_state", "feature_cols"}}`. `predict_service.py`, `predict_cli.py` and `predict_delay.predict_flight` each rebuild the feature row by hand. **If you change features or encoders, update all three** and retrain.
- `model_common.py` + `model_*.py` (each exposes `run(limit)`) + `run_models.py` form the modular per-model benchmark. `run_models.py` imports them through `MODEL_REGISTRY`.
- `compare_*.py` and `analyze_imbalance.py` are large standalone scripts that duplicate their own load/preprocess code and produce self-contained Plotly HTML reports. `model_common.py` mirrors `compare_5_models.py`'s SQL, features and split, so keep the two in sync.

### Documents

`report/main.tex` (article using `report-template.sty`; chapters in `report/content/chap*.tex`) and `slide/main.tex` (Beamer, sections in `slide/Sections/`) are Vietnamese LaTeX (`T5` fontenc). Figures in `report/img/` and `slide/images/` are copied, not generated in place, so re-copy them after regenerating EDA or model plots. `report/REQUIRE.md` contains the assignment requirements and the report outline.
