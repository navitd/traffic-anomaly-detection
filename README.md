# Traffic Sensor Anomaly Detection

**🚧 Work in progress — 3 of 5 pipeline stages built and tested; anomaly detection and the dashboard are not yet implemented. See [Status](#status) below.**

An ETL + anomaly detection pipeline on Caltrans freeway loop-detector data (volume, occupancy, speed). Planned end-to-end scope: sensor-level data quality flags, unsupervised anomaly detection, and a map-based dashboard for drilling into flagged sensors.

## Status

Phase 1 (MVP) in active development, updated regularly.

| Stage | Module | Status |
|---|---|---|
| Ingest | `ingest.py` | ✅ done — raw PeMS-BAY CSV → tidy DuckDB `readings`/`sensor_locations` tables |
| Data-quality validation | `data_validation.py` | ✅ done — completeness, uniqueness, validity, consistency, accuracy, and timeliness checks |
| Feature engineering | `transform.py` | ✅ done — SQL-derived features (deltas, rolling stats, time-of-day) + per-sensor STL decomposition |
| Anomaly detection | `detect.py` | ⬜ not started — planned: PyOD Isolation Forest + STL-residual z-score baseline |
| Dashboard | `app/streamlit_app.py` | ⬜ placeholder page only — planned: map + time-series drill-down, deployed to Streamlit Community Cloud |

Validation against the PeMS-BAY bootstrap data already surfaced real findings worth noting: every sensor is missing exactly the 1-hour DST spring-forward gap (Mar 12, 2017), and ~3,090 stuck-sensor runs consistent with the benchmark's own gap-imputation method rather than genuine hardware faults.

## Data

- **Primary**: raw [Caltrans PeMS](https://dot.ca.gov/programs/traffic-operations/mpr/pems-source) loop-detector data (account pending approval).
- **Fallback / bootstrap**: PeMS-BAY (325 Bay Area sensors, Jan–May 2017), mirrored via `scripts/fetch_pems_bay.sh`.

Raw data is not committed to this repo (see `.gitignore`) — run the fetch script or place your own PeMS export under `data/raw/`.

## Stack

DuckDB, Polars, GeoPandas, PyOD, ruptures, Streamlit, pytest. Managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
uv run pytest
uv run streamlit run app/streamlit_app.py  # currently a placeholder page — see Status
```

## Layout

```
src/traffic_anomaly/   # ingest, data_validation, transform, detect, geo
app/                    # Streamlit dashboard
tests/
scripts/                # data fetch helpers
data/                   # gitignored; raw/interim/processed
```
