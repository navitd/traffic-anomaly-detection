# Traffic Sensor Anomaly Detection

ETL + anomaly detection pipeline on Caltrans freeway loop-detector data (volume, occupancy, speed): sensor-level data quality flags, unsupervised anomaly detection, and a map-based dashboard.

## Status

Phase 1 (MVP) in progress. See `docs/plan.md` — *(not yet written)*.

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
uv run streamlit run app/streamlit_app.py
```

## Layout

```
src/traffic_anomaly/   # ingest, transform, detect, geo
app/                    # Streamlit dashboard
tests/
scripts/                # data fetch helpers
data/                   # gitignored; raw/interim/processed
```
