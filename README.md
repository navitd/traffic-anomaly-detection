# Traffic Sensor Anomaly Detection

**🚧 Work in progress — all 5 pipeline stages built and tested locally; not yet deployed. See [Status](#status) below.**

An ETL + anomaly detection pipeline on Caltrans freeway loop-detector data (volume, occupancy, speed). Planned end-to-end scope: sensor-level data quality flags, unsupervised anomaly detection, and a map-based dashboard for drilling into flagged sensors.

## Status

Phase 1 (MVP) in active development, updated regularly.

| Stage | Module | Status |
|---|---|---|
| Ingest | `ingest.py` | ✅ done — raw PeMS-BAY CSV → tidy DuckDB `readings`/`sensor_locations` tables |
| Data-quality validation | `data_validation.py` | ✅ done — completeness, uniqueness, validity, consistency, accuracy, and timeliness checks |
| Feature engineering | `transform.py` | ✅ done — SQL-derived features (deltas, rolling stats, time-of-day) + per-sensor STL decomposition |
| Anomaly detection | `detect.py` | ✅ done — PyOD Isolation Forest (multivariate) + STL-residual z-score baseline, qualitative agreement summary |
| Dashboard | `app/streamlit_app.py` | ✅ done — map, time-series drill-down, STL decomposition; not yet deployed |

Validation against the PeMS-BAY bootstrap data already surfaced real findings worth noting: every sensor is missing exactly the 1-hour DST spring-forward gap (Mar 12, 2017), and ~3,090 stuck-sensor runs consistent with the benchmark's own gap-imputation method rather than genuine hardware faults.

## Data

- **Primary**: raw [Caltrans PeMS](https://dot.ca.gov/programs/traffic-operations/mpr/pems-source) loop-detector data (account pending approval).
- **Fallback / bootstrap**: PeMS-BAY (325 Bay Area sensors, Jan–May 2017), mirrored via `scripts/fetch_pems_bay.sh`.

Raw data is not committed to this repo (see `.gitignore`) — run the fetch script or place your own PeMS export under `data/raw/`.

## Stack

DuckDB, Polars, GeoPandas, PyOD, ruptures, Streamlit, pytest. Managed with [uv](https://docs.astral.sh/uv/).

## Setup

Create the virtual environment **outside this repository** (e.g. `~/venvs/anomaly-detection`), not as an in-repo `.venv`. If the repo lives on a Windows-mounted drive (`/mnt/c/...` under WSL), an in-repo venv's symlinks can break Windows-based copies/backups of the folder.

```bash
python -m venv ~/venvs/anomaly-detection
source ~/venvs/anomaly-detection/bin/activate

uv sync --active
pytest
streamlit run app/streamlit_app.py
```

`uv sync --active` installs into the currently activated venv instead of managing its own in-repo one — run it (or plain `uv sync`) without an activated venv and it will create/use an in-repo `.venv` instead. In each new terminal, re-run the `source` line before working.

## Layout

```
src/traffic_anomaly/   # ingest, data_validation, transform, detect, geo
app/                    # Streamlit dashboard
tests/
scripts/                # data fetch helpers
data/                   # gitignored; raw/interim/processed
```
