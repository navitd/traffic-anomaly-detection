# Traffic Sensor Anomaly Detection

**🚧 Work in progress — all 5 pipeline stages built and tested; dashboard live at [traffic-anomaly-detection2609.streamlit.app](https://traffic-anomaly-detection2609.streamlit.app/). See [Status](#status) below.**

An end-to-end ETL + anomaly detection pipeline on Caltrans freeway loop-detector data: sensor-level data-quality checks, unsupervised anomaly detection (Isolation Forest + STL-residual z-scores), and a map-based dashboard for drilling into flagged sensors.

## Status

Phase 1 (MVP) is complete and deployed, running on the PeMS-BAY bootstrap data. Next: switch to raw Caltrans PeMS data once account access is approved.

| Stage | Module | Status |
|---|---|---|
| Ingest | `ingest.py` | ✅ done — raw PeMS-BAY CSV → tidy DuckDB `readings`/`sensor_locations` tables |
| Data-quality validation | `data_validation.py` | ✅ done — completeness, uniqueness, validity, consistency, accuracy, and timeliness checks |
| Feature engineering | `transform.py` | ✅ done — SQL-derived features (deltas, rolling stats, time-of-day) + per-sensor STL decomposition |
| Anomaly detection | `detect.py` | ✅ done — PyOD Isolation Forest (multivariate) + STL-residual z-score baseline, qualitative agreement summary |
| Dashboard | `app/streamlit_app.py` | ✅ done — map, time-series drill-down, STL decomposition; [deployed on Streamlit Community Cloud](https://traffic-anomaly-detection2609.streamlit.app/) |

Validation against the PeMS-BAY bootstrap data already surfaced real findings worth noting: every sensor is missing exactly the 1-hour DST spring-forward gap (Mar 12, 2017), and ~3,090 stuck-sensor runs consistent with the benchmark's own gap-imputation method rather than genuine hardware faults.

## Data

- **Primary**: raw [Caltrans PeMS](https://dot.ca.gov/programs/traffic-operations/mpr/pems-source) loop-detector data (account pending approval).
- **Fallback / bootstrap**: PeMS-BAY (325 Bay Area sensors, Jan–May 2017), mirrored via `scripts/fetch_pems_bay.sh`.

Raw data is not committed to this repo (see `.gitignore`) — run the fetch script or place your own PeMS export under `data/raw/`.

## Stack

DuckDB, Polars, statsmodels (STL), PyOD, GeoPandas, Streamlit + pydeck, pytest. Managed with [uv](https://docs.astral.sh/uv/).

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

If `data/processed/traffic.duckdb` doesn't exist, the dashboard downloads the prebuilt database (~1.3 GB) from [Hugging Face](https://huggingface.co/datasets/Navit6/traffic-anomaly-db) on first launch, so you can explore it without running the pipeline.

To rebuild the database from raw data instead:

```bash
bash scripts/fetch_pems_bay.sh
bash scripts/fetch_pems_bay_locations.sh
```

Then run the stages from a script (the STL step uses a process pool, so keep the `__main__` guard — it won't work pasted into a REPL or notebook):

```python
# build_db.py — run with: python build_db.py
from pathlib import Path

from traffic_anomaly.data_validation import run_validation
from traffic_anomaly.detect import run_detection
from traffic_anomaly.ingest import build_pems_bay_db
from traffic_anomaly.transform import build_features, build_stl_components

if __name__ == "__main__":
    raw = Path("data/raw/pems_bay")
    con = build_pems_bay_db(
        Path("data/processed/traffic.duckdb"), raw / "vel_pems_bay.csv", raw / "sensor_locations.csv"
    )
    print(run_validation(con))  # data-quality report
    build_features(con)
    build_stl_components(con)   # per-sensor STL, in parallel
    print(run_detection(con))   # Isolation Forest + z-score tables; returns a summary
```

## Deployment

The dashboard runs on [Streamlit Community Cloud](https://traffic-anomaly-detection2609.streamlit.app/) and redeploys on every push to `master`. Cloud containers start empty, so the app fetches the prebuilt DuckDB file from the public Hugging Face dataset at startup (downloaded to a temporary file and renamed only when complete). Streamlit Cloud installs from `requirements.txt`, generated with `uv export --no-dev --no-hashes --format requirements-txt`.

## Layout

```
src/traffic_anomaly/   # ingest, data_validation, transform, detect, geo
app/                    # Streamlit dashboard
tests/
scripts/                # data fetch helpers
data/                   # gitignored; raw/interim/processed
```
