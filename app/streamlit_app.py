"""Traffic Sensor Anomaly Detection — map + time-series drill-down dashboard."""

from pathlib import Path

import duckdb
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from traffic_anomaly.detect import detection_summary, run_detection
from traffic_anomaly.geo import sensor_locations_gdf

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "traffic.duckdb"
REMOTE_DB_URL = "https://huggingface.co/datasets/Navit6/traffic-anomaly-db/resolve/main/traffic.duckdb"
# DuckDB sizes itself from the host's RAM/cores, not the container's; cap it so the
# app stays inside Community Cloud's limits instead of being killed and restarted
# (a restart re-downloads the whole DB).
DUCKDB_MEMORY_LIMIT = "512MB"
DUCKDB_THREADS = 2
FLAGGED_ROWS_PER_DETECTOR = 100

st.set_page_config(page_title="Traffic Sensor Anomaly Detection", layout="wide")


def _download_db() -> None:
    """Fetch the prebuilt DB from Hugging Face — a fresh deploy container starts with no local copy."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Download to a sibling temp file and rename only once complete, so an interrupted
    # download never leaves a truncated traffic.duckdb that later loads would trust.
    part_path = DB_PATH.with_name(DB_PATH.name + ".part")
    with st.spinner("Downloading prebuilt database (~1.3GB, first load only)..."):
        response = requests.get(REMOTE_DB_URL, stream=True, timeout=300)
        response.raise_for_status()
        with open(part_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        part_path.replace(DB_PATH)


@st.cache_resource
def get_connection() -> duckdb.DuckDBPyConnection:
    """Open the pipeline's DuckDB file, downloading it first if this container doesn't have it yet."""
    if not DB_PATH.exists():
        _download_db()

    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads = {DUCKDB_THREADS}")
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "readings" not in tables or "stl_components" not in tables:
        st.error(
            "No pipeline data found. Run ingest -> data_validation -> transform "
            "(see README Setup) before launching the dashboard."
        )
        st.stop()
    if not {"zscore_anomalies", "iforest_anomalies"} <= tables:
        with st.spinner("Running anomaly detection (first load only — cached on disk after this)..."):
            run_detection(con)
    return con


@st.cache_data
def load_sensor_summary(_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per sensor: location + anomaly counts from both detectors, plus map styling columns."""
    gdf = sensor_locations_gdf(_con)
    zscore_counts = _con.execute("""
        SELECT sensor_id, count(*) FILTER (WHERE is_anomaly) AS zscore_count
        FROM zscore_anomalies GROUP BY sensor_id
    """).fetchdf()
    iforest_counts = _con.execute("""
        SELECT sensor_id, count(*) FILTER (WHERE is_anomaly) AS iforest_count
        FROM iforest_anomalies GROUP BY sensor_id
    """).fetchdf()

    df = pd.DataFrame(gdf.drop(columns="geometry")).merge(zscore_counts, on="sensor_id", how="left")
    df = df.merge(iforest_counts, on="sensor_id", how="left")
    df[["zscore_count", "iforest_count"]] = df[["zscore_count", "iforest_count"]].fillna(0)
    df["total_anomalies"] = df["zscore_count"] + df["iforest_count"]

    max_total = max(df["total_anomalies"].max(), 1)
    intensity = df["total_anomalies"] / max_total
    df["radius"] = 80 + 400 * intensity
    df["color_g"] = (255 - 255 * intensity).astype(int)
    return df


@st.cache_data
def load_detection_summary(_con: duckdb.DuckDBPyConnection) -> dict:
    """Cached: the agreement counts scan all ~17M rows of both detector tables."""
    return detection_summary(_con)


@st.cache_data(max_entries=20)
def load_sensor_timeseries(_con: duckdb.DuckDBPyConnection, sensor_id: str) -> pd.DataFrame:
    """Raw speed + STL components + both detectors' flags for one sensor, joined on ts."""
    return _con.execute("""
        SELECT f.ts, f.speed_mph, s.trend, s.seasonal, s.resid,
               COALESCE(z.is_anomaly, false) AS zscore_flag,
               COALESCE(i.is_anomaly, false) AS iforest_flag,
               i.iforest_score
        FROM features f
        JOIN stl_components s USING (sensor_id, ts)
        LEFT JOIN zscore_anomalies z USING (sensor_id, ts)
        LEFT JOIN iforest_anomalies i USING (sensor_id, ts)
        WHERE f.sensor_id = ?
        ORDER BY f.ts
    """, [sensor_id]).fetchdf()


def render_map(summary: pd.DataFrame) -> None:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=summary,
        get_position="[longitude, latitude]",
        get_radius="radius",
        get_fill_color="[255, color_g, 40, 180]",
        pickable=True,
    )
    view_state = pdk.ViewState(
        latitude=summary["latitude"].mean(),
        longitude=summary["longitude"].mean(),
        zoom=9,
    )
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>{sensor_id}</b><br/>z-score anomalies: {zscore_count}"
                    "<br/>Isolation Forest anomalies: {iforest_count}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        },
    ))


FULL_RESOLUTION_MAX_DAYS = 7  # longer windows are plotted hourly to keep charts responsive


def _hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly means, except the STL residual, which keeps each hour's value farthest from
    zero — a mean would average a short anomaly spike away."""
    hourly = df.resample("1h").mean(numeric_only=True)
    resid = df["resid"].resample("1h")
    high, low = resid.max(), resid.min()
    hourly["resid"] = high.where(high.abs() >= low.abs(), low)
    return hourly


def _strongest_flags(flagged: pd.DataFrame) -> pd.DataFrame:
    """Top flags from each detector by its own strength, so neither detector crowds out the other."""
    by_zscore = flagged[flagged["zscore_flag"]].assign(_strength=lambda d: d["resid"].abs())
    by_iforest = flagged[flagged["iforest_flag"]].assign(_strength=lambda d: d["iforest_score"])
    top = pd.concat([
        by_zscore.nlargest(FLAGGED_ROWS_PER_DETECTOR, "_strength"),
        by_iforest.nlargest(FLAGGED_ROWS_PER_DETECTOR, "_strength"),
    ])
    return top.drop(columns="_strength").drop_duplicates("ts").sort_values("ts")


def render_drilldown(con: duckdb.DuckDBPyConnection, sensor_id: str) -> None:
    df = load_sensor_timeseries(con, sensor_id)

    st.subheader(f"Sensor {sensor_id} — speed & flagged anomalies")
    first_day, last_day = df["ts"].min().date(), df["ts"].max().date()
    start, end = st.slider("Date range", first_day, last_day, (first_day, last_day))
    df = df[(df["ts"].dt.date >= start) & (df["ts"].dt.date <= end)]

    # A full sensor history is ~52k 5-minute readings per chart, which the browser
    # redraws on every interaction; plot hourly unless the window is short.
    if (end - start).days < FULL_RESOLUTION_MAX_DAYS:
        chart_df = df.set_index("ts")
        st.caption("5-minute readings.")
    else:
        chart_df = _hourly(df.set_index("ts"))
        st.caption(
            f"Hourly means (residual: each hour's most extreme value). Narrow the range "
            f"to under {FULL_RESOLUTION_MAX_DAYS} days to see individual 5-minute readings."
        )
    st.line_chart(chart_df[["speed_mph"]])

    flagged = df[df["zscore_flag"] | df["iforest_flag"]]
    st.caption(
        f"{len(flagged)} flagged readings out of {len(df)} "
        f"({int(df['zscore_flag'].sum())} z-score, {int(df['iforest_flag'].sum())} Isolation Forest)"
    )
    if not flagged.empty:
        columns = ["ts", "speed_mph", "resid", "iforest_score", "zscore_flag", "iforest_flag"]
        shown = _strongest_flags(flagged)
        if len(shown) < len(flagged):
            st.caption(
                f"Showing {len(shown)} of {len(flagged)}: the {FLAGGED_ROWS_PER_DETECTOR} strongest "
                "flags per detector (largest |residual| for z-score, highest score for Isolation Forest)."
            )
        st.dataframe(shown[columns], width="stretch", hide_index=True)
        st.download_button(
            "Download all flagged readings (CSV)",
            flagged[columns].to_csv(index=False),
            file_name=f"sensor_{sensor_id}_flagged.csv",
            mime="text/csv",
        )

    st.subheader("STL decomposition")
    st.caption("Plotted separately — trend, seasonal, and residual sit on very different scales.")
    st.line_chart(chart_df[["trend"]])
    st.line_chart(chart_df[["seasonal"]])
    st.line_chart(chart_df[["resid"]])


@st.fragment
def render_sensor_section(con: duckdb.DuckDBPyConnection, summary: pd.DataFrame) -> None:
    """Sensor picker + drill-down as a fragment: changing the sensor reruns only this part,
    not the map and summary metrics above it."""
    sensor_ids = sorted(summary["sensor_id"])
    default_sensor = summary.sort_values("total_anomalies", ascending=False)["sensor_id"].iloc[0]
    sensor_id = st.selectbox("Sensor", sensor_ids, index=sensor_ids.index(default_sensor))
    render_drilldown(con, sensor_id)


def main() -> None:
    st.title("Traffic Sensor Anomaly Detection")
    # get_connection() is cached (st.cache_resource) and shared across every session/thread;
    # DuckDB connections aren't safe for concurrent use, so each script run gets its own
    # cursor — a lightweight thread-safe handle onto the same underlying database.
    con = get_connection().cursor()
    summary = load_sensor_summary(con)

    overall = load_detection_summary(con)
    cols = st.columns(3)
    cols[0].metric("Z-score anomalies", overall["zscore_anomalies"])
    cols[1].metric("Isolation Forest anomalies", overall["iforest_anomalies"])
    cols[2].metric("Flagged by both", overall["flagged_by_both"])
    st.caption(
        "No ground-truth labels exist for this dataset — these are detector "
        "agreement counts, not precision/recall."
    )

    render_map(summary)
    render_sensor_section(con, summary)


if __name__ == "__main__":
    main()
