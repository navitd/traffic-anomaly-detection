"""Traffic Sensor Anomaly Detection — map + time-series drill-down dashboard."""

from pathlib import Path

import duckdb
import pandas as pd
import pydeck as pdk
import streamlit as st

from traffic_anomaly.detect import detection_summary, run_detection
from traffic_anomaly.geo import sensor_locations_gdf

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "traffic.duckdb"

st.set_page_config(page_title="Traffic Sensor Anomaly Detection", layout="wide")


@st.cache_resource
def get_connection() -> duckdb.DuckDBPyConnection:
    """Open the pipeline's DuckDB file, building the anomaly-detection tables if they're missing."""
    con = duckdb.connect(str(DB_PATH))
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
def load_sensor_timeseries(_con: duckdb.DuckDBPyConnection, sensor_id: str) -> pd.DataFrame:
    """Raw speed + STL components + both detectors' flags for one sensor, joined on ts."""
    return _con.execute("""
        SELECT f.ts, f.speed_mph, s.trend, s.seasonal, s.resid,
               COALESCE(z.is_anomaly, false) AS zscore_flag,
               COALESCE(i.is_anomaly, false) AS iforest_flag
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


def render_drilldown(con: duckdb.DuckDBPyConnection, sensor_id: str) -> None:
    df = load_sensor_timeseries(con, sensor_id)

    st.subheader(f"Sensor {sensor_id} — speed & flagged anomalies")
    st.line_chart(df.set_index("ts")[["speed_mph"]])

    flagged = df[df["zscore_flag"] | df["iforest_flag"]]
    st.caption(
        f"{len(flagged)} flagged readings out of {len(df)} "
        f"({int(df['zscore_flag'].sum())} z-score, {int(df['iforest_flag'].sum())} Isolation Forest)"
    )
    if not flagged.empty:
        st.dataframe(
            flagged[["ts", "speed_mph", "resid", "zscore_flag", "iforest_flag"]],
            use_container_width=True,
        )

    st.subheader("STL decomposition")
    st.caption("Plotted separately — trend, seasonal, and residual sit on very different scales.")
    stl_df = df.set_index("ts")
    st.line_chart(stl_df[["trend"]])
    st.line_chart(stl_df[["seasonal"]])
    st.line_chart(stl_df[["resid"]])


def main() -> None:
    st.title("Traffic Sensor Anomaly Detection")
    # get_connection() is cached (st.cache_resource) and shared across every session/thread;
    # DuckDB connections aren't safe for concurrent use, so each script run gets its own
    # cursor — a lightweight thread-safe handle onto the same underlying database.
    con = get_connection().cursor()
    summary = load_sensor_summary(con)

    overall = detection_summary(con)
    cols = st.columns(3)
    cols[0].metric("Z-score anomalies", overall["zscore_anomalies"])
    cols[1].metric("Isolation Forest anomalies", overall["iforest_anomalies"])
    cols[2].metric("Flagged by both", overall["flagged_by_both"])
    st.caption(
        "No ground-truth labels exist for this dataset — these are detector "
        "agreement counts, not precision/recall."
    )

    render_map(summary)

    sensor_ids = sorted(summary["sensor_id"])
    default_sensor = summary.sort_values("total_anomalies", ascending=False)["sensor_id"].iloc[0]
    sensor_id = st.selectbox("Sensor", sensor_ids, index=sensor_ids.index(default_sensor))
    render_drilldown(con, sensor_id)


if __name__ == "__main__":
    main()
