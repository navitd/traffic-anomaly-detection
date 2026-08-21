"""Anomaly detection: PyOD Isolation Forest and STL-residual z-score baseline."""

import duckdb
import polars as pl
from pyod.models.iforest import IForest

ZSCORE_THRESHOLD = 3.0
IFOREST_CONTAMINATION = 0.01
IFOREST_FEATURES = [
    "speed_mph",
    "delta_speed",
    "rolling_mean",
    "rolling_std",
    "trend",
    "seasonal",
    "resid",
]


MIN_RESID_STD_MPH = 1e-6  # floor for STDDEV_SAMP(resid); below this, treat variance as noise, not signal


def build_zscore_anomalies(con: duckdb.DuckDBPyConnection, threshold: float = ZSCORE_THRESHOLD) -> None:
    """Baseline: flag STL residuals more than `threshold` std deviations from their per-sensor mean.

    STL's loess arithmetic never produces an exactly-zero residual, even for a
    perfectly flat input series (floating-point noise on the order of 1e-14) — so a
    stuck/dead sensor's near-zero residual variance would otherwise blow up into
    spurious large z-scores rather than being recognized as "no real signal here."
    Flooring the std at MIN_RESID_STD_MPH (well below any real speed sensor's
    measurement resolution) avoids that instability.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE zscore_anomalies AS
        SELECT sensor_id, ts, resid,
               (resid - AVG(resid) OVER w) / GREATEST(STDDEV_SAMP(resid) OVER w, {MIN_RESID_STD_MPH}) AS resid_zscore,
               abs((resid - AVG(resid) OVER w) / GREATEST(STDDEV_SAMP(resid) OVER w, {MIN_RESID_STD_MPH})) > {threshold} AS is_anomaly
        FROM stl_components
        WINDOW w AS (PARTITION BY sensor_id)
    """)


def build_iforest_anomalies(
    con: duckdb.DuckDBPyConnection,
    contamination: float = IFOREST_CONTAMINATION,
    random_state: int = 0,
) -> None:
    """Fit an Isolation Forest on the multivariate feature matrix and flag outliers.

    Joins `features` (speed, delta, rolling stats, time-of-day) with `stl_components`
    (trend/seasonal/resid) so the model sees deviations that only show up in
    combination, not just a single feature crossing a threshold on its own.
    Rows with no prior reading (delta_speed IS NULL, i.e. each sensor's first row)
    are dropped rather than imputed.
    """
    feature_cols = ", ".join(IFOREST_FEATURES)
    df = con.execute(f"""
        SELECT f.sensor_id, f.ts, {feature_cols}
        FROM features f
        JOIN stl_components s USING (sensor_id, ts)
        WHERE f.delta_speed IS NOT NULL
    """).pl()

    model = IForest(contamination=contamination, random_state=random_state)
    model.fit(df.select(IFOREST_FEATURES).to_numpy())

    result = df.select(["sensor_id", "ts"]).with_columns(
        pl.Series("iforest_score", model.decision_scores_),
        pl.Series("is_anomaly", model.labels_.astype(bool)),
    )
    con.execute("CREATE OR REPLACE TABLE iforest_anomalies AS SELECT * FROM result")


def detection_summary(con: duckdb.DuckDBPyConnection) -> dict:
    """Qualitative evaluation: counts per method plus agreement between them.

    There's no ground truth for this dataset, so this can't report precision/recall —
    only how many anomalies each method flags and how much the two independently-built
    methods agree, as a sanity check rather than a rigorous accuracy measure.
    """
    (zscore_n,) = con.execute("SELECT count(*) FROM zscore_anomalies WHERE is_anomaly").fetchone()
    (iforest_n,) = con.execute("SELECT count(*) FROM iforest_anomalies WHERE is_anomaly").fetchone()
    (both_n,) = con.execute("""
        SELECT count(*) FROM zscore_anomalies z
        JOIN iforest_anomalies i USING (sensor_id, ts)
        WHERE z.is_anomaly AND i.is_anomaly
    """).fetchone()
    return {"zscore_anomalies": zscore_n, "iforest_anomalies": iforest_n, "flagged_by_both": both_n}


def run_detection(
    con: duckdb.DuckDBPyConnection,
    zscore_threshold: float = ZSCORE_THRESHOLD,
    iforest_contamination: float = IFOREST_CONTAMINATION,
) -> dict:
    """Run both detectors and return the qualitative-evaluation summary."""
    build_zscore_anomalies(con, threshold=zscore_threshold)
    build_iforest_anomalies(con, contamination=iforest_contamination)
    return detection_summary(con)
