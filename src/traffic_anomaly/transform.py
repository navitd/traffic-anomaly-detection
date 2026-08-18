"""Feature engineering on validated readings (rolling stats, STL decomposition prep) — runs after data_validation."""

import concurrent.futures
import multiprocessing

import duckdb
import polars as pl
from statsmodels.tsa.seasonal import STL

ROLLING_WINDOW_INTERVALS = 12  # 12 * 5min = 1 hour
DAILY_PERIOD = 288  # 5-min intervals -> one full day


def build_features(con: duckdb.DuckDBPyConnection, window: int = ROLLING_WINDOW_INTERVALS) -> None:
    """Create a `features` table: rate of change, trailing rolling mean/std, time-of-day."""
    con.execute(f"""
        CREATE OR REPLACE TABLE features AS
        SELECT
            sensor_id,
            ts,
            speed_mph,
            speed_mph - LAG(speed_mph) OVER w AS delta_speed,
            AVG(speed_mph) OVER (w ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) AS rolling_mean,
            STDDEV_SAMP(speed_mph) OVER (w ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) AS rolling_std,
            EXTRACT(hour FROM ts) AS hour_of_day,
            EXTRACT(dow FROM ts) AS day_of_week
        FROM readings
        WINDOW w AS (PARTITION BY sensor_id ORDER BY ts)
    """)


def _stl_fit(args: tuple[str, list, list, int]) -> pl.DataFrame:
    """Run in a worker process: STL-decompose one sensor's speed series.

    robust=False — the robust (iteratively-reweighted) fit took minutes per
    sensor on the 6-month PeMS-BAY series, making a 325-sensor batch run for
    hours; the plain fit takes ~13s/sensor and parallelizes across sensors.
    """
    sensor_id, ts_values, speed_values, period = args
    stl = STL(speed_values, period=period, robust=False).fit()
    return pl.DataFrame({
        "sensor_id": sensor_id,
        "ts": ts_values,
        "trend": stl.trend,
        "seasonal": stl.seasonal,
        "resid": stl.resid,
    })


def stl_decompose_sensor(con: duckdb.DuckDBPyConnection, sensor_id: str, period: int = DAILY_PERIOD) -> pl.DataFrame:
    """STL-decompose a single sensor's speed series into trend/seasonal/resid."""
    df = con.execute(
        "SELECT ts, speed_mph FROM readings WHERE sensor_id = ? ORDER BY ts", [sensor_id]
    ).pl()
    return _stl_fit((sensor_id, df["ts"].to_list(), df["speed_mph"].to_numpy(), period))


def build_stl_components(
    con: duckdb.DuckDBPyConnection,
    sensor_ids: list[str] | None = None,
    period: int = DAILY_PERIOD,
    max_workers: int = 8,
) -> None:
    """Create an `stl_components` table by STL-decomposing each sensor in parallel."""
    if sensor_ids is None:
        sensor_ids = [r[0] for r in con.execute("SELECT DISTINCT sensor_id FROM readings").fetchall()]

    tasks = []
    for sid in sensor_ids:
        df = con.execute("SELECT ts, speed_mph FROM readings WHERE sensor_id = ? ORDER BY ts", [sid]).pl()
        tasks.append((sid, df["ts"].to_list(), df["speed_mph"].to_numpy(), period))

    # spawn, not the Linux default fork: forking while DuckDB's internal worker
    # threads are alive leaves child processes deadlocked on inherited lock state.
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        parts = list(executor.map(_stl_fit, tasks))

    combined = pl.concat(parts)
    con.execute("CREATE OR REPLACE TABLE stl_components AS SELECT * FROM combined")
