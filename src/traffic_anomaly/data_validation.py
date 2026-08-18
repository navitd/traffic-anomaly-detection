"""Data quality checks on `readings`/`sensor_locations`: completeness, uniqueness, validity, consistency, accuracy, timeliness."""

import duckdb

BAY_AREA_LAT_RANGE = (37.2, 37.9)
BAY_AREA_LON_RANGE = (-122.5, -121.6)
SPEED_RANGE_MPH = (0, 100)
STUCK_RUN_THRESHOLD = 12  # 12 * 5min = 1 hour of an identical reading


def null_counts(con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> dict[str, int]:
    """Completeness: null count per column."""
    exprs = ", ".join(f"count(*) FILTER (WHERE {c} IS NULL) AS {c}" for c in columns)
    row = con.execute(f"SELECT {exprs} FROM {table}").fetchone()
    return dict(zip(columns, row))


def duplicate_reading_count(con: duckdb.DuckDBPyConnection) -> int:
    """Uniqueness: rows in `readings` sharing a (sensor_id, ts) key."""
    (count,) = con.execute("""
        SELECT count(*) FROM (
            SELECT sensor_id, ts FROM readings
            GROUP BY sensor_id, ts HAVING count(*) > 1
        )
    """).fetchone()
    return count


def out_of_range_speed_count(con: duckdb.DuckDBPyConnection, bounds: tuple[float, float] = SPEED_RANGE_MPH) -> int:
    """Validity: readings outside a physically plausible speed range."""
    (count,) = con.execute(
        "SELECT count(*) FROM readings WHERE speed_mph < ? OR speed_mph > ?", list(bounds)
    ).fetchone()
    return count


def invalid_coordinate_count(
    con: duckdb.DuckDBPyConnection,
    lat_range: tuple[float, float] = BAY_AREA_LAT_RANGE,
    lon_range: tuple[float, float] = BAY_AREA_LON_RANGE,
) -> int:
    """Validity: sensor coordinates outside the expected geographic bounding box."""
    (count,) = con.execute(
        "SELECT count(*) FROM sensor_locations "
        "WHERE latitude < ? OR latitude > ? OR longitude < ? OR longitude > ?",
        [*lat_range, *lon_range],
    ).fetchone()
    return count


def referential_integrity(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Consistency: sensor_ids present in one table but not the other."""
    (readings_only,) = con.execute("""
        SELECT count(DISTINCT r.sensor_id) FROM readings r
        LEFT JOIN sensor_locations s USING (sensor_id) WHERE s.sensor_id IS NULL
    """).fetchone()
    (locations_only,) = con.execute("""
        SELECT count(DISTINCT s.sensor_id) FROM sensor_locations s
        LEFT JOIN readings r USING (sensor_id) WHERE r.sensor_id IS NULL
    """).fetchone()
    return {"readings_only": readings_only, "locations_only": locations_only}


def sensor_id_cardinality(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Consistency: distinct sensor_id count per table."""
    (readings_n,) = con.execute("SELECT count(DISTINCT sensor_id) FROM readings").fetchone()
    (locations_n,) = con.execute("SELECT count(DISTINCT sensor_id) FROM sensor_locations").fetchone()
    return {"readings": readings_n, "sensor_locations": locations_n}


def stuck_sensor_runs(con: duckdb.DuckDBPyConnection, min_run_length: int = STUCK_RUN_THRESHOLD) -> list[tuple]:
    """Accuracy: runs of consecutive identical speed_mph readings per sensor (flatline signature).

    Uses the SQL "gaps and islands" pattern: a boolean flag for "same as previous
    reading", cumulative-summed to assign a group id to each run of identical values.
    """
    rows = con.execute(f"""
        WITH flagged AS (
            SELECT sensor_id, ts, speed_mph,
                   speed_mph IS NOT DISTINCT FROM LAG(speed_mph) OVER w AS same_as_prev
            FROM readings
            WINDOW w AS (PARTITION BY sensor_id ORDER BY ts)
        ),
        grouped AS (
            SELECT sensor_id, ts, speed_mph,
                   SUM(CASE WHEN same_as_prev THEN 0 ELSE 1 END) OVER (
                       PARTITION BY sensor_id ORDER BY ts
                   ) AS run_id
            FROM flagged
        )
        SELECT sensor_id, run_id, count(*) AS run_length, min(ts) AS run_start, max(ts) AS run_end
        FROM grouped
        GROUP BY sensor_id, run_id
        HAVING count(*) >= {min_run_length}
        ORDER BY run_length DESC
    """).fetchall()
    return rows


def cadence_gaps(con: duckdb.DuckDBPyConnection, interval_seconds: int = 300) -> list[tuple]:
    """Timeliness: sensors missing readings against the expected fixed-interval cadence.

    Expected count per sensor = (max(ts) - min(ts)) / interval + 1, compared against
    the actual row count for that sensor.
    """
    rows = con.execute(f"""
        WITH bounds AS (SELECT min(ts) AS min_ts, max(ts) AS max_ts FROM readings),
        expected AS (
            SELECT CAST(epoch(max_ts) - epoch(min_ts) AS BIGINT) / {interval_seconds} + 1 AS expected_count
            FROM bounds
        )
        SELECT r.sensor_id, count(*) AS actual_count, e.expected_count,
               e.expected_count - count(*) AS missing_count
        FROM readings r, expected e
        GROUP BY r.sensor_id, e.expected_count
        HAVING count(*) <> e.expected_count
        ORDER BY missing_count DESC
    """).fetchall()
    return rows


def run_validation(con: duckdb.DuckDBPyConnection) -> dict:
    """Run all checks and return a summary report keyed by dimension."""
    return {
        "completeness": {
            "readings_nulls": null_counts(con, "readings", ["ts", "sensor_id", "speed_mph"]),
            "sensor_locations_nulls": null_counts(con, "sensor_locations", ["sensor_id", "latitude", "longitude"]),
        },
        "uniqueness": {"duplicate_readings": duplicate_reading_count(con)},
        "validity": {
            "out_of_range_speed": out_of_range_speed_count(con),
            "invalid_coordinates": invalid_coordinate_count(con),
        },
        "consistency": {
            "referential_integrity": referential_integrity(con),
            "sensor_id_cardinality": sensor_id_cardinality(con),
        },
        "accuracy": {"stuck_sensor_runs": stuck_sensor_runs(con)},
        "timeliness": {"cadence_gaps": cadence_gaps(con)},
    }
