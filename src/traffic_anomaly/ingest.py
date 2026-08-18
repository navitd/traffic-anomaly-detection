"""Load raw PeMS / PeMS-BAY sensor data into DuckDB."""

from pathlib import Path

import duckdb


def load_wide_csv(con: duckdb.DuckDBPyConnection, csv_path: Path, table: str) -> None:
    """Load a wide timestamp-by-sensor CSV (first column is the timestamp) into `table`.

    DuckDB's auto-generated name for the unnamed first column depends on the
    total column count (e.g. "column0" vs "column000"), so it's renamed to
    "ts" positionally rather than assumed.
    """
    con.execute(
        f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto(?, header=true)",
        [str(csv_path)],
    )
    (first_col,) = con.execute(f"SELECT column_name FROM (DESCRIBE {table}) LIMIT 1").fetchone()
    con.execute(f'ALTER TABLE {table} RENAME "{first_col}" TO ts')


def unpivot_to_readings(con: duckdb.DuckDBPyConnection, wide_table: str, value_name: str) -> None:
    """Reshape a wide sensor-by-timestamp table into a tidy (ts, sensor_id, value) table."""
    con.execute(f"""
        CREATE OR REPLACE TABLE readings AS
        UNPIVOT {wide_table}
        ON COLUMNS(* EXCLUDE (ts))
        INTO NAME sensor_id VALUE {value_name}
    """)


def build_pems_bay_db(db_path: Path, velocity_csv: Path) -> duckdb.DuckDBPyConnection:
    """Build a DuckDB database with a tidy `readings(ts, sensor_id, speed_mph)` table."""
    con = duckdb.connect(str(db_path))
    load_wide_csv(con, velocity_csv, "raw_velocity_wide")
    unpivot_to_readings(con, "raw_velocity_wide", value_name="speed_mph")
    return con
