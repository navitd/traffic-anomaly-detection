from pathlib import Path

import duckdb

from traffic_anomaly.ingest import build_pems_bay_db


def test_build_pems_bay_db_unpivots_to_tidy_readings(tmp_path: Path):
    csv_path = tmp_path / "vel.csv"
    csv_path.write_text(
        ",400001,400017\n"
        "2017-01-01 00:00:00,71.4,67.8\n"
        "2017-01-01 00:05:00,71.6,67.5\n"
    )

    con = build_pems_bay_db(tmp_path / "traffic.duckdb", csv_path)

    rows = con.execute("SELECT ts, sensor_id, speed_mph FROM readings ORDER BY ts, sensor_id").fetchall()

    assert len(rows) == 4
    assert rows[0][1] == "400001"
    assert rows[0][2] == 71.4


def test_readings_table_has_no_nulls(tmp_path: Path):
    csv_path = tmp_path / "vel.csv"
    csv_path.write_text(",400001\n2017-01-01 00:00:00,71.4\n")

    con = build_pems_bay_db(tmp_path / "traffic.duckdb", csv_path)

    (null_count,) = con.execute("SELECT count(*) FROM readings WHERE speed_mph IS NULL").fetchone()
    assert null_count == 0


def test_sensor_locations_join_key_matches_readings(tmp_path: Path):
    vel_csv = tmp_path / "vel.csv"
    vel_csv.write_text(",400001,400017\n2017-01-01 00:00:00,71.4,67.8\n")
    loc_csv = tmp_path / "loc.csv"
    loc_csv.write_text(
        "sensor_id,latitude,longitude\n400001,37.364085,-121.901149\n400017,37.253303,-121.94544\n"
    )

    con = build_pems_bay_db(tmp_path / "traffic.duckdb", vel_csv, loc_csv)

    (unmatched,) = con.execute(
        "SELECT count(*) FROM readings r LEFT JOIN sensor_locations s "
        "USING (sensor_id) WHERE s.sensor_id IS NULL"
    ).fetchone()
    assert unmatched == 0
