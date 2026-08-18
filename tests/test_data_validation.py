from pathlib import Path

from traffic_anomaly.data_validation import (
    cadence_gaps,
    duplicate_reading_count,
    invalid_coordinate_count,
    null_counts,
    out_of_range_speed_count,
    referential_integrity,
    sensor_id_cardinality,
    stuck_sensor_runs,
)
from traffic_anomaly.ingest import build_pems_bay_db


def _db(tmp_path: Path, vel_rows: str, loc_rows: str = "sensor_id,latitude,longitude\n400001,37.36,-121.90\n"):
    vel_csv = tmp_path / "vel.csv"
    vel_csv.write_text(vel_rows)
    loc_csv = tmp_path / "loc.csv"
    loc_csv.write_text(loc_rows)
    return build_pems_bay_db(tmp_path / "traffic.duckdb", vel_csv, loc_csv)


def test_null_counts_reports_zero_on_clean_data(tmp_path: Path):
    con = _db(tmp_path, ",400001\n2017-01-01 00:00:00,71.4\n")
    assert null_counts(con, "readings", ["ts", "sensor_id", "speed_mph"]) == {
        "ts": 0,
        "sensor_id": 0,
        "speed_mph": 0,
    }


def test_duplicate_reading_count_flags_repeated_key(tmp_path: Path):
    con = _db(
        tmp_path,
        ",400001\n2017-01-01 00:00:00,71.4\n2017-01-01 00:00:00,68.0\n2017-01-01 00:05:00,70.0\n",
    )
    # DuckDB's read_csv_auto rejects duplicate (ts, sensor) as-is fine here since sensor
    # is a column not a row key; two rows share the same ts, giving one duplicate pair.
    assert duplicate_reading_count(con) == 1


def test_out_of_range_speed_count_flags_negative_and_extreme_values(tmp_path: Path):
    con = _db(tmp_path, ",400001,400017\n2017-01-01 00:00:00,-5.0,250.0\n2017-01-01 00:05:00,60.0,55.0\n")
    assert out_of_range_speed_count(con) == 2


def test_invalid_coordinate_count_flags_point_outside_bounding_box(tmp_path: Path):
    con = _db(
        tmp_path,
        ",400001\n2017-01-01 00:00:00,71.4\n",
        loc_rows="sensor_id,latitude,longitude\n400001,0.0,0.0\n",
    )
    assert invalid_coordinate_count(con) == 1


def test_referential_integrity_flags_orphan_sensor_ids(tmp_path: Path):
    con = _db(
        tmp_path,
        ",400001,400099\n2017-01-01 00:00:00,71.4,60.0\n",
        loc_rows="sensor_id,latitude,longitude\n400001,37.36,-121.90\n400200,37.30,-121.80\n",
    )
    result = referential_integrity(con)
    assert result == {"readings_only": 1, "locations_only": 1}


def test_sensor_id_cardinality_counts_distinct_ids(tmp_path: Path):
    con = _db(
        tmp_path,
        ",400001,400017\n2017-01-01 00:00:00,71.4,60.0\n",
        loc_rows="sensor_id,latitude,longitude\n400001,37.36,-121.90\n400017,37.30,-121.80\n",
    )
    assert sensor_id_cardinality(con) == {"readings": 2, "sensor_locations": 2}


def test_stuck_sensor_runs_flags_long_identical_streak(tmp_path: Path):
    rows = ",400001\n"
    ts_values = [f"2017-01-01 {h:02d}:00:00" for h in range(13)]
    for ts in ts_values:
        rows += f"{ts},70.0\n"
    con = _db(tmp_path, rows)

    runs = stuck_sensor_runs(con, min_run_length=13)
    assert len(runs) == 1
    assert runs[0][0] == "400001"
    assert runs[0][2] == 13  # run_length


def test_stuck_sensor_runs_ignores_short_streaks(tmp_path: Path):
    con = _db(tmp_path, ",400001\n2017-01-01 00:00:00,70.0\n2017-01-01 00:05:00,70.0\n2017-01-01 00:10:00,72.0\n")
    assert stuck_sensor_runs(con, min_run_length=3) == []


def test_cadence_gaps_flags_sensor_missing_an_interval(tmp_path: Path):
    con = _db(
        tmp_path,
        ",400001,400017\n"
        "2017-01-01 00:00:00,71.4,60.0\n"
        "2017-01-01 00:05:00,71.6,\n"
        "2017-01-01 00:10:00,71.0,61.0\n",
    )
    gaps = cadence_gaps(con)
    assert len(gaps) == 1
    sensor_id, actual_count, expected_count, missing_count = gaps[0]
    assert sensor_id == "400017"
    assert missing_count == 1
