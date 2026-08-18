from pathlib import Path

from traffic_anomaly.ingest import build_pems_bay_db
from traffic_anomaly.transform import build_features, build_stl_components, stl_decompose_sensor


def _db_with_hourly_series(tmp_path: Path, n_hours: int, sensor_id: str = "400001"):
    vel_csv = tmp_path / "vel.csv"
    header = f",{sensor_id}\n"
    rows = "\n".join(f"2017-01-{1 + h // 24:02d} {h % 24:02d}:00:00,{65 + (h % 12)}" for h in range(n_hours))
    vel_csv.write_text(header + rows + "\n")
    return build_pems_bay_db(tmp_path / "traffic.duckdb", vel_csv)


def test_build_features_computes_delta_and_rolling_stats(tmp_path: Path):
    con = _db_with_hourly_series(tmp_path, n_hours=5)
    build_features(con, window=2)

    rows = con.execute("SELECT delta_speed, rolling_mean, rolling_std FROM features ORDER BY ts").fetchall()

    assert rows[0][0] is None  # no prior reading to diff against
    assert rows[1][0] == 1  # 66 - 65
    assert rows[1][1] == 65.5  # mean of (65, 66)


def test_build_features_extracts_hour_and_day_of_week(tmp_path: Path):
    con = _db_with_hourly_series(tmp_path, n_hours=3)
    build_features(con)

    hours = [r[0] for r in con.execute("SELECT hour_of_day FROM features ORDER BY ts").fetchall()]
    assert hours == [0, 1, 2]


def test_stl_decompose_sensor_reconstructs_the_series(tmp_path: Path):
    con = _db_with_hourly_series(tmp_path, n_hours=48, sensor_id="400001")
    period = 24

    result = stl_decompose_sensor(con, "400001", period=period)

    assert set(result.columns) == {"sensor_id", "ts", "trend", "seasonal", "resid"}
    assert len(result) == 48
    reconstructed = result["trend"] + result["seasonal"] + result["resid"]
    original = con.execute(
        "SELECT speed_mph FROM readings WHERE sensor_id = '400001' ORDER BY ts"
    ).fetchall()
    original = [v[0] for v in original]
    assert all(abs(a - b) < 1e-6 for a, b in zip(reconstructed.to_list(), original))


def test_build_stl_components_runs_across_multiple_sensors_in_parallel(tmp_path: Path):
    vel_csv = tmp_path / "vel.csv"
    rows = ["," + "400001" + "," + "400017"]
    for h in range(48):
        ts = f"2017-01-{1 + h // 24:02d} {h % 24:02d}:00:00"
        rows.append(f"{ts},{65 + (h % 12)},{70 + (h % 8)}")
    vel_csv.write_text("\n".join(rows) + "\n")

    con = build_pems_bay_db(tmp_path / "traffic.duckdb", vel_csv)
    build_stl_components(con, period=24, max_workers=2)

    sensor_ids = {r[0] for r in con.execute("SELECT DISTINCT sensor_id FROM stl_components").fetchall()}
    assert sensor_ids == {"400001", "400017"}
    (count,) = con.execute("SELECT count(*) FROM stl_components").fetchone()
    assert count == 96
