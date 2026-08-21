from pathlib import Path

from traffic_anomaly.detect import (
    build_iforest_anomalies,
    build_zscore_anomalies,
    detection_summary,
    run_detection,
)
from traffic_anomaly.ingest import build_pems_bay_db
from traffic_anomaly.transform import build_features, build_stl_components

SPIKE_HOUR = 50
SPIKE_SPEED = 5.0  # far outside the ~65-76 mph normal range this series oscillates in
ZSCORE_THRESHOLD_FOR_TEST = 3.0


def _db_with_injected_spike(tmp_path: Path, n_hours: int = 96, sensor_id: str = "400001"):
    vel_csv = tmp_path / "vel.csv"
    header = f",{sensor_id}\n"
    lines = []
    for h in range(n_hours):
        speed = SPIKE_SPEED if h == SPIKE_HOUR else 65 + (h % 12)
        lines.append(f"2017-01-{1 + h // 24:02d} {h % 24:02d}:00:00,{speed}")
    vel_csv.write_text(header + "\n".join(lines) + "\n")
    return build_pems_bay_db(tmp_path / "traffic.duckdb", vel_csv)


def _prepared_db(tmp_path: Path):
    con = _db_with_injected_spike(tmp_path)
    build_features(con)
    build_stl_components(con, period=24)
    return con


def test_build_zscore_anomalies_flags_the_injected_spike(tmp_path: Path):
    con = _prepared_db(tmp_path)
    build_zscore_anomalies(con)

    row = con.execute(
        "SELECT is_anomaly, resid_zscore FROM zscore_anomalies "
        "WHERE ts = '2017-01-03 02:00:00'"  # hour 50 -> day 3, hour 2
    ).fetchone()
    is_anomaly, resid_zscore = row
    assert is_anomaly is True
    assert abs(resid_zscore) > ZSCORE_THRESHOLD_FOR_TEST


def test_build_zscore_anomalies_handles_zero_variance_sensor(tmp_path: Path):
    vel_csv = tmp_path / "vel.csv"
    header = ",400001\n"
    rows = "\n".join(f"2017-01-{1 + h // 24:02d} {h % 24:02d}:00:00,70.0" for h in range(48))
    vel_csv.write_text(header + rows + "\n")
    con = build_pems_bay_db(tmp_path / "traffic.duckdb", vel_csv)
    build_stl_components(con, period=24)

    build_zscore_anomalies(con)  # must not raise on a zero-variance residual

    flags = [r[0] for r in con.execute("SELECT is_anomaly FROM zscore_anomalies").fetchall()]
    assert not any(flags)


def test_build_iforest_anomalies_flags_the_injected_spike(tmp_path: Path):
    con = _prepared_db(tmp_path)
    build_iforest_anomalies(con)

    rows = con.execute("SELECT ts, is_anomaly, iforest_score FROM iforest_anomalies ORDER BY iforest_score DESC").fetchall()
    top_ts, top_is_anomaly, _ = rows[0]

    assert top_ts.strftime("%Y-%m-%d %H:%M:%S") == "2017-01-03 02:00:00"
    assert top_is_anomaly is True


def test_build_iforest_anomalies_drops_rows_with_no_prior_reading(tmp_path: Path):
    con = _prepared_db(tmp_path)
    build_iforest_anomalies(con)

    (first_ts,) = con.execute("SELECT min(ts) FROM readings").fetchone()
    (count_at_first_ts,) = con.execute(
        "SELECT count(*) FROM iforest_anomalies WHERE ts = ?", [first_ts]
    ).fetchone()
    assert count_at_first_ts == 0


def test_detection_summary_counts_and_agreement(tmp_path: Path):
    con = _prepared_db(tmp_path)
    build_zscore_anomalies(con)
    build_iforest_anomalies(con)

    summary = detection_summary(con)

    assert summary["zscore_anomalies"] >= 1
    assert summary["iforest_anomalies"] >= 1
    assert summary["flagged_by_both"] >= 1


def test_run_detection_builds_both_tables_and_returns_summary(tmp_path: Path):
    con = _prepared_db(tmp_path)

    summary = run_detection(con)

    assert set(summary) == {"zscore_anomalies", "iforest_anomalies", "flagged_by_both"}
    (zscore_table_count,) = con.execute("SELECT count(*) FROM zscore_anomalies").fetchone()
    (iforest_table_count,) = con.execute("SELECT count(*) FROM iforest_anomalies").fetchone()
    assert zscore_table_count > 0
    assert iforest_table_count > 0
