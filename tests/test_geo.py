from pathlib import Path

from traffic_anomaly.geo import sensor_locations_gdf
from traffic_anomaly.ingest import build_pems_bay_db


def test_sensor_locations_gdf_has_point_geometry(tmp_path: Path):
    vel_csv = tmp_path / "vel.csv"
    vel_csv.write_text(",400001\n2017-01-01 00:00:00,71.4\n")
    loc_csv = tmp_path / "loc.csv"
    loc_csv.write_text("sensor_id,latitude,longitude\n400001,37.364085,-121.901149\n")

    con = build_pems_bay_db(tmp_path / "traffic.duckdb", vel_csv, loc_csv)
    gdf = sensor_locations_gdf(con)

    assert gdf.crs.to_epsg() == 4326
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].x == -121.901149
    assert gdf.geometry.iloc[0].y == 37.364085
