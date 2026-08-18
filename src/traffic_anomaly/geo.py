"""Sensor metadata / geospatial join utilities (GeoPandas)."""

import duckdb
import geopandas as gpd


def sensor_locations_gdf(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    """Load the `sensor_locations` table as a GeoDataFrame of sensor points (EPSG:4326)."""
    df = con.execute("SELECT sensor_id, latitude, longitude FROM sensor_locations").fetchdf()
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
