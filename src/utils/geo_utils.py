"""
Georeferencing helpers built on GeoPandas / Shapely.

In the production pipeline, drone orthomosaics carry real-world CRS
metadata (from onboard RTK-GPS + photogrammetry) so every detected tree
crown can be mapped to a GIS-ready point/polygon layer. Here we operate
on a synthetic, arbitrary local coordinate system (no real coordinates)
purely to demonstrate the pixel-to-map workflow.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Arbitrary synthetic reference frame - NOT a real survey site.
# In production this comes from the orthomosaic's embedded geotransform.
MOCK_ORIGIN_LON = 0.0
MOCK_ORIGIN_LAT = 0.0
MOCK_METERS_PER_PIXEL = 0.05  # ~5cm/px, typical of drone ortho GSD


def pixel_to_local_meters(x_px: float, y_px: float) -> tuple[float, float]:
    """Convert tile-space pixel coordinates to a local metric plane (meters)."""
    return x_px * MOCK_METERS_PER_PIXEL, y_px * MOCK_METERS_PER_PIXEL


def detections_to_geodataframe(
    detections: list[dict],
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Convert a list of detection dicts (with pixel_x/pixel_y/species/confidence)
    into a GeoDataFrame anchored at a synthetic origin.

    Each detection dict is expected to look like:
        {"pixel_x": 512, "pixel_y": 300, "species": "unknown", "confidence": 0.83}
    """
    records = []
    for det in detections:
        dx_m, dy_m = pixel_to_local_meters(det["pixel_x"], det["pixel_y"])
        # Rough small-angle offset from the mock origin; fine for demo purposes,
        # not geodetically rigorous.
        lon = MOCK_ORIGIN_LON + dx_m / 111_320.0
        lat = MOCK_ORIGIN_LAT + dy_m / 110_540.0
        records.append(
            {
                "species": det.get("species", "unknown"),
                "confidence": det.get("confidence"),
                "cluster_id": det.get("cluster_id"),
                "geometry": Point(lon, lat),
            }
        )

    return gpd.GeoDataFrame(pd.DataFrame(records), geometry="geometry", crs=crs)


def export_detections_geojson(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    """Persist a detections GeoDataFrame to a GeoJSON file for GIS tools (QGIS, ArcGIS)."""
    gdf.to_file(out_path, driver="GeoJSON")
