"""Export georeferenced detections to GeoPackage / GeoJSON / Shapefile / CSV."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

DETECTION_COLUMNS = ["det_id", "image_name", "flight_id", "class", "score", "tier", "x1", "y1", "x2", "y2",
                     "lon", "lat", "utm_x", "utm_y", "utm_crs", "box_area_m2", "georef_method",
                     "georef_error_est_m", "n_views", "status", "model_version"]

# Shapefile column names are limited to 10 characters.
_SHP_RENAME = {"image_name": "image", "flight_id": "flight", "box_area_m2": "area_m2", "georef_method": "geo_meth",
               "georef_error_est_m": "geo_err_m", "model_version": "model", "utm_crs": "utm_crs"}


def detections_to_gdf(df: pd.DataFrame, footprints_lonlat: list[list[tuple[float, float]]] | None = None) -> gpd.GeoDataFrame:
    """Point geometry by default; footprint polygons (WGS84) when supplied."""
    if footprints_lonlat is not None:
        geom = [Polygon(ring) for ring in footprints_lonlat]
    else:
        geom = gpd.points_from_xy(df["lon"], df["lat"])
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")


def export_all(gdf: gpd.GeoDataFrame, out_dir: str | Path, stem: str, formats: list[str]) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if fmt == "gpkg":
            gdf.to_file(path, driver="GPKG", layer="detections")
        elif fmt == "geojson":
            gdf.to_file(path, driver="GeoJSON")
        elif fmt == "shp":
            gdf.rename(columns=_SHP_RENAME).to_file(path, driver="ESRI Shapefile")
        elif fmt == "csv":
            pd.DataFrame(gdf.drop(columns="geometry")).to_csv(path, index=False)
        else:
            raise ValueError(f"unknown export format {fmt}")
        written.append(path)
    return written


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class summary. Area is the footprint of detection BOXES, not canopy cover."""
    if df.empty:
        return pd.DataFrame(columns=["class", "tier", "n_detections", "candidate_region_area_m2", "mean_score"])
    return (df.groupby(["class", "tier"])
              .agg(n_detections=("det_id", "count"), candidate_region_area_m2=("box_area_m2", "sum"),
                   mean_score=("score", "mean"))
              .reset_index())
