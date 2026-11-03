"""Configuration loading. The config is a plain nested dict with attribute-style helpers."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class Config(dict):
    """Dict with attribute access for nested keys: cfg.tiling.tile_px."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if overrides:
        _deep_update(data, overrides)
    cfg = Config(data)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def _deep_update(base: dict, new: dict) -> None:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_alias_map(cfg: dict) -> dict[str, str]:
    """normalised alias -> canonical class name."""
    amap: dict[str, str] = {}
    for canonical in cfg["classes"]:
        amap[_norm(canonical)] = canonical
    for canonical, aliases in cfg.get("class_aliases", {}).items():
        for alias in aliases:
            amap[_norm(alias)] = canonical
    return amap


def canonical_class(name: str, alias_map: dict[str, str]) -> str | None:
    return alias_map.get(_norm(name))


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve cfg.paths[key] relative to the config file's directory."""
    base = Path(cfg["_config_path"]).parent
    p = Path(cfg["paths"][key])
    return (p if p.is_absolute() else (base / p)).resolve()


def camera_for_flight(cfg: dict, flight_id: str) -> dict:
    """Camera parameters (sensor_width_mm, focal_mm) for a flight folder name."""
    cam = dict(cfg["camera"]["default"])
    for key, params in (cfg["camera"].get("per_flight") or {}).items():
        if key in flight_id:
            cam.update(params)
            break
    return cam
