"""Satellite imagery acquisition: how a raster is obtained, never how it is analysed.

A provider's only job is to turn an area of interest into a GeoTIFF on disk plus the metadata
describing where it came from. What happens next -- validation, intents, planner, executor, tools,
VLM -- is the existing pipeline, unchanged. Nothing in `satquery/agent/` or `satquery/specialists/`
imports this package.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SceneMetadata:
    """Provenance for one retrieved raster. Everything here is safe to show a user."""

    provider: str           # "Copernicus Data Space Ecosystem"
    collection: str         # "sentinel-2-l2a"
    satellite: str          # "Sentinel-2"
    product_level: str      # "L2A"
    acquired: str           # ISO date of the scene, e.g. "2026-09-14"
    acquired_datetime: str  # full ISO timestamp as the catalogue reported it
    cloud_cover: float | None
    bbox_wgs84: tuple[float, float, float, float]
    crs: str
    resolution_m: float
    bands: list[str]
    width: int
    height: int
    scene_id: str | None = None
    attribution: str = "Contains modified Copernicus Sentinel data"
    cached: bool = False
    alternatives_considered: int = 0  # how many scenes the search returned before selection

    def as_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class RetrievedScene:
    """A raster on disk plus where it came from."""

    path: Path
    metadata: SceneMetadata
    extra: dict = field(default_factory=dict)


class SatelliteDataProvider(Protocol):
    """Any imagery source. One method, so a second provider stays cheap to add."""

    name: str

    def retrieve(self, bbox_wgs84: tuple[float, float, float, float], bands: list[str],
                 destination: Path) -> RetrievedScene:
        """Fetch imagery covering `bbox_wgs84` and write a GeoTIFF to `destination`."""
        ...
