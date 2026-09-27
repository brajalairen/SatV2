"""Is a retrieved optical scene usable over the selected area? (D-030)

Judged pixel by pixel from Sentinel-2 L2A's own Scene Classification Layer (SCL), rendered on exactly
the grid of the retrieved bands, which is the drawn rectangle itself. Every figure here is therefore
about the user's area, never the catalogue's whole-tile cloud cover: at Loktak Lake on 2026-09-19 a
tile listed at 16.6% cloud left the selected area 42% under cloud or cloud shadow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCL_CLASSES = {0: "no data", 1: "saturated or defective", 2: "dark area", 3: "cloud shadow", 4: "vegetation",
               5: "not vegetated", 6: "water", 7: "unclassified", 8: "cloud, medium probability",
               9: "cloud, high probability", 10: "thin cirrus", 11: "snow or ice"}
# Pixels where the optical signal does not show the ground. They are counted as affected and are
# left out of every optical figure. "Unclassified" and "dark area" stay in: often real ground.
AFFECTED_CLASSES = (0, 1, 3, 8, 9, 10)
METHOD = ("Sentinel-2 L2A scene classification (SCL) over the selected area; affected = no data, saturated or "
          "defective, cloud shadow, cloud (medium or high probability) or thin cirrus")


@dataclass(frozen=True)
class OpticalQuality:
    """How much of the selected area the optical scene actually shows, and the verdict."""

    pixels: int
    clear_pixels: int
    affected_fraction: float
    class_fractions: dict[str, float]  # every SCL class present in the area, by name
    max_affected_fraction: float       # the configured limit (a SatQuery heuristic)
    min_clear_pixels: int
    usable: bool
    reason: str | None                 # why it is not usable; None when it is
    method: str = METHOD

    def as_dict(self) -> dict:
        return asdict(self)


def affected_mask(scl: np.ndarray) -> np.ndarray:
    return np.isin(scl, AFFECTED_CLASSES)


def assess(scl_path: Path, *, max_affected_fraction: float, min_clear_pixels: int) -> OpticalQuality:
    """The verdict for a water question: unusable when the affected share reaches the limit, or when
    too few clear pixels remain for a reliable optical figure, whatever the share."""
    import rasterio

    with rasterio.open(scl_path) as src:
        scl = src.read(1)
    pixels = int(scl.size)
    affected = affected_mask(scl)
    affected_fraction = float(affected.mean()) if pixels else 1.0
    clear = pixels - int(affected.sum())
    classes, counts = np.unique(scl, return_counts=True)
    fractions = {SCL_CLASSES.get(int(c), f"class {int(c)}"): round(float(n) / pixels, 4) for c, n in zip(classes, counts)}

    reason = None
    if affected_fraction >= max_affected_fraction:
        reason = (f"{affected_fraction:.1%} of the selected area is cloud, cloud shadow or without valid data "
                  f"(Sentinel-2 scene classification), at or above the {max_affected_fraction:.0%} limit")
    elif clear < min_clear_pixels:
        reason = (f"only {clear} clear optical pixels remain in the selected area, fewer than the "
                  f"{min_clear_pixels} needed for a reliable optical figure")
    return OpticalQuality(pixels=pixels, clear_pixels=clear, affected_fraction=round(affected_fraction, 4),
                          class_fractions=fractions, max_affected_fraction=max_affected_fraction,
                          min_clear_pixels=min_clear_pixels, usable=reason is None, reason=reason)


def mask_affected(scene_path: Path, scl_path: Path, destination: Path) -> Path:
    """A copy of the scene with affected pixels set to NaN, the pipeline's own nodata: every tool
    then leaves them out, coverage counts clear pixels only, and overlays are transparent there."""
    import rasterio

    with rasterio.open(scene_path) as src, rasterio.open(scl_path) as classified:
        if (src.width, src.height) != (classified.width, classified.height):
            raise ValueError("the scene classification does not share the scene's pixel grid")
        data = src.read(masked=True).astype(np.float32).filled(np.nan)
        data[:, affected_mask(classified.read(1))] = np.nan
        profile = {"driver": "GTiff", "width": src.width, "height": src.height, "count": src.count,
                   "dtype": "float32", "crs": src.crs, "transform": src.transform, "nodata": float("nan"),
                   "compress": "deflate"}
        descriptions = src.descriptions
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(data)
        for index, description in enumerate(descriptions, start=1):
            if description:
                dst.set_band_description(index, description)
    return destination
