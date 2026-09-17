"""Pixel to geographic conversion for map display. Pure raster code: no ML, no agent imports.

Only images that carry both a CRS and an affine transform can be placed on a map. Everything
here returns None when georeferencing is missing, and never guesses a location (CLAUDE.md §7).

Coordinate spaces are kept distinct by name (CLAUDE.md §8):
- `pixel`: column/row of the loaded (possibly decimated) grid
- `crs`: the image's own projected or geographic coordinates
- `wgs84`: longitude/latitude in EPSG:4326, what the web map consumes
"""

from dataclasses import dataclass
from pathlib import Path

from satquery.imaging import RasterImage
from satquery.schemas import ImageSummary

WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class Georeference:
    """Where a loaded raster sits on the earth, ready for a web map."""

    bounds_wgs84: tuple[float, float, float, float]  # west, south, east, north
    corners_wgs84: list[tuple[float, float]]  # top-left, top-right, bottom-right, bottom-left
    crs: str
    approximate: bool  # the grid is rotated, so 4-corner placement is an approximation
    note: str | None = None


def _affine(image: RasterImage) -> tuple[float, ...] | None:
    """The image's affine coefficients (a, b, c, d, e, f), or None if it is not georeferenced."""
    return None if image.crs is None or image.transform is None else tuple(image.transform[:6])


def _apply(coefficients: tuple[float, ...], column: float, row: float) -> tuple[float, float]:
    """Apply an affine to a pixel position. Written out rather than using the `*` operator, which
    the affine package deprecates for matrix multiplication."""
    a, b, c, d, e, f = coefficients
    return a * column + b * row + c, d * column + e * row + f


def _invert(coefficients: tuple[float, ...]) -> tuple[float, ...] | None:
    a, b, c, d, e, f = coefficients
    determinant = a * e - b * d
    if determinant == 0:
        return None
    ia, ib, id_, ie = e / determinant, -b / determinant, -d / determinant, a / determinant
    return ia, ib, -(ia * c + ib * f), id_, ie, -(id_ * c + ie * f)


def pixel_to_crs(image: RasterImage, x: float, y: float) -> tuple[float, float] | None:
    """Pixel (column, row) -> the image's own CRS coordinates."""
    coefficients = _affine(image)
    return None if coefficients is None else _apply(coefficients, x, y)


def georeference(image: RasterImage) -> Georeference | None:
    """Locate `image` on the earth, or None when it carries no CRS/transform.

    The four corners are reprojected individually, so a raster in any projected CRS can be pinned
    to a web map. When the transform has rotation terms the pinning is only an approximation,
    because a web map draws the quadrilateral with a linear interpolation of those corners.
    """
    transform = _affine(image)
    if transform is None:
        return None
    from rasterio.warp import transform as warp_points

    width, height = image.width, image.height
    pixel_corners = [(0, 0), (width, 0), (width, height), (0, height)]
    xs, ys = zip(*(_apply(transform, column, row) for column, row in pixel_corners))
    try:
        lons, lats = warp_points(image.crs, WGS84, list(xs), list(ys))
    except Exception:
        return None  # unknown or unprojectable CRS: say nothing rather than place it wrongly
    if not all(map(_finite, lons + lats)):
        return None

    corners = [(round(lon, 8), round(lat, 8)) for lon, lat in zip(lons, lats)]
    rotated = transform[1] != 0 or transform[3] != 0
    return Georeference(
        bounds_wgs84=(min(lons), min(lats), max(lons), max(lats)),
        corners_wgs84=corners,
        crs=image.crs,
        approximate=rotated,
        note="the raster grid is rotated; corner placement on the map is approximate" if rotated else None,
    )


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def bbox_wgs84_to_pixel(image: RasterImage, bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int] | None:
    """WGS84 (west, south, east, north) -> a pixel window (x_min, y_min, x_max, y_max).

    Clipped to the image. Returns None when the image is not georeferenced or the box misses it.
    """
    transform = _affine(image)
    if transform is None:
        return None
    from rasterio.warp import transform as warp_points

    west, south, east, north = bbox
    try:
        xs, ys = warp_points(WGS84, image.crs, [west, east, east, west], [north, north, south, south])
    except Exception:
        return None
    inverse = _invert(transform)
    if inverse is None:
        return None
    columns, rows = zip(*(_apply(inverse, x, y) for x, y in zip(xs, ys)))
    x_min, x_max = max(0, int(min(columns))), min(image.width, int(round(max(columns))))
    y_min, y_max = max(0, int(min(rows))), min(image.height, int(round(max(rows))))
    return (x_min, y_min, x_max, y_max) if x_max > x_min and y_max > y_min else None


def summarize(image: RasterImage, index: int) -> "ImageSummary":
    """`image.summary()` plus map placement. The one place that fills the WGS84 fields.

    Kept here rather than on `RasterImage` so that `imaging.py` stays free of projection code.
    """
    summary = image.summary(index)
    located = georeference(image)
    if located:
        summary.bounds_wgs84 = located.bounds_wgs84
        summary.corners_wgs84 = located.corners_wgs84
        summary.georeference_note = located.note
    return summary


@dataclass(frozen=True)
class Crop:
    """The result of restricting a raster to a drawn area."""

    width: int
    height: int
    source_width: int
    source_height: int

    @property
    def is_whole_image(self) -> bool:
        return self.width == self.source_width and self.height == self.source_height


MIN_CROP_PIXELS = 16  # validation.check_images rejects anything smaller


def crop_to_bbox(source: str | Path, bbox_wgs84: tuple[float, float, float, float],
                 destination: Path) -> Crop | None:
    """Write the part of `source` inside `bbox_wgs84` to `destination`, preserving georeferencing.

    Works on the source file at full resolution, not on a decimated in-memory copy, so the crop is
    as sharp as the data allows. Returns None when the raster has no CRS, the box misses it, or the
    overlap is too small to analyse; the caller then falls back to the whole image and says so.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds, intersection

    with rasterio.open(source) as src:
        if src.crs is None:
            return None
        try:
            bounds = transform_bounds(WGS84, src.crs, *bbox_wgs84, densify_pts=21)
            requested = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            window = intersection(requested, Window(0, 0, src.width, src.height))
        except Exception:
            return None  # no overlap, or an unprojectable box

        width, height = int(window.width), int(window.height)
        if width < MIN_CROP_PIXELS or height < MIN_CROP_PIXELS:
            return None
        if width == src.width and height == src.height:
            return Crop(width, height, src.width, src.height)  # nothing to cut

        profile = src.profile | {
            "width": width,
            "height": height,
            "transform": src.window_transform(window),
            "driver": "GTiff",
        }
        data = src.read(window=window)
        descriptions = src.descriptions

    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(data)
        for index, description in enumerate(descriptions):
            if description:
                dst.set_band_description(index + 1, description)

    with rasterio.open(source) as src:
        return Crop(width, height, src.width, src.height)
