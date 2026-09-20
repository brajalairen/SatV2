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


# --------------------------------------------------------------------------- drawn areas
# A drawn area arrives as a GeoJSON Polygon or MultiPolygon in WGS84: rectangles and polygons as
# drawn, circles as the polygon approximation the map drew. Positions are (longitude, latitude).


def _polygons(geometry: dict) -> list:
    return [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]


def _positions(geometry: dict) -> list[tuple[float, float]]:
    return [(p[0], p[1]) for polygon in _polygons(geometry) for ring in polygon for p in ring]


def geometry_problem(geometry: dict) -> str | None:
    """Why `geometry` is not a usable WGS84 area, or None when it is."""
    if geometry.get("type") not in ("Polygon", "MultiPolygon"):
        return "the area must be a Polygon or MultiPolygon"
    try:
        polygons = _polygons(geometry)
        if not polygons:
            return "the area has no polygons"
        for polygon in polygons:
            if not polygon:
                return "a polygon has no rings"
            for ring in polygon:
                if len(ring) < 4:
                    return "a polygon ring needs at least 4 positions"
                for position in ring:
                    longitude, latitude = float(position[0]), float(position[1])
                    if not (_finite(longitude) and _finite(latitude)):
                        return "the area has a non-finite coordinate"
                    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                        return "the area has a coordinate outside longitude/latitude range"
    except (TypeError, IndexError, KeyError, ValueError):
        return "the area coordinates are malformed"
    return None


def geometry_bounds(geometry: dict) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a drawn area."""
    longitudes, latitudes = zip(*_positions(geometry))
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def is_bounding_box(geometry: dict, tolerance: float = 1e-9) -> bool:
    """True when the area is exactly its own WGS84 bounding box, i.e. a drawn rectangle.

    Such an area is analysed by the plain box crop, so rectangles keep their original code path.
    """
    if geometry["type"] != "Polygon" or len(geometry["coordinates"]) != 1:
        return False
    west, south, east, north = geometry_bounds(geometry)
    near = lambda a, b: abs(a - b) <= tolerance
    corners = set()
    for lon, lat in _positions(geometry):
        if not ((near(lon, west) or near(lon, east)) and (near(lat, south) or near(lat, north))):
            return False  # a vertex that is not a corner of the box
        corners.add((near(lon, east), near(lat, north)))
    ring = _positions(geometry)
    # Every edge runs along the box (a diagonal would make a triangle or a bow-tie), and all four
    # corners are visited.
    axis_aligned = all(near(a[0], b[0]) or near(a[1], b[1]) for a, b in zip(ring, ring[1:]))
    return axis_aligned and len(corners) == 4


@dataclass(frozen=True)
class Crop:
    """The result of restricting a raster to a drawn area."""

    width: int
    height: int
    source_width: int
    source_height: int
    masked: bool = False  # pixels outside a non-rectangular area were set to nodata

    @property
    def is_whole_image(self) -> bool:
        return self.width == self.source_width and self.height == self.source_height and not self.masked


MIN_CROP_PIXELS = 16  # validation.check_images rejects anything smaller


class AreaNotUsable(Exception):
    """Why a raster cannot be restricted to a drawn area. The message is written for the user."""


def _inside(geometry_wgs84: dict, crs, transform, shape: tuple[int, int]):
    """bool (height, width): pixels whose centre lies inside the area, or None if it cannot be projected."""
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom

    try:
        projected = transform_geom(WGS84, crs, geometry_wgs84)
        return geometry_mask([projected], out_shape=shape, transform=transform, invert=True)
    except Exception:
        return None


def crop_to_bbox(source: str | Path, bbox_wgs84: tuple[float, float, float, float],
                 destination: Path, geometry_wgs84: dict | None = None) -> Crop:
    """Write the part of `source` inside `bbox_wgs84` to `destination`, preserving georeferencing.

    Works on the source file at full resolution, not on a decimated in-memory copy, so the crop is
    as sharp as the data allows. Raises AreaNotUsable, saying why, when the raster has no CRS, the
    area misses it, or the overlap is too small to analyse; the caller then falls back to the whole
    image and passes the reason on.

    With `geometry_wgs84` (a circle or polygon), pixels whose centre lies outside it become nodata,
    so every tool analyses the drawn shape rather than its bounding box. The masked crop is written
    as float32 with NaN nodata, which is how `imaging.load_image` represents nodata anyway.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds, intersection

    with rasterio.open(source) as src:
        if src.crs is None:
            raise AreaNotUsable("the image has no coordinate reference system, so a map area cannot be placed on it")
        try:
            bounds = transform_bounds(WGS84, src.crs, *bbox_wgs84, densify_pts=21)
        except Exception as error:
            raise AreaNotUsable("the drawn area could not be projected onto the image's coordinate system") from error
        # Overlap is decided on real coordinates: after rounding to whole pixels a thin overlap
        # becomes an empty window, which rasterio would report as disjoint.
        west, south, east, north = bounds
        if east <= src.bounds.left or west >= src.bounds.right or north <= src.bounds.bottom or south >= src.bounds.top:
            raise AreaNotUsable("the drawn area does not overlap the image")
        too_small = AreaNotUsable(f"the part of the image inside the drawn area is smaller than "
                                  f"{MIN_CROP_PIXELS}x{MIN_CROP_PIXELS} pixels")
        try:
            requested = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            window = intersection(requested, Window(0, 0, src.width, src.height))
        except Exception as error:  # an overlap narrower than one pixel rounds to an empty window
            raise too_small from error

        width, height = int(window.width), int(window.height)
        if width < MIN_CROP_PIXELS or height < MIN_CROP_PIXELS:
            raise too_small

        inside = None
        if geometry_wgs84 is not None:
            inside = _inside(geometry_wgs84, src.crs, src.window_transform(window), (height, width))
            if inside is None:
                raise AreaNotUsable("the drawn shape could not be projected onto the image's coordinate system")
            if not inside.any():
                raise AreaNotUsable("no pixel of the image lies inside the drawn shape")
            if inside.all():
                inside = None  # the shape covers every pixel of the window: nothing to mask

        if width == src.width and height == src.height and inside is None:
            return Crop(width, height, src.width, src.height)  # nothing to cut

        descriptions = src.descriptions
        if inside is None:
            profile = src.profile | {
                "width": width,
                "height": height,
                "transform": src.window_transform(window),
                "driver": "GTiff",
            }
            data = src.read(window=window)
        else:
            # A fresh profile: the source's compression (e.g. JPEG) or predictor may not suit float32.
            profile = {
                "driver": "GTiff", "width": width, "height": height, "count": src.count,
                "dtype": "float32", "crs": src.crs, "transform": src.window_transform(window),
                "nodata": float("nan"), "compress": "deflate",
            }
            data = src.read(window=window, masked=True).astype(np.float32).filled(np.nan)
            data[:, ~inside] = np.nan
        source_width, source_height = src.width, src.height

    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(data)
        for index, description in enumerate(descriptions):
            if description:
                dst.set_band_description(index + 1, description)

    return Crop(width, height, source_width, source_height, masked=inside is not None)
