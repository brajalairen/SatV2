"""Image loading and rendering. Pure raster code: no ML, no agent imports.

Band-name conventions (Round 1). Band descriptions stored in the GeoTIFF always win.
Otherwise the names below are ASSUMED, `band_names_assumed` is set, and validation warns:
- PNG/JPEG optical -> red, green, blue
- TIFF optical: 1 band -> pan; 3 bands -> red, green, blue; 4 bands -> blue, green, red, nir
  (BigEarthNet export order; Cartosat-2S MX order is UNVERIFIED)
- SAR: 1 band -> copol; 2 bands -> copol, crosspol (VV/VH or HH/HV: polarization UNKNOWN)
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from satquery.schemas import ImageSummary

RASTER_SUFFIXES = {".tif", ".tiff"}
PLAIN_SUFFIXES = {".png", ".jpg", ".jpeg"}
SUPPORTED_SUFFIXES = RASTER_SUFFIXES | PLAIN_SUFFIXES

BAND_ALIASES = {
    "red": ("red", "b04", "b4", "r"),
    "green": ("green", "b03", "b3", "g"),
    "blue": ("blue", "b02", "b2", "b"),
    "nir": ("nir", "b08", "b8", "b8a"),
    "pan": ("pan", "panchromatic", "gray", "grey"),
    "copol": ("copol", "vv", "hh"),
    "crosspol": ("crosspol", "vh", "hv"),
}


@dataclass
class RasterImage:
    data: np.ndarray  # float32, shape (bands, height, width); NaN marks nodata
    band_names: list[str]
    modality: str  # "optical" or "sar"
    name: str
    crs: str | None = None
    transform: tuple[float, ...] | None = None  # affine (a, b, c, d, e, f) of the loaded grid
    acquired: str | None = None
    decimation: float = 1.0
    band_names_assumed: bool = False
    display_ready: bool = False  # 8-bit PNG/JPEG: render without stretching

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

    def band(self, role: str) -> np.ndarray | None:
        """Return the band playing `role` (e.g. "red", "nir", "copol"), or None."""
        lookup = {n.lower(): i for i, n in enumerate(self.band_names)}
        for alias in BAND_ALIASES.get(role, (role,)):
            if alias in lookup:
                return self.data[lookup[alias]]
        return None

    def summary(self, index: int) -> ImageSummary:
        return ImageSummary(index=index, name=self.name, modality=self.modality, width=self.width,
                            height=self.height, bands=self.band_names, crs=self.crs,
                            acquired=self.acquired, decimation=round(self.decimation, 3))


def default_band_names(modality: str, count: int, plain: bool) -> list[str]:
    if modality == "sar":
        return ["copol", "crosspol"][:count] + [f"band{i + 1}" for i in range(2, count)]
    if plain or count == 3:
        return ["red", "green", "blue"][:count]
    if count == 1:
        return ["pan"]
    if count == 4:
        return ["blue", "green", "red", "nir"]
    return [f"band{i + 1}" for i in range(count)]


def load_image(path: str | Path, modality: str, acquired: str | None = None,
               max_pixels: int = 2048 * 2048) -> RasterImage:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in RASTER_SUFFIXES:
        return _load_tiff(path, modality, acquired, max_pixels)
    if suffix in PLAIN_SUFFIXES:
        return _load_plain(path, modality, acquired, max_pixels)
    raise ValueError(f"unsupported format '{suffix}'")


def _load_tiff(path: Path, modality: str, acquired: str | None, max_pixels: int) -> RasterImage:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        scale = max(1.0, math.sqrt(src.width * src.height / max_pixels))
        out_h, out_w = max(1, int(src.height / scale)), max(1, int(src.width / scale))
        resampling = Resampling.average if scale > 1 else Resampling.nearest
        data = src.read(out_shape=(src.count, out_h, out_w), resampling=resampling, masked=True)
        data = data.astype(np.float32).filled(np.nan)
        transform = src.transform @ src.transform.scale(src.width / out_w, src.height / out_h)
        descriptions = list(src.descriptions)
        crs = src.crs.to_string() if src.crs else None
        has_georef = src.crs is not None or not src.transform.is_identity

    named = all(descriptions) and len(descriptions) == data.shape[0]
    names = [d.strip() for d in descriptions] if named else default_band_names(modality, data.shape[0], plain=False)
    return RasterImage(data=data, band_names=names, modality=modality, name=path.name, crs=crs,
                       transform=tuple(transform)[:6] if has_georef else None, acquired=acquired,
                       decimation=scale, band_names_assumed=not named)


def _load_plain(path: Path, modality: str, acquired: str | None, max_pixels: int) -> RasterImage:
    with Image.open(path) as im:
        im = im.convert("L" if modality == "sar" else "RGB")
        scale = max(1.0, math.sqrt(im.width * im.height / max_pixels))
        if scale > 1:
            im = im.resize((int(im.width / scale), int(im.height / scale)), Image.Resampling.BILINEAR)
        array = np.asarray(im, dtype=np.float32)
    data = array[None, ...] if array.ndim == 2 else np.transpose(array, (2, 0, 1))
    names = ["intensity"] if modality == "sar" else default_band_names(modality, data.shape[0], plain=True)
    return RasterImage(data=np.ascontiguousarray(data), band_names=names, modality=modality, name=path.name,
                       acquired=acquired, decimation=scale, band_names_assumed=True, display_ready=True)


def percentile_bounds(values: np.ndarray, low: float = 2, high: float = 98) -> tuple[float, float] | None:
    """The (low, high) percentile pair that `stretch` scales by, or None when nothing is finite.

    Computed once from several arrays, it puts them all on one scale: `raster_analysis.change_map`
    needs both dates of a pair measured against the same reference.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    lo, hi = np.percentile(finite, [low, high])
    return float(lo), float(hi)


def stretch(band: np.ndarray, low: float = 2, high: float = 98,
            bounds: tuple[float, float] | None = None) -> np.ndarray:
    """Percentile stretch to uint8 for display; NaN becomes 0.

    `bounds` applies a scale computed elsewhere instead of this band's own percentiles.
    """
    limits = bounds if bounds is not None else percentile_bounds(band, low, high)
    if limits is None:
        return np.zeros(band.shape, dtype=np.uint8)
    lo, hi = limits
    scaled = (band - lo) / (hi - lo) if hi > lo else np.zeros_like(band)
    return (np.nan_to_num(np.clip(scaled, 0, 1)) * 255).astype(np.uint8)


def sar_db(image: RasterImage) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Co- and cross-pol backscatter in dB. Units are inferred, and the inference is reported.

    A negative median means the data are already in dB: backscatter in dB sits mostly below zero,
    while linear intensity never has a negative median, even when noise subtraction leaves a few
    negative samples (a single negative value is therefore not evidence of dB). Otherwise the data
    are treated as linear intensity and converted to dB. RISAT calibration is UNKNOWN, so callers
    must not rely on absolute thresholds.
    """
    co = image.band("copol")
    co = image.data[0] if co is None else co
    cross = image.band("crosspol")
    if cross is None and image.data.shape[0] > 1 and image.band_names_assumed:
        cross = image.data[1]
    if image.display_ready:
        return co, cross, "display-scaled 8-bit intensity (not backscatter)"
    if np.isfinite(co).any() and np.nanmedian(co) < 0:
        return co, cross, "dB (inferred: most values are negative)"
    to_db = lambda b: 10 * np.log10(np.clip(b, 1e-6, None)) if b is not None else None
    return to_db(co), to_db(cross), "linear intensity converted to dB (assumed)"


def render_rgb(image: RasterImage) -> np.ndarray:
    """uint8 array (height, width, 3) for VLM input and evidence overlays."""
    if image.modality == "sar":
        co_db, cross_db, _ = sar_db(image)
        if image.display_ready or cross_db is None:
            gray = co_db.astype(np.uint8) if image.display_ready else stretch(co_db)
            return np.dstack([gray] * 3)
        # Standard dual-pol false colour: R = co-pol, G = cross-pol, B = co/cross ratio (dB difference).
        return np.dstack([stretch(co_db), stretch(cross_db), stretch(co_db - cross_db)])

    red, green, blue = image.band("red"), image.band("green"), image.band("blue")
    if red is None or green is None or blue is None:
        if image.data.shape[0] >= 3:
            red, green, blue = image.data[0], image.data[1], image.data[2]
        else:
            red = green = blue = image.data[0]
    if image.display_ready:
        return np.dstack([np.nan_to_num(b).clip(0, 255).astype(np.uint8) for b in (red, green, blue)])
    # One stretch shared by all three bands preserves colour balance (per-band stretching distorts true colour).
    stacked = np.stack([red, green, blue])
    return np.transpose(stretch(stacked), (1, 2, 0))
