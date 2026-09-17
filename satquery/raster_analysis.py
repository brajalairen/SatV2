"""Deterministic raster analysis (no ML). These are the labelled heuristic floors of D-009.

Thresholds adapt to each image (Otsu, percentiles) instead of fixed dB or reflectance values,
because target sensors (Cartosat-2S, RISAT) differ from Sentinel and their calibration is UNKNOWN.
Ported from the former `remote_sensing/` package: NDVI (optical/ndvi.py), backscatter
statistics (sar/backscatter.py), and the pixel-difference change detection (change_detection.py).
"""

import numpy as np
from scipy import ndimage

from satquery.imaging import RasterImage, sar_db, stretch


def otsu(values: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """Return (threshold, separability in [0, 1]) for finite values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0 or finite.min() == finite.max():
        return float("nan"), 0.0
    hist, edges = np.histogram(finite, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weights = hist / hist.sum()
    w0 = np.cumsum(weights)
    mu = np.cumsum(weights * centers)
    mu_total = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_total * w0 - mu) ** 2 / (w0 * (1 - w0))
    between = np.nan_to_num(between)
    best = int(np.argmax(between))
    total_var = float(np.var(finite))
    return float(centers[best]), float(between[best] / total_var) if total_var > 0 else 0.0


def band_stats(band: np.ndarray) -> dict:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return {"mean": None, "min": None, "max": None, "std": None}
    return {"mean": float(finite.mean()), "min": float(finite.min()), "max": float(finite.max()), "std": float(finite.std())}


def spectral_indices(image: RasterImage) -> dict | None:
    """NDVI and NDWI (McFeeters); None when red/green/NIR bands are unavailable."""
    red, green, nir = image.band("red"), image.band("green"), image.band("nir")
    if nir is None or red is None or green is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
        ndwi = (green - nir) / (green + nir)
    return {"ndvi": np.where(np.isfinite(ndvi), ndvi, np.nan), "ndwi": np.where(np.isfinite(ndwi), ndwi, np.nan)}


def backscatter_stats(image: RasterImage) -> dict:
    co_db, cross_db, units = sar_db(image)
    stats = {"units": units, "copol": band_stats(co_db)}
    if cross_db is not None:
        stats["crosspol"] = band_stats(cross_db)
        stats["copol_minus_crosspol"] = band_stats(co_db - cross_db)
    return stats


def sar_water_mask(image: RasterImage, smoothing_window: int = 5) -> tuple[np.ndarray, dict]:
    """Water appears dark (specular reflection). Speckle is averaged in the linear domain, then Otsu is applied in dB."""
    co_db, _, units = sar_db(image)
    linear = np.power(10.0, np.nan_to_num(co_db, nan=np.nanmin(co_db)) / 10)
    smoothed_db = 10 * np.log10(np.clip(ndimage.uniform_filter(linear, size=smoothing_window), 1e-6, None))
    smoothed_db[~np.isfinite(co_db)] = np.nan
    threshold, separability = otsu(smoothed_db)
    mask = np.nan_to_num(smoothed_db < threshold, nan=False).astype(bool)
    mask = ndimage.binary_opening(mask, iterations=1)
    return mask, {"threshold_db": threshold, "separability": separability, "units": units,
                  "fraction": float(mask.mean())}


def sar_bright_mask(image: RasterImage, percentile: float = 90) -> tuple[np.ndarray, dict]:
    """Strong backscatter (e.g. double-bounce from buildings). A heuristic candidate for built-up area, not a classifier."""
    co_db, _, units = sar_db(image)
    threshold = float(np.nanpercentile(co_db, percentile))
    mask = ndimage.binary_opening(np.nan_to_num(co_db > threshold, nan=False).astype(bool))
    return mask, {"threshold_db": threshold, "percentile": percentile, "units": units, "fraction": float(mask.mean())}


def change_map(before: RasterImage, after: RasterImage, min_region_px: int = 16) -> tuple[np.ndarray, dict]:
    """Change magnitude with an Otsu threshold.

    Optical: change-vector magnitude over per-image percentile-normalised bands.
    SAR: absolute log-ratio (dB difference).
    """
    if before.modality == "sar":
        magnitude = np.abs(sar_db(after)[0] - sar_db(before)[0])
        method = "SAR log-ratio |dB2 - dB1| + Otsu"
    else:
        n = min(before.data.shape[0], after.data.shape[0])
        norm = lambda img, k: stretch(img.data[k]).astype(np.float32) / 255
        magnitude = np.sqrt(sum((norm(after, k) - norm(before, k)) ** 2 for k in range(n)))
        method = "change-vector magnitude on percentile-normalised bands + Otsu"
    threshold, separability = otsu(magnitude)
    mask = np.nan_to_num(magnitude > threshold, nan=False).astype(bool)
    mask = remove_small_regions(ndimage.binary_opening(mask), min_region_px)
    return mask, {"method": method, "threshold": threshold, "separability": separability, "fraction": float(mask.mean())}


def remove_small_regions(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 0 or not mask.any():
        return mask
    labels, count = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    keep = np.isin(labels, np.flatnonzero(sizes >= min_px) + 1)
    return keep


def regions(mask: np.ndarray, max_regions: int = 5) -> list[dict]:
    """Largest connected regions: pixel bbox, share of image, location phrase."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return []
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    order = np.argsort(sizes)[::-1][:max_regions]
    slices = ndimage.find_objects(labels)
    height, width = mask.shape
    found = []
    for k in order:
        sy, sx = slices[k]
        bbox = (float(sx.start), float(sy.start), float(sx.stop), float(sy.stop))
        found.append({"bbox": bbox, "fraction": float(sizes[k] / mask.size),
                      "location": location_phrase(bbox, width, height)})
    return found


def location_phrase(bbox: tuple[float, float, float, float], width: int, height: int) -> str:
    cx, cy = (bbox[0] + bbox[2]) / 2 / width, (bbox[1] + bbox[3]) / 2 / height
    vertical = "north" if cy < 1 / 3 else "south" if cy > 2 / 3 else ""
    horizontal = "west" if cx < 1 / 3 else "east" if cx > 2 / 3 else ""
    return f"{vertical}-{horizontal}" if vertical and horizontal else vertical or horizontal or "centre"


def iou(a: np.ndarray, b: np.ndarray) -> float | None:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else None
