"""Deterministic raster analysis (no ML). These are the labelled heuristic floors of D-009.

Thresholds adapt to each image (Otsu, percentiles) instead of fixed dB or reflectance values,
because target sensors (Cartosat-2S, RISAT) differ from Sentinel and their calibration is UNKNOWN.
Ported from the former `remote_sensing/` package: NDVI (optical/ndvi.py), backscatter
statistics (sar/backscatter.py), and the pixel-difference change detection (change_detection.py).
"""

import numpy as np
from scipy import ndimage

from satquery.imaging import RasterImage, percentile_bounds, sar_db, stretch


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


def coverage(mask: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Share of the analysed pixels that `mask` covers.

    Pixels outside `valid` (nodata, including pixels outside a drawn circle or polygon) are not
    counted, so a masked-out area cannot dilute the figure. Returns a NumPy float, exactly what
    `mask.mean()` returned before nodata was accounted for.
    """
    if valid is None or valid.all():
        return mask.mean()
    count = int(valid.sum())
    return (mask & valid).sum() / count if count else np.float64(0.0)


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
    valid = np.isfinite(co_db)
    if valid.all():
        linear = np.power(10.0, np.nan_to_num(co_db, nan=np.nanmin(co_db)) / 10)
        smoothed = ndimage.uniform_filter(linear, size=smoothing_window)
    else:
        # Average over valid neighbours only (normalised convolution), so nodata, including the
        # outside of a drawn circle or polygon, does not darken the pixels next to it into "water".
        linear = np.where(valid, np.power(10.0, np.where(valid, co_db, 0.0) / 10), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            smoothed = (ndimage.uniform_filter(linear, size=smoothing_window)
                        / ndimage.uniform_filter(valid.astype(np.float64), size=smoothing_window))
    smoothed_db = 10 * np.log10(np.clip(np.nan_to_num(smoothed, nan=1e-6), 1e-6, None))
    smoothed_db[~valid] = np.nan
    threshold, separability = otsu(smoothed_db)
    mask = np.nan_to_num(smoothed_db < threshold, nan=False).astype(bool)
    mask = ndimage.binary_opening(mask, iterations=1)
    return mask, {"threshold_db": threshold, "separability": separability, "units": units,
                  "fraction": float(coverage(mask, valid))}


def sar_bright_mask(image: RasterImage, percentile: float = 90) -> tuple[np.ndarray, dict]:
    """Strong backscatter (e.g. double-bounce from buildings). A heuristic candidate for built-up area, not a classifier."""
    co_db, _, units = sar_db(image)
    threshold = float(np.nanpercentile(co_db, percentile))
    mask = ndimage.binary_opening(np.nan_to_num(co_db > threshold, nan=False).astype(bool))
    return mask, {"threshold_db": threshold, "percentile": percentile, "units": units,
                  "fraction": float(coverage(mask, np.isfinite(co_db)))}


def change_map(before: RasterImage, after: RasterImage, min_region_px: int = 16) -> tuple[np.ndarray, dict]:
    """Change magnitude with an Otsu threshold.

    Optical: change-vector magnitude over bands put on one scale shared by both dates.
    SAR: absolute log-ratio (dB difference), already a common scale.

    The two dates must be measured against the same reference. Scaling each date by its own
    percentiles (as this did before 2026-09-20) makes unchanged ground look different whenever the
    other date's distribution moves, and cancels a change that shifts the whole scene.
    """
    if before.modality == "sar":
        magnitude = np.abs(sar_db(after)[0] - sar_db(before)[0])
        method = "SAR log-ratio |dB2 - dB1| + Otsu"
    else:
        n = min(before.data.shape[0], after.data.shape[0])
        squares = np.zeros(before.data.shape[1:], dtype=np.float32)
        for k in range(n):
            # One percentile range per band, from both dates pooled: unchanged ground then maps to
            # the same value on both dates, and a real difference survives.
            bounds = percentile_bounds(np.concatenate([before.data[k].ravel(), after.data[k].ravel()]))
            scaled = lambda band: stretch(band, bounds=bounds).astype(np.float32) / 255
            squares += (scaled(after.data[k]) - scaled(before.data[k])) ** 2
        magnitude = np.sqrt(squares)
        # stretch() renders nodata as 0; keep it out of the Otsu histogram and out of the result.
        missing = ~(np.isfinite(before.data).any(axis=0) & np.isfinite(after.data).any(axis=0))
        if missing.any():
            magnitude[missing] = np.nan
        method = "change-vector magnitude on bands scaled by one range shared by both dates + Otsu"
    threshold, separability = otsu(magnitude)
    mask = np.nan_to_num(magnitude > threshold, nan=False).astype(bool)
    mask = remove_small_regions(ndimage.binary_opening(mask), min_region_px)
    return mask, {"method": method, "threshold": threshold, "separability": separability,
                  "fraction": float(coverage(mask, np.isfinite(magnitude)))}


def remove_small_regions(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 0 or not mask.any():
        return mask
    labels, count = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    keep = np.isin(labels, np.flatnonzero(sizes >= min_px) + 1)
    return keep


def regions(mask: np.ndarray, max_regions: int = 5, valid: np.ndarray | None = None) -> list[dict]:
    """Largest connected regions: pixel bbox, share of the analysed pixels (see `coverage`), location phrase."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return []
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    order = np.argsort(sizes)[::-1][:max_regions]
    slices = ndimage.find_objects(labels)
    height, width = mask.shape
    total = mask.size if valid is None or valid.all() else max(int(valid.sum()), 1)
    found = []
    for k in order:
        sy, sx = slices[k]
        bbox = (float(sx.start), float(sy.start), float(sx.stop), float(sy.stop))
        found.append({"bbox": bbox, "fraction": float(sizes[k] / total),
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
