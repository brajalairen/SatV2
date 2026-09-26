"""Deterministic input validation (D-008). Its errors are authoritative: routing never overrides them."""

from datetime import date
from pathlib import Path

import numpy as np

from satquery.imaging import PLAIN_SUFFIXES, SUPPORTED_SUFFIXES, RasterImage, modality_from_band_names
from satquery.schemas import AnalysisRequest, InputConfig, ValidationIssue

GRID_TOLERANCE_PX = 1.0


def issue(code: str, message: str, severity: str = "error", image_index: int | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, severity=severity, message=message, image_index=image_index)


def check_request(request: AnalysisRequest) -> list[ValidationIssue]:
    """Checks that need no pixel reads: image count, file existence, format."""
    issues = []
    if not request.query.strip():
        issues.append(issue("empty_query", "Please enter a question or instruction."))
    if not 1 <= len(request.images) <= 2:
        issues.append(issue("image_count", f"Provide 1 or 2 images; got {len(request.images)}."))
    for i, image in enumerate(request.images):
        path = Path(image.path)
        if not path.is_file():
            issues.append(issue("file_missing", f"File not found: {path.name}", image_index=i))
        elif path.suffix.lower() not in SUPPORTED_SUFFIXES:
            issues.append(issue("unsupported_format",
                                f"{path.name}: use GeoTIFF/TIFF (or PNG/JPEG for benchmark images).", image_index=i))
        elif path.suffix.lower() in PLAIN_SUFFIXES:
            issues.append(issue("non_geospatial_format", f"{path.name}: PNG/JPEG carry no georeferencing; "
                                "SIH accepts them only for benchmark datasets.", "warning", i))
    return issues


def detect_input_config(modalities: list[str]) -> InputConfig | None:
    if len(modalities) == 1:
        return "single_sar" if modalities[0] == "sar" else "single_optical"
    if len(modalities) == 2:
        return "pair_bitemporal" if modalities[0] == modalities[1] else "pair_cross_modal"
    return None


def check_images(images: list[RasterImage]) -> list[ValidationIssue]:
    """Checks on loaded pixels, plus pair compatibility."""
    issues = []
    for i, image in enumerate(images):
        if image.width < 16 or image.height < 16:
            issues.append(issue("image_too_small", f"{image.name}: {image.width}x{image.height} px is too small.", image_index=i))
        if not np.isfinite(image.data).any():
            issues.append(issue("no_valid_pixels", f"{image.name}: every pixel is nodata.", image_index=i))
        if image.crs is None and not image.display_ready:
            issues.append(issue("no_crs", f"{image.name}: no CRS; results stay in pixel coordinates.", "warning", i))
        if image.band_names_assumed and not image.display_ready:
            issues.append(issue("band_order_assumed", f"{image.name}: no band descriptions; assumed "
                                f"{', '.join(image.band_names)}.", "warning", i))
        if image.decimation > 1:
            issues.append(issue("decimated", f"{image.name}: large raster read at 1/{image.decimation:.1f} resolution.",
                                "warning", i))
        if image.modality == "optical" and image.data.shape[0] == 2:
            issues.append(issue("modality_suspicious", f"{image.name}: 2 bands looks like SAR, but it was declared optical.",
                                "warning", i))
        if image.modality == "sar" and image.data.shape[0] > 2 and not image.display_ready:
            issues.append(issue("modality_suspicious", f"{image.name}: {image.data.shape[0]} bands is unusual for SAR.",
                                "warning", i))
        # Band descriptions stored in the file are authoritative: SAR analysed as optical (or the
        # reverse) would compute NDWI on backscatter, or backscatter on reflectance.
        stated = None if image.band_names_assumed else modality_from_band_names(image.band_names)
        if stated and stated != image.modality:
            issues.append(issue("modality_conflict", f"{image.name}: its bands ({', '.join(image.band_names)}) are "
                                f"{stated.upper() if stated == 'sar' else stated}, but it was declared {image.modality}. "
                                f"Mark it as {stated.upper() if stated == 'sar' else stated} and ask again.", image_index=i))
    if len(images) == 2:
        issues.extend(_check_pair(*images))
    return issues


def _check_pair(first: RasterImage, second: RasterImage) -> list[ValidationIssue]:
    issues = []
    if (first.height, first.width) != (second.height, second.width):
        issues.append(issue("grid_shape_mismatch", f"Pair must share a pixel grid: {first.width}x{first.height} vs "
                            f"{second.width}x{second.height}. Co-registration is not performed (D-008)."))
    if first.crs and second.crs and first.crs != second.crs:
        issues.append(issue("crs_mismatch", f"Pair uses different CRS: {first.crs} vs {second.crs}."))
    elif (first.crs is None) != (second.crs is None):
        issues.append(issue("georef_partial", "Only one image of the pair is georeferenced.", "warning"))
    if first.transform and second.transform and not issues:
        a1, b1, c1, d1, e1, f1 = first.transform[:6]
        a2, b2, c2, d2, e2, f2 = second.transform[:6]
        pixel_size = abs(a1) or 1.0
        # Same shape and origin is not enough: a different pixel size (or rotation) covers a different
        # extent, and the error grows across the image. Measured at the far corner of the grid.
        drift_px = max(abs((a1 - a2) * first.width + (b1 - b2) * first.height),
                       abs((d1 - d2) * first.width + (e1 - e2) * first.height)) / pixel_size
        offset_px = max(abs(c1 - c2), abs(f1 - f2)) / pixel_size
        if drift_px > GRID_TOLERANCE_PX:
            issues.append(issue("grid_resolution_mismatch",
                                f"Pair pixel sizes differ ({abs(a1):g} x {abs(e1):g} vs {abs(a2):g} x {abs(e2):g} CRS units), "
                                f"so the grids diverge by {drift_px:.1f} px across the image; images are not co-registered. "
                                "Resampling is not performed (D-008)."))
        elif offset_px > GRID_TOLERANCE_PX:
            issues.append(issue("grid_offset", f"Pair grids are offset by {offset_px:.1f} px; images are not co-registered."))
    same_shape = (first.height, first.width) == (second.height, second.width)
    if same_shape and not (np.isfinite(first.data).any(axis=0) & np.isfinite(second.data).any(axis=0)).any():
        issues.append(issue("no_common_valid_pixels", "The two images have no valid pixel in common "
                            "(each one's data lies where the other has nodata), so they cannot be compared."))
    if first.modality != second.modality and first.crs is None and second.crs is None and same_shape:
        # Joint analysis compares pixel by pixel. Without a CRS that alignment cannot be confirmed,
        # so it is assumed from the shared grid shape and the assumption is stated, never silent.
        issues.append(issue("coregistration_unverified", "Neither image has a coordinate reference system, so "
                            "co-registration cannot be confirmed from metadata; the pair is assumed co-registered "
                            f"because both share a {first.width}x{first.height} pixel grid.", "warning"))
    if first.modality != second.modality and first.acquired and second.acquired:
        first_date, second_date = _parse_date(first.acquired), _parse_date(second.acquired)
        if first_date and second_date and first_date != second_date:
            issues.append(issue("acquisition_gap", f"The optical and SAR images were acquired "
                                f"{abs((second_date - first_date).days)} days apart ({first.acquired} and "
                                f"{second.acquired}); anything that changed in between shows up as disagreement "
                                "between the sensors.", "warning"))
    if first.modality == second.modality:
        if not (first.acquired and second.acquired):
            issues.append(issue("dates_unknown", "Acquisition dates missing: assuming image 1 is earlier.", "warning"))
        elif _parse_date(first.acquired) and _parse_date(second.acquired) and _parse_date(first.acquired) > _parse_date(second.acquired):
            issues.append(issue("dates_reversed", "Image 1 is later than image 2; 'before' and 'after' follow upload order.",
                                "warning"))
    return issues


def _parse_date(text: str) -> date | None:
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None
