"""Map placement of loaded rasters (satquery/geo.py). Synthetic data only."""

import numpy as np
import pytest

from satquery import geo
from satquery.imaging import load_image

# tests/conftest.py writes EPSG:32643 (UTM zone 43N) at origin (500000, 2800000) with 10 m pixels.
# That easting is the zone's central meridian, so the top-left corner sits at 75 deg E.
CENTRAL_MERIDIAN = 75.0


@pytest.fixture
def scene():
    return np.arange(3 * 16 * 16, dtype=np.uint16).reshape(3, 16, 16)


def test_georeferences_a_utm_tiff(write_tiff, scene):
    image = load_image(write_tiff("utm.tif", scene), "optical")
    located = geo.georeference(image)

    assert located is not None
    assert located.crs == "EPSG:32643"
    assert not located.approximate and located.note is None
    assert len(located.corners_wgs84) == 4

    west, south, east, north = located.bounds_wgs84
    assert west < east and south < north
    assert west == pytest.approx(CENTRAL_MERIDIAN, abs=1e-6)  # origin is on the central meridian
    assert 25 < north < 26  # 2 800 000 m north of the equator
    # 16 px x 10 m = 160 m across, which is well under a hundredth of a degree.
    assert east - west == pytest.approx(0.0016, abs=5e-4)


def test_corners_are_ordered_clockwise_from_top_left(write_tiff, scene):
    image = load_image(write_tiff("utm.tif", scene), "optical")
    top_left, top_right, bottom_right, bottom_left = geo.georeference(image).corners_wgs84

    assert top_left[1] > bottom_left[1]  # north edge is above the south edge
    assert top_right[0] > top_left[0]  # east edge is right of the west edge
    assert bottom_right[0] > bottom_left[0]


def test_no_crs_means_no_placement(write_tiff, scene):
    """A plain TIFF without georeferencing must be reported as unplaceable, never guessed at."""
    image = load_image(write_tiff("plain.tif", scene, crs=None), "optical")

    assert image.crs is None
    assert geo.georeference(image) is None
    assert geo.pixel_to_crs(image, 0, 0) is None
    assert geo.bbox_wgs84_to_pixel(image, (74.9, 25.2, 75.1, 25.4)) is None


def test_png_is_never_placed(write_png):
    image = load_image(write_png("scene.png", np.zeros((8, 8, 3), np.uint8)), "optical")
    assert geo.georeference(image) is None


def test_summarize_fills_the_wire_fields(write_tiff, scene):
    image = load_image(write_tiff("utm.tif", scene), "optical")
    summary = geo.summarize(image, 3)

    assert summary.index == 3
    assert summary.bounds_wgs84 == geo.georeference(image).bounds_wgs84
    assert summary.corners_wgs84 is not None and len(summary.corners_wgs84) == 4
    assert summary.georeference_note is None


def test_summarize_leaves_the_wire_fields_empty_without_a_crs(write_tiff, scene):
    summary = geo.summarize(load_image(write_tiff("plain.tif", scene, crs=None), "optical"), 0)

    assert summary.bounds_wgs84 is None
    assert summary.corners_wgs84 is None


def test_bbox_round_trips_to_a_pixel_window(write_tiff, scene):
    image = load_image(write_tiff("utm.tif", scene), "optical")
    bounds = geo.georeference(image).bounds_wgs84

    assert geo.bbox_wgs84_to_pixel(image, bounds) == (0, 0, image.width, image.height)


def test_bbox_outside_the_image_is_rejected(write_tiff, scene):
    image = load_image(write_tiff("utm.tif", scene), "optical")
    assert geo.bbox_wgs84_to_pixel(image, (10.0, 50.0, 10.1, 50.1)) is None


def test_decimation_does_not_move_the_footprint(write_tiff):
    """A decimated read rescales the transform, so the scene must still cover the same ground."""
    big = np.zeros((1, 200, 200), dtype=np.uint16)
    path = write_tiff("big.tif", big)
    full = geo.georeference(load_image(path, "optical"))
    reduced = load_image(path, "optical", max_pixels=64 * 64)

    assert reduced.decimation > 1
    assert geo.georeference(reduced).bounds_wgs84 == pytest.approx(full.bounds_wgs84, abs=1e-6)


def test_crop_keeps_georeferencing_and_band_names(write_tiff, tmp_path):
    data = np.arange(2 * 64 * 64, dtype=np.uint16).reshape(2, 64, 64)
    path = write_tiff("wide.tif", data, band_names=["copol", "crosspol"])
    full = geo.georeference(load_image(path, "sar"))
    west, south, east, north = full.bounds_wgs84

    destination = tmp_path / "cropped.tif"
    middle = (west + (east - west) * 0.25, south + (north - south) * 0.25,
              west + (east - west) * 0.75, south + (north - south) * 0.75)
    crop = geo.crop_to_bbox(path, middle, destination)

    assert crop is not None and not crop.is_whole_image
    assert (crop.source_width, crop.source_height) == (64, 64)
    assert crop.width == pytest.approx(32, abs=1) and crop.height == pytest.approx(32, abs=1)

    cropped = load_image(destination, "sar")
    assert cropped.crs == "EPSG:32643"
    assert cropped.band_names == ["copol", "crosspol"]
    assert geo.georeference(cropped).bounds_wgs84 == pytest.approx(middle, abs=1e-4)


def test_crop_declines_rather_than_guessing(write_tiff, tmp_path, scene):
    path = write_tiff("utm.tif", scene)
    west, south, east, north = geo.georeference(load_image(path, "optical")).bounds_wgs84
    destination = tmp_path / "out.tif"

    # No overlap at all.
    assert geo.crop_to_bbox(path, (10.0, 50.0, 10.1, 50.1), destination) is None
    # Overlap smaller than validation's 16 px floor.
    sliver = (west, south, west + (east - west) * 0.02, south + (north - south) * 0.02)
    assert geo.crop_to_bbox(path, sliver, destination) is None
    # A raster with no CRS cannot be cut by a geographic box.
    assert geo.crop_to_bbox(write_tiff("plain.tif", scene, crs=None), (west, south, east, north), destination) is None


def test_crop_covering_everything_reports_the_whole_image(write_tiff, tmp_path, scene):
    path = write_tiff("utm.tif", scene)
    bounds = geo.georeference(load_image(path, "optical")).bounds_wgs84
    crop = geo.crop_to_bbox(path, bounds, tmp_path / "out.tif")

    assert crop is not None and crop.is_whole_image
