import numpy as np
import pytest

from satquery import raster_analysis as ra
from satquery.imaging import RasterImage
from satquery.specialists.falcon import (clean_text, composite_pair, parse_boxes, parse_polygons, polygons_to_mask,
                                         to_single_frame)


def sar_image(data):
    return RasterImage(data=data, band_names=["copol", "crosspol"], modality="sar", name="sar")


def test_otsu_separates_bimodal():
    values = np.concatenate([np.full(500, -20.0), np.full(500, -5.0)])
    threshold, separability = ra.otsu(values)
    assert -20 < threshold < -5 and separability > 0.9


def test_sar_water_mask_finds_dark_block(sar_scene):
    mask, info = ra.sar_water_mask(sar_image(sar_scene))
    assert mask[50, 50] and not mask[20, 20]
    assert 0.1 < info["fraction"] < 0.25


def test_change_map_and_regions(optical_scene):
    before = RasterImage(data=optical_scene, band_names=["blue", "green", "red", "nir"], modality="optical", name="t1")
    changed = optical_scene.copy()
    changed[:, 40:60, 40:60] = changed[:, :20, :20].mean(axis=(1, 2), keepdims=True)  # new water block bottom-right
    after = RasterImage(data=changed, band_names=before.band_names, modality="optical", name="t2")
    mask, info = ra.change_map(before, after)
    assert mask[50, 50] and not mask[30, 5]
    top = ra.regions(mask)[0]
    assert top["location"] == "south-east"


def test_spectral_indices_water(optical_scene):
    image = RasterImage(data=optical_scene, band_names=["blue", "green", "red", "nir"], modality="optical", name="s2")
    indices = ra.spectral_indices(image)
    assert indices["ndwi"][5, 5] > 0 > indices["ndwi"][40, 40]


def test_falcon_box_and_polygon_parsing():
    raw = "</s><s>stadium<655><421><767><557></s>"
    (box,) = parse_boxes(raw, 800, 800)
    assert box == (524.4, 337.2, 614.0, 446.0)
    poly_raw = "<poly><0><0><499><0><499><499><0><499></poly>"
    (polygon,) = parse_polygons(poly_raw, 100, 100)
    mask = polygons_to_mask([polygon], 100, 100)
    assert mask[10, 10] and not mask[80, 80]
    assert clean_text("</s><s>Yes</s>") == "Yes"


def test_bitemporal_composite_and_frame_mapping():
    a = np.zeros((10, 20, 3), np.uint8)
    composite = composite_pair(a, a + 1)
    assert composite.shape == (20, 40, 3) and composite[15, 30, 0] == 1 and composite[15, 5, 0] == 0
    assert to_single_frame([[25.0, 12.0, 5.0, 3.0]], 20, 10) == [[5.0, 2.0, 5.0, 3.0]]


def disk_valid(size=64, radius=0.45):
    """bool mask of a centred disk: the valid pixels of a crop to a drawn circle."""
    yy, xx = np.mgrid[:size, :size]
    return (xx - size / 2 + 0.5) ** 2 + (yy - size / 2 + 0.5) ** 2 <= (radius * size) ** 2


def test_coverage_counts_only_valid_pixels():
    mask = np.zeros((4, 4), bool)
    mask[:2] = True
    valid = np.zeros((4, 4), bool)
    valid[:2, :2] = True
    assert ra.coverage(mask) == 0.5
    assert ra.coverage(mask, valid) == 1.0, "every valid pixel is covered"
    assert ra.coverage(mask, np.ones((4, 4), bool)) == mask.mean()
    assert ra.coverage(mask, np.zeros((4, 4), bool)) == 0.0


def test_sar_water_mask_ignores_nodata_outside_a_drawn_shape(sar_scene):
    valid = disk_valid()
    data = sar_scene.copy()
    data[:, ~valid] = np.nan
    mask, info = ra.sar_water_mask(sar_image(data))

    assert not mask[~valid].any(), "nodata is never water"
    assert mask[50, 50], "the dark block inside the disk is still found"
    assert info["fraction"] == pytest.approx(mask.sum() / valid.sum())


def test_change_map_keeps_nodata_out_of_the_result(optical_scene):
    valid = disk_valid()
    before_data, after_data = optical_scene.copy(), optical_scene.copy()
    after_data[:, 40:50, 28:38] = after_data[:, :20, :20].mean(axis=(1, 2), keepdims=True)  # change inside the disk
    before_data[:, ~valid] = np.nan
    after_data[:, ~valid] = np.nan
    names = ["blue", "green", "red", "nir"]
    mask, info = ra.change_map(RasterImage(before_data, names, "optical", "t1"), RasterImage(after_data, names, "optical", "t2"))

    assert not mask[~valid].any()
    assert mask[45, 33]
    assert info["fraction"] == pytest.approx(mask.sum() / valid.sum())


# --------------------------------------------------------------------------- change-map normalisation


def textured_pair(shift=None, brighten=None):
    """Two dates of a textured scene. `brighten` raises one quadrant of the second date."""
    rng = np.random.default_rng(7)
    base = (np.full((4, 64, 64), 1000, np.float32) + rng.normal(0, 50, (4, 64, 64))).astype(np.float32)
    after = base.copy()
    if brighten is not None:
        after[:, :32, :32] += brighten
    if shift is not None:
        after += shift
    names = ["blue", "green", "red", "nir"]
    return (RasterImage(base, names, "optical", "t1"), RasterImage(after, names, "optical", "t2"))


def test_change_is_not_invented_where_the_ground_did_not_change():
    """The old per-date stretch moved unchanged pixels whenever the other date's distribution moved."""
    before, after = textured_pair(brighten=1000)
    mask, info = ra.change_map(before, after)

    changed, unchanged = mask[:32, :32], mask[32:, 32:]
    assert changed.mean() > 0.95, "the quadrant that really changed is found"
    assert unchanged.mean() < 0.01, f"unchanged ground must stay unflagged, got {unchanged.mean():.3f}"
    assert "shared by both dates" in info["method"]


def test_a_scene_wide_difference_is_not_cancelled():
    """Scaling each date on its own erased any change that moved the whole scene."""
    before, after = textured_pair(shift=800, brighten=1500)
    mask, _ = ra.change_map(before, after)

    assert mask.mean() > 0.2, "a scene-wide brightening plus a bright block is a real difference"


def test_identical_dates_produce_no_change():
    before, after = textured_pair()
    mask, info = ra.change_map(before, after)

    assert not mask.any() and info["fraction"] == 0.0
