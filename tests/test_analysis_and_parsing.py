import numpy as np

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
