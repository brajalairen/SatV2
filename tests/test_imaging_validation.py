import numpy as np

from satquery.imaging import load_image, render_rgb, sar_db
from satquery.schemas import AnalysisRequest, ImageInput
from satquery.validation import check_images, check_request, detect_input_config


def codes(issues):
    return {i.code for i in issues}


def test_band_descriptions_define_roles(write_tiff, optical_scene):
    path = write_tiff("s2.tif", optical_scene, band_names=["B02", "B03", "B04", "B08"])
    image = load_image(path, "optical")
    assert not image.band_names_assumed
    assert np.allclose(image.band("nir"), optical_scene[3])
    assert render_rgb(image).shape == (64, 64, 3)


def test_missing_descriptions_are_assumed_and_flagged(write_tiff, optical_scene):
    image = load_image(write_tiff("mx.tif", optical_scene), "optical")
    assert image.band_names == ["blue", "green", "red", "nir"]
    assert "band_order_assumed" in codes(check_images([image]))


def test_png_is_display_ready_without_crs_warning(write_png):
    image = load_image(write_png("bench.png", np.full((32, 32, 3), 120)), "optical")
    assert image.display_ready and image.crs is None
    assert "no_crs" not in codes(check_images([image]))


def test_sar_units_detected(write_tiff, sar_scene):
    db_image = load_image(write_tiff("sar_db.tif", sar_scene), "sar")
    assert "dB" in sar_db(db_image)[2]
    linear = np.power(10, sar_scene / 10).astype(np.float32)
    lin_image = load_image(write_tiff("sar_lin.tif", linear), "sar")
    assert np.allclose(sar_db(lin_image)[0], sar_scene[0], atol=1e-3)


def test_large_raster_is_decimated(write_tiff):
    image = load_image(write_tiff("big.tif", np.ones((1, 200, 200), np.float32)), "optical", max_pixels=50 * 50)
    assert image.width == 50 and image.decimation > 1
    assert "decimated" in codes(check_images([image]))


def test_request_checks(tmp_path, write_png):
    good = write_png("a.png", np.zeros((20, 20, 3)))
    bad = AnalysisRequest(query="", images=[ImageInput(path=str(tmp_path / "missing.tif"), modality="optical"),
                                            ImageInput(path=good, modality="optical"),
                                            ImageInput(path=good, modality="optical")])
    found = codes(check_request(bad))
    assert {"empty_query", "image_count", "file_missing", "non_geospatial_format"} <= found


def test_pair_grid_checks(write_tiff, optical_scene):
    a = load_image(write_tiff("a.tif", optical_scene), "optical")
    shifted = load_image(write_tiff("b.tif", optical_scene, origin=(501000.0, 2800000.0)), "optical")
    other_crs = load_image(write_tiff("c.tif", optical_scene, crs="EPSG:4326"), "optical")
    smaller = load_image(write_tiff("d.tif", optical_scene[:, :32, :32]), "optical")
    assert "grid_offset" in codes(check_images([a, shifted]))
    assert "crs_mismatch" in codes(check_images([a, other_crs]))
    assert "grid_shape_mismatch" in codes(check_images([a, smaller]))
    assert "dates_unknown" in codes(check_images([a, a]))


def test_input_configuration():
    assert detect_input_config(["optical"]) == "single_optical"
    assert detect_input_config(["sar"]) == "single_sar"
    assert detect_input_config(["optical", "optical"]) == "pair_bitemporal"
    assert detect_input_config(["optical", "sar"]) == "pair_cross_modal"
