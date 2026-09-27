"""D-030: NDWI-first water grounding, the selected area's optical quality from Sentinel-2's scene
classification, the Sentinel-1 fallback it triggers, and explicit radar-only questions.

Offline. The end-to-end retrieval cases (clear, cloudy, Loktak-like, too few pixels, explicit SAR)
run through the HTTP API in tests/test_server.py.
"""

from datetime import date

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.agent.intents import needs_optical_and_sar, needs_sar_only, route_query
from satquery.api import analyze
from satquery.imaging import load_image
from satquery.providers import quality
from satquery.providers.copernicus import (SAR_MAX_DAYS_APART, CopernicusSentinelProvider,
                                           scl_evalscript)
from satquery.providers.errors import NoSarImagery
from satquery.schemas import AnalysisRequest, ImageInput
from satquery.settings import Settings
from satquery.specialists.vlm import FakeVLM, VLMResult

S2 = ["B02", "B03", "B04", "B08"]
VEGETATION, WATER, CLOUD_HIGH, CLOUD_MEDIUM, SHADOW, NO_DATA, UNCLASSIFIED, DARK = 4, 6, 9, 8, 3, 0, 7, 2


def write_scl(path, classes):
    classes = np.asarray(classes, np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=classes.shape[1], height=classes.shape[0], count=1,
                       dtype="uint8", crs="EPSG:4326", transform=from_origin(93.77, 24.60, 1e-4, 1e-4)) as out:
        out.write(classes[None])
    return path


def scl_with(fraction: float, cls: int = CLOUD_HIGH, size: int = 100) -> np.ndarray:
    """A size x size area, clear vegetation, with the first `fraction` of its pixels set to `cls`."""
    classes = np.full(size * size, VEGETATION, np.uint8)
    classes[: round(fraction * size * size)] = cls
    return classes.reshape(size, size)


def judge(tmp_path, classes, **limits):
    limits = {"max_affected_fraction": 0.20, "min_clear_pixels": 256} | limits
    return quality.assess(write_scl(tmp_path / "scl.tif", classes), **limits)


# --------------------------------------------------------------------- 1. the selected area's optical quality

def test_a_clear_area_is_usable(tmp_path):
    verdict = judge(tmp_path, scl_with(0.0))
    assert verdict.usable and verdict.reason is None
    assert (verdict.affected_fraction, verdict.clear_pixels, verdict.pixels) == (0.0, 10_000, 10_000)


@pytest.mark.parametrize("fraction, usable", [(0.19, True), (0.20, False), (0.42, False)])
def test_the_limit_is_at_or_above_20_percent_of_the_area(tmp_path, fraction, usable):
    verdict = judge(tmp_path, scl_with(fraction))
    assert verdict.usable is usable and verdict.affected_fraction == pytest.approx(fraction)
    if not usable:
        assert f"{fraction:.1%}" in verdict.reason and "20% limit" in verdict.reason


@pytest.mark.parametrize("cls", [CLOUD_HIGH, CLOUD_MEDIUM, 10, SHADOW, NO_DATA, 1])
def test_cloud_shadow_and_missing_data_all_count_as_affected(tmp_path, cls):
    assert judge(tmp_path, scl_with(0.30, cls)).usable is False


@pytest.mark.parametrize("cls", [WATER, UNCLASSIFIED, DARK, 5, 11])
def test_ground_classes_do_not_count_as_affected(tmp_path, cls):
    assert judge(tmp_path, scl_with(0.90, cls)).usable is True


def test_too_few_clear_pixels_is_unusable_even_below_20_percent(tmp_path):
    verdict = judge(tmp_path, scl_with(0.10, size=16))  # 256 pixels, 26 cloudy: 230 clear
    assert not verdict.usable and verdict.affected_fraction < 0.20
    assert "only 230 clear optical pixels" in verdict.reason and "256" in verdict.reason


def test_the_limits_are_settings_not_constants(tmp_path):
    assert judge(tmp_path, scl_with(0.30), max_affected_fraction=0.50).usable is True
    assert Settings().optical_max_affected_fraction == 0.20 and Settings().optical_min_clear_pixels == 256


def test_every_class_present_is_reported_by_name(tmp_path):
    classes = scl_with(0.25, CLOUD_HIGH)
    classes[-1, :] = WATER
    fractions = judge(tmp_path, classes).class_fractions
    assert fractions["cloud, high probability"] == 0.25 and fractions["water"] == 0.01
    assert "SCL" in quality.METHOD and "cloud shadow" in quality.METHOD


def test_affected_pixels_become_nodata_and_nothing_else_changes(tmp_path, write_tiff, optical_scene):
    scene = write_tiff("scene.tif", optical_scene, band_names=["blue", "green", "red", "nir"])
    classes = np.full((64, 64), VEGETATION, np.uint8)
    classes[:10, :] = CLOUD_HIGH
    with rasterio.open(scene) as src:
        grid = {"crs": src.crs, "transform": src.transform}
    scl = tmp_path / "scl.tif"
    with rasterio.open(scl, "w", driver="GTiff", width=64, height=64, count=1, dtype="uint8", **grid) as out:
        out.write(classes[None])

    clear = load_image(quality.mask_affected(scene, scl, tmp_path / "clear.tif"), "optical")
    assert np.isnan(clear.data[:, :10, :]).all() and np.allclose(clear.data[:, 10:, :], optical_scene[:, 10:, :])
    assert clear.band_names == ["blue", "green", "red", "nir"] and not clear.band_names_assumed
    assert clear.crs == "EPSG:32643"


# --------------------------------------------------------------------- 2. provider requests

class Recorder:
    def __init__(self, provider, monkeypatch, features=()):
        self.calls, self.features = [], list(features)
        monkeypatch.setattr(provider, "_render", lambda bbox, data, script: self._render(bbox, data, script))
        monkeypatch.setattr(provider, "search_sar", lambda bbox, start, end: self._search(bbox, start, end))
        monkeypatch.setattr(provider, "_download_sar", lambda bbox, scene, destination, alternatives: scene)

    def _render(self, bbox, data, script):
        self.calls.append(("render", bbox, data, script))
        return b"TIFF"

    def _search(self, bbox, start, end):
        self.calls.append(("search", start, end))
        return self.features


def _s1(scene_id, when, polarization="DV"):
    return {"id": scene_id, "properties": {"datetime": when, "s1:polarization": polarization}}


@pytest.fixture
def provider():
    return CopernicusSentinelProvider("id", "secret")


def test_the_scene_classification_comes_from_the_same_day_and_tiles_as_the_bands(provider, monkeypatch, tmp_path):
    recorder = Recorder(provider, monkeypatch)
    provider.retrieve_scene_classification((93.77, 24.50, 93.87, 24.60), "2026-09-19", tmp_path / "scl.tif")
    provider._process((93.77, 24.50, 93.87, 24.60), S2, "2026-09-19")
    (_, scl_bbox, scl_data, script), (_, bands_bbox, bands_data, _) = recorder.calls
    assert scl_bbox == bands_bbox and scl_data == bands_data, "same area, same day, same tile order"
    assert scl_data["dataFilter"]["timeRange"]["from"].startswith("2026-09-19")
    assert script == scl_evalscript() and '"SCL"' in script and "UINT8" in script
    assert (tmp_path / "scl.tif").read_bytes() == b"TIFF"


def test_the_radar_fallback_looks_around_the_optical_date(provider, monkeypatch, tmp_path):
    recorder = Recorder(provider, monkeypatch, [_s1("S1_0913", "2026-09-13T11:00:00Z"), _s1("S1_0925", "2026-09-25T11:00:00Z")])
    scene = provider.retrieve_sar_near((93.77, 24.50, 93.87, 24.60), date(2026, 9, 19), tmp_path / "sar.tif")
    _, start, end = recorder.calls[0]
    assert start == date(2026, 9, 19 - SAR_MAX_DAYS_APART) and end <= date(2026, 10, 1)
    assert scene["id"] == "S1_0913", "equal gaps: the earlier scene, as the pair retrieval does"


def test_a_radar_question_takes_the_most_recent_dual_polarisation_scene(provider, monkeypatch, tmp_path):
    Recorder(provider, monkeypatch, [_s1("old", "2026-09-06T01:00:00Z"), _s1("single", "2026-09-24T01:00:00Z", "SV"),
                                     _s1("new", "2026-09-18T01:00:00Z")])
    assert provider.retrieve_sar((93.77, 24.50, 93.87, 24.60), tmp_path / "sar.tif")["id"] == "new"


def test_no_recent_radar_scene_is_reported(provider, monkeypatch, tmp_path):
    Recorder(provider, monkeypatch, [])
    with pytest.raises(NoSarImagery) as caught:
        provider.retrieve_sar((93.77, 24.50, 93.87, 24.60), tmp_path / "sar.tif", days_back=30)
    assert "last 30 days" in caught.value.message


def test_the_radar_cache_key_is_its_own(provider):
    bbox = (93.77, 24.50, 93.87, 24.60)
    key = provider.sar_cache_key(bbox)
    assert key not in (provider.cache_key(bbox, S2), provider.optical_sar_cache_key(bbox, S2))
    assert key == provider.sar_cache_key(bbox) and key != provider.sar_cache_key(bbox, days_back=7)


# --------------------------------------------------------------------- 3. explicit radar questions

@pytest.mark.parametrize("query", ["Use radar to find the water.", "Using SAR, highlight the water.",
                                   "Find water using Sentinel-1.", "Where is the backscatter lowest?"])
def test_radar_only_questions_are_recognised(query):
    assert needs_sar_only(query) and not needs_optical_and_sar(query)
    assert route_query(query)[0] == "imagery", "a satellite question, not weather"


@pytest.mark.parametrize("query", ["Highlight the water body.", "Use the optical and SAR images together to find water.",
                                   "Compare optical and SAR evidence to find water.", "Fuse the radar and optical data"])
def test_plain_and_joint_questions_are_not_radar_only(query):
    assert needs_sar_only(query) is None


# --------------------------------------------------------------------- 4. NDWI first in single-image grounding

class BlindVLM(FakeVLM):
    """Finds nothing, as Falcon did at Loktak Lake on a dark, cloud-stretched render."""

    def segment(self, rgb, target):
        return VLMResult(text="The object does not exist.", mask=np.zeros(rgb.shape[:2], bool))


def ground(settings, path, query="Highlight the water body in this image.", vlm=None):
    return analyze(AnalysisRequest(query=query, images=[ImageInput(path=path, modality="optical")]),
                   settings=settings, vlm=vlm or BlindVLM())


@pytest.fixture
def settings(tmp_path):
    return Settings(vlm_backend="fake", runs_dir=tmp_path / "runs")


def test_ndwi_answers_even_when_the_vlm_finds_nothing(settings, write_tiff, optical_scene):
    response = ground(settings, write_tiff("s2.tif", optical_scene, band_names=S2))
    assert [p.tool for p in response.trace.plan] == ["optical.spectral_indices", "vlm.segment"]
    assert "primary evidence" in response.trace.plan[0].purpose and "secondary" in response.trace.plan[1].purpose
    assert response.answer.startswith("Highlighted water: 9.8% of the analysed area by NDWI > 0 (spectral index)")
    assert "north-west" in response.answer and "The VLM segmentation, a secondary check, found 0.0%" in response.answer
    overlay = next(e for e in response.evidence if e.kind == "overlay")
    assert overlay.label == "water (NDWI > 0, highlighted)" and overlay.source_step == response.trace.plan[0].step_id


def test_vegetation_uses_ndvi(settings, write_tiff, optical_scene):
    answer = ground(settings, write_tiff("s2.tif", optical_scene, band_names=S2), "Highlight the vegetation.").answer
    assert answer.startswith("Highlighted vegetation:") and "NDVI > 0.3" in answer


def test_no_water_is_said_plainly_by_the_index(settings, write_tiff, optical_scene):
    dry = optical_scene.copy()
    dry[:, :20, :20] = dry[:, 30:50, 30:50]  # the water square becomes vegetation
    answer = ground(settings, write_tiff("dry.tif", dry, band_names=S2)).answer
    assert answer.startswith("No water was found in this image by NDWI > 0 (spectral index).")


def test_masked_cloud_is_not_counted_and_is_said(settings, write_tiff, optical_scene):
    clouded = optical_scene.copy()
    clouded[:, 50:, :] = np.nan
    answer = ground(settings, write_tiff("clouded.tif", clouded, band_names=S2)).answer
    assert "of the image had no usable data (for example, masked cloud) and is not counted" in answer
    assert "Highlighted water: 12.5%" in answer, "400 water pixels of the 3,200 clear ones"


def test_rgb_input_without_nir_still_uses_the_vlm(settings, write_tiff, optical_scene):
    response = ground(settings, write_tiff("rgb.tif", optical_scene[[2, 1, 0]], band_names=["red", "green", "blue"]),
                      vlm=FakeVLM())
    assert [p.tool for p in response.trace.plan] == ["vlm.segment"] and response.answer.startswith("Highlighted water")
