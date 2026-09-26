"""Optical + SAR joint analysis through analyze(): routing, refusals, tools, fusion, answer, evidence, trace.

Synthetic co-registered scenes with known geometry, so every figure can be checked against the truth.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from satquery.api import analyze
from satquery.schemas import AnalysisRequest, ImageInput
from satquery.settings import Settings
from satquery.specialists.tools import FusionParams
from satquery.specialists.vlm import FakeVLM

QUERY = "Use the optical and SAR images together to identify built-up and water-covered regions."
S2 = ["B02", "B03", "B04", "B08"]
WATER = (slice(0, 20), slice(0, 20))
BUILT = (slice(40, 52), slice(40, 52))


@pytest.fixture
def settings(tmp_path):
    return Settings(vlm_backend="fake", runs_dir=tmp_path / "runs")


@pytest.fixture
def optical_truth():
    """Vegetation everywhere, water top-left, a concrete-like block (low NDVI and NDWI) at BUILT."""
    rng = np.random.default_rng(0)
    bands = [np.full((64, 64), v, np.float32) for v in (400, 700, 500, 3000)]
    for band, water, built in zip(bands, (600, 900, 400, 150), (1800, 1900, 2000, 2200)):
        band[WATER], band[BUILT] = water, built
    return (np.stack(bands) + rng.normal(0, 20, (4, 64, 64))).astype(np.float32)


@pytest.fixture
def sar_truth():
    """Co-registered with optical_truth: dark water top-left, strong scatterers at BUILT (dB)."""
    rng = np.random.default_rng(1)
    co = np.full((64, 64), -8.0, np.float32)
    co[WATER], co[BUILT] = -22.0, 2.0
    return (np.stack([co, co - 7.0]) + rng.normal(0, 0.8, (2, 64, 64))).astype(np.float32)


def run(settings, query, *images):
    request = AnalysisRequest(query=query, images=[ImageInput(path=p, modality=m) for p, m in images])
    return analyze(request, settings=settings, vlm=FakeVLM())


def codes(response):
    return {issue.code for issue in response.trace.validation}


def step(response, tool):
    return next(s for s in response.trace.steps if s.tool == tool)


def plan_step(response, step_id):
    return next(p for p in response.trace.plan if p.step_id == step_id)


# --------------------------------------------------------------------- refusals: a modality is missing

def test_optical_only_is_refused_not_answered_by_single_image_vqa(settings, write_tiff, optical_truth):
    response = run(settings, QUERY, (write_tiff("opt.tif", optical_truth, band_names=S2), "optical"))
    assert response.status == "invalid_input" and "missing_sar" in codes(response)
    assert response.trace.plan == [] and response.trace.steps == []
    assert "SAR" in response.answer and "optical and SAR" in response.answer  # quotes the user's phrase


def test_sar_only_is_refused_as_missing_optical_not_as_a_temporal_question(settings, write_tiff, sar_truth):
    """'Compare ... to' also matches the temporal cue; the cross-modal refusal must take precedence."""
    response = run(settings, "Compare optical and SAR evidence to find water.",
                   (write_tiff("sar.tif", sar_truth, band_names=["VV", "VH"]), "sar"))
    assert response.status == "invalid_input"
    assert codes(response) == {"missing_optical"}


def test_two_optical_images_are_refused_not_routed_to_change_analysis(settings, write_tiff, optical_truth):
    """The web upload used to declare every file optical: this pair must never become a date comparison."""
    a = write_tiff("a.tif", optical_truth)
    b = write_tiff("b.tif", optical_truth)
    response = run(settings, QUERY, (a, "optical"), (b, "optical"))
    assert response.status == "invalid_input" and "missing_sar" in codes(response)
    assert response.task is None and response.trace.intent is None


def test_a_sar_file_declared_optical_is_refused_from_its_band_names(settings, write_tiff, optical_truth, sar_truth):
    optical = write_tiff("opt.tif", optical_truth, band_names=S2)
    sar = write_tiff("s1.tif", sar_truth, band_names=["VV", "VH"])
    response = run(settings, QUERY, (optical, "optical"), (sar, "optical"))
    assert response.status == "invalid_input" and "modality_conflict" in codes(response)
    assert "VV, VH" in response.answer


def test_optical_only_ordinary_question_is_not_cross_modal(settings, write_tiff, optical_truth):
    response = run(settings, "Highlight the water body.", (write_tiff("opt.tif", optical_truth, band_names=S2), "optical"))
    assert response.status == "ok" and response.task == "grounding"
    assert "fusion.cross_modal" not in [s.tool for s in response.trace.steps]


def test_sar_only_ordinary_question_is_not_cross_modal(settings, write_tiff, sar_truth):
    response = run(settings, "Where is water visible in this SAR image?",
                   (write_tiff("sar.tif", sar_truth, band_names=["VV", "VH"]), "sar"))
    assert response.status == "ok" and response.task == "grounding"
    assert "fusion.cross_modal" not in [s.tool for s in response.trace.steps]


# --------------------------------------------------------------------- refusals: the pair is incompatible

@pytest.mark.parametrize("sar_kwargs, code", [
    ({"crs": "EPSG:32644"}, "crs_mismatch"),
    ({"pixel": 20.0}, "grid_resolution_mismatch"),
    ({"origin": (500100.0, 2800000.0)}, "grid_offset"),
])
def test_an_incompatible_pair_is_refused_before_planning(settings, write_tiff, optical_truth, sar_truth, sar_kwargs, code):
    optical = write_tiff("opt.tif", optical_truth, band_names=S2)
    sar = write_tiff("sar.tif", sar_truth, band_names=["VV", "VH"], **sar_kwargs)
    response = run(settings, QUERY, (optical, "optical"), (sar, "sar"))
    assert response.status == "invalid_input" and code in codes(response)
    assert response.trace.plan == []


def test_a_sar_image_without_valid_pixels_is_refused(settings, write_tiff, optical_truth):
    optical = write_tiff("opt.tif", optical_truth, band_names=S2)
    sar = write_tiff("sar.tif", np.full((2, 64, 64), np.nan, np.float32), band_names=["VV", "VH"])
    response = run(settings, QUERY, (optical, "optical"), (sar, "sar"))
    assert response.status == "invalid_input" and "no_valid_pixels" in codes(response)


def test_a_pair_without_a_crs_runs_but_states_that_coregistration_is_assumed(settings, write_tiff, optical_truth, sar_truth):
    optical = write_tiff("opt.tif", optical_truth, band_names=S2, crs=None)
    sar = write_tiff("sar.tif", sar_truth, band_names=["VV", "VH"], crs=None)
    response = run(settings, QUERY, (optical, "optical"), (sar, "sar"))
    assert response.status == "ok"
    warning = next(i for i in response.trace.validation if i.code == "coregistration_unverified")
    assert warning.severity == "warning" and "assumed" in warning.message


# --------------------------------------------------------------------- routing and plan

@pytest.mark.parametrize("query, tools", [
    (QUERY, ["optical.spectral_indices", "sar.water_mask", "sar.bright_mask", "fusion.cross_modal"]),
    ("Compare optical and SAR evidence to find water.", ["optical.spectral_indices", "sar.water_mask", "fusion.cross_modal"]),
    ("Using both sensors, identify built-up areas.", ["optical.spectral_indices", "sar.bright_mask", "fusion.cross_modal"]),
    ("Where are the water-covered and built-up regions using optical and radar information?",
     ["optical.spectral_indices", "sar.water_mask", "sar.bright_mask", "fusion.cross_modal"]),
])
def test_the_plan_runs_optical_and_sar_specialists_for_the_classes_asked(settings, write_tiff, optical_truth, sar_truth,
                                                                      query, tools):
    optical = write_tiff("opt.tif", optical_truth, band_names=S2)
    sar = write_tiff("sar.tif", sar_truth, band_names=["VV", "VH"])
    response = run(settings, query, (optical, "optical"), (sar, "sar"))
    assert response.status == "ok" and response.task == "cross_modal_analysis"
    assert response.trace.input_config == "pair_cross_modal"
    assert [s.tool for s in response.trace.steps] == tools
    assert all(s.status == "ok" for s in response.trace.steps)
    for s in response.trace.plan:  # each specialist reads the image of its own modality
        expected = {"optical.spectral_indices": [0], "sar.water_mask": [1], "sar.bright_mask": [1],
                    "fusion.cross_modal": [0, 1]}[s.tool]
        assert s.image_indices == expected, s


def test_the_routing_rule_records_the_cue_and_classes(settings, write_tiff, optical_truth, sar_truth):
    optical = write_tiff("opt.tif", optical_truth, band_names=S2)
    sar = write_tiff("sar.tif", sar_truth, band_names=["VV", "VH"])
    rule = run(settings, QUERY, (optical, "optical"), (sar, "sar")).trace.intent.matched_rule
    assert "pair_cross_modal -> cross_modal_analysis" in rule
    assert "cue 'optical and SAR'" in rule and "classes water, building" in rule


def test_sar_listed_first_still_routes_each_tool_to_its_modality(settings, write_tiff, optical_truth, sar_truth):
    optical = write_tiff("opt.tif", optical_truth, band_names=S2)
    sar = write_tiff("sar.tif", sar_truth, band_names=["VV", "VH"])
    response = run(settings, QUERY, (sar, "sar"), (optical, "optical"))
    assert response.status == "ok"
    assert plan_step(response, step(response, "optical.spectral_indices").step_id).image_indices == [1]
    assert plan_step(response, step(response, "sar.water_mask").step_id).image_indices == [0]
    assert plan_step(response, step(response, "fusion.cross_modal").step_id).image_indices == [1, 0]


def test_rgb_optical_without_nir_uses_the_vlm_for_optical_evidence(settings, write_tiff, optical_truth, sar_truth):
    optical = write_tiff("rgb.tif", optical_truth[[2, 1, 0]], band_names=["red", "green", "blue"])
    sar = write_tiff("sar.tif", sar_truth, band_names=["VV", "VH"])
    response = run(settings, QUERY, (optical, "optical"), (sar, "sar"))
    assert response.status == "ok"
    segments = [s for s in response.trace.steps if s.tool == "vlm.segment"]
    assert [s.params["target"] for s in segments] == ["water", "building"]
    assert all(s.model == FakeVLM.model_id for s in segments)  # the model is named in the trace
    assert "VLM segmentation" in response.answer


# --------------------------------------------------------------------- fusion

def test_fusion_receives_the_optical_and_the_sar_outputs(settings, write_tiff, optical_truth, sar_truth):
    optical = write_tiff("opt.tif", optical_truth, band_names=S2)
    sar = write_tiff("sar.tif", sar_truth, band_names=["VV", "VH"])
    response = run(settings, QUERY, (optical, "optical"), (sar, "sar"))
    fusion, spectral = step(response, "fusion.cross_modal"), step(response, "optical.spectral_indices")
    sar_water, sar_bright = step(response, "sar.water_mask"), step(response, "sar.bright_mask")

    assert fusion.params["optical_water_step"] == spectral.step_id and fusion.params["optical_water_key"] == "water"
    assert fusion.params["optical_building_step"] == spectral.step_id
    assert fusion.params["optical_building_key"] == "built_up_proxy"
    assert fusion.params["sar_water_step"] == sar_water.step_id and fusion.params["sar_bright_step"] == sar_bright.step_id

    water, built = fusion.outputs["water"], fusion.outputs["built_up"]
    assert water["optical_percent"] == pytest.approx(spectral.outputs["water_fraction"] * 100, abs=0.01)
    assert water["sar_percent"] == pytest.approx(sar_water.outputs["fraction"] * 100, abs=0.01)
    assert built["optical_percent"] == pytest.approx(spectral.outputs["built_up_proxy_fraction"] * 100, abs=0.01)
    assert built["sar_percent"] == pytest.approx(sar_bright.outputs["fraction"] * 100, abs=0.01)
    # The truth: water is 400 of 4096 px and the block 144 px, seen by both sensors in the same place.
    assert water["agreement"] == "high" and water["both_percent"] == pytest.approx(400 / 4096 * 100, abs=2.0)
    assert built["agreement"] == "high" and built["both_percent"] == pytest.approx(144 / 4096 * 100, abs=1.0)
    assert water["both_percent"] + water["optical_only_percent"] == pytest.approx(water["optical_percent"], abs=0.02)


def test_disagreement_is_reported_not_hidden(settings, write_tiff, optical_truth, sar_scene):
    """conftest's sar_scene puts its water bottom-right, away from the optical water: IoU 0."""
    optical = write_tiff("opt.tif", optical_truth, band_names=S2)
    sar = write_tiff("sar.tif", sar_scene, band_names=["VV", "VH"])
    response = run(settings, "Compare optical and SAR evidence to find water.", (optical, "optical"), (sar, "sar"))
    water = step(response, "fusion.cross_modal").outputs["water"]
    assert water["agreement"] == "low" and water["both_percent"] == 0
    assert "no water is confirmed by both sensors" in response.answer
    assert "low agreement for water" in response.answer and "lower confidence" in response.answer
    assert response.confidence.value == pytest.approx(0.0, abs=0.01)


def test_nodata_in_one_sensor_is_not_counted_as_disagreement(settings, write_tiff, optical_truth, sar_scene):
    optical = write_tiff("opt.tif", optical_truth, band_names=S2)
    left_missing = sar_scene.copy()
    left_missing[:, :, :32] = np.nan  # the optical water (top-left) lies where SAR has no data
    sar = write_tiff("sar.tif", left_missing, band_names=["VV", "VH"])
    response = run(settings, "Compare optical and SAR evidence to find water.", (optical, "optical"), (sar, "sar"))
    water = step(response, "fusion.cross_modal").outputs["water"]
    assert water["optical_only_percent"] == 0 and water["sar_only_percent"] > 0
    layer = next(e for e in response.evidence if e.kind == "overlay" and e.label.startswith("FUSED water"))
    painted = np.asarray(Image.open(layer.file))
    assert painted[10, 10, 3] == 0  # optical water under SAR nodata: not painted as "optical only"
    assert painted[50, 50, 3] > 0  # SAR water where both have data: painted


def test_fusion_accepts_only_permitted_parameters():
    with pytest.raises(ValidationError):
        FusionParams(sar_water_step="s1", threshold=0.5)
    with pytest.raises(ValidationError):
        FusionParams(optical_water_key="../etc")


# --------------------------------------------------------------------- answer, evidence, trace

def test_the_answer_separates_optical_sar_and_fused_evidence(settings, write_tiff, optical_truth, sar_truth):
    optical = write_tiff("opt.tif", optical_truth, band_names=S2)
    sar = write_tiff("sar.tif", sar_truth, band_names=["VV", "VH"])
    lines = run(settings, QUERY, (optical, "optical"), (sar, "sar")).answer.split("\n")
    assert lines[0].startswith("Using the optical and SAR images together")
    assert lines[1].startswith("Optical evidence (opt.tif): water by NDWI > 0")
    assert "built-up proxy" in lines[1] and "bare soil" in lines[1]
    assert lines[2].startswith("SAR evidence (sar.tif): low backscatter")
    assert "not a building detector" in lines[2] and "heuristic" in lines[2]
    assert lines[3].startswith("Cross-modal evidence: water agreement IoU")


def test_visual_evidence_is_one_pinned_layer_per_fused_class(settings, write_tiff, optical_truth, sar_truth):
    optical = write_tiff("opt.tif", optical_truth, band_names=S2)
    sar = write_tiff("sar.tif", sar_truth, band_names=["VV", "VH"])
    response = run(settings, QUERY, (sar, "sar"), (optical, "optical"))
    overlays = [e for e in response.evidence if e.kind == "overlay"]
    pinned = [e for e in overlays if e.image_index is not None]
    assert [e.label.split(":")[0] for e in pinned] == ["FUSED water", "FUSED built-up"]
    assert all(e.image_index == 1 for e in pinned)  # the optical image's grid, shared by the pair
    assert "optical" in pinned[0].label and "SAR only" in pinned[0].label
    assert len(overlays) == 3 and overlays[-1].image_index is None  # plus the side-by-side report figure

    water = np.asarray(Image.open(pinned[0].file))
    assert water.shape == (64, 64, 4)
    assert tuple(water[10, 10]) == (30, 120, 255, 210)  # inside the water: both sensors agree (solid blue)
    assert water[30, 5, 3] == 0  # vegetation: transparent, the imagery shows through
    single = water[(water[..., 3] > 0) & (water[..., 3] < 210)]  # one-sensor pixels at the water's edge
    assert len(single) and (single[:, 3] == 75).all(), "one-sensor evidence is drawn faintly"


def test_the_trace_and_report_show_both_modalities_and_every_tool(settings, write_tiff, optical_truth, sar_truth):
    optical = write_tiff("opt.tif", optical_truth, band_names=S2)
    sar = write_tiff("sar.tif", sar_truth, band_names=["VV", "VH"])
    response = run(settings, QUERY, (optical, "optical"), (sar, "sar"))
    trace = response.trace
    assert [(i.name, i.modality) for i in trace.images] == [("opt.tif", "optical"), ("sar.tif", "sar")]
    assert [s.step_id for s in trace.steps] == [p.step_id for p in trace.plan]
    assert all(s.model is None for s in trace.steps)  # deterministic tools on multispectral input: no model claimed
    assert "SAR evidence" in plan_step(response, step(response, "sar.water_mask").step_id).purpose
    report = Path(response.report_html).read_text(encoding="utf-8")
    assert "#1 optical" in report and "#2 sar" in report and "fusion.cross_modal" in report
    assert "white-space:pre-line" in report
