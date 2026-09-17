"""End-to-end through analyze() with the fake VLM: every input configuration, plus rejection paths."""

from pathlib import Path

import numpy as np
import pytest

from satquery.agent.executor import execute
from satquery.api import analyze
from satquery.schemas import AnalysisRequest, ImageInput, PlanStep
from satquery.settings import Settings
from satquery.specialists.tools import ToolContext
from satquery.specialists.vlm import FakeVLM


@pytest.fixture
def settings(tmp_path):
    return Settings(vlm_backend="fake", runs_dir=tmp_path / "runs")


def run(settings, query, *images, task=None):
    request = AnalysisRequest(query=query, images=[ImageInput(path=p, modality=m, acquired=d) for p, m, d in images],
                              forced_task=task)
    return analyze(request, settings=settings, vlm=FakeVLM())


def assert_complete(response):
    trace = response.trace
    assert [s.step_id for s in trace.steps] == [p.step_id for p in trace.plan]
    assert all(s.status in ("ok", "skipped") for s in trace.steps), [s.error for s in trace.steps]
    assert Path(response.report_html).is_file() and Path(response.report_json).is_file()
    assert any(e.kind == "overlay" for e in response.evidence)
    assert response.confidence is not None and response.confidence.method


def test_single_optical_caption_uses_indices(settings, write_tiff, optical_scene):
    path = write_tiff("s2.tif", optical_scene, band_names=["B02", "B03", "B04", "B08"])
    response = run(settings, "Describe the land-cover and major objects visible in this image.", (path, "optical", None))
    assert response.status == "ok" and response.task == "caption"
    assert [s.tool for s in response.trace.steps] == ["vlm.caption", "optical.spectral_indices"]
    assert "NDVI" in response.answer
    assert_complete(response)


def test_single_optical_grounding(settings, write_tiff, optical_scene):
    path = write_tiff("s2.tif", optical_scene, band_names=["B02", "B03", "B04", "B08"])
    response = run(settings, "Highlight the water body.", (path, "optical", None))
    assert response.task == "grounding" and "water" in response.answer
    assert_complete(response)


def test_single_sar_vqa_adds_sar_context(settings, write_tiff, sar_scene):
    response = run(settings, "Is there flooding in this area?", (write_tiff("sar.tif", sar_scene), "sar", None))
    assert response.task == "vqa" and "SAR context" in response.answer
    assert response.confidence.note  # warns that the VLM saw a SAR rendering
    assert_complete(response)


def test_bitemporal_comparative(settings, write_tiff, optical_scene):
    t1 = write_tiff("t1.tif", optical_scene, band_names=["B02", "B03", "B04", "B08"])
    changed = optical_scene.copy()
    changed[:, 40:60, 40:60] = np.array([1800, 1900, 2000, 2200], np.float32)[:, None, None]  # vegetation -> concrete-like
    t2 = write_tiff("t2.tif", changed, band_names=["B02", "B03", "B04", "B08"])
    response = run(settings, "Has the built-up area increased, decreased, or remained unchanged?",
                   (t1, "optical", "2019-01-01"), (t2, "optical", "2023-01-01"))
    assert response.task == "change_analysis" and response.status == "ok"
    tools = [s.tool for s in response.trace.steps]
    assert tools.count("optical.spectral_indices") == 2 and "vlm.segment" not in tools  # multispectral input -> spectral path
    compare = next(s for s in response.trace.steps if s.tool == "change.compare_areas")
    assert compare.outputs["verdict"] == "increased" and compare.params["mask_key"] == "built_up_proxy"
    assert "Between 2019-01-01 and 2023-01-01" in response.answer
    assert_complete(response)


def test_compare_areas_is_inconclusive_when_nothing_detected(write_tiff, sar_scene):
    from satquery.imaging import load_image
    from satquery.specialists.tools import ToolOutput
    ctx = ToolContext(images=[load_image(write_tiff("s.tif", sar_scene), "sar")], vlm=FakeVLM())
    empty = np.zeros((64, 64), bool)
    ctx.artifacts = {"a": ToolOutput(outputs={}, masks={"mask": empty}), "b": ToolOutput(outputs={}, masks={"mask": empty})}
    plan = [PlanStep(step_id="s1", tool="change.compare_areas", image_indices=[0, 0], purpose="test",
                     params={"target": "building", "before_step": "a", "after_step": "b"})]
    (result,) = execute(plan, ctx)
    assert result.outputs["verdict"] == "inconclusive"


def test_cross_modal(settings, write_tiff, optical_scene, sar_scene):
    optical = write_tiff("opt.tif", optical_scene, band_names=["B02", "B03", "B04", "B08"])
    sar = write_tiff("sar.tif", sar_scene, band_names=["VV", "VH"])
    response = run(settings, "Use the optical and SAR images together to identify built-up and water-covered regions.",
                   (optical, "optical", None), (sar, "sar", None))
    assert response.task == "cross_modal_analysis" and response.status == "ok"
    assert "Water:" in response.answer and "agreement" in response.confidence.method
    assert_complete(response)


def test_invalid_pair_is_rejected_before_planning(settings, write_tiff, optical_scene):
    a = write_tiff("a.tif", optical_scene)
    b = write_tiff("b.tif", optical_scene[:, :32, :32])
    response = run(settings, "What changed?", (a, "optical", None), (b, "optical", None))
    assert response.status == "invalid_input" and response.trace.plan == []
    assert "grid" in response.answer.lower() or "pixel grid" in response.answer


def test_forced_task_must_match_inputs(settings, write_tiff, optical_scene):
    response = run(settings, "What changed?", (write_tiff("a.tif", optical_scene), "optical", None), task="change_analysis")
    assert response.status == "invalid_input"
    assert any(i.code == "task_input_mismatch" for i in response.trace.validation)


def test_unpermitted_parameter_is_rejected(write_tiff, sar_scene):
    from satquery.imaging import load_image
    ctx = ToolContext(images=[load_image(write_tiff("s.tif", sar_scene), "sar")], vlm=FakeVLM())
    plan = [PlanStep(step_id="s1", tool="sar.water_mask", image_indices=[0], params={"threshold_db": -15}, purpose="test"),
            PlanStep(step_id="s2", tool="nonexistent.tool", image_indices=[0], purpose="test")]
    results = execute(plan, ctx)
    assert [r.status for r in results] == ["failed", "failed"]
    assert "threshold_db" in results[0].error and "registry" in results[1].error
