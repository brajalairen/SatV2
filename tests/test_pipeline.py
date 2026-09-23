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


# --------------------------------------------------------------------------- nodata (e.g. outside a drawn circle)


def _masked_optical(optical_scene):
    """The optical scene cropped to a drawn circle: corners outside it are NaN in every band."""
    from satquery.imaging import RasterImage
    yy, xx = np.mgrid[:64, :64]
    inside = (xx - 31.5) ** 2 + (yy - 31.5) ** 2 <= 29 ** 2
    data = optical_scene.copy()
    data[:, ~inside] = np.nan
    return RasterImage(data=data, band_names=["blue", "green", "red", "nir"], modality="optical", name="circle"), inside


def test_vlm_masks_never_claim_nodata(optical_scene):
    """The VLM sees the corners outside a circle as black, which a model may call water."""
    image, inside = _masked_optical(optical_scene)
    ctx = ToolContext(images=[image], vlm=FakeVLM())
    plan = [PlanStep(step_id="s1", tool="vlm.segment", image_indices=[0], params={"target": "water"}, purpose="test")]
    (result,) = execute(plan, ctx)

    mask = ctx.artifacts["s1"].masks["mask"]
    assert result.status == "ok" and not mask[~inside].any()
    assert result.outputs["fraction"] == pytest.approx(mask.sum() / inside.sum(), abs=1e-4)


def test_boxes_centred_on_nodata_are_dropped(optical_scene):
    from satquery.specialists.vlm import VLMResult

    class TwoBoxes(FakeVLM):
        def detect(self, rgb, target):
            return VLMResult(text="2 ships", boxes=[(28.0, 28.0, 36.0, 36.0), (0.0, 0.0, 4.0, 4.0)])

    image, _ = _masked_optical(optical_scene)
    ctx = ToolContext(images=[image], vlm=TwoBoxes())
    plan = [PlanStep(step_id="s1", tool="vlm.detect", image_indices=[0], params={"target": "ship"}, purpose="test")]
    (result,) = execute(plan, ctx)

    assert result.outputs["count"] == 1 and result.outputs["dropped_outside_area"] == 1
    assert [e.bbox for e in result.evidence] == [(28.0, 28.0, 36.0, 36.0)]


# --------------------------------------------------------------------------- concurrent requests (HTTP thread pool)


def _in_threads(target, count=4):
    import threading
    threads = [threading.Thread(target=target) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_backend_is_created_once_under_concurrent_first_requests(monkeypatch):
    """Two first requests at once must not load the weights twice."""
    import time
    import satquery.api as api

    created = []

    class SlowFake(FakeVLM):
        def __init__(self):
            created.append(self)
            time.sleep(0.05)  # widen the window in which a second request could slip in

    monkeypatch.setattr(api, "FakeVLM", SlowFake)
    monkeypatch.setattr(api, "_VLM_CACHE", {})
    backends = []
    _in_threads(lambda: backends.append(api.get_vlm(Settings(vlm_backend="fake"))))

    assert len(created) == 1 and all(backend is backends[0] for backend in backends)


def test_falcon_generations_run_one_at_a_time(monkeypatch):
    """One model on one GPU: concurrent requests queue for it instead of generating in parallel."""
    import threading
    import time
    from satquery.specialists.falcon import FalconVLM

    vlm = FalconVLM("not-loaded")
    active, peak, guard = 0, 0, threading.Lock()

    def fake_generate(prompt, rgb):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return "</s>water</s>", 0.5

    monkeypatch.setattr(vlm, "_generate_unlocked", fake_generate)
    rgb = np.zeros((8, 8, 3), np.uint8)
    _in_threads(lambda: vlm.caption(rgb))

    assert peak == 1


def test_the_json_report_names_files_within_its_run_folder(settings, write_tiff, optical_scene, tmp_path):
    """A downloaded report must not carry this machine's paths: they break elsewhere and disclose the layout."""
    import json

    path = write_tiff("s2.tif", optical_scene, band_names=["B02", "B03", "B04", "B08"])
    response = run(settings, "Highlight the water body.", (path, "optical", None))
    text = Path(response.report_json).read_text(encoding="utf-8")
    report = json.loads(text)
    run_dir = Path(response.report_json).parent

    assert tmp_path.name not in text, "no absolute path from this machine"
    assert report["report_html"] == "report.html" and report["report_json"] == "report.json"
    files = [e["file"] for e in report["evidence"] if e["file"]]
    assert files and all((run_dir / name).is_file() for name in files)
    assert Path(response.report_html).is_absolute(), "the in-memory response keeps real paths for the CLI and UI"


def test_a_question_too_long_for_the_model_is_rejected_up_front(settings, write_tiff, optical_scene):
    path = write_tiff("s2.tif", optical_scene, band_names=["B02", "B03", "B04", "B08"])
    question = "Is there a water body in this image? " + "Please look carefully at every part. " * 10

    response = run(settings, question, (path, "optical", None))

    assert response.status == "invalid_input" and response.trace.steps == []
    assert any(i.code == "query_too_long" for i in response.trace.validation)
    assert "at most 300" in response.answer


def test_a_long_caption_request_is_not_limited(settings, write_tiff, optical_scene):
    """Captioning never passes the question to the model, so its length does not matter."""
    path = write_tiff("s2.tif", optical_scene, band_names=["B02", "B03", "B04", "B08"])
    response = run(settings, "Describe the land-cover of this image. " + "Include every detail you can. " * 12,
                   (path, "optical", None))
    assert response.status == "ok" and response.task == "caption"


def test_falcon_adapter_setting_defaults_to_the_unadapted_base_model(monkeypatch):
    """Unset SATQUERY_FALCON_ADAPTER must leave model_id and the download target untouched (D-027)."""
    from satquery.settings import load_settings
    from satquery.specialists.falcon import FalconVLM

    monkeypatch.delenv("SATQUERY_FALCON_ADAPTER", raising=False)
    assert load_settings().falcon_adapter == ""

    vlm = FalconVLM("base/model")
    assert vlm.model_id == "base/model"
    assert vlm.base_model_id == "base/model" and vlm.adapter == ""


def test_falcon_adapter_is_named_in_the_model_id(monkeypatch):
    """With an adapter set, every execution trace must name both parts: that is what demonstrates R5."""
    from satquery.settings import load_settings
    from satquery.specialists.falcon import FalconVLM

    monkeypatch.setenv("SATQUERY_FALCON_ADAPTER", "runs/adapter")
    assert load_settings().falcon_adapter == "runs/adapter"

    vlm = FalconVLM("base/model", adapter="runs/adapter")
    assert vlm.model_id == "base/model + runs/adapter"
    # weights still come from the base repo; the adapter is applied on top
    assert vlm.base_model_id == "base/model"


def test_changing_the_adapter_does_not_reuse_the_cached_backend(monkeypatch):
    """The adapter is part of the cache key, or switching it would silently return the wrong model."""
    from satquery import api

    monkeypatch.setattr(api, "_VLM_CACHE", {})
    plain = api.get_vlm(Settings(vlm_backend="fake", falcon_adapter=""))
    adapted = api.get_vlm(Settings(vlm_backend="fake", falcon_adapter="runs/adapter"))

    assert plain is not adapted
    assert api.get_vlm(Settings(vlm_backend="fake", falcon_adapter="")) is plain
