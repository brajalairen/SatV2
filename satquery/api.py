"""The single entry point shared by the UI, CLI and tests (D-005): analyze(request) -> response."""

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from satquery.agent.aggregator import aggregate
from satquery.agent.executor import execute
from satquery.agent.intents import (COMPATIBLE_TASKS, classify, find_target, needs_multiple_dates,
                                    needs_optical_and_sar)
from satquery.agent.planner import build_plan
from satquery.evidence import write_reports
from satquery import geo
from satquery.imaging import load_image
from satquery.schemas import AnalysisRequest, AnalysisResponse, ExecutionTrace, Intent, ValidationIssue
from satquery.settings import Settings, load_settings
from satquery.specialists.tools import MAX_PROMPT_CHARS, ToolContext
from satquery.specialists.vlm import FakeVLM, VLMBackend
from satquery.validation import check_images, check_request, detect_input_config, issue

_VLM_CACHE: dict[tuple, VLMBackend] = {}
# The HTTP server runs analyses on a thread pool: without this, two simultaneous first requests
# could each create a backend and load the weights twice.
_VLM_CACHE_LOCK = threading.Lock()


def get_vlm(settings: Settings) -> VLMBackend:
    key = (settings.vlm_backend, settings.falcon_model_id, settings.falcon_adapter, settings.device,
           settings.num_beams, settings.max_new_tokens)
    with _VLM_CACHE_LOCK:
        if key not in _VLM_CACHE:
            if settings.vlm_backend == "fake":
                _VLM_CACHE[key] = FakeVLM()
            elif settings.vlm_backend == "falcon":
                from satquery.specialists.falcon import FalconVLM
                _VLM_CACHE[key] = FalconVLM(settings.falcon_model_id, settings.device, settings.num_beams,
                                            settings.max_new_tokens, settings.falcon_adapter)
            else:
                raise ValueError(f"unknown SATQUERY_VLM_BACKEND '{settings.vlm_backend}' (use 'falcon' or 'fake')")
        return _VLM_CACHE[key]


def preload(settings: Settings | None = None) -> VLMBackend:
    """Load model weights now (required at import time on ZeroGPU; avoids a slow first request elsewhere)."""
    vlm = get_vlm(settings or load_settings())
    if hasattr(vlm, "load"):
        vlm.load()
    return vlm


def _errors(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    return [i for i in issues if i.severity == "error"]


def _missing_modality(query: str, images) -> ValidationIssue | None:
    """A structured refusal when the query asks for optical + SAR but one modality is absent."""
    phrase = needs_optical_and_sar(query)
    if not phrase:
        return None
    present = {image.modality for image in images}
    if present == {"optical", "sar"}:
        return None
    have, need = ("optical", "SAR") if "optical" in present else ("SAR", "optical")
    count = "an" if have == "optical" else "a"
    provided = f"only {count} {have} image was provided" if len(images) == 1 else f"both images are declared {have}"
    return issue(f"missing_{need.lower()}",
                 f'"{phrase}" asks for a joint optical + SAR analysis, but {provided}, so there is no {need} '
                 f"evidence to combine. Add {'an' if need == 'optical' else 'a'} {need} image of the same area on the "
                 f"same pixel grid, or, if one of these images is {need}, mark it as {need} and ask again.")


def analyze(request: AnalysisRequest, settings: Settings | None = None, vlm: VLMBackend | None = None) -> AnalysisResponse:
    settings = settings or load_settings()
    started = time.perf_counter()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    run_dir = Path(settings.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace = ExecutionTrace(run_id=run_id, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           query=request.query, input_config=None, images=[], validation=[], intent=None, plan=[], steps=[])

    def finish(status, answer, task=None, evidence=(), confidence=None) -> AnalysisResponse:
        trace.total_duration_s = round(time.perf_counter() - started, 3)
        response = AnalysisResponse(status=status, task=task, answer=answer, evidence=list(evidence),
                                    confidence=confidence, trace=trace)
        response.report_html, response.report_json = write_reports(response, run_dir)
        return response

    def reject() -> AnalysisResponse:
        return finish("invalid_input", "Input rejected: " + " ".join(i.message for i in _errors(trace.validation)))

    trace.validation = check_request(request)
    if _errors(trace.validation):
        return reject()

    images = []
    for i, item in enumerate(request.images):
        try:
            images.append(load_image(item.path, item.modality, item.acquired, settings.max_pixels))
        except Exception as error:
            trace.validation.append(issue("unreadable", f"{Path(item.path).name}: cannot read image ({error}).", image_index=i))
    if _errors(trace.validation):
        return reject()
    trace.images = [geo.summarize(image, i) for i, image in enumerate(images)]
    trace.validation += check_images(images)
    if _errors(trace.validation):
        return reject()

    config = detect_input_config([image.modality for image in images])
    trace.input_config = config
    # A question asking for optical and SAR together is refused unless both are present. Checked
    # first: it must never fall through to single-image VQA, to bi-temporal change analysis on two
    # optical images, or to the temporal refusal below ("compare optical and SAR" names no dates).
    missing = None if request.forced_task or config == "pair_cross_modal" else _missing_modality(request.query, images)
    if missing:
        trace.validation.append(missing)
        return reject()
    # One image cannot show a change, so the VLM is never asked to infer one from a single frame.
    # A forced task is the caller's explicit choice and is checked against the inputs below instead.
    phrase = None if request.forced_task or len(images) != 1 else needs_multiple_dates(request.query)
    if phrase:
        trace.validation.append(issue(
            "needs_multiple_dates",
            f'"{phrase}" asks about change over time, which needs two images of the same area from different '
            "dates; one image cannot show a change. Add a second, dated image of the same area, or draw a "
            "rectangle on the map so two Sentinel-2 dates can be retrieved for it."))
        return reject()
    if request.forced_task:
        target, _ = find_target(request.query)
        intent = Intent(task=request.forced_task, target=target, matched_rule="task forced by caller")
    else:
        intent = classify(request.query, config)
    trace.intent = intent
    if intent.task not in COMPATIBLE_TASKS[config]:
        trace.validation.append(issue("task_input_mismatch", f"Task '{intent.task}' cannot run on input configuration '{config}'."))
        return reject()

    trace.plan = build_plan(intent, config, images, request.query)
    # Some plans hand the question to the VLM verbatim; its permitted length would otherwise fail
    # that step mid-run with a generic error. Captions never pass it on, so they are not limited.
    if any(len(str(step.params.get(key, ""))) > MAX_PROMPT_CHARS for step in trace.plan for key in ("question", "description")):
        trace.validation.append(issue("query_too_long", f"The question is {len(request.query)} characters; the model "
                                      f"accepts at most {MAX_PROMPT_CHARS}. Please shorten it."))
        return reject()
    ctx = ToolContext(images=images, vlm=vlm or get_vlm(settings))
    trace.steps = execute(trace.plan, ctx)
    status, answer, evidence, confidence = aggregate(intent, ctx, trace.steps, run_dir)
    return finish(status, answer, intent.task, evidence, confidence)
