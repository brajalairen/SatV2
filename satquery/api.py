"""The single entry point shared by the UI, CLI and tests (D-005): analyze(request) -> response."""

import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from satquery.agent.aggregator import aggregate
from satquery.agent.executor import execute
from satquery.agent.intents import COMPATIBLE_TASKS, classify, find_target
from satquery.agent.planner import build_plan
from satquery.evidence import write_reports
from satquery import geo
from satquery.imaging import load_image
from satquery.schemas import AnalysisRequest, AnalysisResponse, ExecutionTrace, Intent, ValidationIssue
from satquery.settings import Settings, load_settings
from satquery.specialists.tools import ToolContext
from satquery.specialists.vlm import FakeVLM, VLMBackend
from satquery.validation import check_images, check_request, detect_input_config, issue

_VLM_CACHE: dict[tuple, VLMBackend] = {}


def get_vlm(settings: Settings) -> VLMBackend:
    key = (settings.vlm_backend, settings.falcon_model_id, settings.device, settings.num_beams, settings.max_new_tokens)
    if key not in _VLM_CACHE:
        if settings.vlm_backend == "fake":
            _VLM_CACHE[key] = FakeVLM()
        elif settings.vlm_backend == "falcon":
            from satquery.specialists.falcon import FalconVLM
            _VLM_CACHE[key] = FalconVLM(settings.falcon_model_id, settings.device, settings.num_beams, settings.max_new_tokens)
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
        Path(response.report_json).write_text(response.model_dump_json(indent=2), encoding="utf-8")
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
    ctx = ToolContext(images=images, vlm=vlm or get_vlm(settings))
    trace.steps = execute(trace.plan, ctx)
    status, answer, evidence, confidence = aggregate(intent, ctx, trace.steps, run_dir)
    return finish(status, answer, intent.task, evidence, confidence)
