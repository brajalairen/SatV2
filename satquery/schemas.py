"""Shared contracts between modules (D-005). Changing a field here needs team approval.

Coordinate convention: every geometry is in PIXEL coordinates of the loaded image
(`RasterImage` grid, x to the right, y down), unless a field name says otherwise.
"""

from typing import Literal

from pydantic import BaseModel, Field

Modality = Literal["optical", "sar"]
InputConfig = Literal["single_optical", "single_sar", "pair_cross_modal", "pair_bitemporal"]
TaskType = Literal["vqa", "caption", "grounding", "change_analysis", "cross_modal_analysis"]
Severity = Literal["error", "warning"]
StepStatus = Literal["ok", "failed", "skipped"]
ResponseStatus = Literal["ok", "partial", "invalid_input", "error"]


class ImageInput(BaseModel):
    path: str
    modality: Modality
    acquired: str | None = None  # ISO date if known, e.g. "2023-05-01"


class AnalysisRequest(BaseModel):
    query: str
    images: list[ImageInput]
    forced_task: TaskType | None = None  # bypasses intent rules; recorded in the trace


class ValidationIssue(BaseModel):
    code: str
    severity: Severity
    message: str
    image_index: int | None = None


class ImageSummary(BaseModel):
    index: int
    name: str
    modality: Modality
    width: int
    height: int
    bands: list[str]
    crs: str | None
    acquired: str | None
    decimation: float = 1.0  # >1 when the raster was read at reduced resolution
    # Map placement (satquery/geo.py). None whenever the image carries no CRS + transform.
    bounds_wgs84: tuple[float, float, float, float] | None = None  # west, south, east, north
    corners_wgs84: list[tuple[float, float]] | None = None  # TL, TR, BR, BL for a map image source
    georeference_note: str | None = None  # set when the placement is only approximate


class Intent(BaseModel):
    task: TaskType
    target: str | None = None  # e.g. "water", "building"
    comparative: bool = False  # e.g. "has built-up area increased?"
    matched_rule: str


class PlanStep(BaseModel):
    step_id: str
    tool: str
    image_indices: list[int]
    params: dict = Field(default_factory=dict)
    purpose: str


class Evidence(BaseModel):
    kind: Literal["bbox", "mask", "overlay", "metric"]
    label: str
    image_index: int | None = None
    bbox: tuple[float, float, float, float] | None = None  # pixel x_min, y_min, x_max, y_max
    fraction: float | None = None  # share of image pixels covered by a mask
    value: float | str | None = None
    file: str | None = None
    source_step: str


class Confidence(BaseModel):
    value: float | None
    method: str  # how it was computed, shown to users (D-010)
    calibrated: bool = False
    note: str | None = None


class StepResult(BaseModel):
    step_id: str
    tool: str
    model: str | None = None
    status: StepStatus
    params: dict = Field(default_factory=dict)
    outputs: dict = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Confidence | None = None
    error: str | None = None
    duration_s: float = 0.0


class ExecutionTrace(BaseModel):
    run_id: str
    created_at: str
    query: str
    input_config: InputConfig | None
    images: list[ImageSummary]
    validation: list[ValidationIssue]
    intent: Intent | None
    plan: list[PlanStep]
    steps: list[StepResult]
    total_duration_s: float = 0.0


class AnalysisResponse(BaseModel):
    status: ResponseStatus
    task: TaskType | None
    answer: str
    evidence: list[Evidence]
    confidence: Confidence | None
    trace: ExecutionTrace
    report_html: str | None = None
    report_json: str | None = None
