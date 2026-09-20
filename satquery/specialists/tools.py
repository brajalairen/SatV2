"""Tool registry (R9c): every step the planner may schedule, each with its permitted parameters (R9d)."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from satquery import raster_analysis as ra
from satquery.imaging import RasterImage, render_rgb
from satquery.schemas import Confidence, Evidence
from satquery.specialists.vlm import VLMBackend, VLMResult

SAR_VLM_NOTE = "The VLM was trained on optical imagery; this SAR image was rendered as false colour, so reliability is low."


@dataclass
class ToolContext:
    images: list[RasterImage]
    vlm: VLMBackend
    artifacts: dict[str, "ToolOutput"] = field(default_factory=dict)
    _renders: dict[int, np.ndarray] = field(default_factory=dict)
    _valid: dict[int, np.ndarray] = field(default_factory=dict)

    def rgb(self, index: int) -> np.ndarray:
        if index not in self._renders:
            self._renders[index] = render_rgb(self.images[index])
        return self._renders[index]

    def valid(self, *indices: int) -> np.ndarray:
        """bool (height, width): pixels with data in every listed image.

        Nodata is NaN, and that includes the outside of a drawn circle or polygon (geo.crop_to_bbox),
        so masks are restricted to valid pixels and every coverage figure counts only those.
        """
        for index in indices:
            if index not in self._valid:
                self._valid[index] = np.isfinite(self.images[index].data).any(axis=0)
        return np.logical_and.reduce([self._valid[index] for index in indices])


@dataclass
class ToolOutput:
    outputs: dict
    evidence: list[Evidence] = field(default_factory=list)
    confidence: Confidence | None = None
    model: str | None = None
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # in memory only, for later steps and overlays
    skipped_reason: str | None = None


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")  # any parameter not declared here is rejected


class CaptionParams(Params):
    detailed: bool = False


MAX_PROMPT_CHARS = 300  # longest free text (the user's question) a tool passes to the VLM


class QuestionParams(Params):
    question: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)


class TargetParams(Params):
    target: str = Field(min_length=1, max_length=40)


class DescriptionParams(Params):
    description: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)


class WaterMaskParams(Params):
    smoothing_window: int = Field(5, ge=3, le=15)


class BrightMaskParams(Params):
    percentile: float = Field(90, ge=75, le=99)


class IndicesParams(Params):
    ndwi_water_threshold: float = Field(0.0, ge=-0.5, le=0.5)
    ndvi_vegetation_threshold: float = Field(0.3, ge=0.0, le=0.9)
    ndvi_bare_threshold: float = Field(0.2, ge=0.0, le=0.5)


class ChangeMapParams(Params):
    min_region_px: int = Field(16, ge=0, le=10000)


class CompareParams(Params):
    target: str = Field(min_length=1, max_length=40)
    before_step: str
    after_step: str
    mask_key: str = Field("mask", pattern=r"^[a-z_]+$")  # which mask of the referenced steps to compare


class FusionParams(Params):
    sar_water_step: str
    sar_bright_step: str
    optical_water_step: str
    optical_building_step: str


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params_model: type[Params]
    run: Callable[[ToolContext, list[int], Params, str], ToolOutput]


def _vlm_confidence(result: VLMResult, image: RasterImage) -> Confidence:
    note = SAR_VLM_NOTE if image.modality == "sar" else None
    if result.score is None:
        return Confidence(value=None, method="not available from this backend", note=note)
    return Confidence(value=round(result.score, 3), method="beam-search sequence probability (uncalibrated)", note=note)


def _mask_output(mask: np.ndarray, label: str, index: int, step_id: str, extra: dict | None = None,
                 valid: np.ndarray | None = None) -> ToolOutput:
    if valid is not None and valid.shape == mask.shape:
        mask = mask & valid  # a model may paint nodata (e.g. black corners outside a drawn circle)
    else:
        valid = None
    fraction = round(float(ra.coverage(mask, valid)), 4)
    found = ra.regions(mask, valid=valid)
    evidence = [Evidence(kind="mask", label=label, image_index=index, fraction=fraction, source_step=step_id)]
    evidence += [Evidence(kind="bbox", label=f"{label} region ({r['location']})", image_index=index, bbox=r["bbox"],
                          fraction=round(r["fraction"], 4), source_step=step_id) for r in found]
    outputs = {"fraction": fraction, "regions": found, **(extra or {})}
    return ToolOutput(outputs=outputs, evidence=evidence, masks={"mask": mask})


def vlm_caption(ctx, idx, p: CaptionParams, step_id):
    result = ctx.vlm.caption(ctx.rgb(idx[0]), p.detailed)
    return ToolOutput(outputs={"text": result.text}, confidence=_vlm_confidence(result, ctx.images[idx[0]]), model=ctx.vlm.model_id)


def vlm_vqa(ctx, idx, p: QuestionParams, step_id):
    result = ctx.vlm.vqa(ctx.rgb(idx[0]), p.question)
    return ToolOutput(outputs={"text": result.text}, confidence=_vlm_confidence(result, ctx.images[idx[0]]), model=ctx.vlm.model_id)


def _centre_is_valid(box, valid: np.ndarray) -> bool:
    height, width = valid.shape
    column = min(max(int((box[0] + box[2]) / 2), 0), width - 1)
    row = min(max(int((box[1] + box[3]) / 2), 0), height - 1)
    return bool(valid[row, column])


def _vlm_boxes(result: VLMResult, ctx, idx, label, step_id):
    valid = ctx.valid(idx[0])
    boxes = [b for b in result.boxes if _centre_is_valid(b, valid)]  # drop boxes centred on nodata
    evidence = [Evidence(kind="bbox", label=label, image_index=idx[0], bbox=b, source_step=step_id) for b in boxes]
    outputs = {"count": len(boxes), "text": result.text}
    if len(boxes) < len(result.boxes):
        outputs["dropped_outside_area"] = len(result.boxes) - len(boxes)
    return ToolOutput(outputs=outputs, evidence=evidence,
                      confidence=_vlm_confidence(result, ctx.images[idx[0]]), model=ctx.vlm.model_id)


def vlm_detect(ctx, idx, p: TargetParams, step_id):
    return _vlm_boxes(ctx.vlm.detect(ctx.rgb(idx[0]), p.target), ctx, idx, p.target, step_id)


def vlm_ground(ctx, idx, p: DescriptionParams, step_id):
    return _vlm_boxes(ctx.vlm.ground(ctx.rgb(idx[0]), p.description), ctx, idx, "matched region", step_id)


def vlm_segment(ctx, idx, p: TargetParams, step_id):
    result = ctx.vlm.segment(ctx.rgb(idx[0]), p.target)
    out = _mask_output(result.mask, p.target, idx[0], step_id, valid=ctx.valid(idx[0]))
    out.confidence, out.model = _vlm_confidence(result, ctx.images[idx[0]]), ctx.vlm.model_id
    return out


def vlm_change(ctx, idx, p: Params, step_id):
    result = ctx.vlm.change(ctx.rgb(idx[0]), ctx.rgb(idx[1]))
    out = _mask_output(result.mask, "change", idx[1], step_id, valid=ctx.valid(idx[0], idx[1]))
    out.confidence, out.model = _vlm_confidence(result, ctx.images[idx[1]]), ctx.vlm.model_id
    return out


def sar_backscatter(ctx, idx, p: Params, step_id):
    stats = ra.backscatter_stats(ctx.images[idx[0]])
    evidence = [Evidence(kind="metric", label="mean co-pol backscatter (dB)", image_index=idx[0],
                         value=round(stats["copol"]["mean"], 2) if stats["copol"]["mean"] is not None else None, source_step=step_id)]
    return ToolOutput(outputs=stats, evidence=evidence)


def sar_water(ctx, idx, p: WaterMaskParams, step_id):
    mask, info = ra.sar_water_mask(ctx.images[idx[0]], p.smoothing_window)
    out = _mask_output(mask, "water (SAR, low backscatter)", idx[0], step_id, info, valid=ctx.valid(idx[0]))
    out.confidence = Confidence(value=round(info["separability"], 3), method="Otsu histogram separability (heuristic)")
    return out


def sar_bright(ctx, idx, p: BrightMaskParams, step_id):
    mask, info = ra.sar_bright_mask(ctx.images[idx[0]], p.percentile)
    return _mask_output(mask, "strong scatterers (SAR, candidate built-up)", idx[0], step_id, info,
                        valid=ctx.valid(idx[0]))


def spectral(ctx, idx, p: IndicesParams, step_id):
    indices = ra.spectral_indices(ctx.images[idx[0]])
    if indices is None:
        return ToolOutput(outputs={}, skipped_reason="image has no red/green/NIR bands")
    water = np.nan_to_num(indices["ndwi"] > p.ndwi_water_threshold, nan=False).astype(bool)
    vegetation = np.nan_to_num(indices["ndvi"] > p.ndvi_vegetation_threshold, nan=False).astype(bool)
    # Heuristic proxy: neither vegetated nor water. It includes bare soil, so it is not a building classifier.
    built_up_proxy = np.nan_to_num((indices["ndvi"] < p.ndvi_bare_threshold) & (indices["ndwi"] < p.ndwi_water_threshold),
                                   nan=False).astype(bool)
    mean_ndvi = float(np.nanmean(indices["ndvi"]))
    label = "high" if mean_ndvi >= 0.6 else "moderate" if mean_ndvi >= 0.3 else "low"  # ported heuristic
    valid = ctx.valid(idx[0])
    outputs = {"mean_ndvi": round(mean_ndvi, 3), "mean_ndwi": round(float(np.nanmean(indices["ndwi"])), 3),
               "vegetation_level": f"{label} (heuristic on mean NDVI)", "water_fraction": round(float(ra.coverage(water, valid)), 4),
               "vegetation_fraction": round(float(ra.coverage(vegetation, valid)), 4),
               "built_up_proxy_fraction": round(float(ra.coverage(built_up_proxy, valid)), 4)}
    evidence = [Evidence(kind="metric", label="mean NDVI", image_index=idx[0], value=outputs["mean_ndvi"], source_step=step_id)]
    return ToolOutput(outputs=outputs, evidence=evidence,
                      masks={"water": water, "vegetation": vegetation, "built_up_proxy": built_up_proxy})


def change_map(ctx, idx, p: ChangeMapParams, step_id):
    mask, info = ra.change_map(ctx.images[idx[0]], ctx.images[idx[1]], p.min_region_px)
    out = _mask_output(mask, "change (deterministic)", idx[1], step_id, info, valid=ctx.valid(idx[0], idx[1]))
    out.confidence = Confidence(value=round(info["separability"], 3), method="Otsu histogram separability (heuristic)")
    return out


MEASURE_LABELS = {"built_up_proxy": "built-up/bare-surface proxy (low NDVI and NDWI; heuristic)",
                  "water": "water (NDWI > 0)", "vegetation": "vegetation (NDVI > 0.3)"}


def compare_areas(ctx, idx, p: CompareParams, step_id):
    valid = ctx.valid(*idx)
    before = ra.coverage(ctx.artifacts[p.before_step].masks[p.mask_key], valid)
    after = ra.coverage(ctx.artifacts[p.after_step].masks[p.mask_key], valid)
    delta_pp = (after - before) * 100
    if before < 0.001 and after < 0.001:  # nothing detected in either image: do not claim "unchanged"
        verdict = "inconclusive"
    else:
        verdict = "increased" if delta_pp > 1 else "decreased" if delta_pp < -1 else "remained roughly unchanged"
    outputs = {"target": p.target, "measure": MEASURE_LABELS.get(p.mask_key, f"{p.target} (model segmentation)"),
               "before_percent": round(before * 100, 2), "after_percent": round(after * 100, 2),
               "change_percentage_points": round(float(delta_pp), 2), "verdict": verdict, "tolerance_pp": 1.0}
    return ToolOutput(outputs=outputs, evidence=[Evidence(kind="metric", label=f"{p.target} change (percentage points)",
                                                          value=outputs["change_percentage_points"], source_step=step_id)])


def cross_modal(ctx, idx, p: FusionParams, step_id):
    get = lambda step, key="mask": ctx.artifacts[step].masks.get(key) if step in ctx.artifacts else None
    sar_water_mask, sar_bright_mask = get(p.sar_water_step), get(p.sar_bright_step)
    optical_water, optical_building = get(p.optical_water_step, "water"), get(p.optical_building_step)
    if optical_water is None:
        optical_water = get(p.optical_water_step)
    masks, outputs, agreements = {}, {}, []
    valid = ctx.valid(*idx)
    for name, sar_mask, optical_mask in (("water", sar_water_mask, optical_water), ("built_up", sar_bright_mask, optical_building)):
        if sar_mask is None or optical_mask is None:
            outputs[name] = {"available": False}
            continue
        agreement = ra.iou(sar_mask, optical_mask)
        masks[f"{name}_both"] = sar_mask & optical_mask
        masks[f"{name}_sar_only"] = sar_mask & ~optical_mask
        masks[f"{name}_optical_only"] = optical_mask & ~sar_mask
        percent = lambda mask: round(ra.coverage(mask, valid) * 100, 2)
        outputs[name] = {"sar_percent": percent(sar_mask), "optical_percent": percent(optical_mask),
                         "both_percent": percent(masks[f"{name}_both"]), "agreement_iou": agreement,
                         "regions": ra.regions(sar_mask | optical_mask, 3, valid=valid)}
        if agreement is not None:
            agreements.append(agreement)
    confidence = Confidence(value=round(float(np.mean(agreements)), 3) if agreements else None,
                            method="mean optical-SAR agreement IoU (heuristic)")
    evidence = [Evidence(kind="metric", label=f"{name} optical-SAR agreement IoU", value=o.get("agreement_iou"), source_step=step_id)
                for name, o in outputs.items() if o.get("available", True)]
    return ToolOutput(outputs=outputs, evidence=evidence, confidence=confidence, masks=masks)


REGISTRY: dict[str, Tool] = {tool.name: tool for tool in [
    Tool("vlm.caption", "Scene description by the remote-sensing VLM", CaptionParams, vlm_caption),
    Tool("vlm.vqa", "Visual question answering by the remote-sensing VLM", QuestionParams, vlm_vqa),
    Tool("vlm.detect", "Object detection with bounding boxes", TargetParams, vlm_detect),
    Tool("vlm.ground", "Text-guided region grounding with bounding boxes", DescriptionParams, vlm_ground),
    Tool("vlm.segment", "Text-guided segmentation mask", TargetParams, vlm_segment),
    Tool("vlm.change", "Bi-temporal change detection by the VLM", Params, vlm_change),
    Tool("sar.backscatter_stats", "SAR backscatter statistics in dB", Params, sar_backscatter),
    Tool("sar.water_mask", "SAR low-backscatter water mask (Otsu, heuristic)", WaterMaskParams, sar_water),
    Tool("sar.bright_mask", "SAR strong-scatterer mask (candidate built-up, heuristic)", BrightMaskParams, sar_bright),
    Tool("optical.spectral_indices", "NDVI/NDWI masks when NIR is available", IndicesParams, spectral),
    Tool("change.map", "Deterministic change map (change vector / log-ratio + Otsu)", ChangeMapParams, change_map),
    Tool("change.compare_areas", "Compare a class's area between two dates", CompareParams, compare_areas),
    Tool("fusion.cross_modal", "Combine optical and SAR masks with an agreement score", FusionParams, cross_modal),
]}
