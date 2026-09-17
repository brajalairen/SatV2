"""Combines step results into one answer with visual evidence and confidence (R9e)."""

from pathlib import Path

from satquery import evidence as ev
from satquery.raster_analysis import iou
from satquery.schemas import Confidence, Evidence, Intent, StepResult
from satquery.specialists.tools import ToolContext


class Results:
    def __init__(self, results: list[StepResult], ctx: ToolContext):
        self.results, self.ctx = results, ctx

    def get(self, tool: str, image_index: int | None = None) -> StepResult | None:
        for r in self.results:
            if r.tool == tool and r.status == "ok" and (image_index is None or image_index in self._indices(r)):
                return r
        return None

    def mask(self, step: StepResult | None, key: str = "mask"):
        return self.ctx.artifacts[step.step_id].masks.get(key) if step else None

    def _indices(self, result):
        return [e.image_index for e in result.evidence] or [None]


def _pct(fraction) -> str:
    return f"{fraction * 100:.1f}%"


def _places(regions: list[dict]) -> str:
    places = list(dict.fromkeys(r["location"] for r in regions))
    return ", ".join(places) if places else "no distinct location"


def aggregate(intent: Intent, ctx: ToolContext, results: list[StepResult], run_dir: Path):
    r = Results(results, ctx)
    compose = {"caption": _caption, "vqa": _vqa, "grounding": _grounding,
               "change_analysis": _change, "cross_modal_analysis": _cross_modal}[intent.task]
    answer, confidence, overlays = compose(intent, r, ctx, run_dir)
    step_evidence = [e for s in results for e in s.evidence]
    failed = [s for s in results if s.status == "failed"]
    if not answer:
        status, answer = "error", "The analysis could not produce a result; see the execution trace for failed steps."
    else:
        status = "partial" if failed else "ok"
    return status, answer, overlays + step_evidence, confidence


def _overlay_evidence(array, run_dir: Path, name: str, label: str, step_id: str,
                      image_index: int | None = None) -> Evidence:
    """`image_index` is the input image this overlay shares a pixel grid with, so a map can pin it
    to that raster's footprint. It stays None for side-by-side composites, which match no grid."""
    return Evidence(kind="overlay", label=label, file=ev.save_png(array, run_dir / f"{name}.png"),
                    image_index=image_index, source_step=step_id)


def _sar_context(r: Results) -> str:
    water = r.get("sar.water_mask")
    return f" SAR context: {_pct(water.outputs['fraction'])} of the scene shows low backscatter typical of water (heuristic)." if water else ""


def _caption(intent, r, ctx, run_dir):
    caption = r.get("vlm.caption")
    if not caption:
        return "", None, []
    answer = caption.outputs["text"]
    indices = r.get("optical.spectral_indices")
    if indices:
        answer += (f" Spectral indices: mean NDVI {indices.outputs['mean_ndvi']} ({indices.outputs['vegetation_level']}); "
                   f"water-like pixels (NDWI > 0): {_pct(indices.outputs['water_fraction'])}.")
    if ctx.images[0].modality == "sar":
        answer += _sar_context(r)
    return answer, caption.confidence, [_overlay_evidence(ctx.rgb(0), run_dir, "input", "input image (as analysed)", caption.step_id, 0)]


def _vqa(intent, r, ctx, run_dir):
    vqa = r.get("vlm.vqa")
    if not vqa:
        return "", None, []
    answer = vqa.outputs["text"] + (_sar_context(r) if ctx.images[0].modality == "sar" else "")
    return answer, vqa.confidence, [_overlay_evidence(ctx.rgb(0), run_dir, "input", "input image (as analysed)", vqa.step_id, 0)]


def _grounding(intent, r, ctx, run_dir):
    target = intent.target or "described region"
    for tool, color in (("vlm.segment", "red"), ("sar.water_mask", "blue"), ("sar.bright_mask", "red")):
        step = r.get(tool)
        if step:
            fraction = step.outputs["fraction"]
            answer = (f"Highlighted {target}: {_pct(fraction)} of the image, mainly in the {_places(step.outputs['regions'])}."
                      if fraction > 0 else f"No {target} was found in this image.")
            array = ev.overlay(ctx.rgb(0), masks=[(r.mask(step), color)])
            confidence = step.confidence or Confidence(value=None, method="not estimated for this tool")
            return answer, confidence, [_overlay_evidence(array, run_dir, "grounding", f"{target} (highlighted)", step.step_id, 0)]
    for tool in ("vlm.detect", "vlm.ground"):
        step = r.get(tool)
        if step:
            boxes = [e.bbox for e in step.evidence if e.bbox]
            regions = [{"location": _location(b, ctx)} for b in boxes]
            answer = (f"Found {len(boxes)} {target} region(s), located in the {_places(regions)}." if boxes
                      else f"No {target} was located.")
            array = ev.overlay(ctx.rgb(0), boxes=[(b, "red") for b in boxes])
            return answer, step.confidence, [_overlay_evidence(array, run_dir, "grounding", f"{target} (boxes)", step.step_id, 0)]
    return "", None, []


def _location(box, ctx):
    from satquery.raster_analysis import location_phrase
    return location_phrase(box, ctx.images[0].width, ctx.images[0].height)


def _change(intent, r, ctx, run_dir):
    vlm_step, map_step = r.get("vlm.change"), r.get("change.map")
    primary = vlm_step or map_step
    if not primary:
        return "", None, []
    before, after = ctx.images
    period = f"Between {before.acquired} and {after.acquired}, " if before.acquired and after.acquired else ""
    answer = (f"{period}{_pct(primary.outputs['fraction'])} of the scene changed "
              f"({'VLM change detection' if vlm_step else 'deterministic change map'}), mainly in the "
              f"{_places(primary.outputs['regions'])}.")
    confidence = primary.confidence
    if vlm_step and map_step:
        agreement = iou(r.mask(vlm_step), r.mask(map_step))
        answer += f" The deterministic change map flags {_pct(map_step.outputs['fraction'])}"
        answer += f" (agreement IoU {agreement:.2f})." if agreement is not None else "."
        confidence = Confidence(value=round(agreement, 3) if agreement is not None else None,
                                method="agreement IoU between VLM change mask and deterministic change map (heuristic)")
    compare = r.get("change.compare_areas")
    if compare:
        o = compare.outputs
        if o["verdict"] == "inconclusive":
            answer = (f"Could not determine whether {o['target']} area changed: none was detected in either image "
                      f"(it may not be resolvable at this image resolution). ") + answer
        else:
            answer = (f"{o['target'].capitalize()} area {o['verdict']}: {o['before_percent']}% -> {o['after_percent']}% of the scene "
                      f"({o['change_percentage_points']:+.1f} percentage points; measured as {o['measure']}). ") + answer
    masks = [(r.mask(map_step), "yellow"), (r.mask(vlm_step), "red")]
    array = ev.side_by_side(ctx.rgb(0), ev.overlay(ctx.rgb(1), masks=masks))
    label = "before | after with change (red: VLM, yellow: deterministic map)"
    return answer, confidence, [_overlay_evidence(array, run_dir, "change", label, primary.step_id)]


def _cross_modal(intent, r, ctx, run_dir):
    fusion = r.get("fusion.cross_modal")
    if not fusion:
        return "", None, []
    optical = 0 if ctx.images[0].modality == "optical" else 1
    parts = []
    for key, label in (("water", "Water"), ("built_up", "Built-up")):
        o = fusion.outputs.get(key, {})
        if o.get("available", True) and "sar_percent" in o:
            agreement = f"{o['agreement_iou']:.2f}" if o["agreement_iou"] is not None else "n/a"
            parts.append(f"{label}: optical {o['optical_percent']}%, SAR {o['sar_percent']}%, confirmed by both "
                         f"{o['both_percent']}% (agreement IoU {agreement}), mainly in the {_places(o['regions'])}.")
    if not parts:
        return "", None, []
    art = ctx.artifacts[fusion.step_id].masks
    masks = [(art.get("water_optical_only"), "cyan"), (art.get("water_sar_only"), "cyan"), (art.get("water_both"), "blue"),
             (art.get("built_up_optical_only"), "orange"), (art.get("built_up_sar_only"), "orange"), (art.get("built_up_both"), "red")]
    array = ev.side_by_side(ev.overlay(ctx.rgb(optical), masks=masks), ctx.rgb(1 - optical))
    label = "optical with fused masks (blue/red: both sensors agree; cyan/orange: one sensor) | SAR false colour"
    return " ".join(parts), fusion.confidence, [_overlay_evidence(array, run_dir, "cross_modal", label, fusion.step_id)]
