"""Combines step results into one answer with visual evidence and confidence (R9e)."""

from pathlib import Path

from satquery import evidence as ev
from satquery.raster_analysis import iou, regions as mask_regions
from satquery.schemas import Confidence, Evidence, Intent, StepResult
from satquery.specialists.tools import ToolContext
from satquery.specialists.weather import HEAVY_RAIN_CODES


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
    compose = {"caption": _caption, "vqa": _vqa, "grounding": _grounding, "change_analysis": _change,
               "cross_modal_analysis": _cross_modal, "weather_forecast": _weather}[intent.task]
    answer, confidence, overlays = compose(intent, r, ctx, run_dir)
    step_evidence = [e for s in results for e in s.evidence]
    failed = [s for s in results if s.status == "failed"]
    if not answer:
        status, answer = "error", "The analysis could not produce a result; see the execution trace for failed steps."
    else:
        status = "partial" if failed else "ok"
    return status, answer, overlays + step_evidence, confidence


def _overlay_evidence(array, run_dir: Path, name: str, label: str, step_id: str,
                      image_index: int | None = None, ctx: ToolContext | None = None) -> Evidence:
    """`image_index` is the input image this overlay shares a pixel grid with, so a map can pin it
    to that raster's footprint. It stays None for side-by-side composites, which match no grid.
    Pinned overlays are transparent wherever that image has no data (e.g. outside a drawn circle)."""
    alpha = ctx.valid(image_index) if ctx is not None and image_index is not None else None
    return Evidence(kind="overlay", label=label, file=ev.save_png(array, run_dir / f"{name}.png", alpha),
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
    return answer, caption.confidence, [_overlay_evidence(ctx.rgb(0), run_dir, "input", "input image (as analysed)", caption.step_id, 0, ctx)]


def _vqa(intent, r, ctx, run_dir):
    vqa = r.get("vlm.vqa")
    if not vqa:
        return "", None, []
    answer = vqa.outputs["text"] + (_sar_context(r) if ctx.images[0].modality == "sar" else "")
    return answer, vqa.confidence, [_overlay_evidence(ctx.rgb(0), run_dir, "input", "input image (as analysed)", vqa.step_id, 0, ctx)]


# Water and vegetation on multispectral input: the spectral mask is the primary evidence (D-030).
SPECTRAL_GROUNDING = {"water": ("water", "NDWI > {ndwi_water_threshold:g}", "blue"),
                      "vegetation": ("vegetation", "NDVI > {ndvi_vegetation_threshold:g}", "green")}


def _spectral_grounding(target: str, spectral: StepResult, r, ctx, run_dir):
    """The spectral mask decides; the VLM's figure is reported beside it as a secondary check, never
    in its place (it can miss water that the index finds)."""
    key, rule, color = SPECTRAL_GROUNDING[target]
    rule = rule.format(**spectral.params)
    valid = ctx.valid(0)
    mask = r.mask(spectral, key) & valid
    fraction = spectral.outputs[f"{key}_fraction"]
    answer = (f"Highlighted {target}: {_pct(fraction)} of the analysed area by {rule} (spectral index), mainly in the "
              f"{_places(mask_regions(mask, valid=valid))}." if fraction > 0 else
              f"No {target} was found in this image by {rule} (spectral index).")
    vlm = r.get("vlm.segment")
    if vlm:
        answer += f" The VLM segmentation, a secondary check, found {_pct(vlm.outputs['fraction'])}."
    if not valid.all():
        answer += (f" {_pct(1 - valid.mean())} of the image had no usable data (for example, masked cloud) "
                   "and is not counted.")
    confidence = Confidence(value=None, method=f"not estimated: {rule} is a fixed spectral rule (heuristic)")
    array = ev.overlay(ctx.rgb(0), masks=[(mask, color)])
    return answer, confidence, [_overlay_evidence(array, run_dir, "grounding", f"{target} ({rule}, highlighted)",
                                                  spectral.step_id, 0, ctx)]


def _grounding(intent, r, ctx, run_dir):
    target = intent.target or "described region"
    spectral = r.get("optical.spectral_indices")
    if spectral and intent.target in SPECTRAL_GROUNDING:
        return _spectral_grounding(intent.target, spectral, r, ctx, run_dir)
    for tool, color in (("vlm.segment", "red"), ("sar.water_mask", "blue"), ("sar.bright_mask", "red")):
        step = r.get(tool)
        if step:
            fraction = step.outputs["fraction"]
            answer = (f"Highlighted {target}: {_pct(fraction)} of the image, mainly in the {_places(step.outputs['regions'])}."
                      if fraction > 0 else f"No {target} was found in this image.")
            array = ev.overlay(ctx.rgb(0), masks=[(r.mask(step), color)])
            confidence = step.confidence or Confidence(value=None, method="not estimated for this tool")
            return answer, confidence, [_overlay_evidence(array, run_dir, "grounding", f"{target} (highlighted)", step.step_id, 0, ctx)]
    for tool in ("vlm.detect", "vlm.ground"):
        step = r.get(tool)
        if step:
            boxes = [e.bbox for e in step.evidence if e.bbox]
            regions = [{"location": _location(b, ctx)} for b in boxes]
            answer = (f"Found {len(boxes)} {target} region(s), located in the {_places(regions)}." if boxes
                      else f"No {target} was located.")
            array = ev.overlay(ctx.rgb(0), boxes=[(b, "red") for b in boxes])
            return answer, step.confidence, [_overlay_evidence(array, run_dir, "grounding", f"{target} (boxes)", step.step_id, 0, ctx)]
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
    # Each figure is attributed to what produced it: the VLM's interpretation, or the deterministic
    # map, which is a heuristic on pixel differences and not a land-cover classification.
    source = "VLM change detection" if vlm_step else "the deterministic change map (heuristic pixel differencing)"
    answer = (f"{period}{source} flags {_pct(primary.outputs['fraction'])} of the scene as changed, mainly in the "
              f"{_places(primary.outputs['regions'])}.")
    confidence = primary.confidence
    if vlm_step and map_step:
        agreement = iou(r.mask(vlm_step), r.mask(map_step))
        answer += (f" The deterministic change map (heuristic pixel differencing) flags "
                   f"{_pct(map_step.outputs['fraction'])}")
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
    # The same change marks on the later image alone, which shares the pair's pixel grid (validation
    # requires it), so a map can pin it where the change is. The composite matches no single grid.
    on_map = _overlay_evidence(ev.overlay(ctx.rgb(1), masks=masks), run_dir, "change_on_after",
                               "changed areas on the later image (red: VLM, yellow: deterministic map, heuristic)",
                               primary.step_id, 1, ctx)
    return answer, confidence, [on_map, _overlay_evidence(array, run_dir, "change", label, primary.step_id)]


# Per fused class: the words used in the answer, and the map colours (both sensors, optical only, SAR only).
# Confirmed evidence is drawn strongly and one-sensor evidence faintly: disagreement stays visible, but a
# broad single-sensor proxy (e.g. bare desert in the optical built-up proxy) cannot drown the fused result.
CONFIRMED_ALPHA, SINGLE_SENSOR_ALPHA = 210, 75
FUSED_CLASSES = {
    "water": {"noun": "water-covered regions", "short": "water",
              "colors": (("both", "blue", CONFIRMED_ALPHA), ("optical_only", "cyan", SINGLE_SENSOR_ALPHA),
                         ("sar_only", "violet", SINGLE_SENSOR_ALPHA)),
              "legend": "solid blue = optical and SAR agree; faint cyan = optical only, faint violet = SAR only"},
    "built_up": {"noun": "built-up candidates", "short": "built-up",
                 "colors": (("both", "red", CONFIRMED_ALPHA), ("optical_only", "orange", SINGLE_SENSOR_ALPHA),
                            ("sar_only", "yellow", SINGLE_SENSOR_ALPHA)),
                 "legend": "solid red = optical and SAR agree; faint orange = optical only, faint yellow = SAR only "
                           "(proxies, heuristic)"},
}


def _pct1(percent: float) -> str:
    return f"{percent:.1f}%"


def _optical_source(step: StepResult | None, mask_key: str) -> str:
    """What the optical mask measures, from the step and parameters that produced it."""
    if step is None:
        return "optical evidence"
    if step.tool == "optical.spectral_indices" and mask_key == "water":
        return f"water by NDWI > {step.params.get('ndwi_water_threshold', 0.0):g} (spectral index)"
    if step.tool == "optical.spectral_indices" and mask_key == "built_up_proxy":
        return (f"built-up proxy by NDVI < {step.params.get('ndvi_bare_threshold', 0.2):g} and NDWI < "
                f"{step.params.get('ndwi_water_threshold', 0.0):g} (heuristic; also includes bare soil)")
    return f"{step.params.get('target', 'target')} by VLM segmentation ({step.model})"


def _sar_source(step: StepResult | None) -> str:
    if step is None:
        return "SAR evidence"
    o = step.outputs
    unit = "dB" if str(o.get("units", "")).startswith(("dB", "linear")) else "(8-bit display units)"
    if step.tool == "sar.water_mask":
        return f"low backscatter below an adaptive Otsu threshold of {o.get('threshold_db', float('nan')):.1f} {unit} (heuristic)"
    share = 100 - float(step.params.get("percentile", 90))
    return (f"strong scatterers, the brightest {share:g}% of co-pol backscatter (a relative threshold, so it flags "
            f"about that share of any scene; a built-up proxy, not a building detector)")


def _cross_modal(intent, r, ctx, run_dir):
    """Optical evidence, SAR evidence and the fused conclusion, each attributed to what produced it."""
    fusion = r.get("fusion.cross_modal")
    if not fusion:
        return "", None, []
    optical, sar = (0, 1) if ctx.images[0].modality == "optical" else (1, 0)
    steps = {s.step_id: s for s in r.results}
    fused = {name: o for name, o in fusion.outputs.items() if o.get("available", True) and name in FUSED_CLASSES}
    if not fused:
        return "", None, []

    summary, optical_parts, sar_parts, joint_parts, caveats = [], [], [], [], []
    for name, o in fused.items():
        words = FUSED_CLASSES[name]
        places = _places(o["both_regions"])
        if o["both_percent"] > 0:
            summary.append(f"{words['noun']} are confirmed by both sensors over {_pct1(o['both_percent'])} of the analysed "
                           f"area, mainly in the {places}")
        elif o["optical_percent"] or o["sar_percent"]:
            summary.append(f"no {words['short']} is confirmed by both sensors (optical {_pct1(o['optical_percent'])}, "
                           f"SAR {_pct1(o['sar_percent'])})")
        else:
            summary.append(f"neither sensor detected {words['noun']}")
        optical_parts.append(f"{_optical_source(steps.get(o['optical_step']), o['optical_mask'])} covers "
                             f"{_pct1(o['optical_percent'])}")
        sar_parts.append(f"{_sar_source(steps.get(o['sar_step']))} {'cover' if name == 'built_up' else 'covers'} "
                         f"{_pct1(o['sar_percent'])}")
        if o["agreement_iou"] is None:
            joint_parts.append(f"{words['short']}: nothing to compare")
            continue
        joint_parts.append(f"{words['short']} agreement IoU {o['agreement_iou']:.2f} ({o['agreement']}); optical only "
                           f"{_pct1(o['optical_only_percent'])}, SAR only {_pct1(o['sar_only_percent'])}")
        if o["agreement"] != "high":
            caveats.append(f"Optical and SAR evidence show {o['agreement']} agreement for {words['short']} "
                           f"(IoU {o['agreement_iou']:.2f}), so the {words['short']} interpretation has lower confidence.")
    for name, o in fusion.outputs.items():
        if not o.get("available", True):
            caveats.append(f"{FUSED_CLASSES.get(name, {}).get('short', name).capitalize()} could not be fused: {o['reason']}.")
    dates = (ctx.images[optical].acquired, ctx.images[sar].acquired)
    if all(dates) and dates[0] != dates[1]:
        caveats.append(f"The optical image was acquired on {dates[0]} and the SAR image on {dates[1]}; anything "
                       "that changed in between appears as disagreement between the sensors.")
    if intent.target and intent.target not in ("water", "building"):
        caveats.append(f"The cross-modal tools analyse water and built-up surfaces only; '{intent.target}' was not analysed.")

    answer = "\n".join([
        "Using the optical and SAR images together, " + "; ".join(summary) + ".",
        f"Optical evidence ({ctx.images[optical].name}): " + "; ".join(optical_parts) + ".",
        f"SAR evidence ({ctx.images[sar].name}): " + "; ".join(sar_parts) + ".",
        "Cross-modal evidence: " + "; ".join(joint_parts) + ".",
        *caveats,
    ])

    art = ctx.artifacts[fusion.step_id].masks
    colored = lambda name: [(art.get(f"{name}_{part}"), color, alpha)
                            for part, color, alpha in FUSED_CLASSES[name]["colors"]]
    shape = (ctx.images[optical].height, ctx.images[optical].width)
    # One transparent layer per class, pinned to the pair's shared grid: the map shows it over either
    # input, and the colours say which sensor saw what.
    layers = [_overlay_evidence(ev.mask_layer(shape, colored(name)), run_dir, f"fused_{name}",
                                f"FUSED {FUSED_CLASSES[name]['short']}: {FUSED_CLASSES[name]['legend']}",
                                fusion.step_id, optical)
              for name in fused]
    all_masks = [(mask, color) for name in fused for mask, color, _ in colored(name)]
    composite = ev.side_by_side(ev.overlay(ctx.rgb(optical), masks=all_masks), ev.overlay(ctx.rgb(sar), masks=all_masks))
    label = ("optical (left) and SAR false colour (right) with the fused masks: "
             + "; ".join(f"{FUSED_CLASSES[name]['short']}: {FUSED_CLASSES[name]['legend']}" for name in fused))
    return answer, fusion.confidence, layers + [_overlay_evidence(composite, run_dir, "cross_modal", label, fusion.step_id)]


# --------------------------------------------------------------------------- weather (D-029, optional)

HEAVY_RAIN_MM = 64.5  # IMD's "heavy rainfall" threshold for 24 h; said wherever it is used


def _chance(probability: float | None) -> str:
    if probability is None:
        return "not forecast"
    return "likely" if probability >= 70 else "possible" if probability >= 30 else "unlikely"


def _day(iso: str, local_today: str) -> str:
    from datetime import date

    day = date.fromisoformat(iso)
    text = day.strftime("%a %d %b")
    return {0: f"today ({text})", 1: f"tomorrow ({text})"}.get((day - date.fromisoformat(local_today)).days, text)


def _first_upper(text: str) -> str:
    return text[:1].upper() + text[1:]


def _num(value, digits: int = 0, unit: str = "") -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{unit}"


def _coordinate(latitude: float, longitude: float) -> str:
    return (f"{abs(latitude):.3f}°{'N' if latitude >= 0 else 'S'}, "
            f"{abs(longitude):.3f}°{'E' if longitude >= 0 else 'W'}")


def _km_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (latitude, longitude) points, in km."""
    import math

    (lat1, lon1), (lat2, lon2) = (tuple(map(math.radians, p)) for p in (a, b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _rain_summary(days: list[dict], name, period: str, heavy: bool) -> str:
    wettest = max(days, key=lambda d: (d["precipitation_probability_max"] or 0, d["precipitation_sum"] or 0))
    facts = (f"{_num(wettest['precipitation_probability_max'])}% chance, about "
             f"{_num(wettest['precipitation_sum'], 1)} mm, {wettest['condition']}")
    if heavy:
        heavy_days = [d for d in days if d["weather_code"] in HEAVY_RAIN_CODES
                      or (d["precipitation_sum"] or 0) >= HEAVY_RAIN_MM]
        if heavy_days:
            verdict = f"Heavy rain is forecast {', '.join(name(d) for d in heavy_days)} ({facts})."
        elif len(days) == 1:
            verdict = (f"Heavy rain is not expected {period}: about {_num(wettest['precipitation_sum'], 1)} mm is "
                       f"forecast ({_num(wettest['precipitation_probability_max'])}% chance of rain, {wettest['condition']}).")
        else:
            verdict = f"Heavy rain is not expected {period}: the wettest day is {name(wettest)} ({facts})."
        return verdict + (f" (Heavy means the provider's heavy or violent rain code, or at least {HEAVY_RAIN_MM} mm "
                          "in a day, IMD's 'heavy rainfall' threshold.)")
    if len(days) == 1:
        return f"Rain is {_chance(wettest['precipitation_probability_max'])} {name(wettest)}: {facts}."
    likely = [d for d in days if (d["precipitation_probability_max"] or 0) >= 70]
    top = wettest["precipitation_probability_max"] or 0
    if top < 30:
        return f"Rain is unlikely {period}: the highest chance is {_num(top)}% on {name(wettest)}."
    return (f"Rain is {_chance(top)} {period}, most likely {name(wettest)} ({facts})"
            + (f"; rain is likely on {len(likely)} of the {len(days)} days." if likely else "."))


def _temperature_summary(days: list[dict], name, period: str) -> str:
    if len(days) == 1:
        d = days[0]
        return (f"{_first_upper(name(d))}: {_num(d['temperature_min'])}–{_num(d['temperature_max'])} °C, "
                f"{d['condition']}.")
    known = [d for d in days if d["temperature_max"] is not None and d["temperature_min"] is not None]
    if not known:
        return f"No temperatures were forecast {period}."
    warmest = max(known, key=lambda d: d["temperature_max"])
    coolest = min(known, key=lambda d: d["temperature_min"])
    return (f"Temperatures {period} range from {_num(coolest['temperature_min'])} °C to "
            f"{_num(warmest['temperature_max'])} °C: warmest {name(warmest)} (max {_num(warmest['temperature_max'], 1)} °C), "
            f"coolest night {name(coolest)} (min {_num(coolest['temperature_min'], 1)} °C).")


def _weather(intent, r, ctx, run_dir):
    """Short-range forecast answer: the focus of the question first, then one line per day."""
    step = r.get("weather.forecast")
    if not step:
        return "", None, []
    o = step.outputs
    days, today = o["days"], o["local_today"]
    wind_unit = o["units"].get("wind_speed_10m_max", "km/h")
    name = lambda d: _day(d["date"], today)
    first, last = name(days[0]), name(days[-1])
    period = (first if len(days) == 1 else
              f"over the next {len(days)} days ({days[0]['date']} to {days[-1]['date']})" if days[0]["date"] == today
              else f"from {first} to {last}")

    focus = intent.target
    if focus == "temperature":
        summary = _temperature_summary(days, name, period)
    elif focus == "wind":
        windiest = max(days, key=lambda d: d["wind_speed_max"] or 0)
        summary = (f"The strongest wind {period} is forecast {name(windiest)}: up to "
                   f"{_num(windiest['wind_speed_max'])} {wind_unit} ({windiest['condition']}).")
    elif focus in ("rain", "heavy rain"):
        summary = _rain_summary(days, name, period, heavy=focus == "heavy rain")
    else:
        summary = _temperature_summary(days, name, period) + " " + _rain_summary(days, name, period, heavy=False)

    lines = [summary]
    now = o.get("current")
    if now:
        lines.append(f"Now ({now['time'][11:16]} local): {_num(now['temperature'], 1)} °C, {now['condition']}, "
                     f"humidity {_num(now['relative_humidity'])}%, wind {_num(now['wind_speed'])} {wind_unit}.")
    for d in days:
        lines.append(f"{_first_upper(name(d))}: {_num(d['temperature_min'])}–{_num(d['temperature_max'])} °C · rain "
                     f"{_num(d['precipitation_probability_max'])}% ({_num(d['precipitation_sum'], 1)} mm) · "
                     f"{d['condition']}" + (f" · wind {_num(d['wind_speed_max'])} {wind_unit}" if focus == "wind" else ""))
    # Weather models forecast grid cells several km across: say which cell answered, and how far away.
    asked = (step.params["latitude"], step.params["longitude"])
    lines.append(f"Forecast for {_coordinate(*asked)} (marked on the map); the provider answered from its nearest "
                 f"model grid point, {_coordinate(o['latitude'], o['longitude'])}, about "
                 f"{_km_between(asked, (o['latitude'], o['longitude'])):.1f} km away (elevation {_num(o['elevation_m'])} m). "
                 f"Days are local to {o['timezone']}.")
    retrieved = o["retrieved_at"][:16].replace("T", " ")
    lines.append(f"Source: {o['attribution']}, {o['model'].split(' ')[0]} models; retrieved {retrieved} UTC"
                 + (" (cached)." if o["cached"] else "."))
    return "\n".join(lines), step.confidence, []
