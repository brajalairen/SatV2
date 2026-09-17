"""Plan templates keyed by (input configuration, intent). A plan is data, so it appears verbatim in the trace."""

from satquery.imaging import RasterImage
from satquery.schemas import InputConfig, Intent, PlanStep


class PlanBuilder:
    def __init__(self):
        self.steps: list[PlanStep] = []

    def add(self, tool: str, images: list[int], purpose: str, **params) -> str:
        step_id = f"s{len(self.steps) + 1}"
        self.steps.append(PlanStep(step_id=step_id, tool=tool, image_indices=images, params=params, purpose=purpose))
        return step_id


def build_plan(intent: Intent, config: InputConfig, images: list[RasterImage], query: str) -> list[PlanStep]:
    plan = PlanBuilder()
    if config == "single_optical":
        _single_optical(plan, intent, images[0], query)
    elif config == "single_sar":
        _single_sar(plan, intent, query)
    elif config == "pair_bitemporal":
        _bitemporal(plan, intent, images)
    elif config == "pair_cross_modal":
        _cross_modal(plan, images)
    return plan.steps


def _has_nir(image: RasterImage) -> bool:
    return image.band("nir") is not None


def _single_optical(plan, intent, image, query):
    if intent.task == "caption":
        plan.add("vlm.caption", [0], "describe the scene")
        if _has_nir(image):
            plan.add("optical.spectral_indices", [0], "quantify vegetation and water from NIR bands")
    elif intent.task == "vqa":
        plan.add("vlm.vqa", [0], "answer the question", question=query)
    elif intent.target in ("water", "building", "vegetation", "road"):
        plan.add("vlm.segment", [0], f"segment {intent.target}", target=intent.target)
        if intent.target in ("water", "vegetation") and _has_nir(image):
            plan.add("optical.spectral_indices", [0], "cross-check with spectral indices")
    elif intent.target:
        plan.add("vlm.detect", [0], f"detect {intent.target}", target=intent.target)
    else:
        plan.add("vlm.ground", [0], "ground the described region", description=query)


def _single_sar(plan, intent, query):
    plan.add("sar.backscatter_stats", [0], "summarise backscatter")
    if intent.task == "grounding" and intent.target == "water":
        plan.add("sar.water_mask", [0], "delineate low-backscatter water")
        return
    if intent.task == "grounding" and intent.target == "building":
        plan.add("sar.bright_mask", [0], "delineate strong scatterers (candidate built-up)")
        return
    plan.add("sar.water_mask", [0], "context: low-backscatter (water-like) areas")
    if intent.task == "caption":
        plan.add("vlm.caption", [0], "describe the false-colour SAR rendering")
    elif intent.task == "vqa":
        plan.add("vlm.vqa", [0], "answer on the false-colour SAR rendering", question=query)
    else:
        plan.add("vlm.ground", [0], "ground the described region", description=query)


def _bitemporal(plan, intent, images):
    plan.add("vlm.change", [0, 1], "VLM change detection (image 1 = before, image 2 = after)")
    plan.add("change.map", [0, 1], "deterministic change map to corroborate the VLM")
    if not intent.target:
        return
    spectral_masks = {"building": "built_up_proxy", "water": "water", "vegetation": "vegetation"}
    mask_key = "mask"
    if images[0].modality == "sar" and intent.target in ("water", "building"):
        tool = "sar.water_mask" if intent.target == "water" else "sar.bright_mask"
        before = plan.add(tool, [0], f"{intent.target} extent before")
        after = plan.add(tool, [1], f"{intent.target} extent after")
    elif intent.target in spectral_masks and all(_has_nir(image) for image in images):
        # Multispectral input: spectral indices measure land-cover extent more reliably than VLM segmentation,
        # which was trained on sub-metre imagery and misses buildings at ~10 m.
        mask_key = spectral_masks[intent.target]
        before = plan.add("optical.spectral_indices", [0], f"{intent.target} extent before (spectral, multispectral input)")
        after = plan.add("optical.spectral_indices", [1], f"{intent.target} extent after (spectral, multispectral input)")
    else:
        before = plan.add("vlm.segment", [0], f"{intent.target} extent before", target=intent.target)
        after = plan.add("vlm.segment", [1], f"{intent.target} extent after", target=intent.target)
    plan.add("change.compare_areas", [0, 1], f"did {intent.target} increase or decrease?",
             target=intent.target, before_step=before, after_step=after, mask_key=mask_key)


def _cross_modal(plan, images):
    optical = 0 if images[0].modality == "optical" else 1
    sar = 1 - optical
    sar_water = plan.add("sar.water_mask", [sar], "water from SAR (dark, specular)")
    sar_bright = plan.add("sar.bright_mask", [sar], "built-up candidates from SAR (strong scatterers)")
    if _has_nir(images[optical]):
        optical_water = plan.add("optical.spectral_indices", [optical], "water from optical NDWI")
    else:
        optical_water = plan.add("vlm.segment", [optical], "water from optical imagery", target="water")
    optical_building = plan.add("vlm.segment", [optical], "buildings from optical imagery", target="building")
    plan.add("fusion.cross_modal", [optical, sar], "combine complementary optical and SAR evidence",
             sar_water_step=sar_water, sar_bright_step=sar_bright,
             optical_water_step=optical_water, optical_building_step=optical_building)
