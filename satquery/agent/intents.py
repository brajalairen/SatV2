"""Rule-based intent classification (D-006). Every intent records the rule that produced it."""

import re

from satquery.schemas import InputConfig, Intent

# Canonical target names (the vocabulary passed to the VLM) and the query words that map to them.
AREA_TARGETS = {
    "water": ("water", "river", "lake", "pond", "flood", "reservoir", "sea", "coast", "wetland", "canal"),
    "building": ("built-up", "built up", "building", "urban", "settlement", "residential", "house", "city", "town"),
    "vegetation": ("vegetation", "forest", "tree", "crop", "farmland", "agricultur", "grass"),
    "road": ("road", "highway", "street"),
}
OBJECT_TARGETS = ("airplane", "aircraft", "ship", "boat", "vehicle", "car", "storage tank", "bridge", "stadium", "harbor",
                  "tennis court", "baseball field", "basketball court", "windmill", "chimney", "dam", "airport",
                  "train station", "overpass")

GROUNDING_CUES = re.compile(r"\b(highlight|locate|where|find|show|mark|detect|segment|outline|delineate|point out)\b", re.I)
CAPTION_CUES = re.compile(r"\b(describe|description|caption|summari[sz]e|overview|land[- ]?cover)\b", re.I)
COMPARATIVE_CUES = re.compile(r"\b(increas\w*|decreas\w*|more|less|grow\w*|grew|expand\w*|shrink\w*|shrunk|reduc\w*|"
                              r"remain\w*|unchanged)\b", re.I)


def find_target(query: str) -> tuple[str | None, bool]:
    """Return (canonical target, is_area_class). Specific objects win over area classes."""
    text = query.lower()
    for obj in OBJECT_TARGETS:
        if re.search(rf"\b{re.escape(obj)}s?\b", text):
            return obj, False
    for canonical, words in AREA_TARGETS.items():
        if any(re.search(rf"\b{re.escape(w)}", text) for w in words):
            return canonical, True
    return None, False


def classify(query: str, config: InputConfig) -> Intent:
    target, _ = find_target(query)
    if config == "pair_bitemporal":
        comparative = bool(target and COMPARATIVE_CUES.search(query))
        rule = "input configuration pair_bitemporal -> change_analysis"
        if target:
            rule += f"; target '{target}'" + ("; comparative cue" if comparative else "")
        return Intent(task="change_analysis", target=target, comparative=comparative, matched_rule=rule)
    if config == "pair_cross_modal":
        return Intent(task="cross_modal_analysis", target=target,
                      matched_rule="input configuration pair_cross_modal -> cross_modal_analysis")

    grounding = GROUNDING_CUES.search(query)
    caption = CAPTION_CUES.search(query)
    if grounding and target:
        return Intent(task="grounding", target=target, matched_rule=f"grounding cue '{grounding.group(0)}' + target '{target}'")
    if caption:
        return Intent(task="caption", matched_rule=f"caption cue '{caption.group(0)}'")
    if grounding:
        return Intent(task="grounding", matched_rule=f"grounding cue '{grounding.group(0)}' with free-form description")
    return Intent(task="vqa", target=target, matched_rule="no grounding/caption cue -> visual question answering")


COMPATIBLE_TASKS = {
    "single_optical": {"vqa", "caption", "grounding"},
    "single_sar": {"vqa", "caption", "grounding"},
    "pair_bitemporal": {"change_analysis"},
    "pair_cross_modal": {"cross_modal_analysis"},
}
