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

# Does the query need more than one date to answer? Used to refuse single-date imagery requests
# honestly instead of answering them from one image: a single scene can never produce
# change_analysis, which COMPATIBLE_TASKS allows only for pair_bitemporal.
#
# Deliberately narrower than "mentions a change word", because three ordinary cases must stay out:
#   - imperatives are UI commands, not questions ("change the map", "expand the sidebar"), so
#     `change`/`expand` directly followed by a determiner does not count;
#   - `compare`/`different` need a comparison phrase ("over time", "from", "between"), so
#     "a different kind of crop" and "show me a different view" do not count;
#   - present-tense "growing" describes vegetation ("what crops are growing here"), so only the
#     past forms `grew`/`grown` count, while `expansion`/`expanded` do imply two moments.
TEMPORAL_CUES = re.compile(
    r"\b(?:"
    r"chang(?:e[ds]?|ing)\b(?!\s+(?:the|this|that|my|a|an|to|it)\b)"
    r"|unchanged|changes\b"
    r"|compar(?:e|ed|ing|ison)\b\s*(?:.*\b(?:over time|to|with|against|between)\b|$)"
    r"|differen(?:ce|ces|t)\b\s*(?:.*\b(?:from|to|between|since|over time)\b)"
    r"|\b(?:before|after)\b\s*(?:and|/|vs|versus|\?|$)"
    r"|\bover time\b|\bsince\b\s+\d{4}|\bbetween\b\s+\d{4}"
    r"|\b(?:increas|decreas|shrink|shrunk|reduc)\w*\b"
    r"|expan(?:ded|ding|sion)\b|\bexpand\b(?!\s+(?:the|this|that|my|a|an|it)\b)"
    r"|\b(?:grew|grown)\b"
    r"|\bbi-?temporal\b|\btime series\b|\bhistorical\b|\bdeforestation\b"
    r")", re.I)


# Does the query ask for optical and SAR evidence together? Used to refuse, rather than silently
# re-route, a joint question that arrives without both modalities: one optical image would otherwise
# be answered by single-image VQA, and two optical images by bi-temporal change analysis.
# A query counts when it names both a SAR term and an optical term, or asks for fusion outright.
# One sensor alone ("describe this radar scene", "the spectral signature") does not count.
SAR_TERMS = re.compile(r"\b(?:sar|radar|backscatter\w*|sentinel-?1|risat|microwave)\b", re.I)
OPTICAL_TERMS = re.compile(r"\b(?:optical|multi-?spectral|sentinel-?2|cartosat|spectral|ndwi|ndvi)\b", re.I)
JOINT_TERMS = re.compile(r"\b(?:cross[- ]?modal\w*|multi[- ]?modal\w*|multi[- ]?sensor|(?:both|two) sensors|"
                         r"sensor fusion|data fusion|fus(?:e|ed|ing|ion))\b", re.I)

# The classes the cross-modal tools can analyse, in the order they are reported.
CROSS_MODAL_CLASSES = ("water", "building")


def needs_optical_and_sar(query: str) -> str | None:
    """The phrase that asks for joint optical + SAR analysis, or None. Quoted back in refusals."""
    joint = JOINT_TERMS.search(query)
    if joint:
        return joint.group(0)
    sar, optical = SAR_TERMS.search(query), OPTICAL_TERMS.search(query)
    if sar and optical:
        first, last = sorted((sar, optical), key=lambda match: match.start())
        return query[first.start():last.end()]
    return None


def needs_sar_only(query: str) -> str | None:
    """The radar phrase of a question that asks for SAR alone ("use radar to find the water"), or None.

    Naming optical too, or asking for fusion, is the joint optical + SAR path instead (D-030).
    """
    if needs_optical_and_sar(query):
        return None
    sar = SAR_TERMS.search(query)
    return sar.group(0) if sar else None


def find_area_targets(query: str) -> list[str]:
    """Every area class the query names, in AREA_TARGETS order ("built-up and water" -> both)."""
    text = query.lower()
    return [canonical for canonical, words in AREA_TARGETS.items()
            if any(re.search(rf"\b{re.escape(w)}", text) for w in words)]


def cross_modal_classes(query: str) -> list[str]:
    """Which classes a cross-modal query asks about; both when it names neither."""
    named = find_area_targets(query)
    return [c for c in CROSS_MODAL_CLASSES if c in named] or list(CROSS_MODAL_CLASSES)


# Is it a weather question? Weather is an optional capability answered by its own specialist
# (D-029), never by imagery, so it is decided before any imagery is retrieved.
# Unambiguous weather words count on their own ("rainforest" does not: \brain needs a boundary).
# Words that also describe what imagery shows ("snow on the peaks", "cloud cover in this scene",
# "wind turbines") count only with a forecast or time cue.
WEATHER_TERMS = re.compile(r"\b(?:weather|forecasts?|rain(?:s|y|ing|fall|falls|ed)?|precipitation|drizzl\w*|"
                           r"thunder\w*|humid(?:ity)?|temperatures?|heat ?waves?|downpours?|monsoons?)\b", re.I)
AMBIGUOUS_WEATHER_TERMS = re.compile(r"\b(?:snow\w*|cloud(?:s|y)?|fog(?:gy)?|storm(?:s|y)?|wind(?:s|y)?|sunny|"
                                     r"sunshine|hot|cold|warm|chilly|freez\w*|frost\w*|hail\w*|cyclones?)\b", re.I)
FORECAST_CUES = re.compile(r"\b(?:will|going to|gonna|expected|forecast\w*|tomorrow|tonight|today|now|later|"
                           r"this (?:week|weekend|morning|afternoon|evening)|next \w+|upcoming|coming days)\b", re.I)
# Satellite-analysis wording. Together with a weather cue the question asks two specialists at once,
# which is not supported yet, so it is refused with a request to ask separately.
IMAGERY_CUES = re.compile(r"\b(?:images?|imagery|scenes?|satellite|sentinel\S*|sar|radar|optical|multi-?spectral|"
                          r"ndvi|ndwi|land[- ]?cover|built[- ]?up|buildings?|water bod(?:y|ies)|vegetation|"
                          r"what (?:has )?changed|any changes|change detection|highlight|segment\w*)\b", re.I)


def needs_weather(query: str) -> str | None:
    """The phrase that makes `query` a weather question, or None."""
    strong = WEATHER_TERMS.search(query)
    if strong:
        return strong.group(0)
    ambiguous = AMBIGUOUS_WEATHER_TERMS.search(query)
    return ambiguous.group(0) if ambiguous and FORECAST_CUES.search(query) else None


def route_query(query: str) -> tuple[str, str]:
    """("weather" | "imagery" | "mixed", the rule that decided). Wording only: no I/O, no planning."""
    weather = needs_weather(query)
    if not weather:
        return "imagery", "no weather cue -> satellite analysis"
    imagery = IMAGERY_CUES.search(query)
    if imagery:
        return "mixed", f"weather cue '{weather}' and satellite-analysis cue '{imagery.group(0)}'"
    return "weather", f"weather cue '{weather}' -> weather specialist"


def needs_multiple_dates(query: str) -> str | None:
    """The temporal phrase that makes `query` unanswerable from a single image, or None.

    Returns the matched text so a refusal can quote the user's own words back rather than
    being generic.
    """
    match = TEMPORAL_CUES.search(query)
    return match.group(0).strip() if match else None


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
        cue = needs_optical_and_sar(query)
        rule = "input configuration pair_cross_modal -> cross_modal_analysis"
        rule += (f"; cross-modal cue '{cue}'" if cue else "") + f"; classes {', '.join(cross_modal_classes(query))}"
        return Intent(task="cross_modal_analysis", target=target, matched_rule=rule)

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
    "area_only": {"weather_forecast"},
}
