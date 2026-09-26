"""Golden routing cases, starting with the representative queries in the SIH problem statement."""

import pytest

from satquery.agent.intents import COMPATIBLE_TASKS, classify, needs_multiple_dates

CASES = [
    # (query, input configuration, expected task, expected target, comparative)
    ("Describe the land-cover and major objects visible in this image.", "single_optical", "caption", None, False),
    ("Highlight the water body referred to in the query.", "single_optical", "grounding", "water", False),
    ("What changed between these two dates, and where did the change occur?", "pair_bitemporal", "change_analysis", None, False),
    ("Use the optical and SAR images together to identify built-up and water-covered regions.", "pair_cross_modal",
     "cross_modal_analysis", "water", False),
    ("Has the built-up area increased, decreased, or remained unchanged?", "pair_bitemporal", "change_analysis", "building", True),
    ("Where are the airplanes located and what is their type?", "single_optical", "grounding", "airplane", False),
    ("How many storage tanks are there?", "single_optical", "vqa", "storage tank", False),
    ("Is there a river in this image?", "single_optical", "vqa", "water", False),
    ("Show me the forest.", "single_sar", "grounding", "vegetation", False),
    ("Give me an overview of this radar scene.", "single_sar", "caption", None, False),
    ("Did the lake shrink?", "pair_bitemporal", "change_analysis", "water", True),
    ("Point out the area near the coast with ships.", "single_optical", "grounding", "ship", False),
]


@pytest.mark.parametrize("query, config, task, target, comparative", CASES)
def test_routing(query, config, task, target, comparative):
    intent = classify(query, config)
    assert intent.task == task
    assert intent.matched_rule
    if task != "cross_modal_analysis":
        assert intent.target == target
    assert intent.comparative == comparative


# --------------------------------------------------------------- temporal detection
# A single retrieved scene can never produce change_analysis (COMPATIBLE_TASKS allows it only for
# pair_bitemporal), so a query needing two dates must be refused rather than answered from one image.

@pytest.mark.parametrize("query", [
    "What has changed here?",
    "What changed in this area?",
    "Has this area changed?",
    "Compare this area over time",
    "What is different from before?",
    "Has vegetation increased?",
    "Has the water area decreased?",
    # further phrasings that equally need two dates
    "how much has the forest changed",
    "any changes in this region?",
    "compare before and after",
    "urban expansion here?",
    "has the lake shrunk",
    "deforestation since 2020",
    "what is the difference between 2019 and 2024",
    "has the built-up area expanded",
    "show me the historical imagery",
    "bi-temporal analysis please",
])
def test_temporal_queries_are_detected(query):
    assert needs_multiple_dates(query) is not None, f"should need two dates: {query!r}"


@pytest.mark.parametrize("query", [
    # UI commands, not questions about imagery
    "change the map",
    "change the basemap style",
    "Change the opacity",
    "change the layer opacity to 50%",
    "expand the sidebar",
    "expand the details panel",
    # 'different' used attributively, with no comparison across time
    "Is there a different kind of crop here?",
    "Show me a different view",
    "are there different types of buildings",
    # present-tense growth is vegetation, not a change over time
    "what crops are growing here",
    "is this a growing city",
    # ordinary single-image questions
    "What is in this area?",
    "Describe this image",
    "Are there water bodies here?",
    "Where are the buildings?",
    "What is the vegetation condition here?",
    "identify the airport",
    "how many ships are in the harbor",
])
def test_ordinary_queries_are_not_treated_as_temporal(query):
    assert needs_multiple_dates(query) is None, f"should NOT need two dates: {query!r}"


def test_the_matched_phrase_is_returned_so_a_refusal_can_quote_it():
    assert needs_multiple_dates("What has changed here?") == "changed"
    assert needs_multiple_dates("Has vegetation increased?") == "increased"


def test_a_single_image_can_never_be_routed_to_change_analysis():
    """The structural guarantee the refusal rests on, not just the regex."""
    for config in ("single_optical", "single_sar"):
        assert "change_analysis" not in COMPATIBLE_TASKS[config]
        assert classify("What has changed here?", config).task != "change_analysis"
