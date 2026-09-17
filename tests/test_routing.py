"""Golden routing cases, starting with the representative queries in the SIH problem statement."""

import pytest

from satquery.agent.intents import classify

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
