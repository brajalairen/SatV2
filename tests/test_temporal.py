"""Date resolution for temporal questions: which two periods are searched (satquery/providers/temporal.py).

A fixed "today" keeps every expectation exact. Windows are search periods only; the dates a user
sees are the acquisition dates Copernicus returns.
"""

from datetime import date

import pytest

from satquery.agent.intents import needs_multiple_dates
from satquery.providers.errors import TemporalRangeUnsupported
from satquery.providers.temporal import resolve_windows

TODAY = date(2026, 9, 27)


def windows(query):
    w = resolve_windows(query, TODAY)
    return (w.before.start, w.before.end), (w.after.start, w.after.end), w.basis


# ------------------------------------------------------------------ routing: temporal or single-date

@pytest.mark.parametrize("query", [
    "What changed here?",
    "What has changed in this area?",
    "What changed between June and September?",
    "Compare this area between June 2026 and September 2026.",
    "How has this area changed over the last 3 months?",
])
def test_the_required_temporal_questions_route_to_the_temporal_workflow(query):
    assert needs_multiple_dates(query), query
    resolve_windows(query, TODAY)  # and each one resolves to two search windows


@pytest.mark.parametrize("query", ["Are there water bodies here?", "Describe this area.", "change the map"])
def test_ordinary_questions_stay_single_date(query):
    assert needs_multiple_dates(query) is None


# ------------------------------------------------------------------ explicit dates

def test_two_named_months_search_each_month():
    before, after, basis = windows("What changed between June and September?")
    assert before == (date(2026, 6, 1), date(2026, 6, 30))
    assert after == (date(2026, 9, 1), TODAY)  # September has not ended: the window stops today
    assert basis == "two explicit periods"


def test_months_with_years_are_used_as_given():
    before, after, _ = windows("Compare this area between June 2026 and September 2026.")
    assert before == (date(2026, 6, 1), date(2026, 6, 30)) and after[0] == date(2026, 9, 1)


def test_specific_days_search_a_week_either_side():
    before, after, _ = windows("What changed between June 1 and September 1?")
    assert before == (date(2026, 5, 25), date(2026, 6, 8))
    assert after == (date(2026, 8, 25), date(2026, 9, 8))


def test_the_earlier_period_is_always_before_whatever_order_it_is_named_in():
    before, after, _ = windows("September 2026 vs June 2026")
    assert before[0] == date(2026, 6, 1) and after[0] == date(2026, 9, 1)


def test_a_month_range_across_new_year_takes_the_first_month_from_the_year_before():
    before, after, _ = windows("What changed between November and February?")
    assert before == (date(2025, 11, 1), date(2025, 11, 30))
    assert after == (date(2026, 2, 1), date(2026, 2, 28))


def test_two_years_compare_the_same_season():
    before, after, _ = windows("What changed between 2019 and 2024?")
    assert before == (date(2019, 8, 28), date(2019, 10, 27))
    assert after == (date(2024, 8, 28), date(2024, 10, 27))


def test_the_modal_verb_may_is_not_the_month_may():
    _, _, basis = windows("What may have changed here?")
    assert basis == "default"


def test_since_a_month_compares_it_with_the_last_30_days():
    before, after, basis = windows("What changed since June?")
    assert before == (date(2026, 6, 1), date(2026, 6, 30))
    assert after == (date(2026, 8, 29), TODAY)
    assert basis == "one explicit period"


def test_a_recent_single_period_is_split_rather_than_compared_with_itself():
    before, after, _ = windows("What changed in September?")
    assert before[0] == date(2026, 9, 1) and after[1] == TODAY
    assert before[1] < after[0]


# ------------------------------------------------------------------ relative periods

def test_last_three_months_searches_its_first_and_last_third():
    before, after, basis = windows("How has this area changed over the last 3 months?")
    assert basis == "relative period"
    assert before == (date(2026, 6, 30), date(2026, 7, 29))
    assert after == (date(2026, 8, 29), TODAY)
    assert (after[0] - before[1]).days >= 30  # meaningfully apart, not adjacent passes


def test_last_year_caps_each_window_at_60_days():
    before, after, _ = windows("What changed over the last year?")
    assert (before[1] - before[0]).days + 1 == 60 and (after[1] - after[0]).days + 1 == 60
    assert before[0] == date(2025, 9, 28)


# ------------------------------------------------------------------ no dates at all

def test_no_dates_compares_a_year_ago_with_the_last_60_days():
    before, after, basis = windows("What changed here?")
    assert basis == "default"
    assert before == (date(2025, 9, 28), date(2025, 11, 26))
    assert after == (date(2026, 7, 30), TODAY)
    # roughly the same season a year apart, never two adjacent passes
    assert (after[0] - before[1]).days > 200


# ------------------------------------------------------------------ unsupported ranges: actionable refusals

@pytest.mark.parametrize("query, fragment", [
    ("What changed between June and December 2026?", "has not begun"),
    ("What changed since 2012?", "before Sentinel-2"),
    ("What changed over the last week?", "too short"),
    ("What changed over the last 20 years?", "before Sentinel-2"),
])
def test_unsupported_ranges_are_refused_with_a_reason(query, fragment):
    with pytest.raises(TemporalRangeUnsupported) as caught:
        resolve_windows(query, TODAY)
    assert fragment in caught.value.message
    assert caught.value.status == 422


def test_an_impossible_day_is_refused_not_guessed():
    with pytest.raises(TemporalRangeUnsupported):
        resolve_windows("What changed between February 30 and June 1?", TODAY)
