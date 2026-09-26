"""Turn a temporal question into two search windows: one for the earlier scene, one for the later.

These are only the periods searched. No date here is ever shown as an acquisition date: provenance
always carries the acquisition dates Copernicus returns for the scenes actually found in them.

Policy
------
Two explicit periods ("between June and September", "June 2026 vs September 2026",
"between June 1 and September 1", "between 2019 and 2024"): one window per period, and the earlier
period is always "before", whichever order the question names them in.
    month          -> that calendar month
    month and day  -> that day, +- 7 days (a single day rarely has a clear Sentinel-2 pass)
    year alone     -> the same season in that year (today's day of the year, +- 30 days), so seasonal
                      cycles such as greening or snow are not mistaken for change
A month without a year is its most recent occurrence that has already begun; if the earlier-named
month would then fall after the later-named one, it is taken from the year before.

One explicit period ("since June", "since 2020", "what changed in March"): that period against the
last 30 days. If the period is itself recent enough to overlap the last 30 days, it is split instead:
its first third against its last third.

A relative period ("over the last 3 months", "in the past year"): the period ending today. The
earlier scene is searched in its first third, the later in its last third, each third capped at
60 days and at least 5, so the two scenes are at least a third of the period apart.

No dates at all ("What changed here?"): the last 12 months, split the same way. That compares a scene
from about a year ago with one from the last 60 days: roughly the same season a year apart, rather
than two adjacent passes a few days apart.

Limits: every window must have begun by today, and must end after 2015-07-01, when Sentinel-2A's
first data became available. A relative period must be at least 10 days.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

from satquery.providers.errors import TemporalRangeUnsupported

SENTINEL2_START = date(2015, 7, 1)
MIN_RELATIVE_DAYS = 10
MAX_THIRD_DAYS = 60
MIN_THIRD_DAYS = 5
DEFAULT_SPAN_DAYS = 365
RECENT_DAYS = 30
DAY_MARGIN = 7
SEASON_MARGIN = 30

MONTHS = {name: number for number, names in enumerate([
    (), ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",),
    ("jun", "june"), ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"),
    ("oct", "october"), ("nov", "november"), ("dec", "december")]) for name in names}
_MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_YEAR = r"((?:19|20)\d{2})"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
# "1 June 2026" / "June 1, 2026" / "June 2026" / "June"
_DAY_MONTH = re.compile(rf"\b{_DAY}\s+{_MONTH}\b(?:,?\s+{_YEAR}\b)?", re.I)
_MONTH_DAY = re.compile(rf"\b{_MONTH}\b(?:\s+{_DAY}\b(?!\d))?(?:,?\s+{_YEAR}\b)?", re.I)
_BARE_YEAR = re.compile(rf"\b{_YEAR}\b")
_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_RELATIVE = re.compile(r"\b(?:last|past|previous|preceding)\s+(?:(\d+|a|an|one|two|three|four|five|six|seven|eight"
                       r"|nine|ten|eleven|twelve)\s+)?(day|week|month|year)s?\b", re.I)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
# "may" is a month only next to a date or after a word that introduces one: "what may have changed" is not May.
_MAY_CONTEXT = re.compile(r"\b(?:in|since|from|between|and|to|until|till|vs|versus|of|early|late|mid|before|after)\s+$", re.I)


@dataclass(frozen=True)
class Window:
    start: date
    end: date
    label: str

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass(frozen=True)
class TemporalWindows:
    before: Window
    after: Window
    basis: str        # "two explicit periods" | "one explicit period" | "relative period" | "default"
    explanation: str  # one sentence for provenance and the trace


@dataclass(frozen=True)
class _Period:
    """A period the question names, before it is turned into a search window."""

    position: int
    month: int | None
    day: int | None
    year: int | None
    text: str


def _clip(start: date, end: date, today: date) -> tuple[date, date]:
    return start, min(end, today)


def _month_window(month: int, year: int, today: date) -> tuple[date, date]:
    return _clip(date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]), today)


def _find_periods(query: str) -> list[_Period]:
    periods: list[_Period] = []
    taken: list[tuple[int, int]] = []

    def free(span: tuple[int, int]) -> bool:
        return all(span[1] <= a or span[0] >= b for a, b in taken)

    for match in _DAY_MONTH.finditer(query):
        periods.append(_Period(match.start(), MONTHS[match.group(2).lower()], int(match.group(1)),
                               int(match.group(3)) if match.group(3) else None, match.group(0)))
        taken.append(match.span())
    for match in _MONTH_DAY.finditer(query):
        if not free(match.span()):
            continue
        word = match.group(1).lower()
        if word == "may" and not (match.group(2) or match.group(3) or _MAY_CONTEXT.search(query[:match.start()])):
            continue
        periods.append(_Period(match.start(), MONTHS[word], int(match.group(2)) if match.group(2) else None,
                               int(match.group(3)) if match.group(3) else None, match.group(0)))
        taken.append(match.span())
    for match in _BARE_YEAR.finditer(query):
        if free(match.span()):
            periods.append(_Period(match.start(), None, None, int(match.group(1)), match.group(0)))
            taken.append(match.span())
    for period in periods:
        if period.day is not None and not 1 <= period.day <= 31:
            raise TemporalRangeUnsupported(f'"{period.text.strip()}" is not a valid date.')
    return sorted(periods, key=lambda p: p.position)


def _latest_year_for(month: int, today: date) -> int:
    """The most recent year in which `month` has already begun."""
    return today.year if month <= today.month else today.year - 1


def _period_window(period: _Period, year: int, today: date) -> Window:
    if period.month is None:  # a bare year: the same season in that year
        anchor = _safe_date(year, today.month, today.day)
        start, end = anchor - timedelta(days=SEASON_MARGIN), anchor + timedelta(days=SEASON_MARGIN)
        start, end = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        start, end = _clip(start, end, today)
        return Window(start, end, f"{year} (same season: {start:%d %b} to {end:%d %b})")
    if period.day is not None:
        try:
            anchor = date(year, period.month, period.day)
        except ValueError as error:
            raise TemporalRangeUnsupported(f'"{period.text.strip()}" is not a valid date.') from error
        start, end = _clip(anchor - timedelta(days=DAY_MARGIN), anchor + timedelta(days=DAY_MARGIN), today)
        return Window(start, end, f"{anchor:%d %B %Y} (+- {DAY_MARGIN} days)")
    start, end = _month_window(period.month, year, today)
    return Window(start, end, f"{calendar.month_name[period.month]} {year}")


def _safe_date(year: int, month: int, day: int) -> date:
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def _thirds(start: date, end: date) -> tuple[Window, Window]:
    span = (end - start).days + 1
    third = max(MIN_THIRD_DAYS, min(MAX_THIRD_DAYS, span // 3))
    before = Window(start, start + timedelta(days=third - 1), "")
    after = Window(end - timedelta(days=third - 1), end, "")
    before = Window(before.start, before.end, f"{before.start:%Y-%m-%d} to {before.end:%Y-%m-%d}")
    after = Window(after.start, after.end, f"{after.start:%Y-%m-%d} to {after.end:%Y-%m-%d}")
    return before, after


def _recent(today: date) -> Window:
    start = today - timedelta(days=RECENT_DAYS - 1)
    return Window(start, today, f"the last {RECENT_DAYS} days")


def _check(windows: TemporalWindows, today: date) -> TemporalWindows:
    for role, window in (("earlier", windows.before), ("later", windows.after)):
        if window.start > today:
            raise TemporalRangeUnsupported(
                f"The {role} period, {window.label}, has not begun yet. Ask about dates up to today ({today:%Y-%m-%d}).")
        if window.end < SENTINEL2_START:
            raise TemporalRangeUnsupported(
                f"The {role} period, {window.label}, is before Sentinel-2 imagery exists: the first scenes date from "
                f"mid-2015. Ask about a period from mid-2015 onwards.")
    if windows.before.start >= windows.after.end:
        raise TemporalRangeUnsupported(
            "The two periods do not leave room for an earlier and a later scene. Name two different periods, "
            "for example \"between June and September\".")
    return windows


def resolve_windows(query: str, today: date | None = None) -> TemporalWindows:
    """Two search windows for a question that needs two dates. Raises TemporalRangeUnsupported."""
    today = today or date.today()
    periods = _find_periods(query)

    if len(periods) >= 2:
        first, second = periods[0], periods[1]
        second_year = second.year or (_latest_year_for(second.month, today) if second.month else today.year)
        if first.year:
            first_year = first.year
        elif first.month and second.month:
            first_year = second_year if (first.month, first.day or 1) < (second.month, second.day or 1) else second_year - 1
        else:
            first_year = second_year
        a, b = _period_window(first, first_year, today), _period_window(second, second_year, today)
        before, after = (a, b) if (a.start, a.end) <= (b.start, b.end) else (b, a)
        return _check(TemporalWindows(before, after, "two explicit periods",
                                      f"The earlier scene is searched in {before.label} and the later in {after.label}."), today)

    if len(periods) == 1:
        period = periods[0]
        year = period.year or (_latest_year_for(period.month, today) if period.month else today.year)
        named = _period_window(period, year, today)
        recent = _recent(today)
        if named.end < recent.start:
            return _check(TemporalWindows(named, recent, "one explicit period",
                                          f"The earlier scene is searched in {named.label} and the later in {recent.label}."), today)
        before, after = _thirds(named.start, named.end)
        return _check(TemporalWindows(before, after, "one explicit period",
                                      f"{named.label} is recent, so it is split: its first part ({before.label}) "
                                      f"against its last part ({after.label})."), today)

    relative = _RELATIVE.search(query)
    if relative:
        amount = relative.group(1)
        count = int(amount) if amount and amount.isdigit() else _NUMBER_WORDS.get((amount or "one").lower(), 1)
        unit = relative.group(2).lower()
        span_days = count * _UNIT_DAYS[unit]
        if span_days < MIN_RELATIVE_DAYS:
            raise TemporalRangeUnsupported(
                f"{relative.group(0)} is too short to compare two Sentinel-2 scenes (a pass every few days, often "
                f"cloudy). Ask about at least the last {MIN_RELATIVE_DAYS} days.")
        start = today - timedelta(days=span_days - 1)
        if start < SENTINEL2_START:
            raise TemporalRangeUnsupported(
                f"{relative.group(0)} reaches back before Sentinel-2 imagery exists (mid-2015). Ask about a shorter period.")
        before, after = _thirds(start, today)
        return _check(TemporalWindows(before, after, "relative period",
                                      f"Over {relative.group(0)}: the earlier scene comes from its first part "
                                      f"({before.label}), the later from its last part ({after.label})."), today)

    start = today - timedelta(days=DEFAULT_SPAN_DAYS - 1)
    before, after = _thirds(start, today)
    return _check(TemporalWindows(before, after, "default",
                                  f"No period was named, so a scene from about a year ago ({before.label}) is compared "
                                  f"with a recent one ({after.label}): roughly the same season, a year apart."), today)
