"""What a weather question asks about: the period and the focus (D-029, optional capability).

The period is kept abstract ("tomorrow", "Friday", "the next 7 days") and resolved only once the
forecast arrives, against the location's own calendar: "tomorrow" in Mumbai is not tomorrow on the
server. Short range only: anything past the provider's 16 days, and anything in the past, is
refused with the user's own words quoted back, never estimated.
"""

import re
from dataclasses import dataclass
from datetime import date

from satquery.agent.intents import needs_weather
from satquery.schemas import Intent
from satquery.specialists.weather import FORECAST_DAYS as MAX_FORECAST_DAYS  # the provider's horizon
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december")
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16}
_NUMBER = r"(\d+|" + "|".join(NUMBER_WORDS) + r")"

# Named past periods are quoted in a refusal in preference to a bare past tense ("did it").
PAST = re.compile(r"\b(?:yesterday|last (?:night|week|weekend|month|year|\d+ days)|past (?:week|month|\d+ days)|"
                  r"\w+ ago)\b", re.I)
PAST_TENSE = re.compile(r"\b(?:did|was|were) (?:it|there)\b", re.I)
LONG_RANGE = re.compile(r"\b(?:next (?:month|year|season)|this (?:year|season)|in \d+ (?:months?|years?)|"
                        r"\d+ months?|(?:two|three|four|five|six|several|few) months|(?:next|this) monsoon)\b", re.I)
# Seasonal words refuse only when no short period is named ("will the monsoon rain continue this week?" is fine).
SEASONAL = re.compile(r"\b(?:monsoons?|seasons?|seasonal|winter|summer|onset|climate)\b", re.I)
DAYS = re.compile(rf"\b(?:next|coming|following)\s+{_NUMBER}\s+days?\b|\b{_NUMBER}[- ]days?\b", re.I)
WEEKS = re.compile(rf"\b(?:next|coming)\s+{_NUMBER}\s+weeks?\b|\bfortnight\b", re.I)
HOURS = re.compile(r"\b(?:next|coming)\s+(\d+)\s+hours?\b", re.I)
DAY_MONTH = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + "|".join(MONTHS) + r")\b"
                       r"|\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I)
MONTH = re.compile(r"\b(?:in|during|for|by)\s+(" + "|".join(MONTHS) + r")\b", re.I)

HEAVY = re.compile(r"\b(?:heavy|intense|torrential|extreme|very heavy)\b", re.I)
RAIN = re.compile(r"\b(?:rain\w*|precipitation|drizzl\w*|showers?|downpours?|monsoons?|umbrella|wet)\b", re.I)
TEMPERATURE = re.compile(r"\b(?:temperatures?|hot|cold|warm\w*|heat\w*|chilly|freez\w*|frost\w*|degrees?)\b", re.I)
WIND = re.compile(r"\b(?:wind\w*|storm\w*|gusts?|cyclones?)\b", re.I)


@dataclass(frozen=True)
class ForecastHorizon:
    """The period a weather question asks about, before it meets the location's own calendar."""

    label: str                   # how the answer names it, e.g. "tomorrow", "the next 7 days"
    start_day: int = 0           # days after the location's local today
    days: int = 7
    weekday: int | None = None   # the next such day (0 = Monday) in the location's calendar
    weekend: bool = False
    on_date: str | None = None   # ISO date named in the question, e.g. "2026-10-02"
    current: bool = False        # also report current conditions
    unsupported: str | None = None  # why this period cannot be forecast (the question is refused)
    phrase: str | None = None    # the user's words that named the period


def _count(text: str) -> int:
    return int(text) if text.isdigit() else NUMBER_WORDS[text.lower()]


def _refuse(phrase: str, reason: str) -> ForecastHorizon:
    return ForecastHorizon(label=phrase, unsupported=reason, phrase=phrase)


BEYOND = f"the forecast reaches only {MAX_FORECAST_DAYS} days ahead"


def forecast_horizon(query: str, today: date | None = None) -> ForecastHorizon:
    """The period `query` names. `today` (the server's date) only checks a named calendar date's
    distance; the period itself is resolved against the location's calendar after retrieval."""
    today = today or date.today()
    past = PAST.search(query) or PAST_TENSE.search(query)
    if past:
        return _refuse(past.group(0), "that asks about past weather, and this assistant only forecasts")
    long_range = LONG_RANGE.search(query)
    if long_range:
        return _refuse(long_range.group(0), f"that is a seasonal or long-range question, and {BEYOND}")
    for pattern, unit in ((DAYS, 1), (WEEKS, 7)):
        match = pattern.search(query)
        if match:
            count = 14 if match.group(0).lower() == "fortnight" else _count(next(g for g in match.groups() if g)) * unit
            if count > MAX_FORECAST_DAYS:
                return _refuse(match.group(0), BEYOND)
            if count == 0:
                break
            return ForecastHorizon(label=f"the next {count} days", days=count, phrase=match.group(0))
    hours = HOURS.search(query)
    if hours:
        count = int(hours.group(1))
        if count > MAX_FORECAST_DAYS * 24:
            return _refuse(hours.group(0), BEYOND)
        return ForecastHorizon(label=f"the next {count} hours", days=max(1, -(-count // 24) + 1),
                               current=True, phrase=hours.group(0))
    named = DAY_MONTH.search(query)
    if named:
        day, month = (named.group(1), named.group(2)) if named.group(1) else (named.group(4), named.group(3))
        try:
            target = date(today.year, MONTHS.index(month.lower()) + 1, int(day))
        except ValueError:
            return _refuse(named.group(0), "that is not a real calendar date")
        if (today - target).days > 1:  # a day already gone this year means the same day next year
            target = target.replace(year=today.year + 1)
        if (target - today).days > MAX_FORECAST_DAYS:
            return _refuse(named.group(0), BEYOND)
        return ForecastHorizon(label=target.strftime("%a %d %b"), days=1, on_date=target.isoformat(),
                               phrase=named.group(0))
    month = MONTH.search(query)
    if month:
        return _refuse(month.group(0), f"that asks about a whole month, which is a climate question; {BEYOND}")

    text = query.lower()
    short = None
    if re.search(r"\bday after tomorrow\b", text):
        short = ForecastHorizon(label="the day after tomorrow", start_day=2, days=1, phrase="day after tomorrow")
    elif re.search(r"\btomorrow\b", text):
        short = ForecastHorizon(label="tomorrow", start_day=1, days=1, phrase="tomorrow")
    elif re.search(r"\b(?:today|tonight|this (?:morning|afternoon|evening))\b", text):
        short = ForecastHorizon(label="today", days=1, current=True, phrase="today")
    elif re.search(r"\b(?:right now|currently|at the moment|now|current)\b", text):
        short = ForecastHorizon(label="now", days=1, current=True, phrase="now")
    elif any(re.search(rf"\b{name}\b", text) for name in WEEKDAYS):
        index = next(i for i, name in enumerate(WEEKDAYS) if re.search(rf"\b{name}\b", text))
        short = ForecastHorizon(label=WEEKDAYS[index].capitalize(), days=1, weekday=index, phrase=WEEKDAYS[index])
    elif re.search(r"\bweekend\b", text):
        short = ForecastHorizon(label="this weekend", days=2, weekend=True, phrase="weekend")
    elif re.search(r"\bnext week\b", text):
        short = ForecastHorizon(label="next week", start_day=7, days=7, phrase="next week")
    elif re.search(r"\b(?:couple of days)\b", text):
        short = ForecastHorizon(label="the next 2 days", days=2, phrase="couple of days")
    elif re.search(r"\b(?:few days|coming days|next days|several days|next little while)\b", text):
        short = ForecastHorizon(label="the next 3 days", days=3, phrase="the next few days")
    elif re.search(r"\b(?:this week|week)\b", text):
        short = ForecastHorizon(label="the next 7 days", days=7, phrase="this week")
    if short:
        return short
    seasonal = SEASONAL.search(query)
    if seasonal:
        reason = f"that is a seasonal or climate question, and {BEYOND}"
        if seasonal.group(0).lower().startswith(("monsoon", "onset")):
            reason += "; monsoon onset is declared by the national weather service (IMD in India) from its own criteria"
        return _refuse(seasonal.group(0), reason)
    return ForecastHorizon(label="the next 7 days", days=7, current=True)


def weather_focus(query: str) -> str | None:
    """What the question is mainly about: "heavy rain", "rain", "temperature", "wind", or None (general)."""
    if RAIN.search(query):
        return "heavy rain" if HEAVY.search(query) else "rain"
    if TEMPERATURE.search(query):
        return "temperature"
    if WIND.search(query):
        return "wind"
    return None


def classify_weather(query: str, today: date | None = None) -> tuple[Intent, ForecastHorizon]:
    """The weather intent, recorded with the rule that produced it, and the period it asks about."""
    horizon = forecast_horizon(query, today)
    focus = weather_focus(query)
    rule = (f"weather cue '{needs_weather(query) or 'weather'}' -> weather_forecast; period "
            + (f"'{horizon.phrase}' (unsupported)" if horizon.unsupported else horizon.label)
            + (f"; focus {focus}" if focus else ""))
    return Intent(task="weather_forecast", target=focus, matched_rule=rule), horizon
