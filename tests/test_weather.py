"""Weather specialist (D-029, optional capability): routing, forecast period, area -> point, provider
parsing and errors, cache, and the answer, trace and report. Offline: HTTP is faked with the response
shape Open-Meteo returned live on 2026-09-27. The one live call is in tests/test_weather_live.py.
"""

import math
from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from satquery import api, geo
from satquery.agent.forecast import forecast_horizon, weather_focus
from satquery.agent.intents import route_query
from satquery.examples import EXAMPLE_QUERIES, load_scenarios
from satquery.settings import Settings
from satquery.specialists.tools import WeatherParams
from satquery.specialists.weather import (CUSTOMER_FORECAST_URL, FORECAST_URL, InvalidWeatherResponse, NoForecastData,
                                          OpenMeteoWeather, WeatherRateLimited, WeatherTimeout, WeatherUnavailable)

TODAY = date(2026, 9, 27)  # a Sunday
NAVI_MUMBAI = geo.bbox_geometry((72.97, 19.05, 73.02, 19.10))


# --------------------------------------------------------------------- fakes

def payload(start: date = TODAY, days: int = 16, probs=None, sums=None, codes=None, tmax=None, tmin=None) -> dict:
    """An Open-Meteo forecast body, shaped exactly like the live response."""
    dates = [(start + timedelta(days=i)).isoformat() for i in range(days)]
    pad = lambda values, fill: (list(values) + [fill] * days)[:days]
    return {
        "latitude": 19.086115, "longitude": 73.0306, "generationtime_ms": 0.27, "utc_offset_seconds": 19800,
        "timezone": "Asia/Kolkata", "timezone_abbreviation": "GMT+5:30", "elevation": 8.0,
        "current_units": {"time": "iso8601", "interval": "seconds", "temperature_2m": "°C"},
        "current": {"time": f"{start}T02:15", "interval": 900, "temperature_2m": 24.9, "relative_humidity_2m": 88,
                    "precipitation": 0.0, "weather_code": 3, "wind_speed_10m": 7.2},
        "daily_units": {"time": "iso8601", "weather_code": "wmo code", "temperature_2m_max": "°C",
                        "temperature_2m_min": "°C", "precipitation_sum": "mm", "precipitation_probability_max": "%",
                        "wind_speed_10m_max": "km/h"},
        "daily": {"time": dates,
                  "weather_code": pad(codes or [51, 53, 95, 51, 51, 1, 1], 1),
                  "temperature_2m_max": pad(tmax or [30.8, 31.7, 31.3, 32.4, 33.0, 34.7, 34.5], 33.0),
                  "temperature_2m_min": pad(tmin or [24.6, 24.7, 24.9, 24.3, 24.9, 25.9, 26.2], 25.0),
                  "precipitation_sum": pad(sums or [0.8, 1.5, 5.2, 1.5, 0.3, 0.0, 0.0], 0.0),
                  "precipitation_probability_max": pad(probs or [29, 55, 95, 94, 53, 27, 29], 10),
                  "wind_speed_10m_max": pad([12.0, 14.0, 22.0, 15.0, 11.0, 9.0, 8.0], 10.0)},
    }


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=None):
        self.status_code, self._body = status_code, body
        self.text = text if text is not None else str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeClient:
    """Stands in for httpx.Client, recording every GET."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def http(monkeypatch):
    """Point every OpenMeteoWeather at a scripted answer; returns the recording client."""
    client = FakeClient(FakeResponse(200, payload()))
    monkeypatch.setattr(OpenMeteoWeather, "_client", lambda self: client)
    monkeypatch.setattr(api, "_WEATHER", {})  # no forecast cached by an earlier test
    return client


@pytest.fixture
def settings(tmp_path):
    return Settings(runs_dir=tmp_path / "runs")


def ask(settings, query, area=NAVI_MUMBAI, **kwargs):
    return api.answer_weather(query, area, settings=settings, **kwargs)


# --------------------------------------------------------------------- 1. routing

WEATHER_QUERIES = [
    "What's the weather going to be like this week?",
    "Will it rain here in the next 7 days?",
    "When is rain most likely?",
    "What's the temperature going to be this week?",
    "Will there be heavy rainfall tomorrow?",
    "What will the weather be like over the next few days?",
    "Will it snow tomorrow?",
    "Is it going to be windy tonight?",
    "When will monsoon come?",  # a weather question, refused later as long-range
    "Did it rain yesterday?",   # likewise, refused as past
]

IMAGERY_QUERIES = [
    *EXAMPLE_QUERIES, *(scenario["query"] for scenario in load_scenarios()),
    "Compare optical and SAR evidence to find water.", "What changed here over the last 3 months?",
    "Is there flooding in this area?", "Has the built-up area increased?", "Describe this area",
    # weather-like words that describe what imagery shows, with no forecast cue
    "Is this a rainforest?", "Is there snow on the peaks?", "How much cloud cover is in this scene?",
    "Where are the wind turbines?", "Is there fog over the river?", "Highlight the storm drains",
]

MIXED_QUERIES = [
    "What is the weather and what changed here?",
    "Will it rain tomorrow and where are the buildings?",
    "Is it raining in this satellite image?",
]


@pytest.mark.parametrize("query", WEATHER_QUERIES)
def test_weather_questions_route_to_the_weather_specialist(query):
    assert route_query(query)[0] == "weather", query


@pytest.mark.parametrize("query", IMAGERY_QUERIES)
def test_satellite_questions_never_route_to_weather(query):
    assert route_query(query)[0] == "imagery", query


@pytest.mark.parametrize("query", MIXED_QUERIES)
def test_weather_plus_imagery_is_recognised_as_mixed(query):
    decided, rule = route_query(query)
    assert decided == "mixed" and "weather cue" in rule and "satellite-analysis cue" in rule


# --------------------------------------------------------------------- 2. forecast period

@pytest.mark.parametrize("query, start, days", [
    ("What's the weather going to be like this week?", 0, 7),
    ("Will it rain here in the next 7 days?", 0, 7),
    ("What will the weather be like over the next few days?", 0, 3),
    ("Will there be heavy rainfall tomorrow?", 1, 1),
    ("What about the day after tomorrow?", 2, 1),
    ("Weather for next week?", 7, 7),
    ("Rain over the next 16 days?", 0, 16),
    ("Rain over the next two weeks?", 0, 14),
    ("Will the monsoon rain continue this week?", 0, 7),  # a short period named: not a seasonal question
])
def test_short_periods(query, start, days):
    horizon = forecast_horizon(query, TODAY)
    assert (horizon.unsupported, horizon.start_day, horizon.days) == (None, start, days)


def test_named_days_stay_abstract_until_the_local_calendar_is_known():
    assert forecast_horizon("Will it rain on Friday?", TODAY).weekday == 4
    assert forecast_horizon("Weather this weekend?", TODAY).weekend
    assert forecast_horizon("Will it rain on 2 October?", TODAY).on_date == "2026-10-02"
    now = forecast_horizon("Is it raining right now?", TODAY)
    assert now.current and now.days == 1
    default = forecast_horizon("What's the weather like here?", TODAY)
    assert (default.days, default.current) == (7, True)


@pytest.mark.parametrize("query, phrase, reason", [
    ("When will monsoon come?", "monsoon", "IMD"),
    ("Will it rain in January?", "in January", "whole month"),
    ("What will the weather be like six months from now?", "six months", "long-range"),
    ("Will it rain next month?", "next month", "long-range"),
    ("Rain over the next 20 days?", "next 20 days", "16 days"),
    ("Rain over the next 3 weeks?", "next 3 weeks", "16 days"),
    ("Will it rain on October 20?", "October 20", "16 days"),
    ("Will it rain on 30 February?", "30 February", "not a real"),
    ("Did it rain yesterday?", "yesterday", "past"),
    ("How much rain fell last week?", "last week", "past"),
])
def test_long_range_and_past_periods_are_refused_with_the_users_words(query, phrase, reason):
    horizon = forecast_horizon(query, TODAY)
    assert horizon.unsupported and reason in horizon.unsupported
    assert horizon.phrase.lower() == phrase.lower()


@pytest.mark.parametrize("query, focus", [
    ("Will there be heavy rainfall tomorrow?", "heavy rain"),
    ("When is rain most likely?", "rain"),
    ("What's the temperature going to be this week?", "temperature"),
    ("Is it going to be windy tonight?", "wind"),
    ("What's the weather going to be like this week?", None),
])
def test_the_focus_of_the_question(query, focus):
    assert weather_focus(query) == focus


# --------------------------------------------------------------------- 3. area -> forecast point

def _inside(point, ring):
    x, y = point
    crossings = 0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            crossings += 1
    return crossings % 2 == 1


def test_a_rectangle_is_forecast_at_its_centre():
    assert geo.representative_point(NAVI_MUMBAI) == pytest.approx((72.995, 19.075))


def test_a_drawn_circle_is_forecast_at_its_centre():
    ring = [[73.0 + 0.02 * math.cos(a), 19.0 + 0.02 * math.sin(a)] for a in (i * 2 * math.pi / 64 for i in range(64))]
    circle = {"type": "Polygon", "coordinates": [ring + [ring[0]]]}
    assert geo.representative_point(circle) == pytest.approx((73.0, 19.0), abs=1e-6)


def test_a_concave_shape_is_forecast_inside_itself_not_at_its_centroid():
    # A U: its centroid falls in the empty middle
    u = [[0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3], [0, 0]]
    point = geo.representative_point({"type": "Polygon", "coordinates": [u]})
    assert _inside(point, u)


def test_a_hole_is_never_the_forecast_point():
    outer = [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]
    hole = [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]
    point = geo.representative_point({"type": "Polygon", "coordinates": [outer, hole]})
    assert _inside(point, outer) and not _inside(point, hole)


def test_a_multipolygon_is_forecast_in_its_largest_part():
    small = [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
    large = [[[10, 10], [14, 10], [14, 14], [10, 14], [10, 10]]]
    assert geo.representative_point({"type": "MultiPolygon", "coordinates": [small, large]}) == pytest.approx((12, 12))


def test_the_area_extent_is_latitude_corrected():
    width, height = geo.area_extent_km(NAVI_MUMBAI)
    assert width == pytest.approx(0.05 * 111.32 * math.cos(math.radians(19.075)), rel=1e-6)
    assert height == pytest.approx(0.05 * 110.57, rel=1e-6)


# --------------------------------------------------------------------- 4. provider

def test_the_request_asks_for_the_whole_horizon_in_local_days(http):
    forecast = OpenMeteoWeather().forecast(19.07512, 72.99534)
    call = http.calls[0]
    assert call["url"] == FORECAST_URL and "apikey" not in call["params"]
    assert (call["params"]["forecast_days"], call["params"]["timezone"]) == (16, "auto")
    assert (call["params"]["latitude"], call["params"]["longitude"]) == (19.08, 73.0), "rounded, as cached"
    assert "precipitation_probability_max" in call["params"]["daily"]
    assert len(forecast.days) == 16 and forecast.days[0].date == "2026-09-27"
    assert (forecast.latitude, forecast.longitude, forecast.elevation_m) == (19.086115, 73.0306, 8.0)
    assert forecast.days[2].condition == "thunderstorm" and forecast.current.temperature == 24.9
    assert forecast.units["precipitation_sum"] == "mm"


def test_an_api_key_goes_only_to_the_paid_endpoint_and_never_into_an_error(http):
    http.answer = FakeResponse(400, {"error": True, "reason": "apikey sk-secret-123 is invalid"})
    with pytest.raises(WeatherUnavailable) as caught:
        OpenMeteoWeather(api_key="sk-secret-123").forecast(19.0, 73.0)
    assert http.calls[0]["url"] == CUSTOMER_FORECAST_URL and http.calls[0]["params"]["apikey"] == "sk-secret-123"
    assert "sk-secret-123" not in str(caught.value.as_payload())


@pytest.mark.parametrize("answer, error, code", [
    (FakeResponse(429, {}), WeatherRateLimited, "rate_limited"),
    (FakeResponse(503, {"reason": "maintenance"}), WeatherUnavailable, "weather_unavailable"),
    (httpx.ReadTimeout("slow"), WeatherTimeout, "provider_timeout"),
    (httpx.ConnectError("offline"), WeatherUnavailable, "weather_unavailable"),
    (FakeResponse(200, ValueError("not json")), InvalidWeatherResponse, "invalid_provider_response"),
    (FakeResponse(200, ["not", "a", "forecast"]), InvalidWeatherResponse, "invalid_provider_response"),
])
def test_provider_failures_are_typed_errors_never_a_made_up_forecast(http, answer, error, code):
    http.answer = answer
    with pytest.raises(error) as caught:
        OpenMeteoWeather().forecast(19.0, 73.0)
    assert caught.value.code == code and caught.value.message


def _broken(**changes):
    body = payload()
    body["daily"] |= changes
    return body


@pytest.mark.parametrize("body, error", [
    (_broken(precipitation_probability_max=None), InvalidWeatherResponse),  # a variable missing
    (_broken(temperature_2m_max=[30.0]), InvalidWeatherResponse),  # arrays of different lengths
    (_broken(temperature_2m_max=["hot"] * 16), InvalidWeatherResponse),  # not a number
    (_broken(time=[], weather_code=[], temperature_2m_max=[], temperature_2m_min=[], precipitation_sum=[],
             precipitation_probability_max=[], wind_speed_10m_max=[]), NoForecastData),
    (_broken(temperature_2m_max=[None] * 16, precipitation_probability_max=[None] * 16), NoForecastData),
])
def test_malformed_or_empty_forecasts_are_refused(http, body, error):
    http.answer = FakeResponse(200, body)
    with pytest.raises(error):
        OpenMeteoWeather().forecast(19.0, 73.0)


def test_a_forecast_is_reused_within_its_ttl_then_refreshed(http):
    now = [1000.0]
    weather = OpenMeteoWeather(cache_ttl_s=1800, clock=lambda: now[0])
    first = weather.forecast(19.071, 72.991)
    again = weather.forecast(19.0712, 72.9913)  # ~30 m away: the same 0.01-degree cell
    assert len(http.calls) == 1 and (first.cached, again.cached) == (False, True)
    weather.forecast(19.2, 72.8)  # another point
    assert len(http.calls) == 2
    now[0] += 1801  # past the TTL: a model rerun may have changed the forecast
    assert weather.forecast(19.071, 72.991).cached is False and len(http.calls) == 3


def test_the_tool_accepts_only_permitted_parameters():
    with pytest.raises(ValidationError):
        WeatherParams(latitude=19.0, longitude=73.0, units="imperial")
    with pytest.raises(ValidationError):
        WeatherParams(latitude=91.0, longitude=73.0)
    with pytest.raises(ValidationError):
        WeatherParams(latitude=19.0, longitude=73.0, start_day=10, days=7)  # ends past 16 days


# --------------------------------------------------------------------- 5. answer, trace, report

def test_a_weather_answer_runs_the_agent_pipeline_and_records_it(http, settings):
    response = ask(settings, "Will it rain here in the next 7 days?")
    trace = response.trace
    assert (response.status, response.task, trace.input_config) == ("ok", "weather_forecast", "area_only")
    assert trace.images == [] and "weather cue 'rain'" in trace.intent.matched_rule
    (step,) = trace.plan
    assert step.tool == "weather.forecast" and step.image_indices == []
    assert (step.params["latitude"], step.params["longitude"], step.params["days"]) == (19.075, 72.995, 7)
    (result,) = trace.steps
    assert result.status == "ok" and "Open-Meteo" in result.model
    assert result.outputs["period"] == ["2026-09-27", "2026-10-03"] and len(result.outputs["days"]) == 7
    assert any(e.label == "2026-09-29 precipitation probability max (%)" and e.value == 95 for e in response.evidence)
    assert response.confidence.value is None and "not estimated by SatQuery" in response.confidence.method
    report = Path(response.report_html).read_text(encoding="utf-8")
    assert "weather.forecast" in report and "Open-Meteo.com (CC BY 4.0)" in report


def test_rain_wording_names_the_most_likely_day_and_every_figure_comes_from_the_provider(http, settings):
    answer = ask(settings, "When is rain most likely?").answer
    lines = answer.split("\n")
    assert lines[0].startswith("Rain is likely over the next 7 days") and "most likely Tue 29 Sep (95% chance" in lines[0]
    assert "Today (Sun 27 Sep): 25–31 °C · rain 29% (0.8 mm) · light drizzle" in lines
    # The point asked for, and the model grid point that answered, are both named, with their distance.
    assert ("Forecast for 19.075°N, 72.995°E (marked on the map); the provider answered from its nearest model grid "
            "point, 19.086°N, 73.031°E, about 3.9 km away") in answer and "Asia/Kolkata" in answer
    assert "Weather data by Open-Meteo.com (CC BY 4.0)" in answer


def test_tomorrow_is_resolved_on_the_locations_calendar_not_the_servers(http, settings):
    http.answer = FakeResponse(200, payload(start=date(2026, 9, 28)))  # already Monday where the area is
    outputs = ask(settings, "Will there be heavy rainfall tomorrow?").trace.steps[0].outputs
    assert outputs["local_today"] == "2026-09-28" and outputs["period"] == ["2026-09-29", "2026-09-29"]


@pytest.mark.parametrize("query, period", [
    ("Will it rain on Friday?", ["2026-10-02", "2026-10-02"]),
    ("Weather this weekend?", ["2026-09-27", "2026-09-27"]),  # a Sunday: only today is left of the weekend
    ("Will it rain on 2 October?", ["2026-10-02", "2026-10-02"]),
    ("Weather for next week?", ["2026-10-04", "2026-10-10"]),
])
def test_named_days_resolve_to_local_dates(http, settings, query, period):
    assert ask(settings, query).trace.steps[0].outputs["period"] == period


@pytest.mark.parametrize("codes, sums, expected", [
    ([51, 65], [0.8, 20.0], "Heavy rain is forecast tomorrow"),  # the provider's own "heavy rain" code
    ([51, 63], [0.8, 70.0], "Heavy rain is forecast tomorrow"),  # IMD's 64.5 mm/day threshold
    ([51, 53], [0.8, 1.5], "Heavy rain is not expected tomorrow (Mon 28 Sep): about 1.5 mm"),
])
def test_heavy_rain_is_judged_by_a_stated_rule(http, settings, codes, sums, expected):
    http.answer = FakeResponse(200, payload(codes=codes, sums=sums))
    answer = ask(settings, "Will there be heavy rainfall tomorrow?").answer
    assert answer.startswith(expected) and "64.5 mm" in answer


def test_temperature_questions_lead_with_temperature(http, settings):
    first = ask(settings, "What's the temperature going to be this week?").answer.split("\n")[0]
    assert first.startswith("Temperatures over the next 7 days") and "24 °C to 35 °C" in first


def test_a_large_area_is_not_passed_off_as_one_point(http, settings):
    response = ask(settings, "Will it rain this week?", area=geo.bbox_geometry((72.5, 18.5, 73.5, 19.5)))
    assert any(i.code == "point_forecast" and i.severity == "warning" for i in response.trace.validation)
    assert "one point cannot represent all of it" in response.answer


@pytest.mark.parametrize("query, area, code", [
    ("Will it rain tomorrow?", None, "area_missing"),
    ("Will it rain tomorrow?", {"type": "Point", "coordinates": [73.0, 19.0]}, "area_invalid"),
    ("What is the weather and what changed here?", NAVI_MUMBAI, "mixed_question"),
    ("Describe this area", NAVI_MUMBAI, "not_a_weather_question"),
    ("When will monsoon come?", NAVI_MUMBAI, "forecast_horizon_unsupported"),
    ("Will it rain in January?", NAVI_MUMBAI, "forecast_horizon_unsupported"),
    ("Did it rain yesterday?", NAVI_MUMBAI, "forecast_horizon_unsupported"),
    ("   ", NAVI_MUMBAI, "empty_query"),
])
def test_refusals_are_structured_and_never_call_the_provider(http, settings, query, area, code):
    response = ask(settings, query, area=area)
    assert response.status == "invalid_input" and [i.code for i in response.trace.validation] == [code]
    assert response.answer.startswith("Input rejected:") and http.calls == [], "nothing retrieved, nothing estimated"


def test_a_provider_failure_is_reported_never_filled_in(http, settings):
    http.answer = FakeResponse(429, {})
    response = ask(settings, "Will it rain tomorrow?")
    assert response.status == "error" and response.task == "weather_forecast"
    (step,) = response.trace.steps
    assert step.status == "failed" and step.error.startswith("WeatherRateLimited:")
    assert "limiting requests" in response.answer and "No forecast was estimated" in response.answer
    assert "°C" not in response.answer and "%" not in response.answer


def test_a_switched_off_provider_is_reported(http, tmp_path):
    response = ask(Settings(runs_dir=tmp_path / "runs", weather_provider="off"), "Will it rain tomorrow?")
    assert response.status == "error" and "switched off" in response.answer and http.calls == []
