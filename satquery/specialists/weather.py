"""Weather forecasts from Open-Meteo: the backend of the `weather.forecast` tool (D-029).

An OPTIONAL product enhancement, not an SIH requirement. Short range only: up to 16 days.

Open-Meteo (checked 2026-09-27): free for non-commercial use without an API key (600 calls/min,
10,000/day), data under CC BY 4.0, so the attribution travels with every answer. OPEN_METEO_API_KEY
is needed only for the paid customer endpoint; it is sent to that endpoint alone and never logged,
quoted in an error or returned by any API response.

Every failure raises a typed WeatherError. There is no fallback: no forecast is ever made up.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Protocol

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CUSTOMER_FORECAST_URL = "https://customer-api.open-meteo.com/v1/forecast"
PROVIDER = "Open-Meteo"
MODEL = "best_match (Open-Meteo's automatic choice of national and global weather models)"
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"
ATTRIBUTION_URL = "https://open-meteo.com/"
FORECAST_DAYS = 16  # always the whole horizon: one cache entry per point serves every question
DAILY = ("weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
         "precipitation_probability_max", "wind_speed_10m_max")
CURRENT = ("temperature_2m", "relative_humidity_2m", "precipitation", "weather_code", "wind_speed_10m")

# WMO weather interpretation codes, as Open-Meteo documents them.
WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle", 56: "light freezing drizzle",
    57: "dense freezing drizzle", 61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain", 71: "slight snowfall", 73: "moderate snowfall",
    75: "heavy snowfall", 77: "snow grains", 80: "slight rain showers", 81: "moderate rain showers",
    82: "violent rain showers", 85: "slight snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}
HEAVY_RAIN_CODES = {65, 67, 82}  # the code itself says heavy or violent rain


def condition(code: int | None) -> str:
    return WMO_CODES.get(code, f"weather code {code}") if code is not None else "condition not reported"


# --------------------------------------------------------------------------- errors


class WeatherError(Exception):
    """Base: no forecast could be obtained. Carries an HTTP status and a message written for a user."""

    status = 502
    code = "weather_unavailable"

    def __init__(self, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail

    def as_payload(self) -> dict:
        return {"code": self.code, "message": self.message} | ({"detail": self.detail} if self.detail else {})


class WeatherNotConfigured(WeatherError):
    status, code = 503, "weather_not_configured"


class WeatherUnavailable(WeatherError):
    status, code = 502, "weather_unavailable"


class WeatherRateLimited(WeatherError):
    status, code = 429, "rate_limited"


class WeatherTimeout(WeatherError):
    status, code = 504, "provider_timeout"


class NoForecastData(WeatherError):
    status, code = 502, "no_forecast_data"


class InvalidWeatherResponse(WeatherError):
    status, code = 502, "invalid_provider_response"


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class DailyForecast:
    date: str  # local calendar date at the location, ISO
    weather_code: int | None
    condition: str
    temperature_max: float | None
    temperature_min: float | None
    precipitation_sum: float | None
    precipitation_probability_max: float | None
    wind_speed_max: float | None


@dataclass(frozen=True)
class CurrentConditions:
    time: str  # local time at the location, ISO
    temperature: float | None
    relative_humidity: float | None
    precipitation: float | None
    weather_code: int | None
    condition: str
    wind_speed: float | None


@dataclass(frozen=True)
class Forecast:
    provider: str
    model: str
    requested_latitude: float
    requested_longitude: float
    latitude: float  # the model grid point the provider answered for
    longitude: float
    elevation_m: float | None
    timezone: str
    timezone_abbreviation: str | None
    units: dict
    days: tuple[DailyForecast, ...]  # starts at the location's local today
    current: CurrentConditions | None
    retrieved_at: str  # UTC, ISO
    cached: bool = False
    attribution: str = ATTRIBUTION
    attribution_url: str = ATTRIBUTION_URL

    def as_dict(self) -> dict:
        return asdict(self)


class WeatherBackend(Protocol):
    name: str

    def forecast(self, latitude: float, longitude: float) -> Forecast: ...


# --------------------------------------------------------------------------- parsing


def _number(value, what: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidWeatherResponse(f"The weather provider returned a non-numeric {what}.")
    return float(value)


def parse_forecast(payload, *, requested: tuple[float, float], retrieved_at: str) -> Forecast:
    """A validated Forecast from an Open-Meteo JSON body. Anything malformed raises, never guesses."""
    if not isinstance(payload, dict):
        raise InvalidWeatherResponse("The weather provider returned something other than a forecast.")
    daily = payload.get("daily")
    if not isinstance(daily, dict) or not isinstance(daily.get("time"), list):
        raise InvalidWeatherResponse("The weather provider's response has no daily forecast.")
    dates = daily["time"]
    if not dates:
        raise NoForecastData("The weather provider returned no forecast days for this location.")
    for key in DAILY:
        if not isinstance(daily.get(key), list) or len(daily[key]) != len(dates):
            raise InvalidWeatherResponse(f"The weather provider's daily '{key}' is missing or incomplete.")
    if all(daily["precipitation_probability_max"][i] is None and daily["temperature_2m_max"][i] is None
           for i in range(len(dates))):
        raise NoForecastData("The weather provider returned forecast days with no values for this location.")
    try:
        latitude, longitude = float(payload["latitude"]), float(payload["longitude"])
        zone = str(payload["timezone"])
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidWeatherResponse("The weather provider's response has no usable location or time zone.") from error

    days = []
    for i, day in enumerate(dates):
        code = daily["weather_code"][i]
        code = int(code) if isinstance(code, (int, float)) and not isinstance(code, bool) else None
        days.append(DailyForecast(
            date=str(day), weather_code=code, condition=condition(code),
            temperature_max=_number(daily["temperature_2m_max"][i], "temperature"),
            temperature_min=_number(daily["temperature_2m_min"][i], "temperature"),
            precipitation_sum=_number(daily["precipitation_sum"][i], "precipitation amount"),
            precipitation_probability_max=_number(daily["precipitation_probability_max"][i], "precipitation probability"),
            wind_speed_max=_number(daily["wind_speed_10m_max"][i], "wind speed")))

    current = None
    raw = payload.get("current")
    if isinstance(raw, dict) and raw.get("time"):
        code = raw.get("weather_code")
        code = int(code) if isinstance(code, (int, float)) and not isinstance(code, bool) else None
        current = CurrentConditions(
            time=str(raw["time"]), temperature=_number(raw.get("temperature_2m"), "temperature"),
            relative_humidity=_number(raw.get("relative_humidity_2m"), "humidity"),
            precipitation=_number(raw.get("precipitation"), "precipitation"), weather_code=code,
            condition=condition(code), wind_speed=_number(raw.get("wind_speed_10m"), "wind speed"))

    units = {key: value for key, value in (payload.get("daily_units") or {}).items() if key != "time"}
    return Forecast(
        provider=PROVIDER, model=MODEL, requested_latitude=requested[0], requested_longitude=requested[1],
        latitude=latitude, longitude=longitude, elevation_m=_number(payload.get("elevation"), "elevation"),
        timezone=zone, timezone_abbreviation=payload.get("timezone_abbreviation"), units=units,
        days=tuple(days), current=current, retrieved_at=retrieved_at)


# --------------------------------------------------------------------------- client


def _scrub(text: str, secret: str = "", limit: int = 300) -> str:
    """Truncate an upstream message and withhold it entirely if it echoes a credential."""
    cleaned = " ".join(str(text).split())
    if (secret and secret in cleaned) or "apikey" in cleaned.lower():
        return "(upstream message withheld: it contained credential material)"
    return cleaned[:limit]


class OpenMeteoWeather:
    """Open-Meteo forecast client with an in-memory cache.

    Forecasts change as models rerun, so the cache is short-lived (default 30 min) and in memory
    only: unlike imagery, an old forecast must not outlive the process. Keyed by the point rounded to
    0.01 degrees (about 1 km, finer than any weather model grid).
    """

    name = PROVIDER

    def __init__(self, *, api_key: str = "", timeout: float = 20.0, cache_ttl_s: float = 1800.0, clock=time.time):
        self._api_key = api_key
        self.timeout = timeout
        self.cache_ttl_s = cache_ttl_s
        self._clock = clock
        self._cache: dict[tuple[float, float], tuple[float, Forecast]] = {}
        self._lock = threading.Lock()

    def _client(self):
        import httpx

        return httpx.Client(timeout=self.timeout, headers={"User-Agent": "SatQuery (SIH26167 prototype)"})

    def forecast(self, latitude: float, longitude: float) -> Forecast:
        key = (round(latitude, 2), round(longitude, 2))
        with self._lock:
            hit = self._cache.get(key)
            if hit and self._clock() - hit[0] < self.cache_ttl_s:
                return replace(hit[1], cached=True)
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        forecast = parse_forecast(self._get(*key), requested=(latitude, longitude), retrieved_at=retrieved_at)
        with self._lock:
            self._cache[key] = (self._clock(), forecast)
        return forecast

    def _get(self, latitude: float, longitude: float):
        import httpx

        params = {"latitude": latitude, "longitude": longitude, "timezone": "auto", "forecast_days": FORECAST_DAYS,
                  "daily": ",".join(DAILY), "current": ",".join(CURRENT)}
        url = FORECAST_URL
        if self._api_key:  # the paid endpoint; the key goes nowhere else
            url, params = CUSTOMER_FORECAST_URL, params | {"apikey": self._api_key}
        try:
            with self._client() as client:
                response = client.get(url, params=params)
        except httpx.TimeoutException as error:
            raise WeatherTimeout("The weather provider did not respond in time. Try again shortly.") from error
        except httpx.HTTPError as error:
            raise WeatherUnavailable("The weather provider could not be reached.") from error

        if response.status_code == 429:
            raise WeatherRateLimited("The weather provider is limiting requests right now. Try again shortly.")
        if response.status_code in (401, 403):
            raise WeatherNotConfigured("The weather provider refused the request; check OPEN_METEO_API_KEY.")
        if response.status_code >= 400:
            try:
                reason = response.json().get("reason", "")
            except Exception:
                reason = response.text
            raise WeatherUnavailable(f"The weather provider could not answer (HTTP {response.status_code}).",
                                     detail=_scrub(reason, self._api_key) or None)
        try:
            return response.json()
        except ValueError as error:
            raise InvalidWeatherResponse("The weather provider's response is not valid JSON.") from error
