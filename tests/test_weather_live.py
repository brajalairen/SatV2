"""One real Open-Meteo forecast. Opt-in, so the normal suite makes no network calls:

    .venv\\Scripts\\python -m pytest tests/test_weather_live.py -m live

The free endpoint needs no key. Weather is an optional capability (D-029), not an SIH requirement.
"""

import pytest

from satquery import geo
from satquery.api import answer_weather
from satquery.settings import Settings
from satquery.specialists.weather import OpenMeteoWeather

pytestmark = pytest.mark.live

NAVI_MUMBAI = geo.bbox_geometry((72.97, 19.05, 73.02, 19.10))


def test_one_live_forecast_parses_and_answers(tmp_path):
    forecast = OpenMeteoWeather().forecast(19.075, 72.995)
    print(f"\n  grid {forecast.latitude}, {forecast.longitude} | {forecast.timezone} | days {len(forecast.days)}"
          f" from {forecast.days[0].date} | today: {forecast.days[0].condition}")
    assert len(forecast.days) == 16 and forecast.timezone
    assert abs(forecast.latitude - 19.075) < 0.2 and abs(forecast.longitude - 72.995) < 0.2
    assert any(day.precipitation_probability_max is not None for day in forecast.days)

    response = answer_weather("Will it rain here in the next 7 days?", NAVI_MUMBAI,
                              settings=Settings(runs_dir=tmp_path / "runs"))
    assert response.status == "ok" and response.task == "weather_forecast"
    assert "Open-Meteo.com (CC BY 4.0)" in response.answer
    print("  " + response.answer.split("\n")[0])
