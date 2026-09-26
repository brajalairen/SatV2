"""Copernicus provider: OAuth, catalogue search, selection, validation and raster compatibility.

Every test here is offline. HTTP is faked, so the normal suite consumes no Copernicus quota.
The one live test lives in tests/test_copernicus_live.py behind the `live` marker.
"""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.imaging import load_image
from satquery.providers.copernicus import (
    BAND_ROLE,
    CopernicusSentinelProvider,
    bands_for_target,
    bbox_area_km2,
    evalscript,
)
from satquery.providers.errors import (
    AreaInvalid,
    AreaTooLarge,
    AuthenticationFailed,
    CredentialsMissing,
    NoImageryFound,
    RateLimited,
)

BBOX = (72.90, 19.00, 73.00, 19.10)  # ~116 km2 near Navi Mumbai
BANDS = ["B02", "B03", "B04", "B08"]


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b"", text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.content = content
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeClient:
    """Stands in for httpx.Client, recording every POST."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, data=None, json=None, headers=None):
        self.calls.append({"url": url, "data": data, "json": json, "headers": headers or {}})
        return self._responses.pop(0)


@pytest.fixture
def provider():
    return CopernicusSentinelProvider("test-id", "test-secret", max_aoi_km2=400.0)


def _wire(monkeypatch, provider, responses):
    """Point the provider at a scripted set of responses and return the recording client."""
    client = FakeClient(responses)
    monkeypatch.setattr(provider, "_client", lambda: client)
    return client


def _token_response(expires_in=600):
    return FakeResponse(200, {"access_token": "tok-abc", "expires_in": expires_in})


def _feature(scene_id, date, cloud):
    return {"id": scene_id, "properties": {"datetime": date, "eo:cloud_cover": cloud}}


# --------------------------------------------------------------------- 1. OAuth / token handling

def test_missing_credentials_are_refused_before_any_request():
    with pytest.raises(CredentialsMissing):
        CopernicusSentinelProvider("", "")


def test_token_is_requested_once_and_then_reused(monkeypatch, provider):
    """Quota protection: a second call must not sign in again."""
    client = _wire(monkeypatch, provider, [_token_response()])

    assert provider.token() == "tok-abc"
    assert provider.token() == "tok-abc"

    assert len(client.calls) == 1, "the token should be cached, not re-requested"


def test_the_secret_is_sent_only_in_the_token_body(monkeypatch, provider):
    client = _wire(monkeypatch, provider, [_token_response()])
    provider.token()

    call = client.calls[0]
    assert call["data"]["client_secret"] == "test-secret"
    assert call["data"]["grant_type"] == "client_credentials"
    # never in a header, never in a URL
    assert "test-secret" not in json.dumps(call["headers"])
    assert "test-secret" not in call["url"]


def test_an_expired_token_is_refreshed(monkeypatch, provider):
    """The cache has a 30 s minimum lifetime, so expiry is simulated by advancing the clock."""
    client = _wire(monkeypatch, provider, [_token_response(expires_in=600), _token_response()])
    provider.token()

    import satquery.providers.copernicus as module

    real_time = module.time.time
    monkeypatch.setattr(module.time, "time", lambda: real_time() + 3600)
    provider.token()

    assert len(client.calls) == 2, "a token past its expiry must be replaced"


def test_rejected_credentials_raise_authentication_failed(monkeypatch, provider):
    _wire(monkeypatch, provider, [FakeResponse(401, {"error": "invalid_client"})])
    with pytest.raises(AuthenticationFailed):
        provider.token()


def test_an_error_body_containing_a_token_is_not_quoted_back(monkeypatch, provider):
    leaky = json.dumps({"error": "boom", "access_token": "super-secret-value"})
    _wire(monkeypatch, provider, [FakeResponse(500, {}, text=leaky)])
    with pytest.raises(AuthenticationFailed) as caught:
        provider.token()
    assert "super-secret-value" not in str(caught.value.detail or "")


# --------------------------------------------------------------------- 2. catalogue search

def test_search_sends_bbox_collection_and_cloud_filter(monkeypatch, provider):
    client = _wire(monkeypatch, provider, [
        _token_response(),
        FakeResponse(200, {"features": [_feature("s1", "2026-09-14T05:20:00Z", 3.1)]}),
    ])
    features = provider.search(BBOX, days_back=14, max_cloud=25.0)

    assert len(features) == 1
    payload = client.calls[1]["json"]
    assert payload["bbox"] == list(BBOX)
    assert payload["collections"] == ["sentinel-2-l2a"]
    assert "25.0" in payload["filter"]
    # The property name must be quoted. Unquoted, the live API answers HTTP 200 with zero features
    # instead of an error, so a wrong filter looks exactly like "no imagery" (verified 2026-09-24).
    assert '"eo:cloud_cover"' in payload["filter"], "the property name must be quoted for cql2-text"
    assert payload["filter-lang"] == "cql2-text"
    assert "/" in payload["datetime"], "a start/end range is required"
    assert client.calls[1]["headers"]["Authorization"] == "Bearer tok-abc"


def test_the_least_cloudy_scene_wins(provider):
    chosen = provider.select_scene([
        _feature("cloudy", "2026-09-20T05:00:00Z", 44.0),
        _feature("clear", "2026-09-10T05:00:00Z", 2.0),
        _feature("middling", "2026-09-18T05:00:00Z", 17.5),
    ])
    assert chosen["id"] == "clear"


def test_the_newer_scene_wins_when_cloud_cover_ties(provider):
    chosen = provider.select_scene([
        _feature("older", "2026-09-01T05:00:00Z", 5.0),
        _feature("newer", "2026-09-19T05:00:00Z", 5.0),
    ])
    assert chosen["id"] == "newer"


# --------------------------------------------------------------------- 3. no-result handling

def test_an_empty_catalogue_result_is_reported_not_faked(provider):
    with pytest.raises(NoImageryFound) as caught:
        provider.select_scene([])
    message = str(caught.value).lower()
    assert "no sentinel-2" in message
    assert any(hint in message for hint in ("cloud", "window", "area")), "should suggest a remedy"


def test_rate_limiting_is_surfaced_as_its_own_error(monkeypatch, provider):
    _wire(monkeypatch, provider, [_token_response(), FakeResponse(429, {}, text="slow down")])
    with pytest.raises(RateLimited):
        provider.search(BBOX)


# --------------------------------------------------------------------- 4. AOI validation

@pytest.mark.parametrize("bbox, expected", [
    ((73.0, 19.0, 72.9, 19.1), AreaInvalid),      # west >= east
    ((72.9, 19.1, 73.0, 19.0), AreaInvalid),      # south >= north
    ((72.9, 19.0, 72.9, 19.0), AreaInvalid),      # a point encloses nothing (D-025)
    ((-200.0, 19.0, 73.0, 19.1), AreaInvalid),    # outside valid longitude
    ((60.0, 10.0, 80.0, 30.0), AreaTooLarge),     # far above the km2 limit
])
def test_invalid_areas_are_refused_before_any_api_call(provider, bbox, expected):
    with pytest.raises(expected):
        provider.validate_area(bbox)


def test_a_reasonable_area_passes_validation(provider):
    provider.validate_area(BBOX)  # must not raise


def test_area_is_latitude_corrected(provider):
    """One degree of longitude covers less ground near the poles than at the equator."""
    equator = bbox_area_km2((0.0, 0.0, 1.0, 1.0))
    high_latitude = bbox_area_km2((0.0, 60.0, 1.0, 61.0))
    assert high_latitude < equator / 1.5


# --------------------------------------------------------------------- 5. processing request

def test_query_target_selects_the_band_set():
    assert bands_for_target("vegetation") == ["B02", "B03", "B04", "B08"]
    assert bands_for_target("water") == ["B02", "B03", "B04", "B08"]
    assert bands_for_target(None) == ["B02", "B03", "B04", "B08"]
    assert bands_for_target("nonsense-target") == bands_for_target(None)


def test_evalscript_requests_every_band_as_float_reflectance():
    script = evalscript(["B04", "B08"])
    assert '"B04"' in script and '"B08"' in script
    assert "FLOAT32" in script and "REFLECTANCE" in script
    assert "bands: 2" in script
    assert "[sample.B04, sample.B08]" in script


def test_process_request_pins_the_selected_date_and_crs(monkeypatch, provider):
    client = _wire(monkeypatch, provider, [_token_response(), FakeResponse(200, content=b"TIFF")])
    provider._process(BBOX, BANDS, "2026-09-14")

    payload = client.calls[1]["json"]
    time_range = payload["input"]["data"][0]["dataFilter"]["timeRange"]
    assert time_range["from"].startswith("2026-09-14")
    assert time_range["to"].startswith("2026-09-14"), "must not mosaic across dates"
    assert "EPSG/0/4326" in payload["input"]["bounds"]["properties"]["crs"]
    assert payload["output"]["responses"][0]["format"]["type"] == "image/tiff"
    assert payload["output"]["width"] > 0 and payload["output"]["height"] > 0


def test_output_size_is_capped_so_a_huge_area_cannot_blow_up_the_request(provider):
    width, height = provider._output_size((0.0, 0.0, 5.0, 5.0))
    assert width <= 2500 and height <= 2500


# --------------------------------------------------------------------- 6. raster compatibility

def _write_scene(path, bands=BANDS):
    """A GeoTIFF shaped like what the Process API returns: float32, EPSG:4326, no descriptions."""
    data = np.random.default_rng(0).random((len(bands), 24, 24)).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=24, width=24, count=len(bands),
                       dtype="float32", crs="EPSG:4326",
                       transform=from_origin(72.90, 19.10, 0.0001, 0.0001)) as dst:
        dst.write(data)
    return path


def test_band_descriptions_are_written_so_the_pipeline_never_guesses(tmp_path, provider):
    path = _write_scene(tmp_path / "scene.tif")
    width, height = provider._label_bands(path, BANDS)

    assert (width, height) == (24, 24)
    image = load_image(path, modality="optical")
    assert image.band_names == ["blue", "green", "red", "nir"]
    assert image.band_names_assumed is False, "descriptions must remove the assumption"


def test_the_retrieved_raster_loads_through_the_existing_pipeline(tmp_path, provider):
    """Compatibility is what makes this an acquisition layer rather than a second pipeline."""
    path = _write_scene(tmp_path / "scene.tif")
    provider._label_bands(path, BANDS)

    image = load_image(path, modality="optical")
    assert image.modality == "optical"
    assert image.crs is not None and image.transform is not None
    for role in ("red", "green", "blue", "nir"):
        assert image.band(role) is not None, f"{role} must resolve for NDVI/NDWI and rendering"

    from satquery.imaging import render_rgb

    rgb = render_rgb(image)
    assert rgb.shape == (24, 24, 3) and rgb.dtype == np.uint8


def test_a_band_count_mismatch_is_reported(tmp_path, provider):
    from satquery.providers.errors import RasterUnreadable

    path = _write_scene(tmp_path / "scene.tif", bands=["B02", "B03"])
    with pytest.raises(RasterUnreadable):
        provider._label_bands(path, BANDS)  # asked for 4, file has 2


# --------------------------------------------------------------------- 7. metadata propagation

def test_retrieve_returns_full_provenance(monkeypatch, tmp_path, provider):
    scene_bytes = _write_scene(tmp_path / "source.tif").read_bytes()
    _wire(monkeypatch, provider, [
        _token_response(),
        FakeResponse(200, {"features": [
            _feature("S2B_MSIL2A_20260914", "2026-09-14T05:20:11Z", 4.25),
            _feature("S2A_older", "2026-09-02T05:20:11Z", 30.0),
        ]}),
        FakeResponse(200, content=scene_bytes),
    ])

    result = provider.retrieve(BBOX, BANDS, tmp_path / "out" / "scene.tif")
    meta = result.metadata

    assert result.path.is_file()
    assert meta.satellite == "Sentinel-2" and meta.product_level == "L2A"
    assert meta.collection == "sentinel-2-l2a"
    assert meta.provider == "Copernicus Data Space Ecosystem"
    assert meta.acquired == "2026-09-14"
    assert meta.acquired_datetime == "2026-09-14T05:20:11Z"
    assert meta.cloud_cover == 4.25
    assert meta.scene_id == "S2B_MSIL2A_20260914"
    assert meta.bbox_wgs84 == BBOX
    assert meta.crs == "EPSG:4326" and meta.resolution_m == 10.0
    assert meta.bands == ["B02 (blue)", "B03 (green)", "B04 (red)", "B08 (nir)"]
    assert meta.alternatives_considered == 2
    assert "Copernicus Sentinel data" in meta.attribution
    assert meta.as_dict()["satellite"] == "Sentinel-2"


def test_metadata_carries_no_credential_material(monkeypatch, tmp_path, provider):
    scene_bytes = _write_scene(tmp_path / "source.tif").read_bytes()
    _wire(monkeypatch, provider, [
        _token_response(),
        FakeResponse(200, {"features": [_feature("s", "2026-09-14T05:20:11Z", 1.0)]}),
        FakeResponse(200, content=scene_bytes),
    ])
    result = provider.retrieve(BBOX, BANDS, tmp_path / "out" / "scene.tif")

    serialised = json.dumps(result.metadata.as_dict())
    assert "test-secret" not in serialised and "test-id" not in serialised
    assert "tok-abc" not in serialised


# --------------------------------------------------------------------- cache key

def test_the_cache_key_covers_every_input_that_changes_the_result(provider):
    base = provider.cache_key(BBOX, BANDS)
    assert base == provider.cache_key(BBOX, BANDS), "must be stable"
    assert base != provider.cache_key((72.91, 19.0, 73.0, 19.1), BANDS), "AOI"
    assert base != provider.cache_key(BBOX, ["B04", "B08"]), "bands"
    assert base != provider.cache_key(BBOX, BANDS, days_back=90), "date window"
    assert base != provider.cache_key(BBOX, BANDS, max_cloud=50.0), "cloud threshold"

    coarse = CopernicusSentinelProvider("id", "secret", resolution_m=20.0)
    assert base != coarse.cache_key(BBOX, BANDS), "resolution"
