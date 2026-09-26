"""Sentinel-2 L2A retrieval from the Copernicus Data Space Ecosystem (Sentinel Hub APIs).

Scope, deliberately small (MVP): Sentinel-2, Level-2A, optical, one date, rectangle AOI. This is an
acquisition layer only -- it obtains a GeoTIFF and its provenance, then the existing SatQuery
pipeline does the analysis.

Endpoints:
  auth     https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
  catalog  https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search
  process  https://sh.dataspace.copernicus.eu/api/v1/process

Security: the client secret is sent only in the token request body, never logged, never returned,
never given to the frontend. Upstream error bodies are truncated and scrubbed before being quoted.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from satquery.providers import RetrievedScene, SceneMetadata
from satquery.providers.errors import (
    AreaInvalid,
    AreaTooLarge,
    AuthenticationFailed,
    CredentialsMissing,
    NoImageryFound,
    ProcessingFailed,
    ProviderTimeout,
    RasterUnreadable,
    RateLimited,
    RetrievalError,
)

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
COLLECTION = "sentinel-2-l2a"

# Band sets by what the question needs. Keys match satquery.agent.intents.find_target() targets.
# Order is the order written to the GeoTIFF, and the band descriptions make it explicit so
# satquery.imaging never has to assume (imaging.default_band_names would otherwise guess).
BAND_SETS = {
    "vegetation": ["B02", "B03", "B04", "B08"],  # NDVI needs red + NIR
    "water": ["B02", "B03", "B04", "B08"],       # NDWI needs green + NIR
    None: ["B02", "B03", "B04", "B08"],          # RGB + NIR: the pipeline's 4-band optical default
}
BAND_ROLE = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}
MAX_DIMENSION = 2500  # Process API caps output size; also keeps a scene inside settings.max_pixels


def bands_for_target(target: str | None) -> list[str]:
    """Which bands a question needs. Unknown targets get the 4-band optical default."""
    return list(BAND_SETS.get(target, BAND_SETS[None]))


def bbox_area_km2(bbox: tuple[float, float, float, float]) -> float:
    """Approximate area of a WGS84 bbox, latitude-corrected. Good enough for a size guard."""
    west, south, east, north = bbox
    mid = math.radians((south + north) / 2)
    return abs(east - west) * 111.32 * math.cos(mid) * abs(north - south) * 110.57


def evalscript(bands: list[str]) -> str:
    """Return the requested bands as float32, unscaled, so the pipeline sees real reflectance."""
    inputs = ", ".join(f'"{b}"' for b in bands)
    samples = ", ".join(f"sample.{b}" for b in bands)
    return (
        "//VERSION=3\n"
        "function setup() {\n"
        f"  return {{input: [{{bands: [{inputs}], units: 'REFLECTANCE'}}],\n"
        f"          output: {{bands: {len(bands)}, sampleType: 'FLOAT32'}}}};\n"
        "}\n"
        "function evaluatePixel(sample) {\n"
        f"  return [{samples}];\n"
        "}\n"
    )


def _scrub(text: str, limit: int = 300) -> str:
    """Truncate an upstream body and drop anything token-shaped before it reaches a user or log."""
    cleaned = " ".join(str(text).split())
    for marker in ("access_token", "Bearer ", "client_secret"):
        if marker in cleaned:
            return "(upstream error body withheld: it contained credential material)"
    return cleaned[:limit]


class _TokenCache:
    """One OAuth token per process, reused until just before it expires."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._lock = threading.Lock()

    def get(self, fetch) -> str:
        with self._lock:
            if self._token and time.time() < self._expires_at:
                return self._token
            token, expires_in = fetch()
            self._token = token
            # Refresh a minute early so a token cannot expire mid-request.
            self._expires_at = time.time() + max(float(expires_in) - 60.0, 30.0)
            return self._token

    def clear(self) -> None:
        with self._lock:
            self._token, self._expires_at = None, 0.0


class CopernicusSentinelProvider:
    """Sentinel-2 L2A from the Copernicus Data Space Ecosystem."""

    name = "Copernicus Data Space Ecosystem"

    def __init__(self, client_id: str, client_secret: str, *, days_back: int = 30,
                 max_cloud: float = 20.0, max_aoi_km2: float = 400.0, resolution_m: float = 10.0,
                 timeout: float = 90.0, cache_dir: Path | None = None):
        if not client_id or not client_secret:
            raise CredentialsMissing(
                "Copernicus credentials are not configured. Set COPERNICUS_CLIENT_ID and "
                "COPERNICUS_CLIENT_SECRET in the repository-root .env file.")
        self._client_id = client_id
        self._client_secret = client_secret
        self.days_back = days_back
        self.max_cloud = max_cloud
        self.max_aoi_km2 = max_aoi_km2
        self.resolution_m = resolution_m
        self.timeout = timeout
        self.cache_dir = cache_dir
        self._tokens = _TokenCache()

    # ----------------------------------------------------------------- http

    def _client(self):
        import httpx

        return httpx.Client(timeout=self.timeout)

    def _fetch_token(self) -> tuple[str, float]:
        import httpx

        try:
            with self._client() as client:
                response = client.post(TOKEN_URL, data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                })
        except httpx.TimeoutException as error:
            raise ProviderTimeout("Copernicus did not respond while signing in.") from error
        except httpx.HTTPError as error:
            raise AuthenticationFailed("Could not reach the Copernicus sign-in service.") from error

        if response.status_code in (400, 401, 403):
            raise AuthenticationFailed(
                "Copernicus rejected the credentials. Check COPERNICUS_CLIENT_ID and "
                "COPERNICUS_CLIENT_SECRET, and that the OAuth client is still active.",
                detail=_scrub(response.text))
        if response.status_code == 429:
            raise RateLimited("Copernicus is rate limiting sign-in requests. Try again shortly.")
        if response.status_code >= 400:
            raise AuthenticationFailed(f"Copernicus sign-in failed (HTTP {response.status_code}).",
                                       detail=_scrub(response.text))
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise AuthenticationFailed("Copernicus returned no access token.")
        return token, payload.get("expires_in", 600)

    def token(self) -> str:
        """A valid bearer token, reusing the cached one until it is close to expiry."""
        return self._tokens.get(self._fetch_token)

    def _post(self, url: str, payload: dict, *, what: str):
        import httpx

        try:
            with self._client() as client:
                response = client.post(url, json=payload,
                                       headers={"Authorization": f"Bearer {self.token()}"})
        except httpx.TimeoutException as error:
            raise ProviderTimeout(f"Copernicus timed out while {what}.") from error
        except httpx.HTTPError as error:
            raise RetrievalError(f"Could not reach Copernicus while {what}.") from error

        if response.status_code in (401, 403):
            # The cached token may have been revoked: drop it so the next call signs in again.
            self._tokens.clear()
            raise AuthenticationFailed(f"Copernicus refused the request while {what}.",
                                       detail=_scrub(response.text))
        if response.status_code == 429:
            raise RateLimited("Copernicus quota or rate limit reached. Try again later.",
                              detail=_scrub(response.text))
        if response.status_code >= 400:
            raise ProcessingFailed(f"Copernicus failed while {what} (HTTP {response.status_code}).",
                                   detail=_scrub(response.text))
        return response

    # ----------------------------------------------------------------- steps

    def validate_area(self, bbox: tuple[float, float, float, float]) -> None:
        if bbox is None or len(bbox) != 4:
            raise AreaInvalid("Draw a rectangle on the map to choose the area to analyse.")
        west, south, east, north = bbox
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
            raise AreaInvalid("The selected area has invalid coordinates.")
        if west >= east or south >= north:
            raise AreaInvalid("The selected area encloses no ground. Drag to draw a rectangle.")
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise AreaInvalid("The selected area lies outside valid longitude/latitude bounds.")
        area = bbox_area_km2(bbox)
        if area > self.max_aoi_km2:
            raise AreaTooLarge(
                f"The selected area is about {area:,.0f} km2, above the {self.max_aoi_km2:,.0f} km2 "
                "limit for a single request. Draw a smaller rectangle.")

    def search(self, bbox: tuple[float, float, float, float], *, days_back: int | None = None,
               max_cloud: float | None = None, limit: int = 50) -> list[dict]:
        """Catalogue scenes covering the area, newest window first."""
        days = self.days_back if days_back is None else days_back
        cloud = self.max_cloud if max_cloud is None else max_cloud
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        payload = {
            "bbox": list(bbox),
            "datetime": f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "collections": [COLLECTION],
            "limit": limit,
            # The property name must be quoted: an unquoted `eo:cloud_cover < N` is accepted with
            # HTTP 200 but silently matches nothing (verified against the live API, 2026-09-24).
            "filter": f'"eo:cloud_cover" < {cloud}',
            "filter-lang": "cql2-text",
        }
        features = self._post(CATALOG_URL, payload, what="searching the catalogue").json().get("features", [])
        return features

    @staticmethod
    def select_scene(features: list[dict]) -> dict:
        """Least cloudy first, then most recent. Deterministic, and the reason is reportable."""
        if not features:
            raise NoImageryFound(
                "No Sentinel-2 L2A scene matched this area within the search window and cloud "
                "limit. Try a longer date window, a higher cloud threshold, or a different area.")

        def key(feature: dict):
            properties = feature.get("properties", {})
            cloud = properties.get("eo:cloud_cover")
            return (cloud if isinstance(cloud, (int, float)) else 101.0,
                    # newer wins on a cloud tie
                    _negated_timestamp(properties.get("datetime", "")))

        return sorted(features, key=key)[0]

    def _process(self, bbox: tuple[float, float, float, float], bands: list[str],
                 acquired_date: str) -> bytes:
        west, south, east, north = bbox
        width, height = self._output_size(bbox)
        payload = {
            "input": {
                "bounds": {"bbox": list(bbox),
                           "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                "data": [{
                    "type": COLLECTION,
                    "dataFilter": {
                        # One day, so the raster is the scene that was selected, not a mosaic
                        # spanning dates -- provenance must describe exactly what was analysed.
                        "timeRange": {"from": f"{acquired_date}T00:00:00Z",
                                      "to": f"{acquired_date}T23:59:59Z"},
                        "mosaickingOrder": "leastCC",
                    },
                }],
            },
            "output": {"width": width, "height": height,
                       "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": evalscript(bands),
        }
        response = self._post(PROCESS_URL, payload, what="preparing the imagery")
        content = response.content
        if not content:
            raise ProcessingFailed("Copernicus returned an empty image for this area.")
        return content

    def _output_size(self, bbox: tuple[float, float, float, float]) -> tuple[int, int]:
        west, south, east, north = bbox
        mid = math.radians((south + north) / 2)
        metres_x = abs(east - west) * 111_320 * math.cos(mid)
        metres_y = abs(north - south) * 110_570
        width = max(1, min(MAX_DIMENSION, round(metres_x / self.resolution_m)))
        height = max(1, min(MAX_DIMENSION, round(metres_y / self.resolution_m)))
        return width, height

    # ----------------------------------------------------------------- public

    def cache_key(self, bbox, bands: list[str], *, days_back: int | None = None,
                  max_cloud: float | None = None) -> str:
        """Identity of a request: area, collection, window, cloud limit, resolution, bands."""
        material = json.dumps({
            "bbox": [round(float(v), 6) for v in bbox],
            "collection": COLLECTION,
            "days_back": self.days_back if days_back is None else days_back,
            "max_cloud": self.max_cloud if max_cloud is None else max_cloud,
            "resolution_m": self.resolution_m,
            "bands": list(bands),
        }, sort_keys=True)
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def retrieve(self, bbox_wgs84, bands: list[str], destination: Path, *,
                 days_back: int | None = None, max_cloud: float | None = None) -> RetrievedScene:
        """Search, select, download and write a GeoTIFF. Raises a typed RetrievalError on failure."""
        bbox = tuple(float(v) for v in bbox_wgs84)
        self.validate_area(bbox)

        features = self.search(bbox, days_back=days_back, max_cloud=max_cloud)
        scene = self.select_scene(features)
        properties = scene.get("properties", {})
        acquired_datetime = str(properties.get("datetime", ""))
        acquired = acquired_datetime[:10]
        if not acquired:
            raise NoImageryFound("The selected scene has no acquisition date; cannot retrieve it.")

        content = self._process(bbox, bands, acquired)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        width, height = self._label_bands(destination, bands)

        cloud = properties.get("eo:cloud_cover")
        metadata = SceneMetadata(
            provider=self.name, collection=COLLECTION, satellite="Sentinel-2", product_level="L2A",
            acquired=acquired, acquired_datetime=acquired_datetime,
            cloud_cover=round(float(cloud), 2) if isinstance(cloud, (int, float)) else None,
            bbox_wgs84=bbox, crs="EPSG:4326", resolution_m=self.resolution_m,
            bands=[f"{b} ({BAND_ROLE.get(b, b)})" for b in bands],
            width=width, height=height, scene_id=scene.get("id"),
            alternatives_considered=len(features),
        )
        return RetrievedScene(path=destination, metadata=metadata)

    @staticmethod
    def _label_bands(path: Path, bands: list[str]) -> tuple[int, int]:
        """Write band descriptions so satquery.imaging reads roles instead of assuming them.

        Without this, imaging.default_band_names guesses blue/green/red/nir by position and sets
        band_names_assumed, which validation then warns about. Naming them removes the guess.
        """
        import rasterio

        try:
            with rasterio.open(path, "r+") as raster:
                width, height = raster.width, raster.height
                if raster.count != len(bands):
                    raise RasterUnreadable(
                        f"Copernicus returned {raster.count} band(s) but {len(bands)} were requested.")
                for index, band in enumerate(bands, start=1):
                    raster.set_band_description(index, BAND_ROLE.get(band, band))
            return width, height
        except RetrievalError:
            raise
        except Exception as error:
            raise RasterUnreadable(
                "The imagery Copernicus returned could not be read as a GeoTIFF.") from error


def _negated_timestamp(value: str) -> float:
    """Sort key making newer timestamps sort first."""
    try:
        return -datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return 0.0
