"""Sentinel-2 L2A (and Sentinel-1 GRD) retrieval from the Copernicus Data Space Ecosystem (Sentinel Hub APIs).

Scope, deliberately small (MVP): rectangle AOI; one Sentinel-2 date, two dates for a question about
change over time (see providers/temporal.py), or a Sentinel-2 scene plus the Sentinel-1 scene
closest to it in time for a question asking for optical and SAR together. This is an acquisition
layer only -- it obtains GeoTIFFs and their provenance, then the existing SatQuery pipeline does
the analysis.

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
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from satquery.providers import RetrievedScene, SceneMetadata
from satquery.providers.temporal import TemporalWindows
from satquery.providers.errors import (
    AreaInvalid,
    AreaTooLarge,
    AuthenticationFailed,
    CredentialsMissing,
    GridsIncompatible,
    NoEarlierImagery,
    NoImageryFound,
    NoLaterImagery,
    NoSarImagery,
    OnlyOneAcquisition,
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

# Sentinel-1 for questions asking for optical and SAR together. The optical scene is chosen first
# (clouds constrain it; radar sees through them), then the SAR scene closest to it in time.
S1_COLLECTION = "sentinel-1-grd"
S1_BANDS = ["VV", "VH"]  # IW dual polarisation, the standard land acquisition (checked live, 2026-09-27)
# Sentinel-1 revisits a site every 6-12 days, so 12 days finds a scene on any single relative orbit;
# a wider gap would pair ground conditions that may have changed, and the gap is always reported.
SAR_MAX_DAYS_APART = 12
SAR_PROCESSING = ("Sentinel Hub Process API: the selected day only (no multi-date mosaic), IW mode, dual "
                  "polarisation VV+VH, orthorectified with the Copernicus 30 m DEM, sigma0 (ellipsoid) backscatter "
                  "as FLOAT32 linear power, no speckle filter, resampled to the same grid as the optical scene")


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


def sar_evalscript() -> str:
    """VV and VH backscatter as FLOAT32 linear power, which satquery.imaging.sar_db converts to dB."""
    return (
        "//VERSION=3\n"
        "function setup() {\n"
        "  return {input: [{bands: [\"VV\", \"VH\"], units: 'LINEAR_POWER'}],\n"
        "          output: {bands: 2, sampleType: 'FLOAT32'}};\n"
        "}\n"
        "function evaluatePixel(sample) {\n"
        "  return [sample.VV, sample.VH];\n"
        "}\n"
    )


def scl_evalscript() -> str:
    """Sentinel-2 L2A scene classification (SCL) as one UINT8 band: one class per pixel (cloud,
    cloud shadow, water, ...). Resampled by nearest neighbour, so classes stay classes."""
    return (
        "//VERSION=3\n"
        "function setup() {\n"
        "  return {input: [{bands: [\"SCL\"]}],\n"
        "          output: {bands: 1, sampleType: 'UINT8'}};\n"
        "}\n"
        "function evaluatePixel(sample) {\n"
        "  return [sample.SCL];\n"
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
               max_cloud: float | None = None, limit: int = 50,
               start: date | None = None, end: date | None = None) -> list[dict]:
        """Catalogue scenes covering the area: the last `days_back` days, or the dates start..end inclusive."""
        cloud = self.max_cloud if max_cloud is None else max_cloud
        if start is not None and end is not None:
            window = f"{start:%Y-%m-%d}T00:00:00Z/{end:%Y-%m-%d}T23:59:59Z"
        else:
            days = self.days_back if days_back is None else days_back
            now = datetime.now(timezone.utc)
            window = f"{(now - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')}/{now.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        payload = {
            "bbox": list(bbox),
            "datetime": window,
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
        return self._render(bbox, self._sentinel2_day(acquired_date), evalscript(bands))

    @staticmethod
    def _sentinel2_day(acquired_date: str) -> dict:
        """The Sentinel-2 L2A data of one day. Shared by the bands and the scene classification, so
        both come from the same tiles in the same order and describe the same pixels."""
        return {
            "type": COLLECTION,
            "dataFilter": {
                # One day, so the raster is the scene that was selected, not a mosaic
                # spanning dates -- provenance must describe exactly what was analysed.
                "timeRange": {"from": f"{acquired_date}T00:00:00Z",
                              "to": f"{acquired_date}T23:59:59Z"},
                "mosaickingOrder": "leastCC",
            },
        }

    def retrieve_scene_classification(self, bbox_wgs84, acquired_date: str, destination: Path) -> Path:
        """Sentinel-2 L2A's scene classification (cloud, cloud shadow, water, ...) for the selected area
        on `acquired_date`, on exactly the grid of that day's bands (D-030)."""
        bbox = tuple(float(v) for v in bbox_wgs84)
        content = self._render(bbox, self._sentinel2_day(acquired_date), scl_evalscript())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return destination

    def _render(self, bbox: tuple[float, float, float, float], data: dict, script: str) -> bytes:
        """One Process API request for the area. The output grid depends on the bbox only, so every
        collection rendered for the same area lands on the same pixel grid."""
        width, height = self._output_size(bbox)
        payload = {
            "input": {
                "bounds": {"bbox": list(bbox),
                           "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                "data": [data],
            },
            "output": {"width": width, "height": height,
                       "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": script,
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
        return self._download(bbox, bands, scene, destination, alternatives=len(features))

    def _download(self, bbox, bands: list[str], scene: dict, destination: Path, *, alternatives: int) -> RetrievedScene:
        """Fetch one catalogue scene for the area, write it as a GeoTIFF, and describe where it came from."""
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
            alternatives_considered=alternatives,
        )
        return RetrievedScene(path=destination, metadata=metadata)

    # ----------------------------------------------------------------- two dates

    @staticmethod
    def select_pair(before_features: list[dict], after_features: list[dict],
                    max_cloud: float | None = None) -> tuple[dict, dict]:
        """One scene from each window: two different acquisition dates, earlier before later.

        Policy: cloud cover counts in 5-point steps, so 0.01% and 0.3% are equally clear; among
        equally clear pairs the one furthest apart in time wins. A scene is never used twice, and two
        tiles of the same pass (same day) count as one acquisition.
        """
        def candidates(features: list[dict]) -> dict[str, dict]:
            by_day: dict[str, dict] = {}
            for feature in features:
                properties = feature.get("properties", {})
                day = str(properties.get("datetime", ""))[:10]
                cloud = properties.get("eo:cloud_cover")
                if not day or not isinstance(cloud, (int, float)):
                    continue
                if max_cloud is not None and cloud >= max_cloud:
                    continue  # the catalogue filters this already; checked again so the rule cannot drift
                if day not in by_day or cloud < by_day[day]["properties"]["eo:cloud_cover"]:
                    by_day[day] = feature
            return by_day

        before, after = candidates(before_features), candidates(after_features)
        if not before and not after:
            raise NoImageryFound(
                "No Sentinel-2 L2A scene under the cloud limit was found for this area in either period, so a "
                "temporal comparison cannot be performed. Try a longer period, a higher cloud limit, or another area.")
        if not before:
            raise NoEarlierImagery(
                "No suitable Sentinel-2 L2A scene was found for this area in the earlier period, so the earlier date "
                "could not be retrieved. Try an earlier or longer period, or a higher cloud limit.")
        if not after:
            raise NoLaterImagery(
                "No suitable Sentinel-2 L2A scene was found for this area in the later period, so the later date "
                "could not be retrieved. Try a longer period, or a higher cloud limit.")

        def step(feature: dict) -> int:
            return math.ceil(float(feature["properties"]["eo:cloud_cover"]) / 5)

        pairs = [(b_day, a_day) for b_day in before for a_day in after if b_day < a_day]
        if not pairs:
            raise OnlyOneAcquisition(
                "Only one suitable Sentinel-2 acquisition was found for this area and time window, so a temporal "
                "comparison cannot be performed. Name two periods further apart, or a longer period.")
        b_day, a_day = min(pairs, key=lambda p: (step(before[p[0]]) + step(after[p[1]]),
                                                 -(date.fromisoformat(p[1]) - date.fromisoformat(p[0])).days))
        return before[b_day], after[a_day]

    def pair_cache_key(self, bbox, bands: list[str], windows: TemporalWindows, *, max_cloud: float | None = None) -> str:
        """Identity of a two-date request. "mode" keeps it apart from every single-date key."""
        material = json.dumps({
            "mode": "temporal",
            "bbox": [round(float(v), 6) for v in bbox],
            "collection": COLLECTION,
            "before_window": [windows.before.start.isoformat(), windows.before.end.isoformat()],
            "after_window": [windows.after.start.isoformat(), windows.after.end.isoformat()],
            "max_cloud": self.max_cloud if max_cloud is None else max_cloud,
            "resolution_m": self.resolution_m,
            "bands": list(bands),
        }, sort_keys=True)
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def retrieve_pair(self, bbox_wgs84, bands: list[str], windows: TemporalWindows, destination_dir: Path, *,
                      max_cloud: float | None = None) -> tuple[RetrievedScene, RetrievedScene]:
        """Two real acquisitions of the same area, earlier first, on one pixel grid.

        Both are requested for the same bounds and output size, so they share a grid by construction;
        that is still verified on the files, and a mismatch stops the comparison rather than resampling.
        """
        bbox = tuple(float(v) for v in bbox_wgs84)
        self.validate_area(bbox)
        cloud = self.max_cloud if max_cloud is None else max_cloud
        before_features = self.search(bbox, start=windows.before.start, end=windows.before.end, max_cloud=cloud)
        after_features = self.search(bbox, start=windows.after.start, end=windows.after.end, max_cloud=cloud)
        before_scene, after_scene = self.select_pair(before_features, after_features, cloud)

        jobs = [(before_scene, destination_dir / "before.tif", len(before_features)),
                (after_scene, destination_dir / "after.tif", len(after_features))]
        # The two downloads are independent, so they run side by side rather than one after the other.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self._download, bbox, bands, scene, path, alternatives=n) for scene, path, n in jobs]
            before, after = (future.result() for future in futures)
        check_same_grid(before.path, after.path)
        return before, after

    # ----------------------------------------------------------------- optical + SAR

    def search_sar(self, bbox, *, start: date, end: date, limit: int = 50) -> list[dict]:
        """Sentinel-1 GRD scenes in IW mode covering the area, dates start..end inclusive."""
        payload = {
            "bbox": list(bbox),
            "datetime": f"{start:%Y-%m-%d}T00:00:00Z/{end:%Y-%m-%d}T23:59:59Z",
            "collections": [S1_COLLECTION],
            "limit": limit,
            "filter": "\"sar:instrument_mode\" = 'IW'",
            "filter-lang": "cql2-text",
        }
        return self._post(CATALOG_URL, payload, what="searching the Sentinel-1 catalogue").json().get("features", [])

    @staticmethod
    def select_sar_scene(features: list[dict], optical_date: date, max_days_apart: int = SAR_MAX_DAYS_APART) -> dict:
        """The dual-polarisation (VV+VH) scene closest in time to the optical acquisition.

        Ties go to the earlier scene, so the choice is deterministic and reportable.
        """
        def gap(feature: dict) -> int | None:
            try:
                return (date.fromisoformat(str(feature["properties"]["datetime"])[:10]) - optical_date).days
            except (KeyError, ValueError):
                return None

        candidates = [f for f in features
                      if f.get("properties", {}).get("s1:polarization") == "DV"
                      and gap(f) is not None and abs(gap(f)) <= max_days_apart]
        if not candidates:
            raise NoSarImagery(
                f"No Sentinel-1 dual-polarisation (VV+VH) scene of this area was found within {max_days_apart} days of "
                f"the optical scene ({optical_date.isoformat()}), so no optical + SAR pair could be formed. Try "
                "another area, or upload a co-registered SAR GeoTIFF.")
        return min(candidates, key=lambda f: (abs(gap(f)), gap(f)))

    def optical_sar_cache_key(self, bbox, bands: list[str], *, days_back: int | None = None,
                              max_cloud: float | None = None) -> str:
        """Identity of an optical + SAR request. "mode" keeps it apart from single-date and temporal keys."""
        material = json.dumps({
            "mode": "optical_sar",
            "bbox": [round(float(v), 6) for v in bbox],
            "collections": [COLLECTION, S1_COLLECTION],
            "days_back": self.days_back if days_back is None else days_back,
            "max_cloud": self.max_cloud if max_cloud is None else max_cloud,
            "max_days_apart": SAR_MAX_DAYS_APART,
            "resolution_m": self.resolution_m,
            "bands": list(bands) + S1_BANDS,
        }, sort_keys=True)
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def retrieve_optical_sar(self, bbox_wgs84, bands: list[str], destination_dir: Path, *,
                             days_back: int | None = None, max_cloud: float | None = None
                             ) -> tuple[RetrievedScene, RetrievedScene]:
        """A Sentinel-2 scene and the Sentinel-1 scene closest to it in time, on one pixel grid.

        Both are rendered for the same bounds and output size, so they share a grid by construction;
        that is still verified on the files, and a mismatch stops the analysis rather than resampling.
        """
        bbox = tuple(float(v) for v in bbox_wgs84)
        self.validate_area(bbox)
        optical_features = self.search(bbox, days_back=days_back, max_cloud=max_cloud)
        optical_scene = self.select_scene(optical_features)
        try:
            optical_date = date.fromisoformat(str(optical_scene.get("properties", {}).get("datetime", ""))[:10])
        except ValueError as error:
            raise NoImageryFound("The selected scene has no acquisition date; cannot retrieve it.") from error
        today = datetime.now(timezone.utc).date()
        sar_features = self.search_sar(bbox, start=optical_date - timedelta(days=SAR_MAX_DAYS_APART),
                                       end=min(optical_date + timedelta(days=SAR_MAX_DAYS_APART), today))
        sar_scene = self.select_sar_scene(sar_features, optical_date)

        with ThreadPoolExecutor(max_workers=2) as pool:
            optical_job = pool.submit(self._download, bbox, bands, optical_scene, destination_dir / "optical.tif",
                                      alternatives=len(optical_features))
            sar_job = pool.submit(self._download_sar, bbox, sar_scene, destination_dir / "sar.tif",
                                  alternatives=len(sar_features))
            optical, sar = optical_job.result(), sar_job.result()
        check_same_grid(optical.path, sar.path, compare_band_count=False)
        return optical, sar

    def retrieve_sar_near(self, bbox_wgs84, optical_date: date, destination: Path) -> RetrievedScene:
        """The Sentinel-1 VV+VH scene closest in time to an optical acquisition, within
        SAR_MAX_DAYS_APART: the radar fallback for an optical scene too cloudy to use (D-030)."""
        bbox = tuple(float(v) for v in bbox_wgs84)
        today = datetime.now(timezone.utc).date()
        features = self.search_sar(bbox, start=optical_date - timedelta(days=SAR_MAX_DAYS_APART),
                                   end=min(optical_date + timedelta(days=SAR_MAX_DAYS_APART), today))
        return self._download_sar(bbox, self.select_sar_scene(features, optical_date), destination,
                                  alternatives=len(features))

    @staticmethod
    def select_latest_sar(features: list[dict], days_back: int) -> dict:
        """The most recent dual-polarisation (VV+VH) scene."""
        candidates = [f for f in features if f.get("properties", {}).get("s1:polarization") == "DV"
                      and str(f.get("properties", {}).get("datetime", ""))[:10]]
        if not candidates:
            raise NoSarImagery(
                f"No Sentinel-1 dual-polarisation (VV+VH) scene of this area was found in the last {days_back} days. "
                "Try another area, or upload a SAR GeoTIFF.")
        return max(candidates, key=lambda f: str(f["properties"]["datetime"]))

    def sar_cache_key(self, bbox, *, days_back: int | None = None) -> str:
        """Identity of a radar-only request. "mode" keeps it apart from every other key."""
        material = json.dumps({
            "mode": "sar",
            "bbox": [round(float(v), 6) for v in bbox],
            "collection": S1_COLLECTION,
            "days_back": self.days_back if days_back is None else days_back,
            "resolution_m": self.resolution_m,
            "bands": S1_BANDS,
        }, sort_keys=True)
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def retrieve_sar(self, bbox_wgs84, destination: Path, *, days_back: int | None = None) -> RetrievedScene:
        """The most recent Sentinel-1 VV+VH scene of the area, for a question that asks for radar
        alone ("use radar to find the water"). No optical scene is involved (D-030)."""
        bbox = tuple(float(v) for v in bbox_wgs84)
        self.validate_area(bbox)
        days = self.days_back if days_back is None else days_back
        today = datetime.now(timezone.utc).date()
        features = self.search_sar(bbox, start=today - timedelta(days=days), end=today)
        return self._download_sar(bbox, self.select_latest_sar(features, days), destination,
                                  alternatives=len(features))

    def _download_sar(self, bbox, scene: dict, destination: Path, *, alternatives: int) -> RetrievedScene:
        properties = scene.get("properties", {})
        acquired_datetime = str(properties.get("datetime", ""))
        acquired = acquired_datetime[:10]
        if not acquired:
            raise NoSarImagery("The selected Sentinel-1 scene has no acquisition date; cannot retrieve it.")
        content = self._render(bbox, {
            "type": S1_COLLECTION,
            "dataFilter": {"timeRange": {"from": f"{acquired}T00:00:00Z", "to": f"{acquired}T23:59:59Z"},
                           "acquisitionMode": "IW", "polarization": "DV", "resolution": "HIGH"},
            "processing": {"orthorectify": True, "demInstance": "COPERNICUS_30", "backCoeff": "SIGMA0_ELLIPSOID"},
        }, sar_evalscript())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        width, height = self._label_bands(destination, S1_BANDS)
        platform = str(properties.get("platform", ""))  # e.g. "sentinel-1d", as the catalogue reports it
        satellite = f"Sentinel-{platform.split('-', 1)[1].upper()}" if platform.startswith("sentinel-") else "Sentinel-1"
        metadata = SceneMetadata(
            provider=self.name, collection=S1_COLLECTION, satellite=satellite, product_level="GRD",
            acquired=acquired, acquired_datetime=acquired_datetime, cloud_cover=None,
            bbox_wgs84=bbox, crs="EPSG:4326", resolution_m=self.resolution_m,
            bands=["VV (co-pol)", "VH (cross-pol)"], width=width, height=height, scene_id=scene.get("id"),
            alternatives_considered=alternatives, processing=SAR_PROCESSING, modality="sar",
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


def check_same_grid(first: Path, second: Path, *, compare_band_count: bool = True) -> None:
    """Raise GridsIncompatible unless both rasters share size, CRS, transform and (by default) band count.

    An optical + SAR pair legitimately differs in band count, so it is compared on the grid alone.
    """
    import rasterio

    try:
        with rasterio.open(first) as a, rasterio.open(second) as b:
            grids = [(r.width, r.height, r.count, r.crs.to_string() if r.crs else None, tuple(r.transform)[:6])
                     for r in (a, b)]
    except Exception as error:
        raise RasterUnreadable("One of the two retrieved rasters could not be read as a GeoTIFF.") from error
    (w1, h1, n1, crs1, t1), (w2, h2, n2, crs2, t2) = grids
    problems = []
    if (w1, h1) != (w2, h2):
        problems.append(f"size {w1}x{h1} vs {w2}x{h2}")
    if compare_band_count and n1 != n2:
        problems.append(f"{n1} vs {n2} bands")
    if crs1 != crs2:
        problems.append(f"CRS {crs1} vs {crs2}")
    if any(abs(x - y) > 1e-9 for x, y in zip(t1, t2)):
        problems.append("different georeferencing")
    if problems:
        raise GridsIncompatible(
            "The two retrieved scenes do not share a pixel grid (" + "; ".join(problems) + "), so no pixel-wise "
            "analysis was run. They are not resampled onto each other silently.")
