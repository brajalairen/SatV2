"""HTTP layer for the map-first web client (D-023).

Presentation only, the same tier as `ui.py` and `cli.py`: it builds an `AnalysisRequest`, calls the
injected analysis function, and rewrites on-disk artifact paths into URLs the browser can fetch. It
never imports specialists or the agent directly.

Run it locally with the labelled fake model:
    SATQUERY_VLM_BACKEND=fake uvicorn satquery.server:app --reload
"""

import json
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from satquery import geo
from satquery.agent.intents import (find_target, needs_multiple_dates, needs_optical_and_sar, needs_sar_only,
                                    route_query)
from satquery.api import analyze, answer_weather
from satquery.evidence import save_png
from satquery.examples import EXAMPLE_QUERIES, EXAMPLES_DIR, load_scenarios
from satquery.imaging import SUPPORTED_SUFFIXES, detect_modality, load_image, render_rgb
from satquery.schemas import (AnalysisRequest, AnalysisResponse, ImageInput, ImageSummary,
                              Modality, TaskType)
from satquery.settings import Settings, load_settings

# Vite's dev server; the production build is served from this same origin and needs no CORS.
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


# --------------------------------------------------------------------------- upload registry


@dataclass
class Upload:
    """One raster the browser has put on the map. `path` is what the pipeline consumes."""

    id: str
    path: Path
    name: str
    modality: Modality
    acquired: str | None
    summary: ImageSummary
    preview: Path | None = None


@dataclass
class UploadStore:
    """In-process registry. Uploads live for the lifetime of the server, like Gradio's temp files."""

    directory: Path
    _items: dict[str, Upload] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, upload: Upload) -> Upload:
        with self._lock:
            self._items[upload.id] = upload
        return upload

    def get(self, upload_id: str) -> Upload:
        with self._lock:
            upload = self._items.get(upload_id)
        if upload is None:
            raise HTTPException(404, f"unknown upload '{upload_id}'")
        return upload


# --------------------------------------------------------------------------- wire format


class UploadInfo(BaseModel):
    """What the client needs to draw one raster: identity, metadata and map placement."""

    id: str
    name: str
    modality: Modality
    acquired: str | None
    summary: ImageSummary
    preview_url: str
    mappable: bool  # False for PNG/JPEG and CRS-less TIFFs: the client shows those off-map
    # How `modality` was decided, shown beside it: read from the file's band descriptions, assumed,
    # or declared by the caller. None where the source fixes it (demo scenarios, retrieved scenes).
    modality_basis: str | None = None


class AnalyzeImage(BaseModel):
    upload_id: str
    modality: Modality | None = None  # overrides what was set at upload time
    acquired: str | None = None


class AreaGeometry(BaseModel):
    """A drawn area as GeoJSON in WGS84. Rectangles, circles and polygons all arrive as polygons."""

    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list

    @model_validator(mode="after")
    def _usable(self) -> "AreaGeometry":
        problem = geo.geometry_problem(self.model_dump())
        if problem:
            raise ValueError(problem)
        return self


class AnalyzeRequest(BaseModel):
    query: str
    images: list[AnalyzeImage] = Field(min_length=1, max_length=2)
    forced_task: TaskType | None = None
    # A drawn area, if any. The analysis is restricted to the part of each image inside it.
    aoi_bbox: tuple[float, float, float, float] | None = None  # west, south, east, north
    # The drawn shape itself. When present it wins over `aoi_bbox`: a circle or polygon is analysed
    # as that shape, not as its bounding box.
    aoi_geometry: AreaGeometry | None = None


class OverlayLayer(BaseModel):
    """An evidence PNG that shares a pixel grid with one input image, so a map can pin it."""

    url: str
    label: str
    image_index: int
    corners_wgs84: list[tuple[float, float]]
    approximate: bool


class AreaScope(BaseModel):
    """Whether the drawn area actually narrowed the analysis, and if not, why not.

    The client states this on the result, so a selected area is never silently ignored.
    """

    applied: bool
    reason: str | None = None
    width: int | None = None
    height: int | None = None
    source_width: int | None = None
    source_height: int | None = None
    masked: bool = False  # pixels outside a drawn circle or polygon were excluded, not just cropped


class WeatherInfo(BaseModel):
    """Where and when a forecast came from, for the result card and the map marker (D-029)."""

    provider: str
    model: str | None = None
    attribution: str
    attribution_url: str
    area_source: str  # "drawn area" or "image footprint"
    area_bbox_wgs84: tuple[float, float, float, float]
    area_extent_km: tuple[float, float]  # east-west, north-south
    point_wgs84: tuple[float, float]  # (longitude, latitude) forecast for: the marker on the map
    grid_point_wgs84: tuple[float, float] | None = None  # the model grid point the provider answered for
    elevation_m: float | None = None
    timezone: str | None = None
    period: tuple[str, str] | None = None  # local dates
    retrieved_at: str | None = None
    cached: bool = False


class AnalyzeResult(BaseModel):
    """`AnalysisResponse` with artifact paths rewritten as URLs, plus per-overlay map placement."""

    response: AnalysisResponse
    overlay_layers: list[OverlayLayer]
    area: AreaScope | None = None  # present only when the request carried a drawn area
    # The uploads this result ran on, in input order: how the client knows which layers it describes.
    upload_ids: list[str] = Field(default_factory=list)
    weather: WeatherInfo | None = None  # present only for a weather answer


class RouteRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class RouteResult(BaseModel):
    """Which specialist a question is for. Wording only: nothing is retrieved or planned here."""

    route: Literal["weather", "imagery", "mixed"]
    rule: str
    message: str | None = None  # for "mixed": what to do instead


class WeatherRequest(BaseModel):
    """A weather question about an area. Any drawn shape works: only a point inside it is forecast."""

    query: str = Field(min_length=1, max_length=2000)
    aoi_geometry: AreaGeometry | None = None
    aoi_bbox: tuple[float, float, float, float] | None = None  # west, south, east, north
    area_source: Literal["drawn area", "image footprint"] = "drawn area"


class FetchImageryRequest(BaseModel):
    """Retrieve imagery for a drawn area so the user need not upload a GeoTIFF.

    The query is carried so the band set can suit the question, and so a question about change over
    time retrieves two real acquisitions instead of being answered from a single scene.
    """

    query: str = Field(min_length=1, max_length=2000)
    aoi_bbox: tuple[float, float, float, float]  # west, south, east, north
    days_back: int | None = Field(default=None, ge=1, le=365)  # single-date search window only
    max_cloud: float | None = Field(default=None, ge=0, le=100)


class FetchedScene(BaseModel):
    """One retrieved scene, registered as an upload. `role` places it in a comparison or a sensor pair."""

    role: Literal["single", "before", "after", "optical", "sar"]
    upload: UploadInfo
    metadata: dict  # SceneMetadata: provider, satellite, date, cloud cover, bands, CRS, processing, ...


class ComparisonWindow(BaseModel):
    """A period searched for one of the two scenes. Never an acquisition date: see the scene metadata."""

    start: str
    end: str
    label: str


class TemporalInfo(BaseModel):
    """How the two dates of a comparison were chosen (policy: satquery/providers/temporal.py)."""

    basis: str
    explanation: str
    before_window: ComparisonWindow
    after_window: ComparisonWindow
    days_apart: int  # between the two acquisitions actually retrieved


class CrossModalInfo(BaseModel):
    """How the SAR scene was matched to the optical one (policy: satquery/providers/copernicus.py)."""

    days_apart: int  # between the two acquisitions actually retrieved
    max_days_apart: int
    explanation: str


class OpticalQualityInfo(BaseModel):
    """How much of the selected area an optical scene shows, by Sentinel-2's own scene classification,
    and whether it was usable for a water question (D-030). Never the catalogue's tile cloud cover."""

    scene: dict  # SceneMetadata of the optical scene that was assessed
    pixels: int
    clear_pixels: int
    affected_fraction: float  # cloud, cloud shadow or no data, over the selected area
    class_fractions: dict[str, float]
    max_affected_fraction: float  # the configured limit (SATQUERY_OPTICAL_MAX_AFFECTED, a heuristic)
    min_clear_pixels: int
    usable: bool
    reason: str | None = None  # why the optical scene was not used
    method: str
    masked: bool = False  # the affected pixels were left out of the optical analysis


class FetchImageryResult(BaseModel):
    """Retrieved imagery, registered as ordinary uploads the existing pipeline can analyse.

    `mode` is "single" for one scene, "temporal" for a question about change over time,
    "cross_modal" for a question asking for optical and SAR together, "sar" for a question asking for
    radar alone, and "sar_fallback" when a water question's optical scene was too obscured to use and
    the nearest Sentinel-1 scene answers instead. `images` lists every scene to analyse: oldest first,
    or optical then SAR. `upload` and `metadata` are the most recent scene (the optical one for a
    sensor pair), as they were before multi-scene retrieval existed, so a single-date client reads
    them unchanged. `optical_quality` is present whenever a water question's optical scene was assessed.
    """

    mode: Literal["single", "temporal", "cross_modal", "sar", "sar_fallback"] = "single"
    upload: UploadInfo
    metadata: dict  # SceneMetadata: provider, satellite, date, cloud cover, bands, CRS, ...
    images: list[FetchedScene] = Field(default_factory=list)
    temporal: TemporalInfo | None = None
    cross_modal: CrossModalInfo | None = None
    optical_quality: OpticalQualityInfo | None = None
    cached: bool = False


class Example(BaseModel):
    index: int
    label: str
    query: str
    images: list[dict]


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    vlm_backend: str
    device: str
    model_is_fake: bool  # surfaced in the UI: a fake backend must never look like the real model
    # Whether imagery can be fetched for a drawn area. A boolean only: no credential value is ever
    # exposed by this endpoint, and the frontend needs nothing more than "is the feature available".
    imagery_provider: str | None = None
    imagery_available: bool = False
    # The same for weather forecasts (optional capability): a name and a boolean, never a credential.
    weather_provider: str | None = None
    weather_available: bool = False


# --------------------------------------------------------------------------- helpers


def _uploads_dir(settings: Settings) -> Path:
    directory = settings.runs_dir.parent / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


_UNSAFE_NAME = re.compile(r"[^\w.() -]+")
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def _stored_name(filename: str | None, suffix: str) -> str:
    """The user's file name, made safe to store on any OS, with the validated extension.

    Each stored file gets its own folder, so the pipeline sees the name the user chose: that name
    appears in the execution trace and the reports, where a random id would mean nothing.
    """
    stem = _UNSAFE_NAME.sub("_", Path(filename or "upload").stem).strip(" .") or "upload"
    if stem.upper() in _RESERVED_NAMES:
        stem = f"_{stem}"
    return f"{stem[:80]}{suffix}"


def _new_folder(parent: Path) -> Path:
    folder = parent / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    return folder


def _upload_modality(path: Path, declared: Modality | None) -> tuple[Modality, str]:
    """The declared modality, else the one the file's band descriptions state, else optical (said so).

    A SAR file uploaded as optical would pair with an optical image as a bi-temporal pair, so an
    undeclared upload is never silently assumed when the file itself says what it is.
    """
    if declared:
        return declared, "declared at upload"
    try:
        detected, names = detect_modality(path)
    except Exception:  # unreadable: _register reports it when it loads the raster
        detected, names = None, []
    if detected:
        return detected, f"from band descriptions {', '.join(names)}"
    return "optical", "assumed: the file does not name its bands; switch to SAR if it is radar"


def _register(store: UploadStore, source: Path, name: str, modality: Modality,
              acquired: str | None, settings: Settings, modality_basis: str | None = None) -> UploadInfo:
    """Load a raster, summarise it with map placement, and remember it for later analysis."""
    try:
        image = load_image(source, modality, acquired, settings.max_pixels)
    except Exception as error:
        raise HTTPException(400, f"cannot read '{name}': {error}") from error

    upload = store.add(Upload(id=uuid.uuid4().hex[:12], path=source, name=name, modality=modality,
                              acquired=acquired, summary=geo.summarize(image, 0)))
    preview = store.directory / f"{upload.id}-preview.png"
    # Transparent where the raster has no data (e.g. masked cloud), not black; unchanged when all is valid.
    save_png(render_rgb(image), preview, np.isfinite(image.data).any(axis=0))
    upload.preview = preview
    return UploadInfo(id=upload.id, name=name, modality=modality, acquired=acquired,
                      summary=upload.summary, preview_url=f"/api/uploads/{upload.id}/preview.png",
                      mappable=upload.summary.corners_wgs84 is not None, modality_basis=modality_basis)


def _safe_child(root: Path, *parts: str) -> Path:
    """Resolve `parts` under `root`, refusing anything that escapes it."""
    target = (root / Path(*parts)).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise HTTPException(404, "not found")
    return target


def _to_urls(response: AnalysisResponse) -> AnalysisResponse:
    """Rewrite absolute artifact paths as `/api/runs/...` URLs, in place of the local filesystem."""
    run_id = response.trace.run_id
    to_url = lambda path: f"/api/runs/{run_id}/{Path(path).name}"
    for item in response.evidence:
        if item.file:
            item.file = to_url(item.file)
    if response.report_html:
        response.report_html = to_url(response.report_html)
    if response.report_json:
        response.report_json = to_url(response.report_json)
    return response


def _overlay_layers(response: AnalysisResponse) -> list[OverlayLayer]:
    """Map-pinnable overlays: those tied to an input image that is itself georeferenced.

    Overlays with `image_index is None` are side-by-side composites, which match no single pixel
    grid; they stay in the evidence gallery and are deliberately not placed on the map.
    """
    images = response.trace.images
    layers = []
    for item in response.evidence:
        if item.kind != "overlay" or not item.file or item.image_index is None:
            continue
        if item.image_index >= len(images):
            continue
        summary = images[item.image_index]
        if summary.corners_wgs84 is None:
            continue
        layers.append(OverlayLayer(url=item.file, label=item.label, image_index=item.image_index,
                                   corners_wgs84=summary.corners_wgs84,
                                   approximate=summary.georeference_note is not None))
    return layers


def _apply_area(uploads: list[Upload], bbox: tuple[float, float, float, float] | None,
                geometry: AreaGeometry | None, directory: Path) -> tuple[list[Path], AreaScope]:
    """Restrict every image to the drawn area, or none of them.

    A bi-temporal or cross-modal pair must keep sharing a pixel grid (validation.check_pair), so
    the crop is all-or-nothing: if any image cannot be cut to the same shape, the whole request
    falls back to the full images and the reason is reported rather than swallowed.

    A rectangle is cut as a box. A circle or polygon is cut to its bounding box and the pixels
    outside the shape are masked, so the analysis covers the shape that was drawn.
    """
    whole = [u.path for u in uploads]
    mask_shape = None
    if geometry is not None:
        shape = geometry.model_dump()
        bbox = geo.geometry_bounds(shape)
        mask_shape = None if geo.is_bounding_box(shape) else shape
    west, south, east, north = bbox
    if west >= east or south >= north:
        return whole, AreaScope(applied=False, reason="a single point has no area to analyse; draw a rectangle, "
                                                      "circle or polygon to restrict the analysis")

    crops, paths = [], []
    for upload in uploads:
        folder = _new_folder(directory)
        destination = folder / f"{Path(upload.path).stem}-area.tif"
        try:
            crop = geo.crop_to_bbox(upload.path, bbox, destination, mask_shape)
        except geo.AreaNotUsable as problem:
            folder.rmdir()
            return whole, AreaScope(applied=False, reason=f"{problem} ({upload.name})")
        if crop.is_whole_image:
            folder.rmdir()  # nothing was written: the image itself is analysed
        crops.append(crop)
        paths.append(upload.path if crop.is_whole_image else destination)

    if all(crop.is_whole_image for crop in crops):
        return whole, AreaScope(applied=False, reason="the selected area covers the whole image")
    if len({(crop.width, crop.height) for crop in crops}) > 1:
        return whole, AreaScope(applied=False, reason="the images would not share a pixel grid after cropping")

    first = crops[0]
    return paths, AreaScope(applied=True, width=first.width, height=first.height,
                            source_width=first.source_width, source_height=first.source_height,
                            masked=any(crop.masked for crop in crops))


def _weather_info(response: AnalysisResponse, area: dict | None, area_source: str) -> WeatherInfo | None:
    """Provenance of a weather answer: the area and its forecast point, and what the provider returned."""
    if area is None or geo.geometry_problem(area):
        return None
    from satquery.specialists.weather import ATTRIBUTION, ATTRIBUTION_URL, PROVIDER

    longitude, latitude = geo.representative_point(area)
    info = WeatherInfo(provider=PROVIDER, attribution=ATTRIBUTION, attribution_url=ATTRIBUTION_URL,
                       area_source=area_source, area_bbox_wgs84=geo.geometry_bounds(area),
                       area_extent_km=tuple(round(v, 2) for v in geo.area_extent_km(area)),
                       point_wgs84=(round(longitude, 6), round(latitude, 6)))
    step = next((s for s in response.trace.steps if s.tool == "weather.forecast" and s.status == "ok"), None)
    if step:
        o = step.outputs
        info.model, info.elevation_m, info.timezone = o["model"], o["elevation_m"], o["timezone"]
        info.grid_point_wgs84 = (o["longitude"], o["latitude"])
        info.period, info.retrieved_at, info.cached = tuple(o["period"]), o["retrieved_at"], o["cached"]
    return info


def _examples() -> list[Example]:
    """Demo scenarios from demo/examples/examples.json (format in demo/examples/README.md)."""
    return [Example(index=i, label=e["label"], query=e["query"], images=e["images"])
            for i, e in enumerate(load_scenarios())]


# --------------------------------------------------------------------------- app


def create_app(run_analysis: Callable[[AnalysisRequest], AnalysisResponse] = analyze) -> FastAPI:
    """`run_analysis` is injected so a host can wrap it (e.g. `spaces.GPU`), as `build_demo` does."""
    settings = load_settings()
    store = UploadStore(directory=_uploads_dir(settings))
    upload_limit = settings.max_upload_mb * 1024 * 1024
    too_large = f"the file is larger than the {settings.max_upload_mb} MB upload limit (SATQUERY_MAX_UPLOAD_MB)"
    app = FastAPI(title="SatQuery AI", description="Agentic remote-sensing analysis (SIH26167)")
    app.add_middleware(CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def refuse_oversized_uploads(request: Request, call_next):
        """Refuse an oversized upload from its declared length, before its body is received and spooled."""
        length = request.headers.get("content-length", "")
        if request.url.path == "/api/uploads" and length.isdigit() and int(length) > upload_limit:
            return JSONResponse(status_code=413, content={"detail": too_large})
        return await call_next(request)

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        current = load_settings()
        return Health(vlm_backend=current.vlm_backend, device=current.device,
                      model_is_fake=current.vlm_backend == "fake",
                      imagery_provider="Copernicus Data Space Ecosystem"
                      if current.copernicus_configured else None,
                      imagery_available=current.copernicus_configured,
                      weather_provider="Open-Meteo" if current.weather_enabled else None,
                      weather_available=current.weather_enabled)

    @app.post("/api/route", response_model=RouteResult)
    def route(request: RouteRequest) -> RouteResult:
        """Weather, imagery, or an unsupported mix of both: decided from the wording alone, so the client
        can send a weather question to the weather specialist before any imagery is retrieved."""
        decided, rule = route_query(request.query)
        message = ("This asks for a weather forecast and a satellite analysis at once, which is not supported yet. "
                   "Please ask the weather question and the imagery question separately.") if decided == "mixed" else None
        return RouteResult(route=decided, rule=rule, message=message)

    @app.post("/api/weather", response_model=AnalyzeResult)
    def weather(request: WeatherRequest) -> AnalyzeResult:
        """A short-range forecast for a point inside the selected area (optional capability, D-029).

        Never retrieves imagery. A switched-off provider is a 503; everything else (no area, a
        long-range or mixed question, a provider failure) is a result the card can explain.
        """
        current = load_settings()
        if not current.weather_enabled:
            return JSONResponse(status_code=503, content={
                "code": "weather_not_configured",
                "message": "Weather forecasts are switched off on this server (SATQUERY_WEATHER=off)."})
        if request.aoi_geometry is not None:
            area = request.aoi_geometry.model_dump()
        elif request.aoi_bbox is not None:
            area = geo.bbox_geometry(request.aoi_bbox)
            problem = geo.geometry_problem(area)
            west, south, east, north = request.aoi_bbox
            if problem or west >= east or south >= north:
                return JSONResponse(status_code=422, content={
                    "code": "area_invalid", "message": f"The selected area cannot be used: {problem or 'it has no extent'}."})
        else:
            area = None
        response = _to_urls(answer_weather(request.query, area, area_source=request.area_source, settings=current))
        return AnalyzeResult(response=response, overlay_layers=[], weather=_weather_info(response, area, request.area_source))

    @app.get("/api/examples", response_model=list[Example])
    def examples() -> list[Example]:
        return _examples()

    @app.get("/api/example-queries", response_model=list[str])
    def example_queries() -> list[str]:
        return list(EXAMPLE_QUERIES)

    @app.post("/api/examples/{index}/load", response_model=list[UploadInfo])
    def load_example(index: int) -> list[UploadInfo]:
        """Register a shipped scenario's rasters as uploads, so it follows the normal analysis path."""
        available = _examples()
        if not 0 <= index < len(available):
            raise HTTPException(404, f"unknown example {index}")
        scenario = available[index]
        return [_register(store, EXAMPLES_DIR / image["path"], Path(image["path"]).name,
                          image.get("modality", "optical"), image.get("acquired") or None, settings)
                for image in scenario.images]

    @app.post("/api/uploads", response_model=UploadInfo)
    def create_upload(file: UploadFile = File(...), modality: Modality | None = Form(None),
                      acquired: str | None = Form(None)) -> UploadInfo:
        """`modality` omitted: read from the file's band descriptions, else optical (and said so)."""
        name = Path(file.filename or "upload").name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(400, f"unsupported format '{suffix}'; accepted: "
                                     + ", ".join(sorted(SUPPORTED_SUFFIXES)))
        target = _new_folder(store.directory) / _stored_name(file.filename, suffix)
        # Streamed in chunks: a satellite scene can be gigabytes, which must not all sit in memory.
        # The limit is enforced here too, for requests that did not declare their length.
        written = 0
        with target.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > upload_limit:
                    break
                out.write(chunk)
        if written > upload_limit:
            target.unlink()
            target.parent.rmdir()
            raise HTTPException(413, too_large)
        modality, basis = _upload_modality(target, modality)
        return _register(store, target, name, modality, (acquired or "").strip() or None, settings, basis)

    @app.post("/api/fetch-imagery", response_model=FetchImageryResult)
    def fetch_imagery(request: FetchImageryRequest) -> FetchImageryResult:
        """Retrieve Sentinel-2 L2A imagery for a drawn area and register it like an upload.

        Deliberately separate from /api/analyze: retrieval failures stay distinguishable from
        analysis failures, and the client can show real progress between the two steps. The result
        is one or two ordinary upload ids, so the analysis path below is completely unchanged: two
        dated uploads are exactly what its existing bi-temporal change analysis takes.
        """
        from satquery.providers.copernicus import CopernicusSentinelProvider, bands_for_target
        from satquery.providers.errors import RetrievalError, SarTemporalUnsupported

        # A weather question never retrieves imagery, not even from a client that skipped /api/route.
        # Checked first, before any provider (or credential) is touched.
        decided, rule = route_query(request.query)
        if decided != "imagery":
            return JSONResponse(status_code=422, content={
                "code": f"{decided}_question",
                "message": "This is a weather question, answered by the weather specialist; no satellite imagery is "
                           "retrieved for it." if decided == "weather" else
                           "This asks for a weather forecast and a satellite analysis at once, which is not supported "
                           "yet. Please ask the two questions separately.",
                "detail": rule})

        current = load_settings()
        try:
            provider = CopernicusSentinelProvider(
                current.copernicus_client_id, current.copernicus_client_secret,
                days_back=current.copernicus_days_back, max_cloud=current.copernicus_max_cloud,
                max_aoi_km2=current.copernicus_max_aoi_km2,
                resolution_m=current.copernicus_resolution_m)

            target, _ = find_target(request.query)
            bands = bands_for_target(target)
            # A question asking for optical and SAR together gets both sensors, checked first: its
            # wording ("compare optical and SAR ...") must not be mistaken for a comparison of dates.
            if needs_optical_and_sar(request.query):
                return fetch_optical_sar(provider, request, bands, current)
            # A question asking for radar alone gets Sentinel-1 alone: no optical scene is involved.
            if needs_sar_only(request.query):
                if needs_multiple_dates(request.query):
                    raise SarTemporalUnsupported(
                        "Comparing two dates with radar is not supported yet, and answering with optical imagery "
                        "instead would ignore what you asked. Ask without radar for an optical comparison, or ask "
                        "about a single date with radar.")
                return fetch_sar(provider, request, current)
            # One scene can never answer a question about change over time, so such a question gets
            # two real acquisitions of the same area. Never one scene used twice (CLAUDE.md section 7).
            if needs_multiple_dates(request.query):
                return fetch_pair(provider, request, bands, current)

            key = provider.cache_key(request.aoi_bbox, bands, days_back=request.days_back,
                                     max_cloud=request.max_cloud)
            cache_folder = current.runs_dir / "imagery-cache" / key
            scene_path, metadata_path = cache_folder / "scene.tif", cache_folder / "metadata.json"

            if scene_path.is_file() and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8")) | {"cached": True}
            else:
                scene = provider.retrieve(request.aoi_bbox, bands, scene_path,
                                          days_back=request.days_back, max_cloud=request.max_cloud)
                metadata = scene.metadata.as_dict()
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            # A water question checks what the optical scene really shows over the selected area, and
            # falls back to radar when too little of it is visible. Other questions are unchanged.
            if target == "water":
                return fetch_water_scene(provider, cache_folder, metadata, current)
        except RetrievalError as problem:
            # Never fall back to fake imagery: the reason is reported instead.
            return JSONResponse(status_code=problem.status, content=problem.as_payload())

        name = f"Sentinel-2 L2A {metadata.get('acquired', '')}".strip()
        upload = _register(store, scene_path, name, "optical", metadata.get("acquired"), current)
        return FetchImageryResult(mode="single", upload=upload, metadata=metadata,
                                  images=[FetchedScene(role="single", upload=upload, metadata=metadata)],
                                  cached=bool(metadata.get("cached")))

    def fetch_water_scene(provider, folder: Path, metadata: dict, current: Settings) -> FetchImageryResult:
        """A water question on one optical scene (D-030).

        Sentinel-2 first. Its scene classification, on the same grid, says how much of the SELECTED
        AREA is cloud, cloud shadow or no data. Usable: the clear pixels are analysed with NDWI (the
        affected ones become nodata). Not usable: the nearest Sentinel-1 scene is retrieved, only
        then, and the water map comes from radar. Every file is cached beside the optical scene.
        """
        from datetime import date

        from satquery.providers import quality
        from satquery.providers.copernicus import check_same_grid
        from satquery.providers.errors import NoSarImagery

        bbox, acquired = metadata["bbox_wgs84"], metadata["acquired"]
        scene_path, scl_path = folder / "scene.tif", folder / "scl.tif"
        cached = bool(metadata.get("cached")) and scl_path.is_file()
        if not scl_path.is_file():
            partial = folder / "scl.partial.tif"  # renamed only once it is known to match the scene
            provider.retrieve_scene_classification(bbox, acquired, partial)
            check_same_grid(scene_path, partial, compare_band_count=False)
            partial.replace(scl_path)
        assessed = quality.assess(scl_path, max_affected_fraction=current.optical_max_affected_fraction,
                                  min_clear_pixels=current.optical_min_clear_pixels)
        info = OpticalQualityInfo(scene=metadata, **assessed.as_dict())

        if assessed.usable:
            analysed = scene_path
            if assessed.affected_fraction > 0:
                analysed = folder / "scene-clear.tif"
                if not analysed.is_file():
                    partial = folder / "scene-clear.partial.tif"
                    quality.mask_affected(scene_path, scl_path, partial).replace(analysed)
                info.masked = True
            name = f"Sentinel-2 L2A {acquired}" + (" (cloud-masked)" if info.masked else "")
            upload = _register(store, analysed, name, "optical", acquired, current)
            return FetchImageryResult(mode="single", upload=upload, metadata=metadata,
                                      images=[FetchedScene(role="single", upload=upload, metadata=metadata)],
                                      optical_quality=info, cached=cached)

        # The optical scene cannot answer: the nearest radar scene is retrieved now, and only now.
        sar_path, sar_record = folder / "sar-fallback.tif", folder / "sar-fallback.json"
        if sar_path.is_file() and sar_record.is_file():
            sar_meta = json.loads(sar_record.read_text(encoding="utf-8")) | {"cached": True}
        else:
            cached = False
            try:
                sar = provider.retrieve_sar_near(bbox, date.fromisoformat(acquired), sar_path)
            except NoSarImagery as problem:
                raise NoSarImagery(f"The optical scene of {acquired} cannot be used here: {assessed.reason}. "
                                   f"{problem.message}") from problem
            check_same_grid(scene_path, sar.path, compare_band_count=False)
            sar_meta = sar.metadata.as_dict()
            sar_record.write_text(json.dumps(sar_meta, indent=2), encoding="utf-8")
        upload = _register(store, sar_path, f"{sar_meta['satellite']} GRD {sar_meta['acquired']} (radar fallback)",
                           "sar", sar_meta["acquired"], current)
        return FetchImageryResult(mode="sar_fallback", upload=upload, metadata=sar_meta,
                                  images=[FetchedScene(role="sar", upload=upload, metadata=sar_meta)],
                                  optical_quality=info, cached=cached and bool(sar_meta.get("cached")))

    def fetch_sar(provider, request: FetchImageryRequest, current: Settings) -> FetchImageryResult:
        """The most recent Sentinel-1 scene for a question asking for radar alone (D-030)."""
        folder = current.runs_dir / "imagery-cache" / provider.sar_cache_key(request.aoi_bbox, days_back=request.days_back)
        path, record = folder / "sar.tif", folder / "sar.json"
        cached = path.is_file() and record.is_file()
        if cached:
            meta = json.loads(record.read_text(encoding="utf-8"))
        else:
            meta = provider.retrieve_sar(request.aoi_bbox, path, days_back=request.days_back).metadata.as_dict()
            record.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        meta |= {"cached": cached}
        upload = _register(store, path, f"{meta['satellite']} GRD {meta['acquired']} (SAR)", "sar", meta["acquired"], current)
        return FetchImageryResult(mode="sar", upload=upload, metadata=meta,
                                  images=[FetchedScene(role="sar", upload=upload, metadata=meta)], cached=cached)

    def fetch_pair(provider, request: FetchImageryRequest, bands: list[str], current: Settings) -> FetchImageryResult:
        """Two real acquisitions for a temporal question, registered oldest first.

        Cached under a key that includes the mode and both search windows, so a single-date entry can
        never be returned for a temporal request, nor one comparison for another.
        """
        from datetime import date

        from satquery.providers.temporal import resolve_windows

        windows = resolve_windows(request.query)  # TemporalRangeUnsupported before any API call
        folder = current.runs_dir / "imagery-cache" / provider.pair_cache_key(
            request.aoi_bbox, bands, windows, max_cloud=request.max_cloud)
        paths = {"before": folder / "before.tif", "after": folder / "after.tif"}
        record = folder / "pair.json"

        cached = record.is_file() and all(path.is_file() for path in paths.values())
        if cached:
            metadata = json.loads(record.read_text(encoding="utf-8"))
        else:
            before, after = provider.retrieve_pair(request.aoi_bbox, bands, windows, folder,
                                                   max_cloud=request.max_cloud)
            metadata = {"before": before.metadata.as_dict(), "after": after.metadata.as_dict()}
            # Written only after both downloads succeeded and share a grid: a failed or mismatched
            # pair is never served from the cache.
            record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        scenes = []
        for role in ("before", "after"):
            meta = metadata[role] | {"cached": cached}
            upload = _register(store, paths[role], f"Sentinel-2 L2A {meta['acquired']} ({role})", "optical",
                               meta["acquired"], current)
            scenes.append(FetchedScene(role=role, upload=upload, metadata=meta))

        days_apart = (date.fromisoformat(metadata["after"]["acquired"])
                      - date.fromisoformat(metadata["before"]["acquired"])).days
        window = lambda w: ComparisonWindow(start=w.start.isoformat(), end=w.end.isoformat(), label=w.label)
        temporal = TemporalInfo(basis=windows.basis, explanation=windows.explanation,
                                before_window=window(windows.before), after_window=window(windows.after),
                                days_apart=days_apart)
        return FetchImageryResult(mode="temporal", upload=scenes[-1].upload, metadata=scenes[-1].metadata,
                                  images=scenes, temporal=temporal, cached=cached)

    def fetch_optical_sar(provider, request: FetchImageryRequest, bands: list[str],
                          current: Settings) -> FetchImageryResult:
        """A Sentinel-2 scene and the Sentinel-1 scene closest to it in time, registered as an
        optical and a SAR upload, so the existing cross-modal analysis runs on them unchanged.

        Cached under a key that names the mode, so no single-date or temporal entry is ever reused.
        """
        from datetime import date

        from satquery.providers.copernicus import SAR_MAX_DAYS_APART

        folder = current.runs_dir / "imagery-cache" / provider.optical_sar_cache_key(
            request.aoi_bbox, bands, days_back=request.days_back, max_cloud=request.max_cloud)
        paths = {"optical": folder / "optical.tif", "sar": folder / "sar.tif"}
        record = folder / "pair.json"

        cached = record.is_file() and all(path.is_file() for path in paths.values())
        if cached:
            metadata = json.loads(record.read_text(encoding="utf-8"))
        else:
            optical, sar = provider.retrieve_optical_sar(request.aoi_bbox, bands, folder,
                                                         days_back=request.days_back, max_cloud=request.max_cloud)
            metadata = {"optical": optical.metadata.as_dict(), "sar": sar.metadata.as_dict()}
            # Written only after both downloads succeeded and share a grid.
            record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        scenes = []
        for role in ("optical", "sar"):
            meta = metadata[role] | {"cached": cached}
            name = f"{meta['satellite']} {meta['product_level']} {meta['acquired']} ({'SAR' if role == 'sar' else role})"
            upload = _register(store, paths[role], name, role, meta["acquired"], current)
            scenes.append(FetchedScene(role=role, upload=upload, metadata=meta))

        optical_meta, sar_meta = metadata["optical"], metadata["sar"]
        days_apart = abs((date.fromisoformat(sar_meta["acquired"]) - date.fromisoformat(optical_meta["acquired"])).days)
        info = CrossModalInfo(
            days_apart=days_apart, max_days_apart=SAR_MAX_DAYS_APART,
            explanation=(f"The least cloudy {optical_meta['satellite']} scene of the search window, paired with the "
                         f"{sar_meta['satellite']} VV+VH scene closest to it in time (within {SAR_MAX_DAYS_APART} days). "
                         f"Both were rendered onto one pixel grid for the selected area; they were acquired "
                         f"{days_apart} day{'s' if days_apart != 1 else ''} apart."))
        return FetchImageryResult(mode="cross_modal", upload=scenes[0].upload, metadata=scenes[0].metadata,
                                  images=scenes, cross_modal=info, cached=cached)

    @app.get("/api/uploads/{upload_id}/preview.png")
    def upload_preview(upload_id: str) -> FileResponse:
        upload = store.get(upload_id)
        if upload.preview is None or not upload.preview.is_file():
            raise HTTPException(404, "preview not available")
        return FileResponse(upload.preview, media_type="image/png")

    @app.post("/api/analyze", response_model=AnalyzeResult)
    def run(request: AnalyzeRequest) -> AnalyzeResult:
        """Blocking on purpose: a plain `def` endpoint runs in FastAPI's threadpool, so the
        single-process pipeline (~0.4 s/task on GPU, ~17 s on CPU) does not block the event loop."""
        uploads = [store.get(item.upload_id) for item in request.images]
        paths = [upload.path for upload in uploads]
        area = None
        if request.aoi_geometry or request.aoi_bbox:
            paths, area = _apply_area(uploads, request.aoi_bbox, request.aoi_geometry, store.directory)

        images = [ImageInput(path=str(path), modality=item.modality or upload.modality,
                             acquired=item.acquired or upload.acquired)
                  for item, upload, path in zip(request.images, uploads, paths)]
        response = run_analysis(AnalysisRequest(query=request.query, images=images,
                                                forced_task=request.forced_task))
        # Rewrite paths to URLs first: _overlay_layers copies `evidence.file`, so doing it the
        # other way round would hand the browser local filesystem paths.
        served = _to_urls(response)
        return AnalyzeResult(response=served, overlay_layers=_overlay_layers(served), area=area,
                             upload_ids=[item.upload_id for item in request.images])

    @app.get("/api/runs/{run_id}/{filename}")
    def run_artifact(run_id: str, filename: str) -> FileResponse:
        """Serves overlay PNGs and the downloadable HTML/JSON reports for one run."""
        path = _safe_child(load_settings().runs_dir, run_id, filename)
        media = {".png": "image/png", ".html": "text/html", ".json": "application/json"}
        return FileResponse(path, media_type=media.get(path.suffix.lower(), "application/octet-stream"))

    if WEB_DIST.is_dir():  # production: one origin serves the API and the built client
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app


app = create_app()
