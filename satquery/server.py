"""HTTP layer for the map-first web client (D-023).

Presentation only, the same tier as `ui.py` and `cli.py`: it builds an `AnalysisRequest`, calls the
injected analysis function, and rewrites on-disk artifact paths into URLs the browser can fetch. It
never imports specialists or the agent directly.

Run it locally with the labelled fake model:
    SATQUERY_VLM_BACKEND=fake uvicorn satquery.server:app --reload
"""

import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from satquery import geo
from satquery.api import analyze
from satquery.evidence import save_png
from satquery.examples import EXAMPLE_QUERIES, EXAMPLES_DIR, load_scenarios
from satquery.imaging import SUPPORTED_SUFFIXES, load_image, render_rgb
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


class AnalyzeResult(BaseModel):
    """`AnalysisResponse` with artifact paths rewritten as URLs, plus per-overlay map placement."""

    response: AnalysisResponse
    overlay_layers: list[OverlayLayer]
    area: AreaScope | None = None  # present only when the request carried a drawn area
    # The uploads this result ran on, in input order: how the client knows which layers it describes.
    upload_ids: list[str] = Field(default_factory=list)


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


def _register(store: UploadStore, source: Path, name: str, modality: Modality,
              acquired: str | None, settings: Settings) -> UploadInfo:
    """Load a raster, summarise it with map placement, and remember it for later analysis."""
    try:
        image = load_image(source, modality, acquired, settings.max_pixels)
    except Exception as error:
        raise HTTPException(400, f"cannot read '{name}': {error}") from error

    upload = store.add(Upload(id=uuid.uuid4().hex[:12], path=source, name=name, modality=modality,
                              acquired=acquired, summary=geo.summarize(image, 0)))
    preview = store.directory / f"{upload.id}-preview.png"
    save_png(render_rgb(image), preview)
    upload.preview = preview
    return UploadInfo(id=upload.id, name=name, modality=modality, acquired=acquired,
                      summary=upload.summary, preview_url=f"/api/uploads/{upload.id}/preview.png",
                      mappable=upload.summary.corners_wgs84 is not None)


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
                      model_is_fake=current.vlm_backend == "fake")

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
    def create_upload(file: UploadFile = File(...), modality: Modality = Form("optical"),
                      acquired: str | None = Form(None)) -> UploadInfo:
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
        return _register(store, target, name, modality, (acquired or "").strip() or None, settings)

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
