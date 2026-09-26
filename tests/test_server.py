"""HTTP contract of satquery/server.py. Uses the fake VLM backend; no GPU, no model weights."""

from pathlib import Path

from satquery.providers.errors import (GridsIncompatible, NoEarlierImagery, NoLaterImagery,
                                       OnlyOneAcquisition)

import numpy as np
import pytest
from fastapi.testclient import TestClient

from satquery.server import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SATQUERY_VLM_BACKEND", "fake")
    monkeypatch.setenv("SATQUERY_RUNS_DIR", str(tmp_path / "runs"))
    return TestClient(create_app())


@pytest.fixture
def scene():
    return np.arange(4 * 32 * 32, dtype=np.uint16).reshape(4, 32, 32) % 3000


def _upload(client, path, modality="optical", acquired=None):
    with open(path, "rb") as handle:
        data = {"modality": modality} | ({"acquired": acquired} if acquired else {})
        response = client.post("/api/uploads", files={"file": (path.split("\\")[-1], handle, "image/tiff")}, data=data)
    assert response.status_code == 200, response.text
    return response.json()


def test_health_reports_the_fake_backend(client):
    """A labelled fake model must be visible to the client, never passed off as the real one."""
    body = client.get("/api/health").json()
    assert body["vlm_backend"] == "fake" and body["model_is_fake"] is True


def test_upload_returns_map_placement_for_a_geotiff(client, write_tiff, scene):
    body = _upload(client, write_tiff("utm.tif", scene, band_names=["blue", "green", "red", "nir"]))

    assert body["mappable"] is True
    assert len(body["summary"]["corners_wgs84"]) == 4
    assert body["summary"]["crs"] == "EPSG:32643"
    assert client.get(body["preview_url"]).headers["content-type"] == "image/png"


def test_upload_without_a_crs_is_flagged_unmappable(client, write_tiff, scene):
    body = _upload(client, write_tiff("plain.tif", scene, crs=None))

    assert body["mappable"] is False
    assert body["summary"]["corners_wgs84"] is None
    assert client.get(body["preview_url"]).status_code == 200  # still viewable off-map


def test_unsupported_format_is_refused(client, tmp_path):
    document = tmp_path / "notes.pdf"
    document.write_bytes(b"%PDF-1.4")
    with open(document, "rb") as handle:
        response = client.post("/api/uploads", files={"file": ("notes.pdf", handle, "application/pdf")})

    assert response.status_code == 400
    assert ".pdf" in response.json()["detail"]


def test_analyze_returns_a_trace_and_servable_artifacts(client, write_tiff, scene):
    upload = _upload(client, write_tiff("utm.tif", scene, band_names=["blue", "green", "red", "nir"]))
    response = client.post("/api/analyze", json={"query": "Describe this image.",
                                                 "images": [{"upload_id": upload["id"]}]})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["response"]["status"] in {"ok", "partial"}
    assert body["response"]["trace"]["steps"], "the execution trace is what SIH evaluates"
    assert body["response"]["report_html"].startswith("/api/runs/")
    assert client.get(body["response"]["report_html"]).status_code == 200
    assert client.get(body["response"]["report_json"]).status_code == 200


def test_overlay_layers_carry_map_corners(client, write_tiff, scene):
    upload = _upload(client, write_tiff("utm.tif", scene, band_names=["blue", "green", "red", "nir"]))
    body = client.post("/api/analyze", json={"query": "Describe this image.",
                                             "images": [{"upload_id": upload["id"]}]}).json()

    assert body["overlay_layers"], "a georeferenced single image should yield a pinnable overlay"
    layer = body["overlay_layers"][0]
    assert layer["url"].startswith("/api/runs/"), "local filesystem paths must never reach the client"
    assert len(layer["corners_wgs84"]) == 4
    assert client.get(layer["url"]).status_code == 200


def test_overlays_are_not_pinned_when_the_input_has_no_crs(client, write_tiff, scene):
    upload = _upload(client, write_tiff("plain.tif", scene, crs=None))
    body = client.post("/api/analyze", json={"query": "Describe this image.",
                                             "images": [{"upload_id": upload["id"]}]}).json()

    assert body["response"]["status"] in {"ok", "partial"}
    assert body["overlay_layers"] == []


def test_unknown_upload_is_rejected(client):
    response = client.post("/api/analyze", json={"query": "hi", "images": [{"upload_id": "nope"}]})
    assert response.status_code == 404


def test_run_artifacts_refuse_path_traversal(client, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    assert client.get("/api/runs/..%2f..%2f/secret.txt").status_code == 404
    assert client.get("/api/runs/anything/missing.png").status_code == 404


def test_examples_are_listed_and_loadable(client):
    examples = client.get("/api/examples").json()
    assert examples, "demo scenarios ship with the repo"
    assert {"index", "label", "query", "images"} <= examples[0].keys()

    loaded = client.post(f"/api/examples/{examples[0]['index']}/load")
    assert loaded.status_code == 200
    uploads = loaded.json()
    assert uploads and all(u["mappable"] for u in uploads), "shipped GeoTIFFs are georeferenced"


def test_example_queries_are_served(client):
    queries = client.get("/api/example-queries").json()
    assert len(queries) == 5 and all(isinstance(q, str) for q in queries)


@pytest.fixture
def area_scene():
    """Large enough that a cropped middle still clears geo.MIN_CROP_PIXELS."""
    return np.arange(4 * 96 * 96, dtype=np.uint16).reshape(4, 96, 96) % 3000


def _bounds(upload):
    return upload["summary"]["bounds_wgs84"]


def _inset(bounds, factor=0.3):
    """A box covering the middle of `bounds`."""
    west, south, east, north = bounds
    return [west + (east - west) * factor, south + (north - south) * factor,
            east - (east - west) * factor, north - (north - south) * factor]


def test_drawn_area_restricts_the_analysis(client, write_tiff, area_scene):
    upload = _upload(client, write_tiff("utm.tif", area_scene, band_names=["blue", "green", "red", "nir"]))
    body = client.post("/api/analyze", json={"query": "Describe this image.",
                                             "images": [{"upload_id": upload["id"]}],
                                             "aoi_bbox": _inset(_bounds(upload))}).json()

    assert body["area"]["applied"] is True
    analysed = body["response"]["trace"]["images"][0]
    assert analysed["width"] < upload["summary"]["width"], "the crop should be smaller than the source"
    assert (body["area"]["width"], body["area"]["height"]) == (analysed["width"], analysed["height"])


def test_area_that_misses_the_image_falls_back_and_says_so(client, write_tiff, scene):
    """A selected area must never be silently ignored (CLAUDE.md §7)."""
    upload = _upload(client, write_tiff("utm.tif", scene, band_names=["blue", "green", "red", "nir"]))
    body = client.post("/api/analyze", json={"query": "Describe this image.",
                                             "images": [{"upload_id": upload["id"]}],
                                             "aoi_bbox": [10.0, 50.0, 10.5, 50.5]}).json()

    assert body["area"]["applied"] is False
    assert "does not overlap" in body["area"]["reason"]
    assert body["response"]["trace"]["images"][0]["width"] == upload["summary"]["width"]


def test_area_covering_everything_is_reported_as_no_narrowing(client, write_tiff, scene):
    upload = _upload(client, write_tiff("utm.tif", scene, band_names=["blue", "green", "red", "nir"]))
    body = client.post("/api/analyze", json={"query": "Describe this image.",
                                             "images": [{"upload_id": upload["id"]}],
                                             "aoi_bbox": _bounds(upload)}).json()

    assert body["area"]["applied"] is False
    assert body["area"]["reason"] == "the selected area covers the whole image"


def test_a_cropped_pair_still_shares_a_pixel_grid(client, write_tiff, area_scene):
    """check_pair rejects mismatched grids, so both images must be cut to the same shape."""
    before = _upload(client, write_tiff("before.tif", area_scene, band_names=["blue", "green", "red", "nir"]))
    after = _upload(client, write_tiff("after.tif", area_scene, band_names=["blue", "green", "red", "nir"]))
    body = client.post("/api/analyze", json={
        "query": "What changed between these two dates?",
        "images": [{"upload_id": before["id"], "acquired": "2019-01-01"},
                   {"upload_id": after["id"], "acquired": "2023-01-01"}],
        "aoi_bbox": _inset(_bounds(before))}).json()

    assert body["area"]["applied"] is True
    shapes = {(i["width"], i["height"]) for i in body["response"]["trace"]["images"]}
    assert len(shapes) == 1, "a cropped pair must stay on one grid"
    assert "grid_shape_mismatch" not in {v["code"] for v in body["response"]["trace"]["validation"]}


def test_area_is_ignored_for_imagery_without_a_crs(client, write_tiff, scene):
    upload = _upload(client, write_tiff("plain.tif", scene, crs=None))
    body = client.post("/api/analyze", json={"query": "Describe this image.",
                                             "images": [{"upload_id": upload["id"]}],
                                             "aoi_bbox": [74.9, 25.2, 75.1, 25.4]}).json()

    assert body["area"]["applied"] is False
    assert "no coordinate reference system" in body["area"]["reason"], "the real reason, not 'does not overlap'"
    assert body["response"]["status"] in {"ok", "partial"}, "the analysis still runs on the whole image"


# --------------------------------------------------------------------------- drawn shapes


def _circle(bounds, scale=0.4, segments=64):
    """A circle-like polygon centred in `bounds`, as the map sends a drawn circle."""
    import math
    west, south, east, north = bounds
    cx, cy = (west + east) / 2, (south + north) / 2
    rx, ry = (east - west) * scale, (north - south) * scale
    ring = [[cx + rx * math.cos(2 * math.pi * k / segments), cy + ry * math.sin(2 * math.pi * k / segments)]
            for k in range(segments)]
    return {"type": "Polygon", "coordinates": [ring + [ring[0]]]}


def _box(west, south, east, north):
    return {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}


@pytest.fixture
def water_scene():
    """Water everywhere (green well above NIR, so NDWI > 0 on every pixel)."""
    rng = np.random.default_rng(3)
    bands = np.stack([np.full((96, 96), v, np.float32) for v in (600, 900, 400, 150)])
    return (bands + rng.normal(0, 5, bands.shape)).astype(np.float32)


def test_a_drawn_circle_is_analysed_as_a_circle_not_its_box(client, write_tiff, water_scene):
    """Coverage must count only pixels inside the circle: water everywhere means 100%, not the ~79%
    a bounding box with masked-out corners would report."""
    upload = _upload(client, write_tiff("water.tif", water_scene, band_names=["blue", "green", "red", "nir"]))
    body = client.post("/api/analyze", json={"query": "Describe this image.",
                                             "images": [{"upload_id": upload["id"]}],
                                             "aoi_geometry": _circle(_bounds(upload))}).json()

    assert body["area"]["applied"] is True and body["area"]["masked"] is True
    indices = next(s for s in body["response"]["trace"]["steps"] if s["tool"] == "optical.spectral_indices")
    assert indices["outputs"]["water_fraction"] == 1.0
    assert "water-like pixels (NDWI > 0): 100.0%" in body["response"]["answer"]


def test_a_drawn_rectangle_sent_as_geometry_matches_the_box_path(client, write_tiff, area_scene):
    upload = _upload(client, write_tiff("utm.tif", area_scene, band_names=["blue", "green", "red", "nir"]))
    box = _inset(_bounds(upload))
    request = {"query": "Describe this image.", "images": [{"upload_id": upload["id"]}]}
    as_box = client.post("/api/analyze", json=request | {"aoi_bbox": box}).json()
    as_shape = client.post("/api/analyze", json=request | {"aoi_geometry": _box(*box)}).json()

    assert as_shape["area"]["applied"] is True and as_shape["area"]["masked"] is False
    assert as_shape["area"] == as_box["area"]
    outputs = lambda body: [s["outputs"] for s in body["response"]["trace"]["steps"]]
    assert outputs(as_shape) == outputs(as_box), "a rectangle keeps the original box crop"


def test_an_unusable_shape_is_refused(client, write_tiff, scene):
    upload = _upload(client, write_tiff("utm.tif", scene, band_names=["blue", "green", "red", "nir"]))
    request = {"query": "Describe this image.", "images": [{"upload_id": upload["id"]}]}
    point = client.post("/api/analyze", json=request | {"aoi_geometry": {"type": "Point", "coordinates": [75.0, 25.3]}})
    open_ring = client.post("/api/analyze", json=request | {"aoi_geometry": {"type": "Polygon",
                                                                             "coordinates": [[[75, 25], [75.1, 25]]]}})
    assert point.status_code == 422 and open_ring.status_code == 422
    assert "4 positions" in open_ring.text


def test_a_circle_over_a_pair_masks_both_images_identically(client, write_tiff, area_scene):
    before = _upload(client, write_tiff("before.tif", area_scene, band_names=["blue", "green", "red", "nir"]))
    after = _upload(client, write_tiff("after.tif", area_scene, band_names=["blue", "green", "red", "nir"]))
    body = client.post("/api/analyze", json={
        "query": "What changed between these two dates?",
        "images": [{"upload_id": before["id"], "acquired": "2019-01-01"},
                   {"upload_id": after["id"], "acquired": "2023-01-01"}],
        "aoi_geometry": _circle(_bounds(before))}).json()

    assert body["area"]["applied"] is True and body["area"]["masked"] is True
    assert len({(i["width"], i["height"]) for i in body["response"]["trace"]["images"]}) == 1
    assert body["response"]["status"] == "ok"


def test_a_pinned_overlay_is_transparent_outside_the_circle(client, write_tiff, water_scene):
    import io
    from PIL import Image

    upload = _upload(client, write_tiff("water.tif", water_scene, band_names=["blue", "green", "red", "nir"]))
    body = client.post("/api/analyze", json={"query": "Describe this image.",
                                             "images": [{"upload_id": upload["id"]}],
                                             "aoi_geometry": _circle(_bounds(upload))}).json()
    overlay = Image.open(io.BytesIO(client.get(body["overlay_layers"][0]["url"]).content))

    assert overlay.mode == "RGBA"
    width, height = overlay.size
    assert overlay.getpixel((0, 0))[3] == 0, "a corner outside the circle is see-through on the map"
    assert overlay.getpixel((width // 2, height // 2))[3] == 255


# --------------------------------------------------------------------------- names and identity


def _upload_as(client, path, filename):
    with open(path, "rb") as handle:
        response = client.post("/api/uploads", files={"file": (filename, handle, "image/tiff")},
                               data={"modality": "optical"})
    assert response.status_code == 200, response.text
    return response.json()


def test_the_trace_shows_the_users_file_name_not_a_storage_id(client, write_tiff, area_scene):
    """The execution trace is what SIH evaluates: it must name the file the user gave it."""
    upload = _upload_as(client, write_tiff("x.tif", area_scene, band_names=["blue", "green", "red", "nir"]),
                        "my scene.tif")
    request = {"query": "Describe this image.", "images": [{"upload_id": upload["id"]}]}
    whole = client.post("/api/analyze", json=request).json()
    cropped = client.post("/api/analyze", json=request | {"aoi_bbox": _inset(_bounds(upload))}).json()

    assert upload["name"] == "my scene.tif"
    assert whole["response"]["trace"]["images"][0]["name"] == "my scene.tif"
    assert cropped["response"]["trace"]["images"][0]["name"] == "my scene-area.tif"


def test_unsafe_file_names_are_neutralised():
    from satquery.server import _stored_name

    assert _stored_name("../../evil:name?.tif", ".tif") == "evil_name_.tif"
    assert _stored_name("CON.tif", ".tif") == "_CON.tif", "a reserved device name on Windows"
    assert _stored_name("...", ".tif") == "upload.tif"
    assert _stored_name(None, ".png") == "upload.png"
    assert _stored_name("dune à Pondichéry.tif", ".tif") == "dune à Pondichéry.tif"


def test_an_upload_named_like_a_path_stays_inside_the_uploads_folder(client, write_tiff, scene, tmp_path):
    upload = _upload_as(client, write_tiff("x.tif", scene, band_names=["blue", "green", "red", "nir"]), "../../escape.tif")
    body = client.post("/api/analyze", json={"query": "Describe this image.",
                                             "images": [{"upload_id": upload["id"]}]}).json()

    assert body["response"]["trace"]["images"][0]["name"] == "escape.tif"
    assert not (tmp_path / "escape.tif").exists() and not (tmp_path.parent / "escape.tif").exists()


def test_a_result_lists_the_uploads_it_ran_on(client, write_tiff, area_scene):
    bands = ["blue", "green", "red", "nir"]
    first = _upload_as(client, write_tiff("a.tif", area_scene, band_names=bands), "same.tif")
    second = _upload_as(client, write_tiff("b.tif", area_scene, band_names=bands), "same.tif")
    body = client.post("/api/analyze", json={
        "query": "What changed between these two dates?",
        "images": [{"upload_id": first["id"], "acquired": "2019-01-01"},
                   {"upload_id": second["id"], "acquired": "2023-01-01"}]}).json()

    assert body["upload_ids"] == [first["id"], second["id"]], "identity survives two files with one name"


def test_a_point_is_reported_as_having_no_area(client, write_tiff, scene):
    upload = _upload(client, write_tiff("utm.tif", scene, band_names=["blue", "green", "red", "nir"]))
    west, south, east, north = _bounds(upload)
    lon, lat = (west + east) / 2, (south + north) / 2
    body = client.post("/api/analyze", json={"query": "Describe this image.",
                                             "images": [{"upload_id": upload["id"]}],
                                             "aoi_bbox": [lon, lat, lon, lat]}).json()

    assert body["area"]["applied"] is False
    assert "single point has no area" in body["area"]["reason"], "a point on the image does overlap it"
    assert body["response"]["status"] in {"ok", "partial"}


def test_an_area_too_small_to_analyse_says_so(client, write_tiff, scene):
    upload = _upload(client, write_tiff("utm.tif", scene, band_names=["blue", "green", "red", "nir"]))
    west, south, east, north = _bounds(upload)
    tiny = [west, south, west + (east - west) * 0.05, south + (north - south) * 0.05]
    body = client.post("/api/analyze", json={"query": "Describe this image.",
                                             "images": [{"upload_id": upload["id"]}], "aoi_bbox": tiny}).json()

    assert body["area"]["applied"] is False and "smaller than 16x16 pixels" in body["area"]["reason"]


# --------------------------------------------------------------------------- upload size


@pytest.fixture
def small_limit_client(tmp_path, monkeypatch):
    monkeypatch.setenv("SATQUERY_VLM_BACKEND", "fake")
    monkeypatch.setenv("SATQUERY_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SATQUERY_MAX_UPLOAD_MB", "1")
    return TestClient(create_app())


def test_an_oversized_upload_is_refused_from_its_declared_length(small_limit_client):
    response = small_limit_client.post("/api/uploads", files={"file": ("big.tif", b"\0" * (2 * 1024 * 1024), "image/tiff")})
    assert response.status_code == 413 and "upload limit" in response.json()["detail"]


def test_an_upload_without_a_declared_length_is_capped_while_streaming(small_limit_client, tmp_path):
    boundary = "satquery-test"
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"big.tif\"\r\n"
            "Content-Type: image/tiff\r\n\r\n").encode()

    def chunked_body():  # a generator body is sent chunked, with no Content-Length
        yield head
        for _ in range(3):
            yield b"\0" * (1024 * 1024)
        yield f"\r\n--{boundary}--\r\n".encode()

    response = small_limit_client.post("/api/uploads", content=chunked_body(),
                                       headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert response.status_code == 413
    assert not any((tmp_path / "uploads").rglob("big.tif")), "the partial file is removed"


# ------------------------------------------------------------- imagery retrieval (/api/fetch-imagery)
# Offline: the provider is replaced, so the suite never touches Copernicus or its quota.

BBOX = [72.90, 19.00, 73.00, 19.10]


def test_health_reports_imagery_availability_without_exposing_credentials(client, monkeypatch):
    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "public-looking-id")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "super-secret-value")
    body = client.get("/api/health").json()

    assert body["imagery_available"] is True
    assert body["imagery_provider"] == "Copernicus Data Space Ecosystem"
    serialised = str(body)
    assert "super-secret-value" not in serialised and "public-looking-id" not in serialised


def test_health_reports_imagery_unavailable_without_credentials(client, monkeypatch):
    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "")
    body = client.get("/api/health").json()
    assert body["imagery_available"] is False and body["imagery_provider"] is None


@pytest.mark.parametrize("query", [
    "What has changed here?",
    "Has vegetation increased?",
    "Compare this area over time",
])
def test_a_query_needing_two_dates_never_takes_the_single_scene_path(client, monkeypatch, query):
    """A temporal question is never answered from one scene: it is routed to two-date retrieval."""
    def single(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a temporal question must not retrieve a single scene")

    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "id")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "secret")
    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve", single)
    routed = []

    def pair(self, bbox, bands, windows, destination_dir, **kwargs):
        routed.append(windows)
        raise NoLaterImagery("stop here: routing is what this test checks")

    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve_pair", pair)
    response = client.post("/api/fetch-imagery", json={"query": query, "aoi_bbox": BBOX})

    assert len(routed) == 1, "routed to the two-date retrieval"
    assert response.json()["code"] == "no_later_imagery"


def test_an_ordinary_question_is_not_refused_as_temporal(client, monkeypatch):
    """The guard must not swallow normal queries: this one fails later, for a different reason."""
    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "")
    response = client.post("/api/fetch-imagery",
                           json={"query": "Are there water bodies here?", "aoi_bbox": BBOX})

    assert response.status_code == 503
    assert response.json()["code"] == "credentials_missing"


def test_missing_credentials_are_reported_not_faked(client, monkeypatch):
    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "")
    body = client.post("/api/fetch-imagery",
                       json={"query": "What is in this area?", "aoi_bbox": BBOX}).json()

    assert body["code"] == "credentials_missing"
    assert "COPERNICUS_CLIENT_ID" in body["message"], "should say what to configure"


def test_a_retrieval_failure_never_falls_back_to_fake_imagery(client, monkeypatch):
    from satquery.providers.errors import NoImageryFound

    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "id")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "secret")

    def no_scenes(*args, **kwargs):
        raise NoImageryFound("No Sentinel-2 L2A scene matched this area.")

    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve", no_scenes)
    response = client.post("/api/fetch-imagery",
                           json={"query": "What is in this area?", "aoi_bbox": BBOX})

    assert response.status_code == 404
    assert response.json()["code"] == "no_imagery_found"


def test_a_retrieved_scene_becomes_an_ordinary_upload_and_analyses(client, monkeypatch, tmp_path, write_tiff, scene):
    """The whole point of the acquisition layer: retrieval produces a normal upload id."""
    from satquery.providers import RetrievedScene, SceneMetadata

    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "id")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "secret")
    source = write_tiff("fetched.tif", scene, ["blue", "green", "red", "nir"])

    def fake_retrieve(self, bbox, bands, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
        return RetrievedScene(path=destination, metadata=SceneMetadata(
            provider="Copernicus Data Space Ecosystem", collection="sentinel-2-l2a",
            satellite="Sentinel-2", product_level="L2A", acquired="2026-09-14",
            acquired_datetime="2026-09-14T05:20:11Z", cloud_cover=4.25, bbox_wgs84=tuple(bbox),
            crs="EPSG:4326", resolution_m=10.0, bands=["B02 (blue)"], width=32, height=32,
            scene_id="S2B_TEST", alternatives_considered=3))

    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve", fake_retrieve)
    fetched = client.post("/api/fetch-imagery",
                          json={"query": "Are there water bodies here?", "aoi_bbox": BBOX})
    assert fetched.status_code == 200, fetched.text
    body = fetched.json()

    assert body["metadata"]["satellite"] == "Sentinel-2"
    assert body["metadata"]["acquired"] == "2026-09-14"
    assert body["metadata"]["cloud_cover"] == 4.25
    upload_id = body["upload"]["id"]

    # the existing analysis path, unchanged
    analysed = client.post("/api/analyze", json={
        "query": "Are there water bodies here?",
        "images": [{"upload_id": upload_id, "modality": "optical"}],
    })
    assert analysed.status_code == 200, analysed.text
    assert analysed.json()["response"]["status"] in ("ok", "inconclusive")


def test_a_second_identical_request_is_served_from_cache(client, monkeypatch, write_tiff, scene):
    from satquery.providers import RetrievedScene, SceneMetadata

    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "id")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "secret")
    source = write_tiff("cached.tif", scene, ["blue", "green", "red", "nir"])
    calls = []

    def fake_retrieve(self, bbox, bands, destination, **kwargs):
        calls.append(bbox)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
        return RetrievedScene(path=destination, metadata=SceneMetadata(
            provider="Copernicus Data Space Ecosystem", collection="sentinel-2-l2a",
            satellite="Sentinel-2", product_level="L2A", acquired="2026-09-14",
            acquired_datetime="2026-09-14T05:20:11Z", cloud_cover=1.0, bbox_wgs84=tuple(bbox),
            crs="EPSG:4326", resolution_m=10.0, bands=["B02 (blue)"], width=32, height=32))

    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve", fake_retrieve)
    payload = {"query": "What is in this area?", "aoi_bbox": BBOX}

    first = client.post("/api/fetch-imagery", json=payload).json()
    second = client.post("/api/fetch-imagery", json=payload).json()

    assert len(calls) == 1, "the second identical request must not hit the API again"
    assert first["cached"] is False and second["cached"] is True


# ------------------------------------------------------------- two-date (temporal) retrieval

TEMPORAL_QUERY = "What changed here over the last 3 months?"


@pytest.fixture
def pair_files(write_tiff, scene):
    """Two different dates of one area on one grid: the later one has a block that changed."""
    later = scene.copy()
    later[:, 8:20, 8:20] = 2900
    names = ["blue", "green", "red", "nir"]
    return write_tiff("jul.tif", scene, names), write_tiff("sep.tif", later, names)


def _scene_metadata(acquired, cloud, scene_id, bbox):
    from satquery.providers import SceneMetadata

    return SceneMetadata(provider="Copernicus Data Space Ecosystem", collection="sentinel-2-l2a",
                         satellite="Sentinel-2", product_level="L2A", acquired=acquired,
                         acquired_datetime=f"{acquired}T06:20:00Z", cloud_cover=cloud, bbox_wgs84=tuple(bbox),
                         crs="EPSG:4326", resolution_m=10.0, bands=["B02 (blue)"], width=32, height=32,
                         scene_id=scene_id, alternatives_considered=4)


@pytest.fixture
def fake_pair(monkeypatch, pair_files):
    """Replace only the network: two real, distinct rasters, as retrieve_pair would write them."""
    from satquery.providers import RetrievedScene

    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "id")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "secret")
    calls = []

    def retrieve_pair(self, bbox, bands, windows, destination_dir, **kwargs):
        calls.append({"bbox": bbox, "windows": windows})
        destination_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for role, source, acquired, cloud, scene_id in (
                ("before", pair_files[0], "2026-07-05", 1.5, "S2A_MSIL2A_20260705"),
                ("after", pair_files[1], "2026-09-19", 0.02, "S2B_MSIL2A_20260919")):
            path = destination_dir / f"{role}.tif"
            path.write_bytes(Path(source).read_bytes())
            out.append(RetrievedScene(path=path, metadata=_scene_metadata(acquired, cloud, scene_id, bbox)))
        return tuple(out)

    def single(*args, **kwargs):  # pragma: no cover - must never run on the temporal path
        raise AssertionError("single-scene retrieval used for a temporal question")

    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve_pair", retrieve_pair)
    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve", single)
    return calls


def test_a_temporal_question_retrieves_two_scenes_with_both_provenance_records(client, fake_pair):
    response = client.post("/api/fetch-imagery", json={"query": TEMPORAL_QUERY, "aoi_bbox": BBOX})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["mode"] == "temporal"
    assert [image["role"] for image in body["images"]] == ["before", "after"]
    before, after = (image["metadata"] for image in body["images"])
    assert (before["acquired"], after["acquired"]) == ("2026-07-05", "2026-09-19")
    assert (before["cloud_cover"], after["cloud_cover"]) == (1.5, 0.02)
    assert before["scene_id"] != after["scene_id"]
    assert body["images"][0]["upload"]["id"] != body["images"][1]["upload"]["id"]
    # both for the drawn area, and the single-date fields still describe the latest scene
    assert before["bbox_wgs84"] == after["bbox_wgs84"] == BBOX
    assert body["upload"]["id"] == body["images"][1]["upload"]["id"] and body["metadata"]["acquired"] == "2026-09-19"
    temporal = body["temporal"]
    assert temporal["basis"] == "relative period" and temporal["days_apart"] == 76
    assert temporal["before_window"]["end"] < temporal["after_window"]["start"]
    assert fake_pair[0]["bbox"] == tuple(BBOX)


def test_a_single_date_question_still_takes_the_single_scene_path(client, monkeypatch, write_tiff, scene):
    from satquery.providers import RetrievedScene

    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "id")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "secret")
    source = write_tiff("single.tif", scene, ["blue", "green", "red", "nir"])

    def retrieve(self, bbox, bands, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
        return RetrievedScene(path=destination, metadata=_scene_metadata("2026-09-19", 0.02, "S2B", bbox))

    def pair(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("two-date retrieval used for a single-date question")

    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve", retrieve)
    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve_pair", pair)
    body = client.post("/api/fetch-imagery", json={"query": "Are there water bodies here?", "aoi_bbox": BBOX}).json()

    assert body["mode"] == "single" and body["temporal"] is None
    assert [image["role"] for image in body["images"]] == ["single"]
    assert body["images"][0]["upload"]["id"] == body["upload"]["id"]  # the fields single-date clients read
    assert body["metadata"]["acquired"] == "2026-09-19"


def test_the_existing_change_pipeline_receives_the_real_pair(client, fake_pair):
    fetched = client.post("/api/fetch-imagery", json={"query": TEMPORAL_QUERY, "aoi_bbox": BBOX}).json()
    images = [{"upload_id": image["upload"]["id"], "modality": "optical"} for image in fetched["images"]]

    result = client.post("/api/analyze", json={"query": TEMPORAL_QUERY, "images": images}).json()
    response = result["response"]
    trace = response["trace"]

    assert response["status"] in ("ok", "partial") and response["task"] == "change_analysis"
    assert trace["input_config"] == "pair_bitemporal"
    assert [image["acquired"] for image in trace["images"]] == ["2026-07-05", "2026-09-19"]
    assert {"vlm.change", "change.map"} <= {step["tool"] for step in trace["plan"]}
    assert response["answer"].startswith("Between 2026-07-05 and 2026-09-19")
    assert "heuristic" in response["answer"], "the deterministic map is labelled a heuristic"
    # the change result is pinned on the map, on the later image
    assert any(layer["image_index"] == 1 and "changed areas" in layer["label"] for layer in result["overlay_layers"])


@pytest.mark.parametrize("problem, status, code", [
    (NoEarlierImagery("No suitable earlier scene."), 404, "no_earlier_imagery"),
    (NoLaterImagery("No suitable later scene."), 404, "no_later_imagery"),
    (OnlyOneAcquisition("Only one suitable Sentinel-2 acquisition was found."), 404, "only_one_acquisition"),
    (GridsIncompatible("The two retrieved scenes do not share a pixel grid."), 502, "grids_incompatible"),
])
def test_temporal_failures_are_structured_and_never_fall_back(client, monkeypatch, problem, status, code):
    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "id")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "secret")
    single_calls = []

    def pair(*args, **kwargs):
        raise problem

    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve_pair", pair)
    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve",
                        lambda *a, **k: single_calls.append(1))
    response = client.post("/api/fetch-imagery", json={"query": TEMPORAL_QUERY, "aoi_bbox": BBOX})

    assert response.status_code == status and response.json()["code"] == code
    assert "upload" not in response.json(), "no imagery is registered on failure"
    assert single_calls == [], "no fallback to one scene"


def test_an_unsupported_period_is_refused_before_any_api_call(client, monkeypatch):
    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "id")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "secret")
    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve_pair",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no API call")))
    response = client.post("/api/fetch-imagery", json={"query": "What changed since 2012?", "aoi_bbox": BBOX})

    assert response.status_code == 422 and response.json()["code"] == "temporal_range_unsupported"
    assert "Sentinel-2" in response.json()["message"]


def test_a_too_large_area_is_still_refused_for_a_temporal_question(client, monkeypatch):
    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "id")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "secret")
    response = client.post("/api/fetch-imagery", json={"query": TEMPORAL_QUERY, "aoi_bbox": [60, 10, 80, 30]})
    assert response.status_code == 413 and response.json()["code"] == "aoi_too_large"


def test_missing_credentials_are_still_reported_for_a_temporal_question(client, monkeypatch):
    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "")
    response = client.post("/api/fetch-imagery", json={"query": TEMPORAL_QUERY, "aoi_bbox": BBOX})
    assert response.status_code == 503 and response.json()["code"] == "credentials_missing"


def test_a_repeated_temporal_request_is_served_from_its_own_cache(client, fake_pair):
    payload = {"query": TEMPORAL_QUERY, "aoi_bbox": BBOX}
    first = client.post("/api/fetch-imagery", json=payload).json()
    second = client.post("/api/fetch-imagery", json=payload).json()

    assert len(fake_pair) == 1, "the second identical request downloads nothing"
    assert first["cached"] is False and second["cached"] is True
    assert [i["metadata"]["acquired"] for i in second["images"]] == ["2026-07-05", "2026-09-19"]


def test_a_single_date_cache_entry_is_never_returned_for_a_temporal_request(client, monkeypatch, fake_pair, write_tiff, scene):
    from satquery.providers import RetrievedScene

    source = write_tiff("single.tif", scene, ["blue", "green", "red", "nir"])

    def retrieve(self, bbox, bands, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
        return RetrievedScene(path=destination, metadata=_scene_metadata("2026-09-19", 0.02, "S2B", bbox))

    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve", retrieve)
    client.post("/api/fetch-imagery", json={"query": "Are there water bodies here?", "aoi_bbox": BBOX})
    temporal = client.post("/api/fetch-imagery", json={"query": TEMPORAL_QUERY, "aoi_bbox": BBOX}).json()

    assert len(fake_pair) == 1, "the temporal request did its own retrieval"
    assert temporal["mode"] == "temporal" and len(temporal["images"]) == 2


# --------------------------------------------------------------------- optical + SAR uploads

def _upload_undeclared(client, path):
    """As the web client uploads: no modality, so the server reads it from the file."""
    with open(path, "rb") as handle:
        response = client.post("/api/uploads", files={"file": (Path(path).name, handle, "image/tiff")})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def optical_sar_pair(write_tiff, optical_scene, sar_scene):
    """Two rasters on one grid, each naming its bands as real Sentinel exports do."""
    return (write_tiff("s2.tif", optical_scene, band_names=["B02", "B03", "B04", "B08"]),
            write_tiff("s1.tif", sar_scene, band_names=["VV", "VH"]))


def test_an_undeclared_upload_takes_its_modality_from_its_band_descriptions(client, optical_sar_pair, write_tiff, scene):
    optical, sar = (_upload_undeclared(client, path) for path in optical_sar_pair)
    unnamed = _upload_undeclared(client, write_tiff("unnamed.tif", scene))

    assert (sar["modality"], sar["modality_basis"]) == ("sar", "from band descriptions VV, VH")
    assert optical["modality"] == "optical" and "B02" in optical["modality_basis"]
    assert unnamed["modality"] == "optical" and unnamed["modality_basis"].startswith("assumed")


def test_a_declared_modality_still_wins(client, optical_sar_pair):
    body = _upload(client, optical_sar_pair[1], modality="optical")
    assert body["modality"] == "optical" and body["modality_basis"] == "declared at upload"


def test_an_uploaded_optical_sar_pair_is_analysed_jointly_with_pinned_fused_layers(client, optical_sar_pair):
    optical, sar = (_upload_undeclared(client, path) for path in optical_sar_pair)
    body = client.post("/api/analyze", json={
        "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "images": [{"upload_id": optical["id"]}, {"upload_id": sar["id"]}]}).json()
    response = body["response"]

    assert response["status"] == "ok" and response["task"] == "cross_modal_analysis"
    assert [s["tool"] for s in response["trace"]["steps"]] == [
        "optical.spectral_indices", "sar.water_mask", "sar.bright_mask", "fusion.cross_modal"]
    assert [i["modality"] for i in response["trace"]["images"]] == ["optical", "sar"]
    layers = body["overlay_layers"]
    assert [layer["label"].split(":")[0] for layer in layers] == ["FUSED water", "FUSED built-up"]
    for layer in layers:
        assert layer["corners_wgs84"] == optical["summary"]["corners_wgs84"]
        image = client.get(layer["url"])
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"


def test_two_undeclared_uploads_of_one_modality_are_refused_not_compared_as_dates(client, write_tiff, scene):
    first = _upload_undeclared(client, write_tiff("a.tif", scene))
    second = _upload_undeclared(client, write_tiff("b.tif", scene))
    response = client.post("/api/analyze", json={
        "query": "Compare optical and SAR evidence to find water.",
        "images": [{"upload_id": first["id"]}, {"upload_id": second["id"]}]}).json()["response"]

    assert response["status"] == "invalid_input" and response["task"] is None
    assert {v["code"] for v in response["trace"]["validation"]} >= {"missing_sar"}


# ------------------------------------------------------------- optical + SAR retrieval (Sentinel-2 + Sentinel-1)

CROSS_MODAL_QUERY = "Use the optical and SAR images together to identify built-up and water-covered regions."


@pytest.fixture
def fake_optical_sar(monkeypatch, optical_sar_pair):
    """Replace only the network: a real optical and a real SAR raster on one grid, as retrieve_optical_sar writes them."""
    from dataclasses import replace

    from satquery.providers import RetrievedScene

    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "id")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "secret")
    calls = []

    def retrieve_optical_sar(self, bbox, bands, destination_dir, **kwargs):
        calls.append({"bbox": bbox, "bands": bands})
        destination_dir.mkdir(parents=True, exist_ok=True)
        optical_meta = _scene_metadata("2026-09-14", 3.0, "S2B_MSIL2A_20260914", bbox)
        sar_meta = replace(optical_meta, collection="sentinel-1-grd", satellite="Sentinel-1D", product_level="GRD",
                           acquired="2026-09-18", acquired_datetime="2026-09-18T01:02:38Z", cloud_cover=None,
                           bands=["VV (co-pol)", "VH (cross-pol)"], scene_id="S1D_IW_GRDH_20260918", modality="sar")
        out = []
        for role, source, meta in (("optical", optical_sar_pair[0], optical_meta), ("sar", optical_sar_pair[1], sar_meta)):
            path = destination_dir / f"{role}.tif"
            path.write_bytes(Path(source).read_bytes())
            out.append(RetrievedScene(path=path, metadata=meta))
        return tuple(out)

    def other_mode(*args, **kwargs):  # pragma: no cover - must never run for a joint optical + SAR question
        raise AssertionError("single-date or temporal retrieval used for an optical + SAR question")

    provider = "satquery.providers.copernicus.CopernicusSentinelProvider"
    monkeypatch.setattr(f"{provider}.retrieve_optical_sar", retrieve_optical_sar)
    monkeypatch.setattr(f"{provider}.retrieve", other_mode)
    monkeypatch.setattr(f"{provider}.retrieve_pair", other_mode)
    return calls


def test_a_joint_question_retrieves_an_optical_and_a_sar_scene(client, fake_optical_sar):
    response = client.post("/api/fetch-imagery", json={"query": CROSS_MODAL_QUERY, "aoi_bbox": BBOX})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["mode"] == "cross_modal" and body["temporal"] is None
    assert [image["role"] for image in body["images"]] == ["optical", "sar"]
    assert [image["upload"]["modality"] for image in body["images"]] == ["optical", "sar"]
    optical, sar = (image["metadata"] for image in body["images"])
    assert (optical["satellite"], sar["satellite"], sar["product_level"]) == ("Sentinel-2", "Sentinel-1D", "GRD")
    assert sar["cloud_cover"] is None and sar["modality"] == "sar"
    assert body["cross_modal"]["days_apart"] == 4 and "4 days apart" in body["cross_modal"]["explanation"]
    assert fake_optical_sar[0]["bbox"] == tuple(BBOX)


def test_compare_optical_and_sar_is_not_mistaken_for_a_comparison_of_dates(client, fake_optical_sar):
    """'Compare ... to' matches the temporal cue too; the joint question must win."""
    body = client.post("/api/fetch-imagery", json={"query": "Compare optical and SAR evidence to find water.",
                                                   "aoi_bbox": BBOX}).json()
    assert body["mode"] == "cross_modal"


def test_the_retrieved_pair_runs_the_joint_analysis_and_states_the_date_gap(client, fake_optical_sar):
    fetched = client.post("/api/fetch-imagery", json={"query": CROSS_MODAL_QUERY, "aoi_bbox": BBOX}).json()
    images = [{"upload_id": s["upload"]["id"], "modality": s["upload"]["modality"], "acquired": s["upload"]["acquired"]}
              for s in fetched["images"]]
    body = client.post("/api/analyze", json={"query": CROSS_MODAL_QUERY, "images": images}).json()
    response = body["response"]

    assert response["status"] == "ok" and response["task"] == "cross_modal_analysis"
    assert "fusion.cross_modal" in [s["tool"] for s in response["trace"]["steps"]]
    assert "acquisition_gap" in {v["code"] for v in response["trace"]["validation"]}
    assert "2026-09-14" in response["answer"] and "2026-09-18" in response["answer"]
    assert len(body["overlay_layers"]) == 2


def test_a_repeated_joint_request_is_served_from_its_own_cache(client, fake_optical_sar):
    payload = {"query": CROSS_MODAL_QUERY, "aoi_bbox": BBOX}
    first = client.post("/api/fetch-imagery", json=payload).json()
    second = client.post("/api/fetch-imagery", json=payload).json()
    assert len(fake_optical_sar) == 1, "the second identical request must not hit the API again"
    assert (first["cached"], second["cached"]) == (False, True)
    assert [s["metadata"]["scene_id"] for s in first["images"]] == [s["metadata"]["scene_id"] for s in second["images"]]


def test_no_sar_scene_is_a_structured_failure_never_an_optical_only_answer(client, monkeypatch):
    from satquery.providers.errors import NoSarImagery

    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "id")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "secret")

    def no_sar(self, *args, **kwargs):
        raise NoSarImagery("No Sentinel-1 dual-polarisation (VV+VH) scene of this area was found.")

    monkeypatch.setattr("satquery.providers.copernicus.CopernicusSentinelProvider.retrieve_optical_sar", no_sar)
    response = client.post("/api/fetch-imagery", json={"query": CROSS_MODAL_QUERY, "aoi_bbox": BBOX})
    assert response.status_code == 404 and response.json()["code"] == "no_sar_imagery"
