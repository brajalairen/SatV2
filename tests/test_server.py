"""HTTP contract of satquery/server.py. Uses the fake VLM backend; no GPU, no model weights."""

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
