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
    assert body["response"]["status"] in {"ok", "partial"}, "the analysis still runs on the whole image"
