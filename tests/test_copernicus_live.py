"""One real Copernicus request. Opt-in, so the normal suite never consumes API quota.

    .venv\\Scripts\\python -m pytest tests/test_copernicus_live.py -m live

Needs COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in the repository-root .env.
Skipped automatically when they are absent.
"""

import pytest

from satquery.imaging import load_image, render_rgb
from satquery.providers.copernicus import CopernicusSentinelProvider, bands_for_target
from satquery.settings import load_settings

pytestmark = pytest.mark.live

# Navi Mumbai: coastal, so a water question has something to find. ~116 km2, inside the AOI limit.
BBOX = (72.90, 19.00, 73.00, 19.10)


@pytest.fixture
def provider():
    settings = load_settings()
    if not settings.copernicus_configured:
        pytest.skip("Copernicus credentials are not configured in .env")
    return CopernicusSentinelProvider(
        settings.copernicus_client_id, settings.copernicus_client_secret,
        # 60 days and a generous cloud limit: the test area is monsoon-affected in September,
        # where every available scene is 40-50% cloud. This is a connectivity test, not a quality one.
        days_back=60, max_cloud=80.0, max_aoi_km2=settings.copernicus_max_aoi_km2,
        resolution_m=settings.copernicus_resolution_m)


def test_one_live_retrieval_end_to_end(provider, tmp_path):
    """Sign in, search, select, download, and confirm the raster enters the existing pipeline."""
    token = provider.token()
    assert token and isinstance(token, str)
    assert provider.token() is token, "the token must be reused, not re-requested"

    features = provider.search(BBOX)
    assert features, "no Sentinel-2 L2A scenes found for the test area"

    scene = provider.retrieve(BBOX, bands_for_target(None), tmp_path / "live.tif")
    meta = scene.metadata

    assert scene.path.is_file() and scene.path.stat().st_size > 0
    assert meta.satellite == "Sentinel-2" and meta.product_level == "L2A"
    assert meta.acquired and len(meta.acquired) == 10
    assert meta.width > 0 and meta.height > 0

    # compatibility with the existing pipeline is the whole point of the acquisition layer
    image = load_image(scene.path, modality="optical")
    assert image.band_names == ["blue", "green", "red", "nir"]
    assert image.band_names_assumed is False
    assert image.crs is not None
    assert render_rgb(image).shape[2] == 3


def test_one_live_temporal_retrieval_feeds_the_change_pipeline(tmp_path):
    """Two real acquisitions for one area, on one grid, analysed by the existing bi-temporal pipeline."""
    import time
    from datetime import date

    from satquery.api import analyze
    from satquery.providers.temporal import resolve_windows
    from satquery.schemas import AnalysisRequest, ImageInput
    from satquery.settings import Settings
    from satquery.specialists.vlm import FakeVLM

    settings = load_settings()
    if not settings.copernicus_configured:
        pytest.skip("Copernicus credentials are not configured in .env")
    provider = CopernicusSentinelProvider(  # shipped defaults: 20% cloud limit
        settings.copernicus_client_id, settings.copernicus_client_secret,
        max_cloud=settings.copernicus_max_cloud, max_aoi_km2=settings.copernicus_max_aoi_km2,
        resolution_m=settings.copernicus_resolution_m)
    area = (55.40, 25.05, 55.45, 25.10)  # ~5 x 5.5 km inland of Dubai: usually cloud-free
    windows = resolve_windows("How has this area changed over the last 3 months?")

    started = time.perf_counter()
    before, after = provider.retrieve_pair(area, bands_for_target(None), windows, tmp_path / "pair")
    print(f"\n  retrieved in {time.perf_counter() - started:.1f} s")
    for role, scene in (("before", before), ("after", after)):
        m = scene.metadata
        print(f"  {role}: {m.scene_id} | {m.acquired_datetime} | cloud {m.cloud_cover}% | {m.width}x{m.height}")

    b, a = before.metadata, after.metadata
    assert b.acquired < a.acquired and b.scene_id != a.scene_id, "two different acquisitions"
    assert windows.before.start <= date.fromisoformat(b.acquired) <= windows.before.end
    assert windows.after.start <= date.fromisoformat(a.acquired) <= windows.after.end
    assert b.cloud_cover < settings.copernicus_max_cloud and a.cloud_cover < settings.copernicus_max_cloud
    assert b.bbox_wgs84 == a.bbox_wgs84 == area
    assert before.path.read_bytes() != after.path.read_bytes(), "not one image written twice"

    # the existing pipeline takes the pair as-is: the deterministic change map runs on real pixels
    response = analyze(AnalysisRequest(query="What changed here?", images=[
        ImageInput(path=str(before.path), modality="optical", acquired=b.acquired),
        ImageInput(path=str(after.path), modality="optical", acquired=a.acquired)]),
        settings=Settings(vlm_backend="fake", runs_dir=tmp_path / "runs"), vlm=FakeVLM())
    assert response.trace.input_config == "pair_bitemporal" and response.task == "change_analysis"
    change = next(s for s in response.trace.steps if s.tool == "change.map")
    assert change.status == "ok"
    print(f"  change.map: {change.outputs['fraction'] * 100:.1f}% flagged (heuristic)")


def test_one_live_optical_sar_retrieval_feeds_the_cross_modal_pipeline(tmp_path):
    """A real Sentinel-2 scene and the closest real Sentinel-1 scene, on one grid, analysed jointly."""
    from datetime import date

    from satquery.api import analyze
    from satquery.providers.copernicus import SAR_MAX_DAYS_APART
    from satquery.schemas import AnalysisRequest, ImageInput
    from satquery.settings import Settings
    from satquery.specialists.vlm import FakeVLM

    settings = load_settings()
    if not settings.copernicus_configured:
        pytest.skip("Copernicus credentials are not configured in .env")
    provider = CopernicusSentinelProvider(  # shipped defaults: 30 days, 20% cloud limit
        settings.copernicus_client_id, settings.copernicus_client_secret,
        max_cloud=settings.copernicus_max_cloud, max_aoi_km2=settings.copernicus_max_aoi_km2,
        resolution_m=settings.copernicus_resolution_m)
    area = (55.40, 25.05, 55.45, 25.10)  # the temporal test's area: usually cloud-free

    optical, sar = provider.retrieve_optical_sar(area, bands_for_target(None), tmp_path / "pair")
    o, s = optical.metadata, sar.metadata
    print(f"\n  optical: {o.scene_id} | {o.acquired} | cloud {o.cloud_cover}% | {o.width}x{o.height}")
    print(f"  sar:     {s.scene_id} | {s.acquired} | {s.satellite} {s.product_level} | {s.width}x{s.height}")

    assert o.satellite == "Sentinel-2" and s.satellite.startswith("Sentinel-1") and s.product_level == "GRD"
    assert abs((date.fromisoformat(s.acquired) - date.fromisoformat(o.acquired)).days) <= SAR_MAX_DAYS_APART
    assert (o.width, o.height) == (s.width, s.height) and o.bbox_wgs84 == s.bbox_wgs84 == area
    assert load_image(sar.path, modality="sar").band_names == ["VV", "VH"]

    response = analyze(AnalysisRequest(
        query="Use the optical and SAR images together to identify built-up and water-covered regions.",
        images=[ImageInput(path=str(optical.path), modality="optical", acquired=o.acquired),
                ImageInput(path=str(sar.path), modality="sar", acquired=s.acquired)]),
        settings=Settings(vlm_backend="fake", runs_dir=tmp_path / "runs"), vlm=FakeVLM())
    assert response.status == "ok" and response.task == "cross_modal_analysis"
    fusion = next(step for step in response.trace.steps if step.tool == "fusion.cross_modal")
    assert fusion.status == "ok" and set(fusion.outputs) == {"water", "built_up"}
    print("  " + response.answer.replace("\n", "\n  "))
