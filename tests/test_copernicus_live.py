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
