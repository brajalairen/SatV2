"""Synthetic raster factories: tests never depend on downloaded data or model weights."""

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin


@pytest.fixture
def write_tiff(tmp_path):
    def _write(name, data, band_names=None, crs="EPSG:32643", origin=(500000.0, 2800000.0), pixel=10.0):
        data = np.asarray(data)
        path = tmp_path / name
        transform = from_origin(origin[0], origin[1], pixel, pixel) if crs else from_origin(0, 0, 1, 1)
        with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
                           dtype=data.dtype, crs=crs, transform=transform) as dst:
            dst.write(data)
            for i, name_ in enumerate(band_names or []):
                dst.set_band_description(i + 1, name_)
        return str(path)
    return _write


@pytest.fixture
def write_png(tmp_path):
    def _write(name, rgb):
        path = tmp_path / name
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
        return str(path)
    return _write


@pytest.fixture
def optical_scene():
    """4-band (blue, green, red, nir) 64x64 reflectance-like scene: water square top-left, vegetation elsewhere."""
    rng = np.random.default_rng(0)
    blue, green, red, nir = (np.full((64, 64), v, np.float32) for v in (400, 700, 500, 3000))
    blue[:20, :20], green[:20, :20], red[:20, :20], nir[:20, :20] = 600, 900, 400, 150  # water: NIR very low
    stack = np.stack([blue, green, red, nir]) + rng.normal(0, 20, (4, 64, 64)).astype(np.float32)
    return stack.astype(np.float32)


@pytest.fixture
def sar_scene():
    """2-band dB scene: dark water block (-22 dB) bottom-right, land around -8 dB, one bright block (+2 dB)."""
    rng = np.random.default_rng(1)
    co = np.full((64, 64), -8.0, np.float32)
    co[40:, 40:] = -22.0
    co[5:12, 5:12] = 2.0
    cross = co - 7.0
    return (np.stack([co, cross]) + rng.normal(0, 0.8, (2, 64, 64))).astype(np.float32)
