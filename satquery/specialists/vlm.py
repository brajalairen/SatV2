"""Vision-language model interface (D-005) and a deterministic fake backend.

Every backend takes uint8 RGB arrays (height, width, 3) and returns geometry in the pixel
frame of the array it received. Backends: `falcon` (real, specialists/falcon.py) and `fake`.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import ndimage

from satquery.raster_analysis import otsu, regions


@dataclass
class VLMResult:
    text: str
    score: float | None = None  # model sequence probability if available: uncalibrated
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)  # pixel x_min, y_min, x_max, y_max
    mask: np.ndarray | None = None  # bool (height, width)
    raw: str = ""


class VLMBackend(Protocol):
    model_id: str

    def caption(self, rgb: np.ndarray, detailed: bool = False) -> VLMResult: ...
    def vqa(self, rgb: np.ndarray, question: str) -> VLMResult: ...
    def detect(self, rgb: np.ndarray, target: str) -> VLMResult: ...
    def ground(self, rgb: np.ndarray, description: str) -> VLMResult: ...
    def segment(self, rgb: np.ndarray, target: str) -> VLMResult: ...
    def change(self, rgb_before: np.ndarray, rgb_after: np.ndarray) -> VLMResult: ...


class FakeVLM:
    """Deterministic stand-in for tests and GPU-free development. Its outputs are NOT model predictions."""

    model_id = "fake-vlm (not a model)"

    @staticmethod
    def _luminance(rgb: np.ndarray) -> np.ndarray:
        return rgb.astype(np.float32).mean(axis=2)

    def caption(self, rgb, detailed=False):
        lum = self._luminance(rgb)
        return VLMResult(text=f"[FAKE VLM] Placeholder description of a {rgb.shape[1]}x{rgb.shape[0]} px scene "
                              f"(mean brightness {lum.mean():.0f}/255).")

    def vqa(self, rgb, question):
        return VLMResult(text="[FAKE VLM] yes")

    def detect(self, rgb, target):
        lum = self._luminance(rgb)
        mask = lum > np.percentile(lum, 90)
        found = regions(mask, max_regions=3)
        return VLMResult(text=f"[FAKE VLM] {len(found)} {target}", boxes=[r["bbox"] for r in found])

    def ground(self, rgb, description):
        return self.detect(rgb, "region")

    def segment(self, rgb, target):
        lum = self._luminance(rgb)
        if "water" in target:
            mask = lum < np.percentile(lum, 25)
        elif "veget" in target or "tree" in target:
            mask = rgb[..., 1].astype(np.int16) > rgb[..., 0].astype(np.int16) + 10
        else:
            mask = lum > np.percentile(lum, 85)
        return VLMResult(text=f"[FAKE VLM] segmented {target}", mask=ndimage.binary_opening(mask))

    def change(self, rgb_before, rgb_after):
        diff = np.abs(self._luminance(rgb_after) - self._luminance(rgb_before))
        threshold, _ = otsu(diff)
        mask = diff > threshold if np.isfinite(threshold) else np.zeros(diff.shape, bool)
        return VLMResult(text="[FAKE VLM] change polygons", mask=ndimage.binary_opening(mask))
