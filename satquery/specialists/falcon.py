"""Falcon-Single-Instruction-Large (0.7B remote-sensing VLM) backend: D-021.

Prompts and output parsing follow TianHuiLab/Falcon's README and inference.py:
coordinates are 1000x1000 bins dequantised to bin centres; bi-temporal change detection
takes one composite image with the 'before' image top-left and the 'after' image bottom-right.
torch/transformers are imported lazily so the rest of the app (and the tests) never needs them.
"""

import re
import threading
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from satquery.specialists.vlm import VLMResult

PROMPTS = {
    "caption": "Describe the image.",
    "caption_detailed": "Describe the image in detail.",
    "detect": "Detect all {target} in the image.\nUse horizontal bounding boxes.",
    "ground": "Detect an area that matches the description.\n{description}\nUse horizontal bounding boxes.",
    "segment": "Segment out {target} in the image.",
    "change": "Find changes in the two images.",
}
BINS = 1000


def dequantize(values: list[int], width: int, height: int) -> list[float]:
    return [(v + 0.5) * (width if i % 2 == 0 else height) / BINS for i, v in enumerate(values)]


def clean_text(raw: str) -> str:
    return re.sub(r"</?s>|<pad>", "", raw).strip()


def parse_boxes(raw: str, width: int, height: int) -> list[tuple[float, float, float, float]]:
    return [tuple(dequantize([int(v) for v in m], width, height)) for m in re.findall(r"<(\d+)><(\d+)><(\d+)><(\d+)>", raw)]


def parse_polygons(raw: str, width: int, height: int) -> list[list[float]]:
    polygons = []
    for instance in re.findall(r"<poly>(.*?)</poly>", raw):
        for part in instance.split("<sep>"):
            values = [int(v) for v in re.findall(r"<(\d+)>", part)]
            if len(values) >= 6:
                polygons.append(dequantize(values[: len(values) // 2 * 2], width, height))
    return polygons


def polygons_to_mask(polygons: list[list[float]], width: int, height: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        draw.polygon(list(zip(polygon[0::2], polygon[1::2])), fill=1)
    return np.asarray(canvas, dtype=bool)


def composite_pair(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    blank = np.zeros_like(before)
    return np.concatenate([np.concatenate([before, blank], axis=0), np.concatenate([blank, after], axis=0)], axis=1)


def to_single_frame(polygons: list[list[float]], width: int, height: int) -> list[list[float]]:
    """Map composite (2W x 2H) coordinates onto one image frame.

    Observed on one WHU-CD sample (2026-09-17): Falcon returns change polygons inside the top-left ('before')
    quadrant. Wrapping coordinates modulo the single-image size keeps that case unchanged and stays safe
    if polygons ever appear in the bottom-right quadrant."""
    return [[(v % width) if i % 2 == 0 else (v % height) for i, v in enumerate(p)] for p in polygons]


class FalconVLM:
    def __init__(self, model_id: str, device: str = "auto", num_beams: int = 3, max_new_tokens: int = 1024,
                 adapter: str = ""):
        # With an adapter, model_id names both parts, so every execution trace shows what actually ran (D-027).
        self.model_id = f"{model_id} + {adapter}" if adapter else model_id
        self.base_model_id = model_id
        self.adapter = adapter
        self.requested_device = device
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        # One model, one GPU: loads and generations run one at a time. The HTTP server calls in from a
        # thread pool, and concurrent generate() calls on a 6 GB card risk running out of memory.
        self._lock = threading.RLock()

    def load(self) -> "FalconVLM":
        with self._lock:
            return self._load()

    def _load(self) -> "FalconVLM":
        if self.model is not None:
            return self
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_imports

        # Falcon's remote code imports flash_attn behind an `if`; transformers' import check still requires it.
        def without_flash_attn(filename):
            return [imp for imp in get_imports(filename) if imp != "flash_attn"]

        self.device = self.requested_device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        local_path = snapshot_download(self.base_model_id)
        with patch("transformers.dynamic_module_utils.get_imports", without_flash_attn):
            model = AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=True, torch_dtype=self.dtype)
            self.processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
        if self.adapter:
            # peft is imported lazily, like torch: the app and its tests never need it unless an adapter is set.
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter)
        self.model = model.to(self.device).eval()
        return self

    def _generate(self, prompt: str, rgb: np.ndarray) -> tuple[str, float | None]:
        with self._lock:
            return self._generate_unlocked(prompt, rgb)

    def _generate_unlocked(self, prompt: str, rgb: np.ndarray) -> tuple[str, float | None]:
        import torch

        self._load()
        inputs = self.processor(text=prompt, images=Image.fromarray(rgb), return_tensors="pt")
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                pixel_values=inputs["pixel_values"].to(self.device, self.dtype),
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        sequences = getattr(output, "sequences", output)
        scores = getattr(output, "sequences_scores", None)
        score = float(torch.exp(scores[0])) if scores is not None else None
        return self.processor.batch_decode(sequences, skip_special_tokens=False)[0], score

    def caption(self, rgb, detailed=False):
        raw, score = self._generate(PROMPTS["caption_detailed" if detailed else "caption"], rgb)
        return VLMResult(text=clean_text(raw), score=score, raw=raw)

    def vqa(self, rgb, question):
        raw, score = self._generate(question.strip(), rgb)
        return VLMResult(text=clean_text(raw), score=score, raw=raw)

    def detect(self, rgb, target):
        raw, score = self._generate(PROMPTS["detect"].format(target=target), rgb)
        return VLMResult(text=clean_text(raw), score=score, raw=raw, boxes=parse_boxes(raw, rgb.shape[1], rgb.shape[0]))

    def ground(self, rgb, description):
        raw, score = self._generate(PROMPTS["ground"].format(description=description.strip()), rgb)
        return VLMResult(text=clean_text(raw), score=score, raw=raw, boxes=parse_boxes(raw, rgb.shape[1], rgb.shape[0]))

    def segment(self, rgb, target):
        raw, score = self._generate(PROMPTS["segment"].format(target=target), rgb)
        height, width = rgb.shape[:2]
        mask = polygons_to_mask(parse_polygons(raw, width, height), width, height)
        return VLMResult(text=clean_text(raw), score=score, raw=raw, mask=mask)

    def change(self, rgb_before, rgb_after):
        composite = composite_pair(rgb_before, rgb_after)
        raw, score = self._generate(PROMPTS["change"], composite)
        height, width = rgb_before.shape[:2]
        polygons = to_single_frame(parse_polygons(raw, composite.shape[1], composite.shape[0]), width, height)
        return VLMResult(text=clean_text(raw), score=score, raw=raw, mask=polygons_to_mask(polygons, width, height))
