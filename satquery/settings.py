"""Runtime settings, read from SATQUERY_* environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    vlm_backend: str = "fake"  # "falcon" (real model) or "fake" (deterministic, for tests/dev)
    falcon_model_id: str = "mehmetbayik/Falcon-Single-Instruction-Large"  # provenance: docs/decisions.md D-021
    falcon_adapter: str = ""  # LoRA adapter path or HF repo id; empty means the unadapted base model (D-027)
    device: str = "auto"  # "auto", "cuda" or "cpu"
    num_beams: int = 3
    max_new_tokens: int = 1024
    max_pixels: int = 2048 * 2048  # larger rasters are read decimated
    runs_dir: Path = Path("runs")
    max_upload_mb: int = 2048  # web uploads above this are refused (the server may be publicly tunnelled)


def load_settings() -> Settings:
    env = os.environ.get
    defaults = Settings()
    return Settings(
        vlm_backend=env("SATQUERY_VLM_BACKEND", defaults.vlm_backend),
        falcon_model_id=env("SATQUERY_FALCON_MODEL_ID", defaults.falcon_model_id),
        falcon_adapter=env("SATQUERY_FALCON_ADAPTER", defaults.falcon_adapter),
        device=env("SATQUERY_DEVICE", defaults.device),
        num_beams=int(env("SATQUERY_NUM_BEAMS", defaults.num_beams)),
        max_new_tokens=int(env("SATQUERY_MAX_NEW_TOKENS", defaults.max_new_tokens)),
        max_pixels=int(env("SATQUERY_MAX_PIXELS", defaults.max_pixels)),
        runs_dir=Path(env("SATQUERY_RUNS_DIR", str(defaults.runs_dir))),
        max_upload_mb=int(env("SATQUERY_MAX_UPLOAD_MB", defaults.max_upload_mb)),
    )
