"""Runtime settings, read from SATQUERY_* environment variables.

Credentials (Copernicus) come from the repository-root `.env`, which is gitignored. They are never
logged, never returned by an API response, and never sent to the frontend: `/api/health` reports
only whether a provider is configured, not the values.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _load_env_file() -> None:
    """Read `.env` once per process. Real environment variables always win over the file."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is only needed for the Copernicus provider
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


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

    # --- Copernicus Data Space imagery retrieval (optional; unset = feature disabled) ---
    copernicus_client_id: str = ""      # secret: never logged or returned by any endpoint
    copernicus_client_secret: str = ""  # secret: never logged or returned by any endpoint
    copernicus_days_back: int = 30      # how far back the catalogue search looks
    copernicus_max_cloud: float = 20.0  # percent; scenes cloudier than this are not considered
    copernicus_max_aoi_km2: float = 400.0  # a larger drawn area is refused before any API call
    copernicus_resolution_m: float = 10.0  # Sentinel-2 native for B02/B03/B04/B08

    @property
    def copernicus_configured(self) -> bool:
        """Whether retrieval can be attempted. Safe to expose: it reveals no credential value."""
        return bool(self.copernicus_client_id and self.copernicus_client_secret)


def load_settings() -> Settings:
    _load_env_file()
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
        copernicus_client_id=env("COPERNICUS_CLIENT_ID", defaults.copernicus_client_id),
        copernicus_client_secret=env("COPERNICUS_CLIENT_SECRET", defaults.copernicus_client_secret),
        copernicus_days_back=int(env("COPERNICUS_DAYS_BACK", defaults.copernicus_days_back)),
        copernicus_max_cloud=float(env("COPERNICUS_MAX_CLOUD", defaults.copernicus_max_cloud)),
        copernicus_max_aoi_km2=float(env("COPERNICUS_MAX_AOI_KM2", defaults.copernicus_max_aoi_km2)),
        copernicus_resolution_m=float(env("COPERNICUS_RESOLUTION_M", defaults.copernicus_resolution_m)),
    )
