"""Web app entry point: Hugging Face ZeroGPU Space (D-020) or local run.

Local, no GPU needed:  SATQUERY_VLM_BACKEND=fake python app.py
Local with Falcon:     SATQUERY_VLM_BACKEND=falcon python app.py
"""

import os

if os.environ.get("SPACE_ID"):  # running on Hugging Face Spaces
    os.environ.setdefault("SATQUERY_VLM_BACKEND", "falcon")
    os.environ.setdefault("SATQUERY_DEVICE", "cuda")
    os.environ.setdefault("SATQUERY_RUNS_DIR", "/tmp/satquery-runs")

from satquery.api import analyze, preload
from satquery.ui import build_demo

try:
    import spaces  # only present on Hugging Face Spaces
except ImportError:
    spaces = None

preload()  # ZeroGPU requires model weights on cuda at import time
run_analysis = spaces.GPU(duration=120)(analyze) if spaces else analyze
demo = build_demo(run_analysis)

if __name__ == "__main__":
    # SATQUERY_SHARE=1 publishes a temporary public *.gradio.live URL. Gradio 5.50 reports it expires
    # after 1 week, and it needs this machine online and the app running (D-020).
    # Ignored on Spaces, which already have their own public URL.
    share = os.environ.get("SATQUERY_SHARE", "").strip().lower() in {"1", "true", "yes"}
    demo.launch(share=share and not os.environ.get("SPACE_ID"))
