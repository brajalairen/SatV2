"""Shipped demo scenarios and example queries, shared by every front end.

Kept separate from `ui.py` so the HTTP server can use them without importing Gradio.
Manifest format: `demo/examples/README.md`.
"""

import json
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "demo" / "examples"

EXAMPLE_QUERIES = [  # representative queries from the SIH problem statement
    "Describe the land-cover and major objects visible in this image.",
    "Highlight the water body in this image.",
    "What changed between these two dates, and where did the change occur?",
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
    "Has the built-up area increased, decreased, or remained unchanged?",
]


def load_scenarios() -> list[dict]:
    """The demo scenarios as written in the manifest, or [] when it is absent."""
    manifest = EXAMPLES_DIR / "examples.json"
    if not manifest.is_file():
        return []
    return json.loads(manifest.read_text(encoding="utf-8"))
