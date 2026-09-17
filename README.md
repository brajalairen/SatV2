# SatQuery AI (SIH26167)

An agentic vision-language assistant for remote-sensing imagery. Supported inputs:
- one optical or SAR image
- a co-registered optical + SAR pair
- a before/after (bi-temporal) pair

You ask a question in plain language. The agent validates the inputs, plans which specialist tools to run, executes them, and returns an answer with visual evidence, a confidence estimate (with its method), a full execution trace, and a downloadable report.

> **Status: Round 1 prototype.** Heuristic components are labelled in the app. See `docs/decisions.md` for scope and priorities.

## How it works
```text
Web client (web/) ─► FastAPI (satquery/server.py) ┐
Gradio UI / CLI ──────────────────────────────────┴─► satquery.api.analyze(request)
   1. load images once (GeoTIFF/TIFF, or PNG/JPEG for benchmarks)       satquery/imaging.py
   2. validate: formats, bands, CRS, pair grid, dates                     satquery/validation.py
   3. input configuration: single_optical | single_sar | pair_cross_modal | pair_bitemporal
   4. rule-based intent: vqa | caption | grounding | change_analysis | cross_modal_analysis   agent/intents.py
   5. plan template: ordered tool calls with permitted parameters        agent/planner.py
   6. execute through the tool registry; record every step               agent/executor.py, specialists/tools.py
   7. aggregate: answer, overlays, confidence, trace, HTML/JSON report   agent/aggregator.py, evidence.py
   8. for the map: pixel -> WGS84 corners so rasters and overlays can be placed   satquery/geo.py
Tools: Falcon 0.7B remote-sensing VLM (VQA, caption, detection, segmentation, change) + deterministic raster tools
       (SAR water / strong-scatterer masks, NDVI/NDWI, change map, optical-SAR fusion)
```

## Quick start (Windows PowerShell; use a short clone path such as `D:\sq\`)
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[ui,dev]"
.venv\Scripts\python -m pytest                      # fast; no GPU or model download
```

### Map-first web app (primary interface)
```powershell
uv pip install --python .venv -e ".[server,dev]"
cd web; npm install; npm run build; cd ..
$env:SATQUERY_VLM_BACKEND="fake"; .venv\Scripts\python -m uvicorn satquery.server:app
```
Open http://127.0.0.1:8000. FastAPI serves both the API and the built client on one port.
Open **Help** in the sidebar and pick a demo scenario for a complete run in two clicks.

For frontend development, run Vite beside the API and use http://localhost:5173 (it proxies `/api`):
```powershell
.venv\Scripts\python -m uvicorn satquery.server:app --reload   # terminal 1
cd web; npm run dev                                             # terminal 2
```

### Gradio UI (fallback)
```powershell
$env:SATQUERY_VLM_BACKEND="fake"; .venv\Scripts\python app.py
```
### Real model
Install the model extras. On a GPU, install the CUDA build of torch first; see `experiments/model_feasibility/README.md`.
```powershell
uv pip install --python .venv -e ".[server,ui,models]"
$env:SATQUERY_VLM_BACKEND="falcon"                       # downloads ~3.4 GB on first run
.venv\Scripts\python -m uvicorn satquery.server:app     # or: .venv\Scripts\python app.py
```
To share a public link for a demo, tunnel port 8000 (for example `cloudflared tunnel --url http://localhost:8000`).

### Command line

```powershell
.venv\Scripts\python -m satquery ask --image opt.tif --modality optical --image sar.tif --modality sar `
    --query "Use the optical and SAR images together to identify built-up and water-covered regions."
```

## What the map does, and what it does not
A GeoTIFF that carries a CRS is placed on the basemap from its own affine transform, and evidence overlays are
pinned to the same footprint. Plain TIFF without a CRS, and PNG/JPEG, cannot be placed, so they open in an
off-map viewer with the identical question and result flow.

Drawing an area **narrows the analysis to the part of your images inside it**: the source raster is cut to that
box at full resolution, keeping its CRS, and the agent runs on the crop. If the area misses your imagery, covers
all of it, or leaves less than 16x16 px, the run falls back to the whole image and the result card says which and
why. A selected area is never silently ignored.

There is **no imagery catalogue and no live retrieval**, so an area drawn over empty map has nothing to analyse. The source, resolution and cloud-cover controls are therefore shipped visibly disabled and
labelled "not connected in this build". Acquisition date and optical/SAR *are* real: they set `ImageInput.acquired`
and `ImageInput.modality`, which decide how the agent routes the request. Place search (Nominatim) moves the
camera only.

## Settings (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `SATQUERY_VLM_BACKEND` | `fake` | `falcon` (real model) or `fake` (tests/dev) |
| `SATQUERY_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `SATQUERY_NUM_BEAMS` | `3` | beam search width (1 is faster) |
| `SATQUERY_RUNS_DIR` | `runs` | where per-request reports and overlays are written |
| `SATQUERY_MAX_PIXELS` | `4194304` | larger rasters are read decimated |

## Other tasks
- **Model feasibility on a laptop GPU:** `experiments/model_feasibility/`
- **Round 1 data subset from local BigEarthNet S1+S2:** `pip install -e ".[data]"`, then `python scripts/build_round1_subset.py --help`
- **Demo scenarios:** `demo/examples/README.md`
- **Deploy the web app to a Hugging Face ZeroGPU Space:** `python scripts/deploy_space.py --space-id <user>/satquery-ai`

## Repository layout
```text
web/                  React + MapLibre client (the primary interface)
satquery/server.py    FastAPI: wraps analyze(), serves web/dist and run artifacts
app.py                Gradio entry point (fallback; local and Hugging Face Space)
satquery/             application package (see "How it works")
tests/                synthetic-data tests (pytest)
scripts/              data subset, dataset inspection, Space deployment
experiments/          isolated one-off experiments (not imported by the app)
demo/examples/        small demo inputs shipped with the app
docs/                 decisions.md (source of truth), architecture audit
CLAUDE.md             rules for AI coding agents
```

## Licence
Apache-2.0 (code). Model and data licences: see `docs/decisions.md` (D-019, D-021) and `demo/examples/README.md`.
