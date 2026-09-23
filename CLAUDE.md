# CLAUDE.md: SatQuery AI (SIH26167)

Working rules for AI coding agents in this repository. Keep this file short; details live in the linked docs.

## 1. Project in brief
- Smart India Hackathon problem **SIH26167 "SatQuery AI"**: an agentic vision-language assistant that answers
  natural-language queries about single remote-sensing images (optical/multispectral **or SAR**), co-registered
  optical+SAR pairs, and bi-temporal pairs.
- Mandatory:
  - single-image VQA
  - captioning or grounding
  - bi-temporal change description or change-VQA
  - optical–SAR joint analysis
  - agentic selection and sequencing of specialist models/tools from a registry, using only permitted parameters
  - input validation, visual evidence, confidence, an auditable execution trace, and downloadable reports
  - at least one component fine-tuned or adapted on BigEarthNet.txt or open-source data
  - a GUI or web app
- Evaluation: public benchmark test subsets (VRSBench, RSVQA, CDVQA) plus a hidden ISRO/SAC set of
  Cartosat-2S optical + RISAT SAR pairs. The observable execution trace is evaluated.

## 2. Sources of truth (priority order)
1. `Problem_Statement.odt`: what SIH requires. Do not invent requirements.
2. The code in this repository: what is actually implemented.
3. `Initial.md`: historical discussion record. It may be outdated or wrong.
4. `docs/architecture-audit-2026-09-16.md`: audit, proposed architecture, risks, assumptions, open questions.

If sources disagree, report the discrepancy instead of silently picking one.

## 3. Current stage, constraints and facts
- **Stage (updated 2026-09-20):** preparing the first-round SIH submission. Deliverables:
  - a PPT
  - a demo video
  - a **working web-app link**

  The full hackathon comes later, only if the team is selected.
  The deadline recorded on 2026-09-17 was "~2 days"; that date has passed, so **confirm the real deadline** rather
  than relying on this line. Current state and next steps: `docs/handoff-2026-09-20.md`.
- **Priority labels:** tag work as **[R1-REQ]** Round 1 required, **[R1-OPT]** Round 1 optional, **[POST-SEL]** after selection, or **[FUTURE]** production.
- **Priority rule:** a convincing, working end-to-end demo beats polish. Build the smallest version of the approved architecture that also extends cleanly later.
- **Compute:** free Colab/Kaggle notebooks + an RTX 4050 Laptop GPU (6 GB total, ~5.2 GB free measured) with 15.6 GB RAM. No paid or institute GPU.
- **Hosting (D-024, supersedes D-020):** the Round 1 link serves the **map-first app from the GPU laptop through a
  tunnel** (`uvicorn satquery.server:app` + `cloudflared`). A free HF Space is **not available**: HF returns HTTP 402
  for both `cpu-basic` Gradio and ZeroGPU without PRO. The code stays ZeroGPU-compatible for later, which is why
  `app.py` still preloads and these constraints still apply to that path:
  - Gradio SDK only
  - Python 3.12
  - torch ≥ 2.8
  - models on `cuda` at module level; GPU work inside `@spaces.GPU` functions
- **Dataset (D-019):** BigEarthNet **S1 + S2, ~110 GB**, on a team laptop. Never load it whole; never deploy it. Round 1 uses a small curated subset.
- **Evaluation:** how SIH runs final evaluation is unknown.
- **Fact (Colab FAQ):** free Colab runtimes disallow "bypassing the notebook UI to interact primarily via a web UI".

## 4. Current state (update when it changes)
<!-- Round 1 decisions of 2026-09-20: D-024 map-first app is the submission, D-025 point tool removed,
     D-026 change map uses one shared scale. Adaptation plan: docs/adaptation-plan.md. -->

| Area | Status |
|---|---|
| GeoChat-7B VQA (4-bit, Colab T4, Python 3.10, transformers 4.31) | Worked in Colab per `Initial.md`. **Deferred to [POST-SEL]** (D-022). The GeoChat Colab notebook is **out of scope** (D-018). |
| Round 1 VLM: Falcon-Single-Instruction-Large (0.7B) | **GPU gate PASSED** 2026-09-17 on the RTX 4050 (CUDA fp16): 6/6 tasks, median 0.42 s, peak reserved VRAM 1.94 GB. Also runs on CPU at ~17 s/task (`experiments/model_feasibility/`, D-021) |
| `satquery/` package: imaging, validation, rule-based agent, tool registry, evidence/reports, Gradio UI, CLI | Implemented (2026-09-17); synthetic-data tests pass. The old `remote_sensing/` package was ported and removed. |
| Map-first web client: `web/` (React + MapLibre + Vite) served by `satquery/server.py` (FastAPI) | Implemented 2026-09-18 (D-023). Primary interface. Georeferenced uploads are placed on a basemap; evidence overlays are pinned to their raster; trace/confidence/reports sit behind "Details". Verified end to end against all demo scenarios with the fake backend. |
| Georeferencing: `satquery/geo.py` | Implemented 2026-09-18. Pixel -> WGS84 corners for map placement, and `crop_to_bbox` to restrict an analysis to a drawn area (all-or-nothing across a pair; falls back with a stated reason). Returns `None` without a CRS + transform. Tests in `tests/test_geo.py`. |
| Drawn areas (`web/src/map/AoiLayer.tsx` -> `aoi_geometry`) | 2026-09-20: rectangle and circle are press-drag-release, anchored at the press point, with the shape visible while dragging. The real geometry reaches the server; circles and polygons are masked to the shape (pixels outside become NaN nodata) and coverage figures count valid pixels only. Rectangles keep the original box crop (outputs verified identical). |
| Gradio UI (`satquery/ui.py`, `app.py`) | Still working; kept as a fallback and for the HF Spaces path. |
| VQA, caption, grounding, change analysis, optical–SAR fusion, single-SAR-image path | Implemented. **All 8 demo scenarios validated with real Falcon on GPU (2026-09-17)**, all status ok, no failed steps: VQA, caption, grounding, bi-temporal change, change VQA, optical–SAR fusion (water agreement IoU 0.97), single-SAR water, and grid-mismatch rejection. |
| Deterministic tools: SAR water/bright masks, NDVI/NDWI, change map, fusion agreement | Implemented; heuristic and labelled as such. The optical change map scales both dates by **one shared percentile range** (D-026, 2026-09-20), so older bi-temporal figures do not apply. |
| Tests | Python **113 passing** (`pytest`, synthetic data, no GPU); web **25 passing** (`cd web; npm test`). Whole-app browser checks were run outside the repo in the 2026-09-20 session: see `docs/handoff-2026-09-20.md` §2. |
| BigEarthNet S1 classifier (`specialists/s1_classifier.py`) | Ported, **not registered** as a tool ([R1-OPT], D-015) |
| Round 1 web-app link | **Map-first app from the GPU laptop through a tunnel** (`uvicorn satquery.server:app` + `cloudflared tunnel --url http://localhost:8000`), per D-024. The Gradio share link (`SATQUERY_SHARE=1 python app.py`, verified 2026-09-17) stays only as a fallback. Either way the laptop must stay online. Checklist: `docs/round1-submission-kit.md` §3. |
| HF ZeroGPU deployment (`app.py`, `requirements.txt`, `scripts/deploy_space.py`) | Written but **cannot deploy on a free account**: HF returns HTTP 402 for both `cpu-basic` Gradio and ZeroGPU, as hosting either now requires PRO. Code stays ZeroGPU-compatible for later. |
| Evaluation harness, batch manifests | Not implemented ([POST-SEL]) |
| Team-performed fine-tuning/adaptation (SIH R5) | **DONE 2026-09-21 (D-027).** LoRA (r=8, 1.57 M params, 0.188%) on Falcon's 96 decoder attention projections, trained by us on 7,360 balanced BigEarthNet.txt binary VQA rows; 27.4 min, peak VRAM 2.09 GB on the RTX 4050. Held-out exact-match **0.4972 -> 0.6633 (+16.61 pp)** on 1,794 rows from unseen patches, every category improved, no answer collapse. Caption/grounding/change re-checked with the adapter on and off: no regression. Evidence: `experiments/adaptation/results/`. Enabled with `SATQUERY_FALCON_ADAPTER`; **default off**. Adapts to Sentinel-2 10 m only: no Cartosat claim. |

## 5. Decisions: `docs/decisions.md` is the source of truth
- **Approved:** D-001–D-017 (all by default, 2026-09-17), with D-016 replaced by D-018.
- **Also recorded:**
  - D-019: dataset
  - D-020: hosting
  - D-021: Falcon, pending its gate
  - D-022: GeoChat deferred
  - D-023: map-first React client (2026-09-18), which **overrides the map/React cuts in D-012** and the hosting route in D-020
  - D-024: Round 1 is demoed and submitted on the map-first app, tunnelled from the GPU laptop (2026-09-20); Gradio is a fallback only
  - D-025: the point tool is removed from area selection (2026-09-20); a point encloses no area
  - D-026: the bi-temporal change map compares both dates on one shared scale (2026-09-20); **older bi-temporal figures are void**
- **Before changing behaviour covered by a decision,** read its entry. **Do not decide open items silently;** raise them.
- **Still excluded (D-012, as amended by D-023):** database, authentication, live imagery retrieval, Docker, co-registration algorithms, LLM planner, multi-agent.
  A map and React are now in scope (D-023); `satquery/server.py` is a thin HTTP layer over `analyze()`, not a separate service tier.
- **No fake capability (D-023).** Controls with no backend (imagery source/resolution/cloud filters, PDF upload) ship visibly disabled and labelled. Never make one look functional.
- **Before a major architectural change or large refactor,** explain the intent and the reason first.

## 6. Scope rules
- Change only what the task needs, in the module that owns it, plus its tests.
- No drive-by refactors, renames, reformatting, or dependency upgrades.
- Ask before changing any of these:
  - shared data contracts
  - function signatures used by other modules
  - package layout
  - the registry or specialist interface
  - environment or dependency pins
- Do not swap one model for another without an approved decision.
- **GeoChat Colab notebook (D-018):** never modify, restructure, commit, or depend on it, and never ask the team to commit it. If an idea from it matters, document the idea only.

## 7. Honesty rules
- Never fabricate benchmark scores, confidence values, training runs, dataset usage, or model capabilities.
- Label heuristic outputs (fixed thresholds, rule-based labels, bucketed confidences) as heuristic, including in UI text and traces.
- Do not claim a model was fine-tuned unless the run, data, and results are recorded in the repo.
- Do not claim imagery is retrieved automatically; inputs are uploaded or preloaded.
- Do not call an answer "evidence-grounded" unless a visual or spatial artifact supports it.
- In analyses and docs, distinguish FACT, ASSUMPTION, RECOMMENDATION, and UNKNOWN.

## 8. Geospatial and sensor rules
- Preserve CRS, transform, and resolution. Check that pairs share a grid before any pixel-wise comparison.
- Keep pixel, normalized-model, and geographic coordinates distinct. Name variables by coordinate space.
- Avoid absolute thresholds tuned to one sensor. The target data is Cartosat-2S (PAN 0.65 m, MX 2 m) and RISAT (C-band).
- Accepted formats per SIH: GeoTIFF/TIFF, plus PNG/JPEG only for benchmark datasets. A TIFF may lack a CRS.

## 9. Model rules
- Library modules never load models at import time; load once per process and reuse. Exception: `app.py` preloads, because ZeroGPU requires it.
- Respect official wrapper APIs. GeoChat needs `chat.upload_img` → `chat.encode_img` → `chat.ask` → `chat.stream_answer`.
- Verify prompt and output formats (e.g. grounding boxes) against real recorded outputs, not memory.

## 10. Do not assume; look it up or ask
- band order and band roles (Cartosat MX, Sentinel-2)
- SAR polarization (RISAT is not necessarily VV/VH), units (DN / reflectance / dB), calibration
- CRS, nodata values, acquisition dates
- VLM prompt syntax, task tokens, and box formats (GeoChat, VRSBench 0–100, BigEarthNet.txt)
- benchmark answer formats and metrics; hidden-set file formats
- free VRAM on the laptop
- channel order and label order of `BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0` (2 channels, 120×120, classes "0".."18")
- anything listed as UNRESOLVED or UNKNOWN in the audit

## 11. Data, models, secrets
- Never commit datasets, checkpoints, run outputs, or credentials (HF, ngrok, or API tokens), including tokens inside notebooks.
- Large data lives outside git: BigEarthNet S1+S2 (~110 GB) on a team laptop.
- Work with subsets, batches and indexes; never whole-dataset loads.
- Model weights come from the Hugging Face cache, never from the repo.
- A `.gitignore` covers `.venv/`, caches, `data/`, `models/`, `runs/`, `build/`, `.gradio/`, and `.env`. Still check `git status` before every commit.

## 12. Testing
- New or changed behavior needs tests. Prefer small synthetic rasters generated inside tests over committed data files.
- Mark tests that need a GPU or model weights (e.g. `gpu`) so they are skipped by default.
- Test runner: `.venv\Scripts\python -m pytest` (see §14). Tests needing a GPU or model weights are marked `gpu` and skipped by default.
- Web tests run under Vitest (`cd web; npm test`). Keep them to pure logic and the store: importing `maplibre-gl` or
  Terra Draw into a test needs a real map, so drawing helpers live in `web/src/map/aoiGeometry.ts` instead.

## 13. Known limitations (Round 1; do not hide them in answers or slides)
- Falcon change polygons come back in the top-left ('before') quadrant of the 2x2 composite (seen on 1 sample); `falcon.to_single_frame` wraps coordinates.
- Falcon runs ~17 s/task on CPU. On the RTX 4050 (CUDA fp16) it is ~0.3-0.4 s/task median with peak reserved VRAM 1.94 GB (gate passed 2026-09-17, D-021).
- SAR masks use per-image adaptive thresholds, so they report relative darkness/brightness, not calibrated classes.
- Band order is assumed when a GeoTIFF has no band descriptions (validation warns).
- Confidence values are uncalibrated; every value carries its `method`.
- The deterministic change map scales both dates by one shared percentile range (D-026), but the **VLM's** change input is
  still two separately stretched renders (`render_rgb` is per image), so the model may still see rescaling artifacts.
- Inside a drawn circle or polygon the VLM still receives a rectangular image, with the outside rendered black
  (nodata). Its masks and boxes are clipped to the shape afterwards, but its free-text answers may still react to the black border.

## 14. Commands and module map
Commands:
- Tests (Python): `.venv\Scripts\python -m pytest`. Fast, synthetic data, no GPU, model tests skipped.
- Tests (web): `cd web; npm test` (Vitest + jsdom, `src/**/*.test.ts`). Geometry and store only: no browser, no server.
- App (map-first, primary): `uvicorn satquery.server:app` after `cd web && npm run build`. Dev: `npm run dev` beside it.
- App (Gradio fallback): `app.py`. Choose the backend with `SATQUERY_VLM_BACKEND=fake|falcon`.
- CLI: `python -m satquery ask --image ... --modality ... --query ...`

| Path | Owns | Must not import |
|---|---|---|
| `satquery/schemas.py` | all shared contracts (**team approval needed to change**) | anything internal |
| `satquery/imaging.py` | loading, band roles, rendering | agent, specialists, torch, geo |
| `satquery/geo.py` | pixel <-> WGS84 for map display; `summarize()` fills the placement fields | agent, specialists, torch, ui |
| `satquery/examples.py` | demo scenarios + example queries, shared by every front end | gradio, fastapi, agent |
| `satquery/validation.py` | request, image and pair checks | agent, specialists |
| `satquery/raster_analysis.py` | deterministic raster math | agent, specialists, torch |
| `satquery/specialists/` | tool registry (`tools.py`), VLM interface + fake (`vlm.py`), Falcon (`falcon.py`) | agent, ui |
| `satquery/agent/` | intents → plan → execute → aggregate | torch, model libraries |
| `satquery/api.py` | `analyze()`, the only entry point for UI, CLI and tests | ui |
| `satquery/ui.py`, `cli.py`, `server.py`, `app.py` | presentation | specialists directly |
| `web/` | React + MapLibre client; talks to `/api/*` only | anything Python |

Growth rule:
- A module becomes a package only when it exceeds ~300 lines or gains backends.
- New tools are registered in `specialists/tools.py`, and the planner references them by name.
- torch and transformers are imported lazily, inside `falcon.py` only.

## 15. Karpathy-inspired coding discipline

- Think before coding: state assumptions; surface ambiguity and tradeoffs.
- Simplicity first: implement the smallest solution that satisfies the task.
- Surgical changes: modify only what the task requires; do not touch unrelated code.
- Goal-driven execution: define a verifiable success criterion and test against it.
- Before a non-trivial change, briefly state the plan and how it will be verified.