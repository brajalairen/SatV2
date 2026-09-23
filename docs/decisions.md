# Decision log

Newest context first. Every entry says who decided and whether it is final.
Priority labels: **[R1-REQ]** Round 1 required · **[R1-OPT]** Round 1 optional · **[POST-SEL]** after selection · **[FUTURE]** production.
Full rationale for D-001–D-017 is in `docs/architecture-audit-2026-09-16.md` (§14 and §16).

---

## Current stage (2026-09-17, team)
In ~2 days we submit the **first-round** entry: a PPT, a demo video, and a **working web-app link**. The full hackathon happens only if we are selected.
- **Rule:** a convincing, working end-to-end demo beats polish. Build the smallest version of the approved architecture that supports the demo and can be extended later.
- **Tie-break:** when polishing an existing part competes with building a missing part the demo needs, build the missing part.

## D-027 · SIH R5 adaptation is a LoRA fine-tune of Falcon on BigEarthNet.txt binary VQA (2026-09-21, executes docs/adaptation-plan.md)
- **Why:** R5 is the one mandatory SIH item with no implementation. Falcon is remote-sensing pre-trained by its
  authors, not by us (D-021), so it does not satisfy R5. LoRA on the *same* component the demo runs on means the
  adapted model appears by name in every execution trace.
- **Feasibility, measured on the RTX 4050 (not estimated):**

  | Check | Result |
  |---|---|
  | Architecture | `FalconForConditionalGeneration`, `is_encoder_decoder=True`, DaViT + 12/12 enc-dec, 837.4 M params (confirms the Florence-2 assumption in the plan §3) |
  | PEFT attaches | 96 decoder attention Linears (`self_attn` + `encoder_attn` q/k/v/out_proj); 1,572,864 trainable params (0.188%) |
  | **Peak reserved VRAM** | **2.09 GB** of 6.44 GB, against a 5.2 GB gate — passes with ~3.2 GB spare |
  | Overfit check | loss 4.46 -> 0.013 on 20 samples over 25 epochs: the label path and gradients work |
  | Environment | torch 2.8.0+cu128, transformers **4.49.0 (pin held)**, peft 0.17.1, accelerate 1.10.1 |

  **Train on the RTX 4050. Colab is not needed.** The processor always resizes to 768x768, so the 2.09 GB figure is
  already at full resolution and gradient checkpointing is unnecessary.
- **Three corrections to `docs/adaptation-plan.md`, forced by what is actually on this machine:**
  1. **`metadata.parquet` does not exist here**, so the plan's `--split train --per-class 200` command
     (`scripts/build_round1_subset.py`) cannot run. Patches are selected from **BigEarthNet.txt's own `split`
     column**, stratified by question `category`. **Land-cover class balance is therefore not controlled**; say so.
  2. The dataset root is **doubly nested** (`...\BigEarthNet-S1\BigEarthNet-S1`), defeating the script's path defaults.
  3. `peft`/`accelerate` must be installed **`--no-deps`**, or they drag transformers past 4.50 and break Falcon's
     remote code (D-021). Recorded as the `adaptation` extra in `pyproject.toml`.
- **Scope:** `type == "binary"` rows only. Bounding-box rows need Falcon's `<bin>` coordinate grammar, a separate
  and riskier target that could regress the working grounding path. Exclusion stated in the write-up.
- **Dataset built 2026-09-21:** 7,360 train / 1,064 validation / 1,794 test rows; 2,000 / 300 / 500 patches;
  **yes-rate exactly 0.500 on every split**; splits disjoint by `patch_id` (asserted, not assumed); zero rows
  dropped for missing imagery. Source parquet SHA-256 recorded in the manifest.
- **Integration:** `SATQUERY_FALCON_ADAPTER` (default empty). Set, `model_id` becomes `"<base> + <adapter>"` so every
  trace names the adapted model; unset, behaviour is identical to today and `peft` is never imported. The adapter is
  part of the VLM cache key, so switching it cannot silently return the wrong model.
- **RESULT (2026-09-21, measured):** trained 27.4 min on 7,360 rows, peak VRAM 2.091 GB, adapter 6.3 MB
  (SHA-256 `b7b1388f...36100`), validation loss 2.0671. Held-out exact-match on 1,794 rows from unseen patches:

  | | Before | After | Change |
  |---|---|---|---|
  | **Overall** | 0.4972 | **0.6633** | **+16.61 pp** |
  | presence (n=504) | 0.5496 | 0.7202 | +17.06 pp |
  | count (n=482) | 0.4315 | 0.6411 | +20.96 pp |
  | area (n=483) | 0.4638 | 0.6190 | +15.52 pp |
  | adjacency (n=325) | 0.5631 | 0.6738 | +11.07 pp |

  The test set is balanced (yes rate 0.500), so the 0.4972 baseline is chance. Predictions after are
  1,015 yes / 779 no: **no answer collapse**. Every category improved.
- **Regression check passed:** with the adapter on vs off, grounding boxes are identical, the change mask differs
  by 0.5% (42,545 -> 42,331 px) with the same polygon, and captions stay coherent. Only binary VQA was trained,
  and the untrained tasks did not degrade. Evidence: `experiments/adaptation/results/regression_check.json`.
- **The demo default stays OFF.** Whether Round 1 runs with the adapter is decided *after* the before/after
  evaluation and the 8-scenario regression check — evidence first (CLAUDE.md §7).
- **Honesty limits:** this adapts to BigEarthNet Sentinel-2 at 120x120 px / 10 m. **No claim may be made that it
  improves Cartosat-2S 0.65 m performance.** Evidence in `experiments/adaptation/results/`; adapter weights are not
  committed (CLAUDE.md §11).

## D-026 · The bi-temporal change map compares both dates on one shared scale (2026-09-20, user decision, FINAL for R1)
- **Was:** each date was stretched by its own 2–98 percentiles per band, then differenced
  (`raster_analysis.change_map`). Two dates were therefore measured against two different references.
- **Why that is wrong:** unchanged ground maps to different values whenever the other date's distribution moves, so a
  large real change invents change everywhere else; and a change that shifts the whole scene cancels out entirely.
  Measured on synthetic pairs: the old method found 17% of a quadrant that had genuinely changed, and 5% of a
  scene-wide change.
- **Now:** one percentile range per band, computed from both dates pooled, applied to both. Unchanged ground maps to
  the same value on both dates. SAR was already correct (dB minus dB) and is unchanged.
- **Effect on the demo:** only the bi-temporal scenarios move. Navi Mumbai 2018→2025: the deterministic map goes from
  27.6% to **25.7%** of the scene, and Otsu separability improves from 0.674 to 0.728. No other scenario's numbers change.
- **Still open [POST-SEL]:** the VLM's own change input is still two separately stretched RGB renders
  (`imaging.render_rgb` is per image), so the model can see the same class of artifact. Fixing that means making the
  render pair-aware, which must not apply to optical–SAR pairs, where a shared scale would be wrong.

## D-024 · Round 1 is demoed and submitted on the map-first web app (2026-09-20, user decision, FINAL for R1)
- **What.** The PPT, the demo video and the submitted link all use the map-first client (`web/`, served by
  `satquery/server.py`). The Gradio UI stays in the repository as a fallback only, and is not what we show.
- **Why it is compliant.** The problem statement asks for "an interactive GUI or web application with an agentic
  remote-sensing AI backend" and names no technology. Both front ends call the same `analyze()`; the map client is the
  primary interface under D-023 and shows the georeferenced evidence and the execution trace that SIH evaluates.
- **How the link is served.** `uvicorn satquery.server:app --port 8000` on the GPU laptop plus a tunnel
  (`cloudflared tunnel --url http://localhost:8000`). Hosting on a free HF Space remains blocked (D-020), and a tunnel
  needs the laptop online either way. Checklist and fallback: `docs/round1-submission-kit.md` §3.
- **Supersedes** the "use the Gradio share link" resolution recorded in D-020.

## D-025 · The point tool is removed from area selection (2026-09-20, user decision, FINAL for R1)
- A point encloses no area, so the analysis pipeline can only fall back to the whole image. Offering it implied a
  capability that does not exist (D-023 "no fake capability"), so rectangle, circle and polygon remain.
- The server still reports "a single point has no area to analyse" if a point geometry arrives, because areas saved in a
  browser before this change may still contain one.
- **[POST-SEL]** If a point should ever mean "analyse around here", it needs a stated radius and its own decision.

## D-018 · GeoChat Colab notebook is out of scope (2026-09-17, team, FINAL)
- **Do not** modify, restructure, commit, or depend on the "GeoChat Collab" notebook. **Do not** ask the team to commit it.
- If an idea from it matters, document the idea; leave the notebook untouched.
- This **replaces D-016**.

## D-019 · Local dataset is BigEarthNet S1+S2, ~110 GB (2026-09-17, team, FINAL)
- Correction to the audit, which cited Initial.md's 51 GiB BigEarthNet-S1 figure; 14.1 GB was the GeoChat checkpoint, not data. Treat ~110 GB as the total until measured.
- **Sentinel-2 imagery is available locally.** Assumption #9 in the audit is resolved.
- **Verified on disk 2026-09-17** (previously unverified assumptions):
  - Root is `D:\Hackathon\Overtime\Big EarthNet`, and the imagery folders are **double-nested**:
    `BigEarthNet-S1\BigEarthNet-S1\<product>\<patch>\` and `BigEarthNet-S2\BigEarthNet-S2\<tile>\<patch>\`.
    So `build_round1_subset.py` needs explicit `--s1-dir` and `--s2-dir`; its default `<root>/BigEarthNet-S1` is one level short.
  - **There is no `metadata.parquet`** anywhere on this machine, so the class-driven selection path cannot run.
  - **`BigEarthNet.txt.parquet` replaces it for pairing.** Its schema is
    `ID, s1_name, patch_id, input, output, type, category, split, latitude, longitude, country, season, climate_zone`
    over 9,553,962 rows, and the `s1_name` column gives the authoritative optical-to-SAR mapping. It lives at
    `D:\Hackathon\satquery remote sensing\SatQuery\data\bigearthnet\txt\BigEarthNet.txt.parquet`, outside this repo.
  - Splits: train 4,674,281 · validation 2,454,690 · test 2,409,962 · bench 15,029.
    Types: binary, mcq, bounding box, captioning. 25,903 test patches have affirmative water answers.
  - **S1 and S2 patch pairs share a pixel grid**: identical CRS, transform and 120x120 shape at 10 m, checked on four pairs.
    S2 is `uint16`, S1 is `float32`. This satisfies D-008 without any co-registration.
  - Caution: patch folder names are **not** a reliable pairing key. S2 writes the tile as `T33UUP` and S1 as `33UUP`, and a
    given S2 patch's real S1 counterpart often sits in a different product folder and date. Use `s1_name`, not name matching.
- **Never** load the whole dataset into RAM/VRAM, and **never** deploy it with the web app.
- **[R1-REQ]** Curated subset only:
  - ~20–40 patches chosen through `metadata.parquet` (demo-relevant classes; no cloud or snow)
  - exported as a 4-band S2 GeoTIFF (B02, B03, B04, B08), a 2-band S1 GeoTIFF (VV, VH), a PNG preview, and the patch's BigEarthNet.txt rows
  - stored in `data/subsets/round1/` (gitignored); 6–10 small examples committed under `demo/examples/`
- **[POST-SEL]**
  - Keep the raw data read-only.
  - Convert to an indexed random-access format (`rico-hdl` → LMDB/safetensors).
  - Build stratified subsets of 20k–100k patches joined with BigEarthNet.txt, pre-rendered into shards of a few GB.
  - Stream in batches for fine-tuning.
  - Evaluate on the BigEarthNet.txt `bench` split.

## D-020 · Round 1 web-app link is hosted on a free HF ZeroGPU Space (2026-09-17, team + recommendation, **BLOCKED 2026-09-17: needs a new team decision**)
- **BLOCKER (measured 2026-09-17, account `Brajalen`, `isPro=False`, `canPay=False`, no orgs).** Creating the Space fails with
  `HTTP 402 Payment Required` in both configurations:
  - default `cpu-basic`: "Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription."
  - `space_hardware="zero-a10g"`: "You must be subscribed to PRO to host Spaces with ZeroGPU. If you recently created your account, please wait 30 days or request a community grant."
  - So the bullet below is **not true for this account**: PRO is now required to host ZeroGPU, not merely an account age of 30 days.
  - The app code itself is unaffected and is still ZeroGPU-compatible. Only the hosting route is blocked.
  - Options considered: subscribe to PRO; request an HF community grant (slow); use a Gradio `share=True` link from the
    GPU laptop; or use a teammate's PRO account.
- **RESOLUTION for Round 1 (2026-09-17, team): use the Gradio share link from the GPU laptop.**
  - Enabled by `SATQUERY_SHARE=1` in `app.py`; ignored automatically when running on a Space.
  - **Verified working 2026-09-17**: public URL served HTTP 200 with the real Falcon model on the RTX 4050, then was shut down.
  - Gradio reports the link **expires after 1 week**. The laptop must stay online with the app running for the link to work.
  - Launch it for the judging window and stop it afterwards; each launch produces a new URL, so paste the current one into the
    submission form and the PPT.
  - **Dependency to know about:** share links need `frpc`, a reverse-proxy client that Gradio downloads to
    `<HF_HOME>/gradio/frpc/`. Endpoint protection flagged it here as a Trojan agent and quarantined it. It was restored and its
    SHA-256 verified byte-for-byte against the value pinned in `gradio/tunneling.py`. The tunnel exposes only local port 7860,
    not the filesystem. Re-check this on any machine used for the demo.
  - **[POST-SEL]** Move to a permanent host: HF PRO, a teammate's PRO account, or another provider.
- The team account is older than 30 days, so it may host up to 2 ZeroGPU Spaces. Verified in HF docs. **Superseded by the blocker above.**
- **Constraints (HF docs):**
  - Gradio SDK only
  - Python 3.12.12 or 3.10.13
  - PyTorch 2.8.0 or newer
  - models placed on `cuda` at module level; GPU work inside `@spaces.GPU` functions
  - daily visitor quotas: 2 min unauthenticated, 5 min for free accounts
- **Why not a free CPU Space:** community reports (not officially confirmed) say new free accounts can no longer host Gradio Spaces on CPU Basic.
- **Why not a tunnel:** a laptop tunnel dies when the laptop is off, and Gradio share links expire.
- **Consequences:**
  - The app code must be ZeroGPU-compatible.
  - The same code must also run on a laptop GPU (for the demo video) and on CPU with a fake model (for tests).
  - Model backends are selected by configuration (see D-005).

## D-023 · Map-first React web client, replacing the Gradio UI as the primary interface (2026-09-18, user decision, overrides D-012 and D-020)

- **What changed.** The wireframe brief asked for a map-first product: the map is the interface, a collapsible
  icon rail holds everyday actions, a floating command bar takes a natural-language question, and results appear
  only after an analysis. Gradio cannot express that, so the primary front end is now a React + MapLibre
  single-page app in `web/`, served by a thin FastAPI layer (`satquery/server.py`) that wraps `analyze()`.
- **Overrides D-012** for two of its cuts, "map" and "React". Every other D-012 cut stands: no database, no auth,
  no live imagery retrieval, no Docker, no co-registration, no LLM planner, no multi-agent.
- **Overrides D-020's hosting route.** A React SPA cannot run on a Gradio-SDK Space. FastAPI serves the built
  bundle and the API on one port; the public link is a tunnel (cloudflared/ngrok) from the GPU laptop, the same
  arrangement as the Gradio share link it replaces.
- **The agent is untouched.** `analyze()`, the planner, the executor, the tool registry and the aggregator are
  unchanged apart from one additive field (see below). The Gradio UI (`satquery/ui.py`, `app.py`) still works and
  is kept as a fallback.
- **Additive changes only:**
  - `satquery/geo.py` (new): pixel -> WGS84 for map placement. Returns `None` whenever an image has no CRS +
    transform; it never guesses a location.
  - `satquery/schemas.py`: `ImageSummary` gains `bounds_wgs84`, `corners_wgs84`, `georeference_note`, all
    defaulting to `None`, so the CLI, the tests and the Gradio UI are unaffected.
  - `satquery/agent/aggregator.py`: single-frame overlays now record `image_index` (a field that already
    existed). Side-by-side composites keep `None`, which is how the client knows not to pin them to the map.
  - `satquery/examples.py` (new): demo scenarios and example queries, shared so the server does not import Gradio.
- **A drawn area narrows the analysis (added 2026-09-18 after user testing).** `geo.crop_to_bbox` cuts each source
  raster to the selected box at full resolution, preserving CRS, transform and band descriptions; the agent then
  runs on the crop. It is all-or-nothing across a pair, because `validation.check_pair` requires a shared pixel
  grid. When the crop cannot be applied (no overlap, whole image, under 16x16 px, or no CRS) the request falls
  back to the full images and `AnalyzeResult.area` carries the reason, which the result card shows.
  Drawing an area still does not *fetch* imagery, because no catalogue exists. Verified: all shipped demo GeoTIFFs carry a CRS (EPSG:32634 / 32633 / 32643) and place correctly; the
  Navi Mumbai pair lands on the NMIA site to within the raster's own pixel grid.
- **What is deliberately not connected, and labelled so in the UI (CLAUDE.md §7):**
  - Satellite/source, resolution and cloud-cover filters: shown disabled under "Imagery search - not connected
    in this build". They are catalogue-search parameters and there is no catalogue.
  - PDF and "other data" in the upload menu: greyed, because the pipeline accepts rasters only.
  - Compare / Temporal change: enabled only for a change analysis over two dated images.
  - Place search (Nominatim) moves the camera and says so; it makes no claim about imagery availability.
- **Non-georeferenced inputs keep working.** Plain TIFF without a CRS and PNG/JPEG benchmark images cannot be
  placed on a map, so they open in an off-map image viewer with the identical question and result flow (D-008).
- **Round 1 deliverable impact:** the execution trace, confidence with its method, input checks and the HTML/JSON
  reports are all still present, one click behind "Details" on the result card (R10).

## D-021 · Round 1 vision-language model: Falcon-Single-Instruction-Large, 0.7B (2026-09-17, recommendation, GATE PASSED 2026-09-17)
- **Why:**
  - It is a remote-sensing model (trained on ~78M RS instruction samples).
  - One model covers VQA, captioning, box detection/grounding, segmentation and **bi-temporal change detection**.
  - It is small enough for the 6 GB laptop and for ZeroGPU.
- **Evidence so far (2026-09-17, dev laptop CPU, fp32, beams=1, 3 tasks):**
  - model loads in 9 s; peak RAM 3.5 GB
  - river image classified "river."
  - stadium box correct
  - change polygon correct, and returned in the top-left quadrant frame
  - latency ~17–18 s per task on CPU
  - Full pipeline via `analyze()` on 6 real-image cases, all status ok:
    - stadium box correct
    - roads traced
    - Navi Mumbai 2018→2025 airport change outlined
    - built-up proxy increased from 26% to 52%
  - Falcon finds **no buildings in 10 m Sentinel-2**, so multispectral inputs now use a spectral proxy for area comparisons, and "nothing detected" is reported as inconclusive.
  - **GPU gate PASSED**; measured figures below.

- **GPU gate result (2026-09-17, RTX 4050 Laptop 6 GB, CUDA fp16, `experiments/model_feasibility/`): PASS.** Both runs produced output on 6 of 6 tasks. Measured, not estimated:

  | Run | Median latency excl. warmup | Peak reserved VRAM | Peak process RAM | Model load |
  |---|---|---|---|---|
  | `--num-beams 3` (Falcon reference) | 0.42 s | 1.943 GB | 3.799 GB | 14.1 s |
  | `--num-beams 1` | 0.32 s | 1.926 GB | 5.265 GB | 8.6 s |

  - Thresholds were at most 5.0 GB reserved VRAM and at most 5 s median latency. Both runs clear them by a wide margin.
  - Free VRAM at start was 4.953 GB of 5.997 GB total, so the ~1.9 GB peak leaves ample headroom.
  - Per task the GPU is roughly 40x faster than the ~17 s measured on CPU.
  - Overlays reviewed: the stadium box is tight and correct. The WHU-CD change polygon outlines the changed plot but is drawn in the top-left "before" quadrant, matching the known limitation.
  - Environment: torch 2.8.0+cu128, transformers 4.49.0, Python 3.12.14. Reports and overlay PNGs in `experiments/model_feasibility/results/`.
  - Windows prerequisite: Developer Mode must be on, otherwise `huggingface_hub` fails with `WinError 1314` when linking cache blobs.
- **Gate:** `experiments/model_feasibility/`. PASS when all hold:
  - loads successfully
  - peak reserved VRAM ≤ 5.0 GB
  - median latency ≤ 5 s on GPU
  - at least 4 of 6 tasks produce output
  - the team judges the overlays sensible
- **Source:**
  - The official repo `TianHuiLab/Falcon-Single-Instruction-Large` is gated (manual approval) and stores its files in a subfolder.
  - Round 1 uses the public re-upload `mehmetbayik/Falcon-Single-Instruction-Large`: code, config and tokenizer are hash-identical; weights have identical byte size, but the hash is unverifiable while the official repo is gated.
  - **[POST-SEL]** Request official access and verify the weight hash.
- **Licence metadata is inconsistent** (GitHub: MIT; official HF card: llama3.2; another mirror: apache-2.0). Acceptable for an academic prototype. Resolve before any production use.
- **Pins:**
  - transformers **< 4.50**: Falcon's remote code has no `GenerationMixin`.
  - Patch transformers' import check for `flash_attn`: the import is guarded by an `if`, but the check still flags it.
  - torch **≥ 2.8** (ZeroGPU).
- **Limits to state honestly:**
  - VQA answers are short (benchmark-style), not conversational.
  - It takes RGB input only, so multispectral and SAR inputs are rendered to RGB first.
- **Fallbacks, in order:**
  1. EarthDial_4B_RGB, 4-bit (RS-adapted; install risk)
  2. a small general VLM (e.g. Qwen2.5-VL-3B) with an explicit "not RS-adapted" label

## D-022 · GeoChat-7B deferred to after selection (2026-09-17, recommendation, FINAL for R1)
- **Memory estimate (4-bit, LLaVA-1.5-7B architecture):**
  - ~3.6 GB quantized decoder weights
  - ~0.5 GB unquantized embeddings and output head
  - ~0.6 GB CLIP vision tower
  - ~0.85 GB KV cache for ~1.6k tokens
  - ~0.3–0.5 GB activations
  - ~0.5–0.7 GB CUDA context
  - **total ~6.4–6.8 GB**, against ~5.2 GB free VRAM on the target laptop (measured)
- Loading also needs ~12 GB free RAM for fp16 checkpoint shards; ~6.5 GB was free.
- The 14 GB download and the legacy Python 3.10 / transformers 4.31 stack on Windows are not worth Round 1 time.
- Not validated locally, by choice: the practical cost is too high for the timeline.
- **[POST-SEL]** Re-evaluate on a larger GPU, or via llama.cpp GGUF with partial offload.

---

## D-001 – D-017 · Approved architecture decisions (2026-09-17, team approved all by default)
| ID | Decision | Round 1 scope |
|---|---|---|
| D-001 | Live demo runs where the GPU is; notebooks are for training/eval only. *Round 1: amended by D-020 (video on laptop, link on ZeroGPU).* | [R1-REQ] |
| D-002 | Model feasibility spike before committing to a VLM. *Executed as D-021/D-022.* | [R1-REQ] |
| D-003 | A team-owned fine-tune/adaptation step (floor: small visual/image-text component). | **[POST-SEL]**; [R1-OPT] quick Falcon fine-tune on a BigEarthNet.txt subset |
| D-004 | Restructure into a single `satquery` package; port `remote_sensing/` code. | [R1-REQ] |
| D-005 | Shared contracts in `satquery/schemas.py`; one entry point `analyze()`; no separate backend server. | [R1-REQ], minimal fields only |
| D-006 | Orchestration = rule-based intents → plan templates → executor → trace; no LLM planner. | [R1-REQ] |
| D-007 | Inputs declared per slot (modality, optional date) plus plausibility checks. | [R1-REQ] |
| D-008 | Accept TIFF/PNG/JPEG without a CRS (flagged); pairs must share a grid; no co-registration algorithms. | [R1-REQ] |
| D-009 | Floor implementations first (labelled heuristics); upgrades only after every mandatory task works. | [R1-REQ] |
| D-010 | Confidence carries its method; no uncalibrated bucket presented as calibrated. | [R1-REQ] |
| D-011 | Evidence = spatial artifacts (boxes, masks, change maps, overlays); report = HTML + JSON. | [R1-REQ] |
| D-012 | Cut: ~~map~~, database, auth, live retrieval, ~~React~~, Docker, co-registration, LLM planner, multi-agent. **Map and React reinstated by D-023 (2026-09-18);** the rest stand. | [FUTURE] |
| D-013 | Headless path shares the agent: small `ask` CLI now; batch manifests and eval harness later. | CLI [R1-REQ]; harness [POST-SEL] |
| D-014 | Replay fallback for frozen demo scenarios, clearly labelled. | [R1-OPT] |
| D-015 | BigEarthNet S1 ResNet18 classifier only as a domain-gated optional tool. | [R1-OPT] |
| D-016 | ~~Commit the GeoChat notebook + environment lock~~, **replaced by D-018**. | — |
| D-017 | CLAUDE.md + short docs set. | [R1-REQ] |
| D-023 | Map-first React + MapLibre client in `web/`, served by FastAPI; overrides the map/React cuts in D-012 and the hosting route in D-020. | [R1-REQ] |

## Open (not decided)
- Exact bi-temporal demo pairs (BigEarthNet likely has no repeat dates; use public change-detection samples or Copernicus exports).
- The missing SIH "Evaluation/Judging Criteria" table (check the portal).
