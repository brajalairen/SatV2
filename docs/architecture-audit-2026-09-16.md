# SatQuery AI (SIH26167): Architecture and Understanding Audit

**Date:** 2026-09-16 · **Repo state audited:** `main` @ `bb7b3c6` · **Mode:** read-only audit. No implementation code was changed.
**Sources:** `Problem_Statement.odt` (PS), `Initial.md`, the full repository (every file read), and external facts checked today (links in §18).

## 0. Context and reading guide

The team asked for a complete audit before building the rest of the system. Three constraints were confirmed during this audit (2026-09-16):

| Constraint | Answer |
|---|---|
| Time to the milestone the full system must serve | **Less than 2 weeks** |
| GPU compute available | **Free Colab / Kaggle + laptop RTX 4050 6 GB** (no paid or institute GPU) |
| How final evaluation is run (benchmarks + hidden ISRO/SAC set) | **Unknown** |

Labels used throughout: **FACT** (verified from the PS text, the repo, or an external source checked today) · **TEAM DECISION** (recorded in Initial.md) · **ASSUMPTION** · **REC** (my recommendation) · **UNKNOWN** · **INFERRED** (my reading, not stated by SIH). §14 uses the requested labels [EXISTING TEAM DECISION] / [YOUR RECOMMENDATION] / [UNRESOLVED].

---

## 1. Executive summary

1. **The only proven AI capability, GeoChat VQA, is not in the repository.** FACT: it exists only in a Colab notebook and a Drive checkpoint (per Initial.md). The repo holds a separate raster-analytics package (`remote_sensing/`, 648 lines, written by a teammate on 14 Sept). Initial.md does not describe that package, and the two lines of work share no interfaces.
2. **Initial.md's implied demo-hosting path is not viable with your compute.** FACT (Colab FAQ): free runtimes disallow *"bypassing the notebook UI to interact primarily via a web UI"*, and all runtimes disallow *"connecting to remote proxies"* and unrelated web services. You also have 12-hour session limits and no GPU guarantee. So the live demo must run on the **laptop**, and notebooks remain for fine-tuning, batch evaluation and pre-computation. **Whether an adequate VLM runs in 6 GB of VRAM is now the most consequential unverified assumption.**
3. **The adaptation requirement is probably not met by the current plan.** The PS says *"The proposed solution must therefore include remote-sensing fine-tuning or domain adaptation"*. GeoChat and the BIFOLD ResNet18 in the repo were both adapted by their authors, not by the team. The safe interpretation is a small, documented fine-tune performed by the team. FACT: BigEarthNet.txt is a text-only parquet (467 MB) with captions, yes/no and multiple-choice VQA, referring-expression boxes and a manually verified `bench` split. The imagery comes separately, and the team already holds BigEarthNet-S1 locally, which makes such a fine-tune tractable.
4. **Several PS requirements are not planned anywhere:**
   - VQA on a **single SAR image**. GeoChat is RGB-only.
   - Accepting **PNG/JPEG and non-georeferenced TIFF** for benchmarks. The current validator rejects both.
   - A **headless batch path** for benchmark evaluation.
   - Enforcing **"only permitted task parameters"**.
   - **Downloadable reports**.
   - **Domain shift** to Cartosat-2S (PAN 0.65 m, MX 2 m) and RISAT (C-band; single, dual, hybrid or quad polarization). Every Sentinel-trained component and every VV/VH assumption is affected.
5. **The existing code is a usable seed for raster tools, but it is not an architecture.**
   - It cannot run from a clean clone: there is no dependency manifest, and `reben_publication` must be obtained from the reBEN git repo.
   - It has no tests and no shared image abstraction. Every function re-reads files by path, one band per Sentinel band.
   - The model is reloaded on every call, and one function is defined twice.
6. **Recommended architecture:** a **modular monolith**, meaning one Python package with a single in-process entry point `analyze(AnalysisRequest) -> AnalysisResponse`. The Gradio UI, a CLI and the evaluation harness all call it. Inside:
   - deterministic validation
   - rule-based intent classification
   - template planning that produces an explicit plan as data
   - an executor that records a structured trace
   - specialists behind one interface, with swappable backends (local model / HTTP worker only if a legacy environment forces it / labelled replay cache / fake).
   
   Deliberately excluded: database, auth, a separate backend server, React, a map, Docker, co-registration algorithms (verification only), and an LLM planner.
7. **Most urgent (days 0–2):**
   - Commit the GeoChat notebook and its environment lock.
   - Run a one-day VLM feasibility spike on the laptop.
   - Freeze the shared contracts.
   - Write CLAUDE.md.
   - Start building the adaptation data subset in parallel.

**Over-engineered in current thinking:** the map UI, a separate "Backend API" tier, a co-registration module, DB/auth placeholders, and "multi-agent" framing.
**Too simplistic today:**
- per-file-only validation
- a fixed raw-unit change threshold
- fixed confidence buckets
- "evidence" meaning a list of labels
- no trace
- hard-coded VV/VH
- no nodata handling

---

## 2. Understanding of the SIH problem (Task 1)

### 2.1 In our own words
Today, analysing satellite imagery requires experts to choose task-specific models and GIS workflows. SIH wants one assistant that works like this:
- A non-expert supplies one image or an image pair and asks a question in plain language.
- The system works out what kind of analysis is needed and checks that the inputs are suitable.
- It runs the right remote-sensing specialist models or tools, sometimes several in sequence, and merges their outputs.
- It returns an answer with visual evidence, a confidence estimate, and an auditable record of what it did.

Single-image VQA is the baseline. The **principal focus is joint reasoning over pairs**: optical+SAR, and two dates.

### 2.2 Users and stakeholders
- FACT: non-expert users in agriculture, disaster management, urban planning, forestry, water resources, infrastructure and environment.
- FACT: ISRO/SAC as evaluator (hidden Cartosat-2S + RISAT set), and the SIH judges.
- INFERRED: analysts who need the trace and report before they can trust an output.

### 2.3 Core objective (FACT, quoted)
*"develop SatQuery AI, a software-based agentic vision-language assistant for analysing single and paired remote-sensing images through natural-language queries. Single-image understanding is a mandatory baseline, while the principal focus is joint reasoning over paired cross-modal and multitemporal imagery."*

### 2.4 Explicitly stated requirements (FACT)

| ID | Requirement (PS) |
|---|---|
| R1 | Inputs: **single** optical/multispectral **or SAR** image (captioning, VQA, grounding); **co-registered optical+SAR pair**; **bi-temporal pair** |
| R2 | Formats: GeoTIFF or TIFF; PNG/JPEG *"only for the prescribed public benchmark datasets"* |
| R3 | ≥1 visual or vision-language component *"fine-tuned or otherwise adapted using BigEarthNet.txt or … any open source training data"*; *"solution must therefore include remote-sensing fine-tuning or domain adaptation"*; *"A generic LLM or VLM without remote-sensing adaptation will not satisfy the requirements"* |
| R4 | Single-image **VQA (mandatory)** |
| R5 | Plus **captioning/scene description OR text-guided grounding** |
| R6 | Bi-temporal **change description OR change-VQA (mandatory)**; change map optional *"where reference masks are available"* |
| R7 | Cross-modal: *"extract complementary information from a co-registered optical/multispectral and SAR image pair"* |
| R8 | Agentic orchestration: *"automatically select, sequence, and execute"* specialists per query and input configuration |
| R9 | Controller must: (a) interpret the query and classify the task; (b) check number, modality, format, metadata and compatibility of the inputs; (c) select tools *"from a predefined registry"*; (d) *"configure only permitted task parameters"*; (e) combine textual and spatial outputs, estimate confidence, return visual evidence; (f) produce an *"auditable execution summary containing the selected task, model/tool names, and key parameters"* |
| R10 | *"only the observable execution trace, including the selected task, models or tools, permitted parameters, and outputs will be evaluated. Internal reasoning text is neither required nor evaluated."* |
| R11 | *"interactive GUI or web application with an agentic remote-sensing AI backend"* |
| R12 | Must include: upload + compatibility checking; an RS-adapted VL component; specialists for VQA, captioning or grounding, change understanding and optical–SAR analysis; an agentic controller; *"visual evidence, confidence information, execution summaries, and downloadable reports"* |
| R13 | Deliverables: the app + backend; *"Codes and models including test and demonstration"* |
| R14 | Evaluation: *"prescribed public benchmark test subsets and an ISRO/SAC evaluation dataset"* (pre-georeferenced, co-registered Cartosat-2S + RISAT pairs; reference answers/labels/boxes/masks; annotations undisclosed). *"Scores will be normalised before combining different metrics."* |
| R15 | Datasets: BigEarthNet.txt (primary adaptation data); VRSBench and RSVQA (single-image captioning/grounding/VQA); CDVQA (change-VQA) |

**FACT:** our copy of the PS contains the placeholder *"Add 'Evaluation/Judging Criteria' table here"*, so **the scoring table is missing**. Check the SIH portal. This is cheap to resolve and could re-weight priorities.

### 2.5 Expected outputs (FACT)
- a natural-language answer
- spatial outputs (boxes, regions, masks, change maps) where the task has them
- visual evidence
- confidence
- an execution summary/trace
- a downloadable report
- for evaluation: task-specific answers, labels, boxes and masks

### 2.6 Constraints (FACT)
- Accepted formats per R2.
- Evaluation annotations are hidden.
- A generic, unadapted VLM is disqualifying.
- The solution is software-only.
- The trace content is prescribed: task, tool names, permitted parameters, outputs.

### 2.7 Requirements that are easy to overlook
1. **A single SAR image is a valid input** (R1), so VQA on SAR alone may be evaluated. Nothing planned handles it.
2. **Plain TIFF without a CRS, and PNG/JPEG for benchmarks, must be accepted.** Validation cannot require georeferencing universally.
3. **"Permitted parameters"** means every tool must declare a parameter schema, and the agent must not invent parameters.
4. **The trace is an evaluated output, not a debug log.** It must be structured, complete and exportable.
5. **Downloadable reports** are explicitly required.
6. **Benchmark evaluation** implies producing answers, boxes and masks for many samples in benchmark-native formats. That needs a headless batch path, not just a GUI.
7. **The hidden set uses different sensors.** FACT (external): Cartosat-2S PAN 0.65 m / MX 2.0 m; RISAT-1A C-band 5.35 GHz, 1–50 m modes, single/dual/hybrid/quad polarization. Sentinel-trained models and VV/VH assumptions will not transfer cleanly.
8. **Hidden-set composition is only partly described.** It is described as optical+SAR *pairs*; whether it also contains bi-temporal pairs is **UNKNOWN**.
9. **Change masks may be scored.** FACT (external): CDVQA/SECOND ships pixel-wise semantic change maps, so *"where reference masks are available"* may apply to benchmark scoring.
10. **Every mandatory task counts.** INFERRED from *"scores will be normalised before combining"*: a missing mandatory task most likely scores zero.
11. **Deliverables include models**, so adapted weights or adapters must be shippable.

### 2.8 Useful but NOT required (INFERRED; do not treat as requirements)
- **Useful:** example-query buttons, live step-by-step progress, labelled replay of frozen demo scenarios, a routing-accuracy test set, geographic coordinates for evidence, model versions in the trace.
- **Not required by SIH:** an interactive map, authentication, a database, live satellite-data retrieval, multi-user support, conversational memory.

---

## 3. Current repository state (Task 3)

### 3.1 Inventory (FACT)
- **Git:** 4 commits, only the `main` branch, remote `github.com/brajalairen/Satellite-Query`.
- **`bf527a5` (2026-09-14, rolenson26):** added `remote_sensing/`. That is 27 files, but only 14 contain code (648 lines). The other **13 are 0-byte**: every `__init__.py`, plus `alignment/coregistration.py`, `geospatial/{crs,raster,reprojection}.py` and `pairs/{bitemporal,optical_sar}.py`.
- **`bb7b3c6` (2026-09-16):** added Initial.md and the PS. README is a one-line title; the licence is MIT.
- **Absent:**
  - no dependency manifest, `.gitignore`, config or `.env` handling
  - no tests, CI or entry point/CLI
  - no UI, no notebooks
  - **no GeoChat or any VLM code**
  - no docs other than Initial.md
- **Environment (this machine):** default Python is 3.14.5. `rasterio`, `pyarrow` and `reben_publication`/`configilm` are **not installed**, so no raster module can import here.

### 3.2 Module status

| File | What it does | Status | Notes / defects |
|---|---|---|---|
| `ingestion/validator.py` | Per file: extension ∈ {.tif,.tiff}; opens with rasterio; checks CRS present, band count > 0, dims > 0 | IMPLEMENTED (per-file only) | **Rejects PNG/JPEG and CRS-less TIFF, which conflicts with R2.** No modality, size or pair checks. |
| `ingestion/loader.py` | Returns an open rasterio dataset | IMPLEMENTED, unused | Handle is never closed |
| `ingestion/inspect_bigearthnet.py` | Prints the first row of `data/bigearthnet/txt/BigEarthNet.txt.parquet` | SCRIPT inside the package | Runs on import; needs pyarrow. **Evidence that the team has the BigEarthNet.txt parquet, which Initial.md does not mention.** |
| `geospatial/metadata.py` | Width, height, bands, resolution, CRS, bounds, dtype of band 1 | IMPLEMENTED | No transform, nodata, band descriptions or acquisition date |
| `optical/bands.py` | S2 band-name table; load one band; stats; `analyze_optical` | IMPLEMENTED, unused by the pipeline | |
| `optical/ndvi.py` | NDVI from separate NIR/Red files; scene-mean label (≥0.6 high, ≥0.3 moderate) | IMPLEMENTED | Nodata is not masked; zero-denominator pixels become NDVI 0 and bias the mean; no raster output; label is a heuristic |
| `optical/cloud.py` | Cloud % from a cloud-probability raster | IMPLEMENTED, unused | Cartosat inputs will not ship such a raster |
| `sar/backscatter.py` | Per-band stats; VV−VH difference | IMPLEMENTED, **defect** | `analyze_s1_backscatter` is **defined twice**; the second silently replaces the first. Assumes values are already in dB. |
| `sar/polarization.py` | Returns `["VV","VH"]` | TRIVIAL | Hard-codes Sentinel polarizations |
| `sar/inference.py` | `S1Inference`: loads BIFOLD `resnet18-s1-v0.2.0`; normalizes with BigEarthNet S1 stats; sigmoid → 19 labels with HIGH/MODERATE/LOW buckets (0.8/0.3); VV/VH `validate_pair`; `analyze()` | IMPLEMENTED, **not runnable from the repo** | FACT (HF config): 2 channels, 120×120 input, 19 classes, class names `"0"`…`"18"`, MIT. **UNVERIFIED:** channel order VV,VH, and whether the repo's alphabetical label list matches the class indices. Per the model card, `reben_publication` must come from `git.tu-berlin.de/rsim/reben-training-scripts` and needs `configilm`. `analyze()` is never called. |
| `change_detection.py` | abs(img2 − img1) on band 1; threshold 100 in raw units; % changed; 4-level label | IMPLEMENTED (naive baseline) | Checks only shape and CRS, not transform or bounds. **The threshold means different things per sensor** (dB vs uint16 reflectance). No change map, no semantics, no description. |
| `fusion.py` | NDVI stats + backscatter stats + S1 classifier + VV/VH alignment, returned side by side; `create_summary`; `analyze_temporal_change` | PARTIAL | **No joint reasoning. No optical↔SAR alignment check.** Loads the S1 model on **every call**. `analyze_temporal_change` duplicates `api.analyze_change`. |
| `api.py` | `analyze_region` (validate 4 files → fusion → summary); `analyze_change` (no validation) | PARTIAL | Imports at module level pull in torch + reben, **even for change detection** |
| `pipeline.py` | Wraps `analyze_region` | TRIVIAL | |
| 6 empty stubs (`alignment`, `crs`, `raster`, `reprojection`, `pairs/*`) | none | PLANNED (intent only) | Empty files suggest these capabilities exist, which misleads developers and AI agents |

### 3.3 Does it work?
- **FACT:** not on this machine's interpreter, and it cannot be reproduced from the repo because there is no manifest.
- **UNKNOWN:** whether it ran on the author's machine. No tests, fixtures or sample outputs were committed.
- **Static reading:** the logic is coherent for Sentinel-style single-band GeoTIFFs. `S1Inference` should work on 120×120 BigEarthNet-S1 patches once its dependencies are installed.

### 3.4 Current implicit architecture
The call chain is `pipeline → api → fusion → {optical.ndvi, sar.backscatter, sar.inference, change_detection}`. Every function takes **file paths** and re-opens files; outputs are ad-hoc dicts with different shapes. Folders mix three organizing principles:
- by modality (`optical/`, `sar/`)
- by pipeline stage (`ingestion/`, `alignment/`, `pairs/`)
- by task (`fusion.py`, `change_detection.py`)

### 3.5 Implicit decisions embedded in the code (FACT)
1. Target sensors are Sentinel-1/2, one band per file (the BigEarthNet layout).
2. "Evidence" means numeric statistics plus classifier labels.
3. Stack: Python, rasterio, numpy, torch.
4. Hard-coded heuristic thresholds: NDVI 0.6/0.3, change 100, confidence 0.8/0.3.
5. No VLM involvement at all.

### 3.6 Technical debt (most serious first)
1. It cannot run from a clone: no manifest, and `reben_publication` is unvendored.
2. There are no tests.
3. There is no shared data model: paths everywhere, dicts with different shapes.
4. Signatures such as `(nir_path, red_path, vv_path, vh_path)` cannot express "one 4-band Cartosat MX GeoTIFF", "RISAT HH/HV" or a PNG benchmark image.
5. The model is loaded on every request.
6. There are duplicate definitions and duplicate APIs.
7. Heavy imports happen at module level.
8. Empty stub files.
9. A script lives inside the package.
10. **There is no `.gitignore`** while the team works with 51 GiB datasets, a 14 GB checkpoint, and tokens in notebooks.

### 3.7 Discrepancies: repo vs Initial.md (FACT)
- Initial.md §3.8 says the repo contents are unknown. It never mentions the S1 classifier, NDVI/backscatter tools, pixel-differencing change detection or the BigEarthNet.txt download. In the other direction, the repo contains **nothing** of the GeoChat work Initial.md calls the "first proven capability".
- The `backend/{agent,router,validators,registry,schemas,evidence,audit,config}.py` modules proposed in Initial.md §4.3 do not exist.
- Initial.md §1.3 paraphrases R3 as "BigEarthNet or other open-source data". The PS says **BigEarthNet.txt**, the image–text dataset. The team downloaded **BigEarthNet-S1 imagery** and the parquet. Both are needed, and S2 imagery is still missing for optical training.

---

## 4. Proposed complete solution, as reconstructed from Initial.md (Task 2)

### 4.1 Overall approach (TEAM DECISION)
- The pipeline is *Ask → Interpret → Validate → Select → Analyze → Verify → Visualize*.
- The **agent is the system and the LLM is only a component.**
- Specialist routing is a core principle.
- Honesty constraints: no fabricated confidence, benchmark numbers, training claims, or imagery-retrieval claims.

### 4.2 Components as intended

| Area | Initial.md intent | Label | In repo? |
|---|---|---|---|
| Frontend | Gradio: uploads, modality display, query box, answer, evidence, confidence, trace, report download; optional map/scene selector | TEAM DECISION (MVP) | No |
| Backend API | "Python orchestration/application layer" between UI and agent; transport unspecified | TEAM DECISION (shape open) | No |
| Agent | Query understanding, validation, requirement determination, model selection, parameters, sequencing, aggregation, confidence/evidence, audit | TEAM DECISION | No |
| Router | Rule-based MVP: two images + SAR → fusion; two images → change; "highlight/where" → grounding; "describe/caption" → caption; else VQA. Must be able to evolve. | TEAM DECISION (MVP) | No |
| Registry | Capability metadata per specialist (task, modality, image count, formats, georef/co-registration needs, params, output schema) | TEAM DECISION + PROPOSAL | No |
| Specialists | GeoChat-7B (VQA/caption/grounding); change model (TBD); SAR processor (TBD); optical/SAR workflow (deterministic first); raster tools | TEAM DECISION (GeoChat) / UNKNOWN (rest) | GeoChat: Colab only. Raster tools and S1 classifier: yes. |
| Contracts | `AnalysisRequest`, `AnalysisResult`, `Evidence`, `ExecutionStep`, `Confidence`; `Specialist.can_handle/analyze` | PROPOSAL | No |
| Evidence/audit | Text, boxes/masks/overlays, derived rasters, metrics, confidence, model metadata, trace; no chain-of-thought | TEAM DECISION | No |
| Database / Auth | None for the MVP | PROPOSAL | None |
| Storage | Drive (GeoChat checkpoint, 14.1 GB); local D: (BigEarthNet-S1, 51 GiB); ephemeral Colab disk | FACT | n/a |
| Model hosting | Colab T4; Python 3.10 uv env; transformers 4.31, torch 2.0.1+cu117; 4-bit loading | TEAM DECISION (verified working) | Not in repo |
| Deployment | Open; HF Spaces suggested but doubted | UNKNOWN | No |
| Data services | Copernicus, Bhoonidhi, EarthExplorer as manual sources only | UNKNOWN | No |
| Evaluation harness | Per-task fixtures, metrics, latency | PROPOSAL | No |
| Adaptation | No team fine-tuning done; compliance question open | UNKNOWN | No |

### 4.3 Intended user workflow
Open the app → supply images (upload or preset) → write a query → the agent checks inputs and configuration → the agent chooses a mode (VQA / grounding / caption / change / optical+SAR) → the specialist runs → evidence is extracted → outputs are combined → the app shows answer + evidence + confidence + trace → the user downloads a report.

### 4.4 Reconstructed end-to-end view (as intended, **not** as recommended)
```text
 LAPTOP (code, preprocessing, UI)                         COLAB T4 (py3.10 env)
 ┌─────────────────────────────────────────────┐          ┌───────────────────────┐
 │ Gradio UI → "Backend API" → SatQuery Agent  │ ──(??)──►│ GeoChat-7B, 4-bit      │
 │   validators · router · registry            │          └───────────────────────┘
 │        │                                    │
 │  change model? · SAR model? · optical/SAR   │
 │        │                                    │
 │  Evidence layer (text, overlays, conf, trace)│
 │        │                                    │
 │  UI panels · report download · (map?)       │
 └─────────────────────────────────────────────┘
```
**Central unresolved contradiction:** the arrow marked `(??)` has no mechanism. TEAM DECISION 5.4 explicitly avoids a live laptop↔Colab dependency, yet the architecture requires one. See §6, issue A1.

---

## 5. Three-source comparison (Task 4)

### 5.1 Requirements-to-system mapping

| Req | Initial.md plan | Repo today | Gap |
|---|---|---|---|
| R1 single optical | GeoChat | No VLM; S2 band stats only | **CRITICAL** |
| R1 single SAR (VQA/caption/grounding) | "SAR processor" (unspecified) | S1 classifier + stats (Sentinel-1 only, no text answers) | **CRITICAL, overlooked** |
| R1/R7 cross-modal pair | Deterministic fusion first | Stats side by side; only VV↔VH alignment checked | IMPORTANT |
| R1/R6 bi-temporal pair | Change specialist TBD | Pixel differencing with % changed | **CRITICAL** (no description or change-VQA) |
| R2 formats | Noted correctly | Only `.tif/.tiff` with CRS accepted | **CONFLICT**, IMPORTANT |
| R3 adaptation | Open question; no team fine-tune | None by team | **CRITICAL** |
| R4 VQA | GeoChat (verified in Colab) | None | **CRITICAL** (not captured in repo) |
| R5 caption/grounding | GeoChat grounding | None | IMPORTANT |
| R8/R9a routing | Rule router | Fixed functions only | **CRITICAL** |
| R9b input checks | Validators | Per-file only | IMPORTANT |
| R9c registry | Planned | None | IMPORTANT |
| R9d permitted parameters | **Not planned** | None | IMPORTANT, overlooked |
| R9e evidence/confidence | Evidence layer; honest confidence | Label lists; fixed buckets | IMPORTANT |
| R9f/R10 trace (evaluated) | Planned | None | **CRITICAL** |
| R11 GUI | Gradio | None | IMPORTANT (the easiest gap to close) |
| R12 downloadable report | Planned | None | IMPORTANT |
| R13 code + models + tests + demo | Implicit | No tests | IMPORTANT |
| R14/R15 benchmark evaluation | Harness proposal | None | **CRITICAL** (evaluation mode unknown) |

### 5.2 Already addressed (fully or partly)
- **In the repo:** GeoTIFF metadata; per-file validation; NDVI and backscatter statistics; the S1 land-cover classifier wrapper; naive change statistics.
- **Outside the repo:** GeoChat VQA (one image, one question).

### 5.3 Possibly unnecessary for SIH
- **Interactive map:** not in the PS. REC: cut.
- **Separate "Backend API" tier:** REC: an in-process function call is enough (§6, A6).
- **Co-registration and reprojection algorithms** (the empty stubs): the PS says ISRO pairs are *pre-georeferenced and co-registered*. REC: **verify** grid compatibility and reject or warn when it fails. Do not implement co-registration.
- **`optical/cloud.py`:** needs a cloud-probability raster that the target data will not have. REC: park it.
- **DB, auth, live retrieval, HF Spaces:** agree with Initial.md; omit.

### 5.4 Contradictions and incorrect assumptions
- **C1.** Decision 5.4 (no live laptop↔Colab link) contradicts the laptop-UI → Colab-model architecture.
- **C2.** "GeoChat covers the single-image tasks" ignores that R1 includes **SAR** single images. FACT: GeoChat documents no SAR or multispectral support.
- **C3.** "Colab T4 is the heavy-inference path" (§13.3 item 2) conflicts with the Colab FAQ for web-UI use on free runtimes and with your compute answer.
- **C4.** The validator requires CRS and TIFF, but the PS allows plain TIFF and benchmark PNG/JPEG.
- **C5.** The code assumes Sentinel VV/VH with one band per file, but evaluation uses Cartosat-2S + RISAT, whose polarizations are not guaranteed to be VV/VH.
- **C6.** The code calls classifier label lists "evidence", but the PS's *"visual evidence"* means spatial/visual artifacts. This terminology collision will mislead people and AI agents.
- **C7.** Initial.md §6.3 rejected laptop GeoChat because of **full precision**. **Laptop + 4-bit was never tested**, and that is exactly the case that now matters.
- **C8.** Initial.md treats BigEarthNet-S1 imagery as progress toward R3. R3 names **BigEarthNet.txt**, the text annotations, which need imagery (S1 and/or S2) plus the parquet.

### 5.5 Missing from both the plan and the repo
- An **image abstraction** (array + band roles + georeferencing + nodata + date) and **renderers** (true colour, PAN, SAR false colour) for VLM input.
- **Modality declaration/detection** and **pair compatibility checks** (optical↔SAR, t1↔t2).
- **Coordinate mapping**: VLM output space → raster pixels → geographic coordinates.
- **Permitted-parameter schemas** per tool.
- A **headless batch CLI** with benchmark adapters and a **report generator**.
- A **run-artifact store**.
- A **replay/fallback path**.
- **Environment management** for two or more Python environments.
- **Test fixtures**.

---

## 6. Architecture audit (Task 5)

Format per issue: **Now** (current approach) · **Problem** · **Consequence** · **Severity** · **Options and tradeoffs** · **REC**.

**A1. Model hosting and process topology: CRITICAL**
- **Now:** GeoChat runs on a Colab T4 in a Python 3.10 environment; the UI and agent would run on the laptop; the transport between them is undefined.
- **Problem:** the only proven GPU host cannot legitimately or reliably serve an interactive demo (FACT: Colab FAQ; 12-hour maximum; no GPU guarantee; venue internet is unknown). GeoChat-7B at 4-bit on 6 GB is **borderline and unverified**. NF4 weights are about 4 GB, plus CLIP ViT-L at 504 px (FACT: GeoChat input is 504×504, so roughly 1.3k image tokens if unpooled, UNVERIFIED), the KV cache, and CUDA overhead.
- **Consequence:** the live demo, or hidden-set evaluation on judges' inputs, could be impossible or cut off mid-session.
- **Options:**

| Option | Advantage | Disadvantage |
|---|---|---|
| (a) GeoChat-7B 4-bit on the laptop | Proven model | Old stack on Windows (bitsandbytes, likely WSL2); no SAR support |
| (b) A ≤4B RS VLM on the laptop (EarthDial-4B) | RS-specific; SAR/MS/bi-temporal training data | Install risk (Python 3.9, flash-attn per README) |
| (c) A small general VLM + team LoRA | Modern stack; also satisfies R3 | Fine-tuning risk in under 2 weeks |
| (d) Colab worker via tunnel | T4 memory | ToS, network and session risk |
| (e) Labelled replay of pre-computed runs | Reliable | Not live |

- **REC:** a one-day laptop spike across (a), (b) and (c) zero-shot. Pick one live VLM by day 2. Keep (e) as a labelled fallback. Put a backend abstraction in front so (d) stays possible without redesign.

**A2. Legacy dependency coupling: IMPORTANT**
- **Now:** GeoChat needs Python 3.10, transformers 4.31, torch 2.0.1 and old tokenizers. Gradio, modern VLMs and configilm want newer stacks.
- **Problem:** one environment forces everything onto the oldest pins.
- **Consequence:** install failures (especially on Windows) and blocked upgrades.
- **Options:** isolate GeoChat as a localhost worker with a three-endpoint JSON contract (adds one process); or choose a live VLM on a modern stack; or port GeoChat to newer transformers (risky, days of work).
- **REC:** if GeoChat is the live VLM, isolate it in `workers/geochat/`. Otherwise run the model in-process. Decide from the A1 spike.

**A3. No shared data contracts: CRITICAL for teamwork and AI agents**
- **Problem:** every integration is bespoke; the UI, trace and report would each need to special-case every tool.
- **REC:** one `schemas.py` (Pydantic; Gradio already depends on it). Specialists return a `StepResult`; `analyze()` returns an `AnalysisResponse`.
- **Cost:** about 200 lines. It prevents the integration crunch of the final week.

**A4. No image abstraction: CRITICAL**
- **Problem:** path-per-band signatures cannot represent R1 inputs. There is repeated I/O, no nodata handling, and no RGB rendering for VLMs.
- **REC:** `load_image(path, declared_modality) -> RasterImage` holding the array, band roles, transform, CRS (optional), nodata and date. Load **once per request**. Renderers produce an RGB view plus the mapping back to raster pixels. Enforce a maximum pixel count and use decimated reads for the VLM view.

**A5. Validation incompatible with the PS: IMPORTANT**
- **REC:** two levels, both deterministic and **authoritative over routing**.
  - **Per image:** format and magic bytes, readability, dimensions, band count, dtype, size limit; CRS is optional and flagged when absent.
  - **Per configuration:** image count and modalities vs the task; for pairs, equal CRS, transform and shape (within tolerance); date order for bi-temporal inputs.
  - **Output:** a `ValidationReport` with machine-readable `code`s.

**A6. "Backend API" tier: overengineering for SIH**
- **Problem:** a UI → HTTP API → agent split adds a server, serialization and deployment for no requirement. Gradio can already expose a programmatic API if one is ever needed.
- **REC:** in-process `analyze()` is the backend API, shared by the UI, CLI and eval harness. Introduce HTTP only at the model-worker boundary (A2), and only if forced.

**A7. Orchestration: rule router (TEAM DECISION), agreed with refinements**
- **Why rules are right here:** with under 2 weeks and 6 GB of VRAM, an LLM planner needs either a second model in memory or a hosted API (network, cost, ISRO data policy). R10 evaluates the **trace**, not internal reasoning.
- **Problem to fix:** single-label keyword routing cannot handle multi-intent queries (e.g., "where did the settlement expand?" is change + localisation).
- **REC:** split routing into four parts:
  1. input-configuration detection (deterministic)
  2. intent classification that returns ranked intents and records the matched rules
  3. **plan templates** keyed by (configuration, intents) that yield ordered steps with permitted parameters
  4. an executor
- Plans are data, so the trace comes for free, multi-step plans are supported, and an LLM classifier can later replace step 2 without touching steps 3–4. Add a golden routing set of 50+ queries that includes the PS's five representative queries.
- **Avoid now:** ReAct loops, LangChain/LangGraph agents, multi-agent designs.

**A8. Confidence: IMPORTANT**
- **Now:** fixed buckets on classifier probabilities.
- **REC:** `Confidence{value | None, method, calibrated=False}`. Methods, depending on the specialist:
  - VLM token probability for closed-form answers (yes/no, multiple choice)
  - classifier probability
  - optical–SAR agreement fraction
  - change-map separability

  The UI always shows the method. Calibrate on the BigEarthNet.txt `bench` split only if time allows.

**A9. Evidence semantics: IMPORTANT**
- **REC:** `Evidence` = box / mask / change_map / overlay image / metric, each with its coordinate space and source step. Classifier label lists become `supporting_observations`.

**A10. Coordinate systems: IMPORTANT (a subtle bug source)**
- **FACT:** box formats differ.
  - VRSBench boxes are normalised 0–100.
  - BigEarthNet.txt boxes are `[x_min y_min, x_max y_max]` (units UNVERIFIED).
  - GeoChat emits rotated boxes (normalisation UNVERIFIED).
- **Problem:** VLM preprocessing may resize or pad.
- **REC:** one tested `coords` module: model space → render pixels → raster pixels → CRS via the affine transform. Variable names state their space (`px_`, `norm100_`, `geo_`).

**A11. Change analysis: CRITICAL**
- **Now:** a raw-unit threshold of 100 on band 1; checks shape and CRS only.
- **Problems:** the threshold is sensor-dependent; there are no semantics, no map and no description; *"has built-up area increased?"* cannot be answered.
- **REC (floor, labelled heuristic):**
  - normalised difference (optical) or log-ratio (SAR) with an **adaptive** threshold (Otsu/percentile) → change mask
  - connected regions → a location phrase (grid quadrants)
  - per-date class proxies (indices when NIR exists; VLM yes/no or grounding otherwise) → templated description and closed-form change-VQA
- **Upgrade options [UNRESOLVED]:** EarthDial bi-temporal; Qwen-VL fine-tuned on CDVQA (FACT: a 2026 study reports recent Qwen VLMs beat earlier specialised CDVQA baselines); TEOChat (LLaMA-2-based, about 7B, too large for the laptop).

**A12. Optical–SAR analysis: IMPORTANT**
- **Now:** results side by side; S1-only classifier; no cross-modal alignment check.
- **REC (floor):**
  - **Water:** low SAR backscatter (adaptive threshold after a speckle filter), cross-checked with optical NDWI where NIR exists.
  - **Built-up:** high SAR backscatter or texture, AND low NDVI (or VLM grounding on the optical image).
  - **Outputs:** masks, area %, **agreement map as confidence**, and notes on disagreements (cloud, shadow, layover).
  - **No absolute dB thresholds.** RISAT calibration and polarization are UNKNOWN.
  - Run the S1 classifier **only when the inputs match its domain** (VV/VH at about 10 m); otherwise record "skipped: domain mismatch" in the trace.

**A13. Model lifecycle: IMPORTANT**
- **Now:** `S1Inference()` is constructed on every call.
- **REC:** lazy singletons loaded once per process, a warm-up at startup, and a health indicator in the UI. Never load a model at import time.

**A14. Failure handling and reliability: IMPORTANT**
- **REC:**
  - The executor catches failures per step and returns `status=partial` with trace entries.
  - Every step has a timeout.
  - Errors reach the UI as structured messages, never raw tracebacks.
  - Non-critical steps (e.g., the optional classifier) never fail the whole run.

**A15. Performance: IMPORTANT**
- **UNKNOWN:** hidden-set image sizes. Cartosat scenes can be very large.
- **REC:**
  - Enforce a maximum pixel count in validation.
  - Use decimated reads for VLM views and windowed processing for masks.
  - Cap each request at **two VLM calls or fewer**.
  - Stream step progress to the UI; this also makes the orchestration visible to judges.

**A16. Security: mostly LOW for a local demo**
- Upload size and pixel limits (decompression bombs); an isolated temp directory.
- No auth on localhost; if a share link is ever used, enable Gradio basic auth.
- **Add `.gitignore` and `.env` handling so tokens never reach git.**
- If an LLM planner is ever added, validate its output against the registry and parameter schemas. R9d requires this anyway.

**A17. Testability, multi-developer work and AI agents: IMPORTANT**
- **Now:** no tests, mixed organization, empty stubs, no contracts. People and agents will guess.
- **REC:** synthetic raster fixtures, fake specialists, golden routing tests, GPU tests marked and skipped by default, and module ownership rules (§7–§8).

**A18. Over-engineering to avoid:** map; backend API tier; microservices; database; auth; co-registration algorithms; LLM planner or multi-agent; Docker/K8s; YAML-driven registry; plugin discovery; task queues; HF Spaces.

**A19. Too simplistic today:**
- validation
- change detection
- confidence buckets
- "evidence" dicts
- missing trace
- missing batch path
- VV/VH hard-coding
- nodata handling
- sensor radiometry (e.g., ESA's L2A reflectance offset from processing baseline 04.00; not verified today, and only relevant if Sentinel-2 L2A is used)

---

## 7. Future project structure: principles (Task 6)

Target: **currently simple, clear boundaries, room to grow.** Each recommendation below states the problem it solves, the complexity it adds, and whether it is needed now.

| # | Recommendation | Problem solved | Complexity added | When |
|---|---|---|---|---|
| S1 | One installable package with `pyproject.toml` (editable install on laptop and notebooks) | The repo is unrunnable from a clone; import paths break between laptop and Colab | 1 file | **Now** |
| S2 | Top level organized by **responsibility** (imaging, validation, agent, specialists, evidence, ui); capability-based inside `specialists/` | Three conflicting organizing principles today | One-time move of ~650 lines | **Now** |
| S3 | A single `schemas.py` for every cross-module type | Ad-hoc dicts; nothing for people or agents to code against | ~200 lines | **Now** |
| S4 | Each specialist implements one interface and **declares** its capabilities and permitted-parameter model | Registry, R9c/R9d, and adding or removing specialists independently | Small base class | **Now** |
| S5 | Backends behind specialists (`local` / `fake` now; `replay` ~day 8; `http` only if needed) | Develop the UI and agent without a GPU; demo fallback; possible worker isolation | ~50 lines per backend | Partly now |
| S6 | `workers/geochat/` with its own locked environment | Legacy pins leaking into the app | One process + README | **Only if** GeoChat is the live VLM |
| S7 | `eval/` kept separate from `tests/` | Benchmark runs are slow, data-dependent and GPU-bound; unit tests must stay fast | One folder | Skeleton now, content ~day 6 |
| S8 | `training/` with a results log per run | R3 evidence must be reproducible and honest | One folder | ~day 3 |
| S9 | `notebooks/` (never imported) + `scripts/` | Scripts inside the package execute on import | Two folders | **Now** |
| S10 | Gitignored `data/`, `models/`, `runs/` (per-request artifacts) | 51 GiB datasets, checkpoints and tokens near git | `.gitignore` | **Now** |
| S11 | Config via one `settings.py` + `.env`; **no YAML registry** | Hard-coded paths and model IDs | 1 file | **Now** |
| S12 | Short `docs/` (architecture, contracts, decisions log, requirements traceability, setup) | Initial.md is historical and long; agents need current, compact truth | 5 short files | **Now** (short) |
| S13 | **Growth rule:** a module starts as one file and is promoted to a package only when it passes ~300 lines or gains backends | Prevents both premature packages and giant files | None | Always |
| — | **Not now:** `src/` layout, monorepo tooling, Docker, microservices, plugin discovery, DB migrations, CI beyond one lint+test action | Value too low for SIH | — | Later, or never |

**Package name.** REC: rename `remote_sensing` → `satquery` **now**. At 648 lines, the rename is cheap today and gets more expensive with every file that imports it. The rename needs your approval. **Flat vs `src/` layout:** REC flat, which is simpler for students and Colab. The main benefit of `src/`, catching packaging mistakes, is small here.

---

## 8. AI-assisted development considerations (Task 7)

### 8.1 Module boundaries (the import rules agents must respect)

| Module | Owns | May import | Must NOT import |
|---|---|---|---|
| `schemas.py` | All shared types and conventions | pydantic, stdlib | Anything internal. **Changes need human approval.** |
| `imaging/` | Loading, rendering, coordinates, indices, SAR utilities | schemas, numpy, rasterio, PIL | agent, specialists, ui, torch |
| `validation.py` | Per-image and pair checks | schemas, imaging | agent, specialists, ui |
| `specialists/<name>` | One capability and its model/backends | schemas, imaging, its own ML libs | agent, ui, **other specialists** |
| `agent/` | Configuration detection, intents, planning, execution, aggregation, trace | schemas, validation, registry, specialists.base | torch or any concrete model library |
| `evidence/` | Overlays, report | schemas, imaging | agent, specialists |
| `ui/`, `cli.py`, `eval/` | Presentation / batch | `satquery.api`, schemas, evidence | specialists directly (**everything goes through `analyze()` so it is traced**) |

REC: enforce this later (~day 5) with one tiny pytest that scans imports. No extra tool is needed.

### 8.2 Worked example: "Implement the satellite-data ingestion module"
- **Where:** `satquery/imaging/io.py` (+ `tests/imaging/test_io.py`).
- **Interface to follow:** `load_image(path, declared_modality) -> RasterImage`, as defined in `schemas.py` and `docs/contracts.md`.
- **May modify:** `imaging/*` and `tests/imaging/*`.
- **Must not modify:** `schemas.py` (ask first), `agent/`, `specialists/`, `ui/`, dependency pins of the worker environment.
- **How to test:** synthetic GeoTIFF/TIFF/PNG fixtures generated in `conftest.py` (with/without CRS, nodata, multiband, float SAR); `pytest -m "not gpu"`.
- **Integration:** the agent calls `load_image` once per request; validation and specialists receive `RasterImage` objects, never paths.

### 8.3 Places where agents must NOT assume (write them down or ask)
- Sensor band order and band roles (Cartosat MX order, S2 band mapping).
- SAR polarization, units (DN vs σ⁰ dB) and calibration.
- CRS, nodata values, date availability.
- VLM prompt syntax and output formats (GeoChat task tokens and box format): verify against **recorded real outputs**.
- Benchmark answer formats and metrics.
- Hidden-set file formats.
- Whether the laptop has free VRAM.
- The S1 classifier's label and channel order.
- Anything labelled UNRESOLVED in `docs/decisions.md`.

### 8.4 Documentation set (kept small)
- `README.md`: install and run.
- `CLAUDE.md`: rules (≤150 lines, links out).
- `docs/architecture.md`: 1–2 pages.
- `docs/contracts.md`: schemas, specialist protocol, coordinate conventions, worker JSON contract if any.
- `docs/decisions.md`: dated, labelled decisions.
- `docs/requirements-traceability.md`: R1–R15 → module → test → demo scenario.
- A short README inside each specialist: capability, parameters, input assumptions, known limits, how to test.

### 8.5 API contracts
- The primary contract is **in-process Python** (Pydantic models).
- If a worker exists, its HTTP JSON contract is documented with examples, and the main app validates responses with the same Pydantic models.
- JSON Schema can be exported from Pydantic for the docs, with no extra tooling.

---

## 9. Risks and failure modes (Task 9)

| ID | Risk | Class | Eventual mitigation |
|---|---|---|---|
| K1 | No legitimate, reliable GPU host for the live demo (Colab FAQ; 6 GB laptop) | **CRITICAL** | Day-1 laptop spike; smaller-VLM fallback; labelled replay; degrade to raster-only specialists |
| K2 | R3 judged unmet (no team-performed adaptation) | **CRITICAL** | Small team fine-tune with documented data, config and before/after metrics on the BigEarthNet.txt `bench` split; precise wording in the report |
| K3 | Hidden-set domain shift (Cartosat 0.65/2 m; RISAT polarization/calibration) breaks Sentinel-tuned logic | **CRITICAL** | Adaptive thresholds; declared band roles; generic co-/cross-pol SAR rendering; domain-gated tools; domain warnings in the trace; test on any Indian sample data you can obtain |
| K4 | A mandatory task (change-VQA, SAR VQA, cross-modal) is still missing at the milestone | **CRITICAL** | Vertical slices: every mandatory task gets a working **floor** by ~day 7 before any upgrade |
| K5 | Evaluation mode surprise (batch predictions, specific output formats) | **CRITICAL** | Manifest-driven headless CLI with JSONL outputs by ~day 6 |
| K6 | The working GeoChat setup is lost (lives only in Colab/Drive) | **CRITICAL** | Commit the notebook + `uv pip freeze` lock now |
| K7 | Late integration failure between two uncoordinated code lines | **CRITICAL** | Contracts first; fakes enable end-to-end runs from day 1 |
| I1 | Router misclassifies multi-intent or unusual queries | IMPORTANT | Golden set; show intents in the UI; optional user task override (recorded in the trace) |
| I2 | Unexpected inputs: huge rasters, float/16-bit scaling, NaN/nodata, RGBA PNG, CRS or size mismatch in pairs | IMPORTANT | Validation codes; normalization rules; fixtures per case |
| I3 | Model load failure or OOM on the laptop (other GPU apps, driver issues) | IMPORTANT | Warm-up + health check; clear error; smaller model or replay fallback |
| I4 | Colab/Kaggle disconnect during fine-tuning or evaluation | IMPORTANT | Checkpoint to Drive/Kaggle output every N steps; resumable eval (skip finished IDs) |
| I5 | Windows install failures (rasterio, bitsandbytes, flash-attn) | IMPORTANT | Pin versions; verify on **the demo laptop** in week 1; WSL2 plan |
| I6 | Multi-step plans too slow | IMPORTANT | ≤2 VLM calls per request; cache by (image hash, prompt); progress streaming |
| I7 | VLM hallucination, especially on SAR renders | IMPORTANT | Templated answers that cite tool evidence; confidence method shown; tool-derived facts preferred for SAR |
| I8 | Wrong label or channel order in the S1 classifier | IMPORTANT | Verify against reBEN/configilm code; test on a known BigEarthNet patch |
| I9 | Coordinate-mapping bugs (resize/pad) | IMPORTANT | Unit tests with synthetic boxes; visual overlay check |
| I10 | Merge conflicts from many people editing shared files | IMPORTANT | Module ownership; `schemas.py` owned by one person; small PRs |
| I11 | Dataset licences (e.g., DOTA images in VRSBench are academic-only) | LOW | Do not redistribute images in the repo |
| L1 | Concurrent requests | LOW | Single-user demo; Gradio queue concurrency 1 |
| L2 | Auth failures / DB failures | LOW | Neither exists by design |
| L3 | Malicious TIFFs | LOW | Local demo; recent GDAL; size limits |
| L4 | `runs/` fills the disk | LOW | Cleanup script |

**Demo-day checklist risks:** GPU driver or power-mode throttling on the laptop; no internet (**nothing may require it**); minutes-long cold model load (start early, warm up); missing example inputs (bundle them); no rehearsal (script it); a total failure (recorded backup video, clearly labelled).

---

## 10. Assumptions (Task 8)
★ marks assumptions that would force architectural change if proven false.

| # | Assumption | Status | Basis |
|---|---|---|---|
| 1 | GeoChat gives useful RS VQA | PARTIALLY VERIFIED | One image, one question (team record) |
| 2 | GeoChat captioning/grounding outputs are usable and parseable | UNVERIFIED | Supported per README; formats untested by the team |
| 3 ★ | GeoChat-7B 4-bit runs acceptably on the RTX 4050 6 GB | UNVERIFIED | Borderline memory estimate; old stack on Windows |
| 4 ★ | A ≤4B RS or general VLM runs well on the laptop | UNVERIFIED | EarthDial-4B exists (MIT) but lists Python 3.9 + flash-attn 2.3.6 |
| 5 ★ | Free Colab can host the interactive demo | **FALSE per Colab FAQ** (free tier); Kaggle UNVERIFIED | FAQ quoted in §1 |
| 6 ★ | Using a model already RS-adapted by others satisfies R3 | UNVERIFIED (risky) | PS wording "solution must include…" |
| 7 | The team can fine-tune a VLM on BigEarthNet.txt within free T4 hours | PARTIALLY VERIFIED | A public Qwen3-VL LoRA on BigEarthNet.txt exists; our throughput is untested |
| 8 | BigEarthNet.txt annotations are valid for S1-only (SAR) training | PARTIALLY VERIFIED | Dataset is co-registered S1+S2; per-modality validity not checked |
| 9 | BigEarthNet-S2 imagery can be obtained in time (team has only S1) | UNVERIFIED | Size and download time unknown |
| 10 | S1 classifier channel order VV,VH and alphabetical label order are correct | UNVERIFIED | HF config lists 2 channels and class names "0".."18" only |
| 11 | S1 classifier transfers to RISAT | UNVERIFIED, likely weak | Different sensor, polarization, resolution, geography |
| 12 ★ | ISRO pairs share an identical pixel grid | PARTIALLY VERIFIED | PS says co-registered; grid equality unstated |
| 13 | The ISRO set has no bi-temporal pairs | UNKNOWN | PS mentions only optical+SAR pairs |
| 14 ★ | ISRO SAR arrives calibrated (dB) with known polarizations | UNKNOWN | RISAT supports many polarization modes |
| 15 | Input image sizes fit in laptop memory | UNKNOWN | |
| 16 ★ | Evaluation runs our system (vs prediction files vs live demo) | UNKNOWN | Team answer |
| 17 | Rule-based orchestration counts as "agentic" | PARTIALLY VERIFIED | R10: only the observable trace is evaluated |
| 18 | Gradio is sufficient as the GUI | PARTIALLY VERIFIED | PS allows "GUI or web application"; usability untested |
| 19 | Deterministic tools count as "specialist tools" | PARTIALLY VERIFIED | PS says "models or tools"; mask scoring quality unknown |
| 20 | Venue internet is available | UNVERIFIED | Design must not depend on it |
| 21 | Benchmark "test subsets" are small enough for free T4 hours | UNKNOWN | |
| 22 | Cartosat-2S MX includes a NIR band | UNVERIFIED | Needed for NDVI/NDWI on the hidden set |

---

## 11. Alternative approaches (Task 10)

**D1. Live-demo VLM** [UNRESOLVED; decide by day 2]

| Option | Advantages | Disadvantages | Effort | SIH fit | When it makes sense |
|---|---|---|---|---|---|
| a. GeoChat-7B 4-bit on laptop (+ worker env) | Proven; RS-specific; grounding | VRAM borderline; legacy stack on Windows; no SAR/MS | 1–3 d | High **if it fits** | Spike shows <6 GB and acceptable latency |
| b. EarthDial-4B on laptop | RS-specific; trained on RGB+SAR+MS; bi-temporal | Install risk (py3.9, flash-attn); formats unknown; no team experience | 2–4 d | High if installable | Spike installs cleanly |
| c. Small general VLM (e.g., Qwen2.5-VL-3B / small Qwen3-VL, sizes to verify) + team QLoRA | Modern stack; documented grounding output; **directly satisfies R3**; fits laptop | Fine-tune risk in <2 weeks; quality vs GeoChat unknown | 3–6 d | High if the fine-tune lands | a/b fail, or R3 needs a VLM fine-tune anyway |
| d. Colab worker via tunnel | T4 memory; proven env | FAQ restrictions; network; 12 h limits | 1 d | Low as primary | Never as the only path |
| e. Labelled replay only | Reliable | Not live; judges' own inputs fail | 1 d | Fallback only | Always, as backup |

**D2. Adaptation (R3)** [UNRESOLVED]
- **a. Rely on pre-adapted models.** Zero effort; high compliance risk.
- **b. LoRA/QLoRA of the live VLM on a BigEarthNet.txt subset** (S1 false-colour renders and/or S2 RGB).
  - For: strongest compliance, and it also **addresses SAR VQA**.
  - Against: 2–4 days; needs an imagery subset and T4 hours.
- **c. Fine-tune a smaller visual or image–text component on BigEarthNet.txt** (e.g., a CLIP-style model on captions, or the BIFOLD classifier).
  - For: 1–3 days, low risk, and matches the PS phrase *"adapting image–text representations"*.
  - Against: less direct impact on VQA quality.
- **d. Fine-tune for change-VQA on CDVQA.** Targets the hardest task; 3–5 days; higher risk.
- **REC:** attempt (b) if a one-day trial run shows it trains on T4; otherwise **(c) as the guaranteed floor**. Report exactly what was trained, on what, with before/after numbers.

**D3. Orchestration**
- **Rules + plan templates (REC).** Deterministic, testable, instant, explainable.
- **Local small LLM classifier.** Costs VRAM and latency.
- **Hosted LLM API.** Network, cost and data-policy risk.
- **Hybrid.** Rules first, with the VLM as a constrained fallback classifier for ambiguous queries; later, if time allows.

**D4. UI**
- **Gradio Blocks, in-process (REC; TEAM DECISION).** Python-only; built-in image and annotated-image components.
- **Streamlit.** Similar effort; weaker for event-driven flows.
- **FastAPI + React.** Best polish, but days of front-end work the team does not want.

**D5. Change analysis** (in order of increasing effort)
1. Deterministic floor (REC first).
2. EarthDial bi-temporal (if D1 = b).
3. Semantic change model trained on SECOND, which also yields CDVQA-style answers.
4. Qwen-VL fine-tuned on CDVQA.
5. TEOChat (about 7B; notebook-only).

**D6. Optical–SAR analysis**
- **Deterministic masks + agreement (REC floor).**
- **Learned dual-branch segmentation.** Needs paired labels; domain shift to Cartosat/RISAT.
- **Multi-sensor VLM.** Only if D1 = b.

**D7. Process topology**
- **Single process (REC default).**
- **App + localhost model worker.** Only if a legacy environment forces it.
- **Microservices.** No.

**D8. Modality determination**
- **User-declared input slots + automatic plausibility checks (REC).** Unambiguous, and the checks still satisfy R9b.
- **Fully automatic detection.** Brittle on arbitrary TIFFs.

**D9. Storage**
- **Filesystem `runs/<run_id>/` (REC).**
- **SQLite.** Only if browsable history becomes a requirement.

### 11b. Hackathon reality check (Task 11)
- **Workload.** The remaining work is large for under 2 weeks; it needs 3–4 parallel tracks plus a strict cut list.
- **Impressive but unnecessary or risky:**
  - map + live imagery retrieval
  - LLM planner / multi-agent designs
  - microservices / Docker
  - learned fusion networks
  - training from scratch
  - custom React UI
  - confidence-calibration research
  - HF Spaces deployment
- **Too simplistic, and will cost points:** validation; change detection; evidence semantics; missing trace; missing batch path; VV/VH hard-coding; no tests.
- **Explaining it to judges.** The recommended pipeline fits on one slide: *Inputs → Validate → Classify intent → Plan (shown) → Execute specialists (traced) → Evidence + Confidence + Report*. Every box maps to R8–R12.
- **Honest demo script.** The four mandatory scenarios, one rejected invalid input, and one multi-step query, all run from a cold start.

---

## 12. Open questions (Task 12)

**CRITICAL (decide before major implementation)**
1. Which VLM runs live on the laptop? (D1 spike, by day 2)
2. Which team-owned adaptation will be done, on what data? Is S1 enough, or must S2 be downloaded? (D2)
3. Do you approve the contracts (§14.3) and the single `analyze()` entry point?
4. Do you approve the restructure and rename to `satquery` and the migration map (§15)?
5. Can anyone obtain the **missing Evaluation/Judging Criteria table** from the SIH portal?
6. Can we ask the SIH/ISRO SPOC (if a channel exists) about:
   - evaluation mode
   - hidden-set file format, SAR polarization and calibration, image sizes
   - whether bi-temporal pairs are included
   - expected box format (axis-aligned vs rotated) and mask encoding
7. Which floor implementation do we accept for change-VQA and cross-modal analysis?
8. Who owns which module, and how many people will write code in parallel?
9. Where exactly are the GeoChat notebook and environment, and can someone reproduce them today?

**IMPORTANT (decide soon)**
10. Declared modality slots vs auto-detection (D8).
11. The confidence method per specialist.
12. Report format: HTML + JSON (REC) or PDF.
13. Maximum input size and downsampling policy.
14. Which benchmark subsets and sample counts to self-evaluate on (RSVQA-LR, VRSBench, CDVQA, BigEarthNet.txt `bench`).
15. For "highlight the water body": boxes, masks, or both?
16. Is a labelled replay fallback acceptable to the team, and how is it labelled?
17. Should the BIFOLD S1 classifier stay in the live system, domain-gated?

**LATER (safe to decide during implementation)**
18. LLM-assisted intent classification.
19. Result caching.
20. GeoJSON/GeoTIFF export of evidence.
21. Map view.
22. Deployment beyond the laptop.
23. CI.
24. Multi-user support.

---

## 13. Recommended development order (Task 13): about 12 working days, parallel tracks

| Stage | Days | Build | Depends on | Test | Do NOT build yet |
|---|---|---|---|---|---|
| **0. Foundation + spike** | 0–1 | `pyproject`, `.gitignore`, package skeleton, `schemas.py`, `analyze()` returning a fake response, fake specialists, CLI stub, pytest setup, CLAUDE.md, `docs/contracts.md`; **commit GeoChat notebook + env lock**. *Parallel:* D1 laptop VLM spike | Approval of this audit | Schema JSON round-trip; CLI runs end-to-end with fakes | Real specialists, UI |
| **1. Imaging + validation** | 1–3 | `load_image` → `RasterImage` (GeoTIFF/TIFF/PNG/JPEG), renderers (RGB/PAN/SAR false colour), coordinate module, per-image + pair validation; migrate metadata/NDVI/backscatter code | Stage 0 contracts | Synthetic fixtures incl. no-CRS TIFF, nodata, float SAR, mismatched grids | Co-registration, reprojection |
| **2. Agent core** | 2–4 | Input-config detection, rule intents, plan templates, executor (permitted params, timeouts, trace), aggregator | Stages 0–1 | **Golden routing set (≥50 queries incl. PS's 5)**; trace completeness; partial-failure behaviour | LLM planner |
| **3. VLM specialist** (parallel) | 2–5 | Chosen backend; VQA/caption/grounding prompts; output parsing; box → pixel mapping; token-probability confidence for closed questions | D1 decision; Stage 1 renderers | Parsing tests on **recorded real outputs**; `gpu`-marked smoke test | Fine-tuned weights |
| **4. Adaptation track** (parallel) | 3–8 | BigEarthNet.txt subset (S1 renders first); QLoRA or floor fine-tune in notebooks with checkpoints; before/after on the `bench` subset; export adapter; `training/RESULTS.md` | D2 decision | Held-out metric improvement is real and documented | Large-scale training |
| **5. First vertical slice + UI** | 4–7 | Single-image VQA end-to-end in Gradio (answer, evidence, confidence, trace, HTML/JSON report); then grounding overlays | Stages 2–3 | Integration test with fakes; manual scenario run | Map, auth, styling polish |
| **6. Change + cross-modal + SAR single image** (parallel) | 4–8 | Deterministic floors: change mask/regions/description/closed-form change-VQA; water/built-up masks + agreement; SAR false-colour → VLM + tool facts; domain-gated S1 classifier | Stages 1–2 | Synthetic pairs with known change and known water/built-up regions; threshold-free-of-units tests | Learned fusion or change models |
| **7. Eval harness** | 6–9 | Manifest CLI → JSONL (answers, boxes, mask paths, trace path); adapters for RSVQA-LR, VRSBench, CDVQA, BigEarthNet.txt `bench` (small subsets); VQA accuracy, grounding Acc@0.5, CDVQA accuracy; resumable; runs in notebooks | Stages 2–6 | Metrics on 20–200 samples/task; routing accuracy report | GPT-judge metrics |
| **8. Hardening** | 8–10 | Frozen `demo/scenarios`, labelled replay cache, structured error UX, cold-start rehearsal **on the demo laptop**, latency budget | Stages 5–7 | Full rehearsal from a reboot, offline | New features |
| **9. Freeze** | 10–12 | Docs, traceability table, deliverables (code, adapters/weights links, test instructions, demo script, backup video) | All | Clean-clone install test | Anything new |

**Rule:** no model upgrade (D5/D6 upgrades, a better VLM) begins until **every mandatory task has a working floor** (end of Stage 6).

---

## 14. Final proposed architecture (Task 14)

### 14.1 Component diagram
```text
 DEMO LAPTOP (RTX 4050 6 GB) — one Python application, runs fully offline
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │  ui/app.py (Gradio)      cli.py (ask | predict manifest)      eval/run.py    │
 │          └──────────────────────┬──────────────────────────────┘              │
 │              satquery.api.analyze(AnalysisRequest) -> AnalysisResponse        │
 │                                 │                                             │
 │  agent/ ───────────────────────────────────────────────────────────────────┐ │
 │  │ 1 load inputs once ............ imaging.io   → RasterImage[]            │ │
 │  │ 2 validate (authoritative) .... validation   → ValidationReport         │ │
 │  │ 3 input configuration ......... single_optical | single_sar |           │ │
 │  │                                 pair_cross_modal | pair_bitemporal      │ │
 │  │ 4 intents (rules) ............. ranked TaskIntents + matched rules      │ │
 │  │ 5 plan (templates + registry) . Plan[steps: tool, permitted params]     │ │
 │  │ 6 execute ..................... StepResults + ExecutionTrace            │ │
 │  │ 7 aggregate ................... answer · evidence · confidence · report │ │
 │  └────────────────────────────────┬────────────────────────────────────────┘ │
 │                       registry.py │ (capabilities + param models)            │
 │   ┌──────────────┬────────────────┼─────────────────┬──────────────────────┐ │
 │   vlm            change           cross_modal        sar_classifier*        │ │
 │   VQA/caption/   map, regions,    water/built-up     BigEarthNet S1 ResNet, │ │
 │   grounding      description,     masks, agreement   domain-gated           │ │
 │   │              change-VQA                                                 │ │
 │   backend: local | http† | replay (labelled) | fake                         │ │
 │                                                                              │
 │   imaging/ io · render · coords · indices · sar        evidence/ overlays · report │
 │   runs/<run_id>/ request.json trace.json response.json evidence/ report.html │
 └──────────────────────────────────────────────────────────────────────────────┘
   † workers/geochat (py3.10 env, localhost HTTP) only if GeoChat is chosen as the live VLM

 COLAB / KAGGLE NOTEBOOKS — interactive use only (policy-compliant)
   training/  → adapters/weights → models/ on laptop
   eval/      → same package, headless → predictions JSONL + metrics
   scripts/make_demo_cache.py → labelled replay cache for frozen scenarios
```

### 14.2 Decisions and labels

| Decision | Label |
|---|---|
| Python as the implementation language | [EXISTING TEAM DECISION] |
| Gradio UI | [EXISTING TEAM DECISION] |
| UI is thin and in-process; calls only `analyze()` | [YOUR RECOMMENDATION] |
| No separate backend API server; one entry point shared by UI/CLI/eval | [YOUR RECOMMENDATION] |
| Agent distinct from the LLM; specialist routing is core | [EXISTING TEAM DECISION] |
| Rule-based routing for the MVP | [EXISTING TEAM DECISION] |
| Intent/plan/executor split; plans as data; golden routing tests | [YOUR RECOMMENDATION] |
| Validation first-class and authoritative | [EXISTING TEAM DECISION] |
| Two-level validation with codes; CRS optional; accept plain TIFF/PNG/JPEG | [YOUR RECOMMENDATION] |
| Registry with machine-readable capabilities | [EXISTING TEAM DECISION] |
| Registry defined in code with Pydantic permitted-parameter models | [YOUR RECOMMENDATION] |
| Evidence + confidence + audit trace as first-class outputs | [EXISTING TEAM DECISION] |
| `Confidence.method`; evidence = spatial artifacts; HTML+JSON report | [YOUR RECOMMENDATION] |
| No chain-of-thought in trace; no fabricated numbers | [EXISTING TEAM DECISION] |
| GeoChat-7B as the RS VLM baseline | [EXISTING TEAM DECISION] |
| **Which VLM runs live on the laptop** | [UNRESOLVED] |
| Demo hosted on the laptop; notebooks for training/eval/precompute only | [YOUR RECOMMENDATION] |
| **Team-owned adaptation approach and data** | [UNRESOLVED] |
| No training from scratch | [EXISTING TEAM DECISION] |
| **Change specialist implementation** (deterministic floor first) | [UNRESOLVED] (floor = [YOUR RECOMMENDATION]) |
| **Cross-modal specialist** (deterministic masks floor) | [UNRESOLVED] (floor aligns with Initial.md proposal) |
| SAR single-image path via false-colour render + VLM + tool facts | [YOUR RECOMMENDATION] |
| S1 classifier kept only as a domain-gated optional tool | [YOUR RECOMMENDATION] |
| No database; filesystem `runs/` | [EXISTING TEAM DECISION] (no DB) + [YOUR RECOMMENDATION] (runs/) |
| No authentication | [EXISTING TEAM DECISION] |
| No map; no live imagery retrieval; honesty about imagery source | [EXISTING TEAM DECISION] (honesty) + [YOUR RECOMMENDATION] (cut map) |
| Pair compatibility verified, never co-registered by us | [YOUR RECOMMENDATION] |
| Headless manifest CLI + eval harness as first-class | [YOUR RECOMMENDATION] |
| Labelled replay fallback | [YOUR RECOMMENDATION] |
| Model worker process only if a legacy env forces it | [YOUR RECOMMENDATION] |
| Declared modality slots + plausibility checks | [UNRESOLVED] (REC given) |

### 14.3 Interfaces (sketch; names are finalized only after approval)
```python
# satquery/schemas.py
Modality    = Literal["optical_rgb", "optical_ms", "optical_pan", "sar"]
InputConfig = Literal["single_optical", "single_sar", "pair_cross_modal", "pair_bitemporal"]
TaskType    = Literal["vqa", "caption", "grounding", "change_description",
                      "change_vqa", "cross_modal_extraction"]

class ImageInput(BaseModel):        path: str; modality: Modality | None; acquired: date | None
class AnalysisRequest(BaseModel):   query: str; images: list[ImageInput]
                                    forced_task: TaskType | None = None   # eval/override; recorded in trace
class ValidationIssue(BaseModel):   code: str; severity: Literal["error", "warning"]; message: str; image_index: int | None
class PlanStep(BaseModel):          step_id: str; tool: str; task: TaskType | None; params: dict; inputs: list[int]
class Evidence(BaseModel):          kind: Literal["bbox", "mask", "change_map", "overlay", "metric"]
                                    label: str; coord_space: Literal["raster_px", "geo"] | None
                                    bbox: tuple[float, float, float, float] | None; file: str | None
                                    value: float | None; source_step: str
class Confidence(BaseModel):        value: float | None; method: str; calibrated: bool = False
class StepResult(BaseModel):        step_id: str; status: Literal["ok", "failed", "skipped"]; outputs: dict
                                    evidence: list[Evidence]; confidence: Confidence | None
                                    error: str | None; duration_s: float
class ExecutionTrace(BaseModel):    run_id: str; input_config: InputConfig | None; validation: list[ValidationIssue]
                                    intents: list[dict]; plan: list[PlanStep]; steps: list[StepResult]
                                    models: dict[str, str]  # tool -> model id + version
class AnalysisResponse(BaseModel):  status: Literal["ok", "partial", "invalid_input", "error"]
                                    answer: str; evidence: list[Evidence]; confidence: Confidence | None
                                    trace: ExecutionTrace; report_path: str | None

# satquery/specialists/base.py
class Capability(BaseModel):        task: TaskType; input_configs: set[InputConfig]; modalities: set[Modality]
                                    requires_georef: bool; params_model: type[BaseModel]
class Specialist(Protocol):
    name: str; version: str; capabilities: list[Capability]
    def run(self, task: TaskType, images: list["RasterImage"], query: str, params: BaseModel) -> StepResult: ...
```

### 14.4 Data flow for the PS's representative queries (plan templates)

| Query | Configuration → intents | Plan steps | Outputs |
|---|---|---|---|
| "Describe the land-cover and major objects…" | single_optical → caption | render_rgb → vlm.caption (→ vlm.ground for named objects, optional) | Text; optional boxes |
| "Highlight the water body…" | single_optical → grounding | RGB only: vlm.ground. MS with NIR: indices.ndwi_mask + vlm.ground | Box/mask overlay + text |
| "What changed between these two dates, and where?" | pair_bitemporal → change_description (+ localisation) | pair checks → change.map → change.regions → change.describe | Change-map overlay, region list, text |
| "Use the optical and SAR images together to identify built-up and water-covered regions." | pair_cross_modal → cross_modal_extraction | pair checks → sar masks → optical indices (if available) → cross_modal.fuse | Two masks, area %, agreement confidence |
| "Has the built-up area increased, decreased, or remained unchanged?" | pair_bitemporal → change_vqa | pair checks → per-date built-up estimate → compare → closed-form answer | Answer + per-date maps |

Deployment: the laptop runs `python -m satquery.ui`. The same package is `pip install -e`'d in notebooks for training and evaluation. No servers, containers or cloud services are required.

---

## 15. Proposed final repository tree (Task 15)
```text
Satellite-Query/
├── README.md                 install, run UI/CLI, run tests
├── CLAUDE.md                 rules for AI agents (short; links to docs/)
├── pyproject.toml            package `satquery`; extras [ui, models, eval, dev]; pytest markers
├── .gitignore                data/ models/ runs/ .env *.ipynb_checkpoints __pycache__
├── .env.example              documented settings (backend choice, model paths)
├── satquery/
│   ├── api.py                analyze(request) -> response  — the only entry point
│   ├── schemas.py            ALL shared contracts (changes need approval)
│   ├── settings.py           configuration from env/.env
│   ├── validation.py         per-image + pair checks          (← ingestion/validator.py)
│   ├── registry.py           builds specialist registry from settings
│   ├── imaging/              pure raster utilities; no ML, no agent imports
│   │   ├── io.py             RasterImage loading, metadata    (← loader.py, geospatial/metadata.py)
│   │   ├── render.py         RGB / PAN / SAR false-colour views
│   │   ├── coords.py         model space ↔ raster px ↔ geo
│   │   ├── indices.py        NDVI, NDWI                        (← optical/ndvi.py, bands.py)
│   │   └── sar.py            dB, speckle filter, statistics    (← sar/backscatter.py)
│   ├── agent/
│   │   ├── inputs.py         input-configuration detection
│   │   ├── intents.py        rule-based intent classification
│   │   ├── planner.py        plan templates
│   │   ├── executor.py       step execution, permitted params, trace
│   │   └── aggregator.py     answer / evidence / confidence composition
│   ├── specialists/
│   │   ├── base.py           Specialist protocol, Capability
│   │   ├── fakes.py          deterministic fakes for tests/dev
│   │   ├── vlm/              prompts.py · parse.py · backends/{local,http,replay}.py
│   │   ├── change.py         (← change_detection.py; promote to package when it grows)
│   │   ├── cross_modal.py    (← fusion.py, rewritten)
│   │   └── sar_classifier.py (← sar/inference.py; domain-gated)
│   ├── evidence/
│   │   ├── overlays.py       boxes, masks, change maps
│   │   └── report.py         HTML + JSON report
│   ├── ui/app.py             Gradio Blocks; calls api.analyze only
│   └── cli.py                `satquery ask …` · `satquery predict manifest.jsonl`
├── workers/geochat/          ONLY if GeoChat is the live VLM: server.py · requirements.lock · README.md
├── training/                 adaptation scripts/configs + RESULTS.md (weights → models/, not git)
├── eval/                     adapters/{rsvqa,vrsbench,cdvqa,bigearthnet_txt,manifest}.py · metrics.py · run.py
├── notebooks/                exploration + Colab/Kaggle launchers (e.g., geochat_inference.ipynb); never imported
├── scripts/                  inspect_bigearthnet.py · download_models.py · make_demo_cache.py
├── demo/scenarios/           one folder per frozen demo case: inputs (small) or fetch refs, query, expected plan
├── tests/
│   ├── conftest.py           synthetic raster factories
│   ├── imaging/ · test_validation.py · agent/ · specialists/ · evidence/
│   ├── golden/routing_cases.yaml
│   └── gpu/                  real-model smoke tests (marker `gpu`, skipped by default)
├── docs/
│   ├── architecture.md · contracts.md · decisions.md · requirements-traceability.md · setup.md
│   └── history/              Initial.md, Problem_Statement.odt (historical context)
├── data/    (gitignored; README.md committed describing expected layout)
├── models/  (gitignored)
└── runs/    (gitignored)
```

| Directory | Purpose |
|---|---|
| `satquery/` | The application: everything the UI, CLI and eval harness execute |
| `satquery/imaging` | Raster I/O and math with no knowledge of tasks; the safest place for parallel work |
| `satquery/agent` | Orchestration logic; owns the trace; never touches model libraries |
| `satquery/specialists` | One capability per module; add or remove without touching the agent (registry only) |
| `workers/` | Separately-environmented model servers, only when dependency pins force isolation |
| `training/` | Reproducible adaptation evidence for R3 |
| `eval/` | Benchmark adapters and metrics; also the prediction-file path if evaluation needs one |
| `tests/` | Fast, GPU-free by default; golden routing cases |
| `demo/` | Frozen, rehearsed scenarios |
| `docs/` | Current truth (contracts, decisions); `history/` holds past context |

**Migration of existing code (for approval):**
- **Moved:**
  - `validator.py` → `validation.py`
  - `loader.py` + `metadata.py` → `imaging/io.py`
  - `ndvi.py` + `bands.py` → `imaging/indices.py`
  - `backscatter.py` → `imaging/sar.py` (duplicate definition removed)
  - `sar/inference.py` → `specialists/sar_classifier.py`
  - `change_detection.py` → `specialists/change.py`
  - `fusion.py` → `specialists/cross_modal.py`
  - `inspect_bigearthnet.py` → `scripts/`
- **Removed:**
  - `polarization.py` (band roles live in `RasterImage`)
  - `api.py` and `pipeline.py` (replaced by `satquery/api.py`)
  - the empty stubs
- **Parked:** `cloud.py`

---

## 16. Decisions that require your approval before implementation

1. **Demo hosting:** the laptop runs the live system; Colab/Kaggle are used only for training, batch evaluation and pre-computation (§6 A1).
2. **Day-1 VLM spike** (options a/b/c) with a decision by day 2 based on loads-in-6-GB, latency, and quality on ~20 fixed samples (D1).
3. **Team-owned adaptation:** QLoRA of the live VLM if a one-day trial trains; otherwise fine-tune a smaller image–text/visual component as the floor (D2).
4. **Restructure and rename** `remote_sensing` → `satquery`, with the migration map in §15.
5. **Contracts:** `schemas.py` + Specialist protocol (§14.3); one entry point `analyze()`; **no separate backend server**.
6. **Orchestration:** rules → plan templates → executor → trace; **no LLM planner now**.
7. **Input configuration:** declared modality slots + plausibility checks.
8. **Validation policy:** accept TIFF/PNG/JPEG without CRS (flagged); pairs must share a grid; **no co-registration algorithms**.
9. **Specialist floors first:** deterministic change and cross-modal baselines labelled as heuristic; model upgrades only after every mandatory task has a floor.
10. **Confidence** object with an explicit method; retire the fixed HIGH/MODERATE/LOW buckets as a presented confidence.
11. **Evidence** = spatial artifacts; **report** = HTML + JSON.
12. **Cut list:** map, database, auth, live retrieval, React, Docker, HF Spaces, co-registration, LLM planner, multi-agent.
13. **Headless manifest CLI + eval harness** as first-class (hedge for unknown evaluation mode).
14. **Labelled replay fallback** for frozen demo scenarios.
15. **S1 classifier** retained only as a domain-gated optional tool.
16. **Immediately commit** the GeoChat notebook + environment lock (needs someone with access to it).
17. **CLAUDE.md and docs set** as described in §17.

---

## 17. What I will do after approval (documentation only, no implementation changes)

This audit task allows documentation files only. `remote_sensing/`, Initial.md and README.md stay untouched, and I make no commits unless asked.

1. **`CLAUDE.md` (repo root, ≤150 lines).** Future agent sessions load it automatically, so its rules prevent large refactors and silent assumptions. It holds only rules that stay true whichever pending decisions you choose. Proposed outline:
   1. The project in five lines, and the source-of-truth order: PS > repo > Initial.md (historical).
   2. A current-state snapshot with status labels (IMPLEMENTED / PARTIAL / PLANNED), updated whenever it changes.
   3. **Honesty rules:** no fabricated scores, confidence or training claims; heuristics labelled by `method`.
   4. **Scope rules:** touch only the module for the task plus its tests; no drive-by refactors, renames, formatting or dependency upgrades; ask before changing contracts, package layout, registry interface or environment pins.
   5. **Geospatial rules:** never assume band order, polarization, units, CRS or nodata; preserve CRS and transform; name coordinate spaces.
   6. **Model rules:** no model loading at import time; one load per process; GeoChat environment pins are frozen; no model swaps without an approved decision.
   7. **Data and secrets:** never commit `data/`, `models/`, `runs/` or tokens.
   8. **Testing:** synthetic fixtures; `gpu` marker; tests with every behaviour change.
   9. The "do not assume" list from §8.3.
   10. Placeholders for commands and the module map, filled in once the structure is approved.
2. **`docs/architecture-audit-2026-09-16.md`:** this audit, so the team can review it in the repo or a PR and future agents can see why decisions were made.
3. **`docs/decisions.md`:** seeded **only** with the decisions you approve from §16, each labelled and dated.
4. *(Optional, on request)* publish this audit as a private shareable page for teammates.

**Verification of that step:**
- `git status` shows only the new documentation files.
- `remote_sensing/` is unchanged.
- CLAUDE.md is at most 150 lines, and every path it references exists or is explicitly marked "planned".

---

## 18. Sources checked during this audit
- Colab FAQ (disallowed activities, runtime limits): https://research.google.com/colaboratory/faq.html
- BigEarthNet.txt dataset card: https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt · paper: https://arxiv.org/abs/2603.29630
- BIFOLD resnet18-s1-v0.2.0 model card and config: https://huggingface.co/BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0
- GeoChat repository: https://github.com/mbzuai-oryx/GeoChat
- VRSBench repository: https://github.com/lx709/VRSBench
- CDVQA paper: https://arxiv.org/abs/2112.06343 · change-VQA with Qwen models (2026): https://arxiv.org/abs/2604.18429
- RSVQA LR/HR statistics (via curriculum-learning paper): https://arxiv.org/pdf/2205.03147
- EarthDial: https://github.com/hiyamdebary/EarthDial
- TEOChat: https://github.com/ermongroup/TEOChat
- Cartosat-2 series (eoPortal): https://www.eoportal.org/satellite-missions/cartosat-2 · RISAT-1/1A (eoPortal): https://www.eoportal.org/satellite-missions/risat-1
