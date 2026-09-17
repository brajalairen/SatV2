# SatQuery AI — Technical Handoff for Claude / Claude Code

**Purpose:** Technical handoff for an AI coding/research agent joining the Smart India Hackathon (SIH) project **SIH26167 — SatQuery AI**.

**Status:** Working prototype is at an early stage. A real remote-sensing VLM (GeoChat-7B) has been successfully loaded and run for at least one image-grounded VQA interaction, but the complete agentic system, specialist workflows, evidence layer, GUI, change analysis, optical–SAR fusion, evaluation harness, and deployment are not yet complete.

**Important reading rule:** Throughout this document, every statement is classified as one of:

- **FACT** — established or experimentally verified in our discussions.
- **DECISION** — a choice the team made or currently intends to follow.
- **ASSUMPTION** — something believed or expected but not yet verified.
- **PROPOSAL** — a recommended implementation direction; not yet a team decision.
- **UNKNOWN** — not determined yet.

Do not silently convert an ASSUMPTION or PROPOSAL into a FACT or DECISION.

---

# 1. SIH Problem Statement

## 1.1 Problem identity

**FACT:** The selected SIH problem is **SIH26167 — SatQuery AI**.

## 1.2 Problem being solved

**FACT:** The problem is about making analysis of remote-sensing / satellite imagery accessible through a query-driven, agentic system rather than forcing users to manually select separate specialist AI tools for separate tasks.

Remote-sensing imagery is used in domains including:

- agriculture,
- disaster management,
- urban planning,
- forest monitoring,
- water resources,
- infrastructure,
- environmental monitoring.

Existing AI systems often isolate capabilities such as:

- land-cover classification,
- object detection,
- visual question answering (VQA),
- image captioning,
- change detection.

This creates a workflow problem: users may need to know which model to use, what input format it expects, whether one or two observations are required, and how to interpret the result.

A single optical image may also be insufficient for some questions. Useful information can require:

- multiple observations over time,
- optical/multispectral imagery,
- SAR imagery,
- or paired optical + SAR observations.

The problem specifically emphasizes that optical and SAR imagery are complementary. Optical/multispectral data provide spectral/context information, while SAR provides structural information and is usable under conditions where optical imagery can be limited, including at night or under cloud cover.

**FACT:** The intended solution should therefore be capable of reasoning over different input configurations rather than treating every query as a single-image VQA problem.

## 1.3 Core requirements from the problem statement

**FACT:** The mandatory/central requirements discussed for the SIH problem are:

1. **Agentic orchestration** — a controller/agent interprets a user query and selects and executes appropriate specialist models/tools.
2. **Input validation** — the system should check image count, modality, file format, metadata, and compatibility/co-registration where relevant.
3. **At least one adapted visual/VLM component** — the problem calls for a visual/VLM component adapted using **BigEarthNet or other open-source training data**.
4. **Single-image VQA.**
5. **Either captioning/scene description or text-guided grounding** in addition to VQA.
6. **Multitemporal change description / change-VQA.**
7. **Cross-modal optical/SAR joint analysis.**
8. **Evidence-grounded responses** that can include visual/spatial evidence.
9. **Confidence estimation.**
10. **Auditable execution summary** containing the task/model/tool names and key configuration details.
11. **GUI/web application** supporting image inputs and natural-language queries.
12. **Demonstrable model/tool selection and orchestration rather than one generic model handling every request.**

## 1.4 Inputs

**FACT:** The problem permits / discusses the following input patterns:

- single optical/multispectral image,
- single SAR image,
- co-registered optical + SAR pair,
- bi-temporal image pair,
- GeoTIFF/TIFF imagery,
- PNG/JPEG for prescribed benchmark datasets.

**FACT:** The hidden ISRO/SAC evaluation set discussed in the problem includes **pre-georeferenced/co-registered Cartosat-2S optical + RISAT SAR pairs**, with task-specific answers/labels/bounding boxes/masks.

## 1.5 Evaluation datasets / data roles

**FACT:** The problem names these datasets/benchmark resources:

- **BigEarthNet.txt** — primary data for adapting image-text representations to multisensor remote sensing.
- **VRSBench** — evaluation of single-image captioning, grounding and VQA.
- **RSVQA** — single-image VQA evaluation.
- **CDVQA** — multitemporal change-based VQA.

**IMPORTANT:** A prior roadmap document described BigEarthNet in simplified terms as paired Sentinel-1/Sentinel-2 data with text annotations and suggested TorchGeo. That roadmap is guidance, not the SIH specification itself. Claude should inspect the official dataset/task documentation before making implementation claims about exact annotations, splits, or formats.

## 1.6 What the final solution is expected to demonstrate

**FACT:** The solution must demonstrate that a user can issue a natural-language remote-sensing query and the system can determine what imagery / modality / temporal configuration and analysis workflow are needed, execute the appropriate specialized model/tool, combine results, and return an evidence-backed response with an execution summary.

**ASSUMPTION:** The judging process will place substantial practical importance on the reliability of the complete workflow and demo. The previous roadmap asserted that judges would evaluate orchestration/UI rather than accuracy; that claim is **not safe to treat as an official judging criterion**. Do not rely on it. The benchmark-specific requirements and actual evaluation outputs matter.

---

# 2. Our Proposed Solution

## 2.1 Overall concept

**DECISION / CONCEPT:** SatQuery is intended to be a conversational remote-sensing analysis system — conceptually similar to “ChatGPT for Earth observation” — but it must not be implemented as merely an LLM chatbot.

The core idea is:

> **Ask → Interpret → Validate → Select → Analyze → Verify → Visualize**

The user supplies imagery and asks a natural-language question. SatQuery understands the question, determines which type of analysis is required, validates the available inputs, selects a specialist model/tool, runs it, aggregates the resulting text/spatial evidence, estimates confidence, and presents the result together with an auditable execution trace.

## 2.2 Conceptual system

```text
User
  │
  │ images + natural-language query
  ▼
SatQuery Agent / Controller
  │
  ├── input validation
  │     ├── image count
  │     ├── modality
  │     ├── format
  │     ├── metadata
  │     └── temporal / co-registration compatibility
  │
  ├── query/task interpretation
  │
  ├── specialist/tool selection
  │
  ▼
Model / Tool Registry
  │
  ├── GeoChat-7B → remote-sensing VQA / captioning / grounding
  ├── change-analysis specialist → bi-temporal tasks
  ├── SAR processor / model → SAR-specific analysis
  ├── optical/SAR fusion workflow → paired modalities
  └── deterministic geospatial/raster tools where appropriate
  │
  ▼
Evidence Layer
  │
  ├── answer text
  ├── spatial evidence / overlay
  ├── derived raster/visualization
  ├── confidence
  └── execution summary / audit trace
  │
  ▼
GUI / Map / Report
```

## 2.3 Important conceptual distinction: agent vs LLM

**DECISION:** The LLM is a component inside the agent. The agent is the larger orchestration system.

```text
LLM = language understanding / planning component
Agent = controller + state + validation + tool/model selection + execution + aggregation + audit
```

SatQuery should therefore not reduce to:

```text
User → LLM → one API → answer
```

The intended design is closer to:

```text
User
  ↓
Agent
  ↓
Validate + determine task
  ↓
Select specialist(s)
  ↓
Execute
  ↓
Combine/verify
  ↓
Evidence + confidence + trace
  ↓
Answer
```

## 2.4 Map and imagery relationship

**DECISION / CLARIFICATION:** The interactive map is primarily **location/context UI**. The satellite imagery is the actual model input.

Do not represent the map itself as though it automatically supplies the model with pixels unless a real imagery-retrieval pipeline is implemented.

For the prototype, imagery can be:

- uploaded by the user,
- preloaded for demo scenarios,
- or loaded from a prepared local/demo scene.

**ASSUMPTION / FUTURE OPTION:** A production system could retrieve imagery for a map-selected area from remote-sensing data services, but autonomous satellite-data retrieval is **not currently implemented** and must not be claimed as implemented.

## 2.5 Intended user journey

1. User opens the SatQuery web application.
2. User selects a location/scene or supplies imagery.
3. User uploads one or more images as applicable.
4. User writes a natural-language query.
5. Agent inspects query and image configuration.
6. Agent determines whether the request requires:
   - single-image VQA,
   - grounding,
   - captioning/scene description,
   - temporal comparison/change analysis,
   - optical + SAR fusion,
   - or another registered workflow.
7. Agent checks compatibility and required metadata.
8. Agent selects specialist model(s)/tool(s).
9. Specialist workflow runs.
10. Evidence is generated or extracted.
11. Agent combines textual and spatial outputs.
12. System returns:
    - natural-language answer,
    - visual/spatial evidence,
    - confidence,
    - execution trace.
13. User can inspect/download a report.

## 2.6 Example workflows

### Mode A — Single-image VQA

```text
Image + “What land cover is visible?”
       ↓
GeoChat / VQA specialist
       ↓
Answer + confidence + trace
```

### Mode B — Text-guided grounding

```text
Image + “Highlight the water body.”
       ↓
Grounding workflow
       ↓
Bounding box / mask / visual overlay + textual explanation
```

### Mode C — Bi-temporal change

```text
Image at t1 + Image at t2
Question: “What changed?”
       ↓
Change-analysis workflow
       ↓
Change description + evidence overlay + confidence
```

### Mode D — Optical + SAR

```text
Optical image + SAR image
Question: “Identify built-up and water-covered regions.”
       ↓
Optical/SAR fusion workflow
       ↓
Joint interpretation + visual evidence + trace
```

These four modes were repeatedly used as the conceptual MVP scope. Only Mode A is currently experimentally demonstrated end-to-end with a real model.

---

# 3. Current Implementation

## 3.1 Overall status

**FACT:** The current implementation is a small prototype, not the complete SatQuery system.

The team explicitly stated that the current Git repository is small and represents only a tiny portion of the final solution.

**CRITICAL:** Claude Code must not infer the final architecture from whatever small amount of code currently exists. The current repo is a starting point, not proof that its structure is the correct architecture for the complete system.

## 3.2 Remote-sensing data exploration actually completed

### Sentinel-2 exploration

**FACT:** The team downloaded Sentinel-2 raw bands from Copernicus Browser for a small ROI.

The downloaded archive contained three TIFF files, with generic filenames (`22.tiff`, `23.tiff`, `24.tiff`). The team examined them and found nonzero 16-bit pixel values. The files were approximately 104 × 166 pixels and shared a geospatial grid.

The band mapping used for visualization was:

- B02 — blue
- B03 — green
- B04 — red

**FACT:** The RGB composite was created by stacking the corresponding bands and percentile-normalizing them for display.

Example conceptual operation:

```python
rgb = np.dstack([
    normalize(red),
    normalize(green),
    normalize(blue)
])
```

**IMPORTANT:** The normalization was only for visualization; it did not alter the source TIFF values.

### GeoTIFF geospatial metadata

**FACT:** The team used `rasterio` to inspect metadata. The Sentinel-2 files had:

- width: 166 pixels,
- height: 104 pixels,
- one raster band each,
- `uint16`,
- CRS: `EPSG:4326`,
- approximately 8.9969e-05 degrees per pixel,
- matching geographic bounds.

`rasterio` was also used to map a row/column position back to geographic coordinates. This established an understanding that a GeoTIFF contains both raster measurements and a spatial transform mapping pixels to geographic coordinates.

### BigEarthNet-S1 v2.0 / Sentinel-1 exploration

**FACT:** The team downloaded and extracted **BigEarthNet-S1 v2.0**, approximately 51 GiB, to a local D: drive.

A sample patch was inspected containing:

- a VV GeoTIFF,
- a VH GeoTIFF.

The sample characteristics established during inspection included:

- 120 × 120 pixels,
- single raster band per file,
- `float32`,
- CRS `EPSG:32633`,
- 10 m × 10 m resolution,
- same spatial bounds/grid for VV and VH.

The team visualized VV and VH as grayscale rasters with a dB colorbar.

**FACT:** The team learned that SAR should not be treated like a natural RGB image. VV/VH are radar-polarization channels, and speckle/noise is expected.

The team also discussed:

- VV = vertical transmit / vertical receive.
- VH = vertical transmit / horizontal receive.
- brighter/darker pixels indicate stronger/weaker backscatter, but pixel brightness alone should not be naively mapped to semantic classes.

## 3.3 GeoChat setup actually completed

**FACT:** The team cloned the official GeoChat repository in Google Colab.

Repository used:

```text
https://github.com/mbzuai-oryx/GeoChat
```

The official model checkpoint used is:

```text
MBZUAI/geochat-7B
```

**FACT:** The model checkpoint was downloaded to Google Drive under:

```text
/content/drive/MyDrive/SatQuery/models/GeoChat-7B
```

The model files total approximately 14.1 GB.

## 3.4 GeoChat environment and compatibility work

**FACT:** The model could not simply be loaded in the initial modern Colab environment because the old GeoChat code expected an older Transformers stack.

The initial environment contained modern versions such as:

- Python 3.13,
- newer Transformers,
- newer Accelerate,
- newer BitsAndBytes,
- newer Torch.

GeoChat failed with a compatibility error involving `_expand_mask` in the Transformers Bloom implementation.

**FACT:** Directly downgrading the old dependency stack under Python 3.13 also failed because the old `tokenizers` version had no suitable Python 3.13 wheel and tried to build from source.

**DECISION:** A separate Python 3.10 environment was created using `uv` inside Colab.

Working environment verified as:

```text
Python:       3.10.21
NumPy:        1.24.4
Torch:        2.0.1+cu117
Transformers: 4.31.0
Accelerate:   0.21.0
BitsAndBytes: imported successfully
CUDA:         True
GPU:          Tesla T4
```

GeoChat was installed into the environment without reinstalling all dependencies:

```bash
uv pip install --python /content/geochat-env/bin/python -e /content/GeoChat --no-deps
```

`einops` was installed separately after an import error.

## 3.5 GPU / loading decisions

**FACT:** The active GPU for the working GeoChat experiment is a **Tesla T4**, approximately 15 GB physical VRAM.

**FACT:** The team successfully loaded GeoChat-7B using **4-bit quantized loading**.

This is important:

- 4-bit loading is a memory/storage optimization for pretrained weights.
- It is **not** 4-bit training.
- No model fine-tuning happened during this step.

**FACT:** The model is stored persistently on Google Drive, while the Colab runtime is ephemeral.

## 3.6 Actual GeoChat inference achieved

**FACT:** GeoChat-7B successfully loaded and performed a real image-grounded VQA test using an official GeoChat demo image:

```text
/content/GeoChat/demo_images/train_2956_0001.png
```

The question used was:

```text
Where are the airplanes located and what is their type?
```

The team eventually achieved an actual generated answer after correcting an image-encoding issue.

The working inference sequence became:

```text
PIL image
  ↓
chat.upload_img(...)
  ↓
chat.encode_img(...)
  ↓
chat.ask(...)
  ↓
chat.stream_answer(...)
  ↓
answer
```

The successful answer included a statement that airplanes were located toward the bottom-left portion of the image and did not invent a type when it was not confident enough.

**FACT:** This is the first proven end-to-end AI capability of the current prototype: real remote-sensing VQA with GeoChat.

## 3.7 Current working technology stack

**FACT:** Technologies actually used during development include:

- Python,
- PyTorch,
- Hugging Face / Transformers,
- BitsAndBytes,
- GeoChat official codebase,
- `rasterio`,
- NumPy,
- Matplotlib,
- Google Colab,
- Google Drive,
- Git/GitHub for project versioning (repository exists, but its detailed current contents were not provided in this handoff conversation).

## 3.8 Current repository state

**FACT:** A Git repository exists and the team stated it is **small and only a tiny portion of the full intended system**.

**UNKNOWN:** The exact current committed tree, branch structure, file names, test coverage, and current Git history have not been inspected in this handoff. Claude Code must inspect the repository before editing.

## 3.9 What currently works

**VERIFIED:**

- T4 CUDA environment works.
- Dedicated Python 3.10 environment works.
- GeoChat can be imported.
- GeoChat-7B checkpoint is available on Drive.
- GeoChat-7B can load in 4-bit.
- GeoChat can process an image after explicit image encoding.
- GeoChat can answer a remote-sensing VQA query.
- Sentinel-1 VV/VH sample loading/visualization works with Rasterio.
- Sentinel-2 RGB composition/visualization works for the downloaded sample.
- GeoTIFF metadata/georeferencing can be inspected with Rasterio.

## 3.10 What is NOT yet implemented / not yet verified

**NOT IMPLEMENTED / UNKNOWN:**

- full SatQuery agent/controller in production form,
- robust task classification/router,
- formal model registry,
- grounding workflow validated end-to-end,
- captioning workflow validated end-to-end,
- multitemporal change detection/change-VQA,
- optical-SAR fusion model/workflow,
- BigEarthNet adaptation/fine-tuning of a visual/VLM component,
- benchmark evaluation harness,
- hidden ISRO/SAC evaluation support beyond conceptual preparation,
- evidence overlay pipeline for model outputs,
- reliable confidence calibration,
- complete audit/execution trace in the final app,
- final Gradio GUI,
- interactive map integration,
- report generation/download in the real implementation,
- production authentication/authorization,
- deployment to Hugging Face Spaces or another host,
- autonomous retrieval of imagery by map coordinates,
- cloud/API-based remote-sensing data ingestion,
- persistent application database.

---

# 4. Architecture

## 4.1 Proposed high-level architecture

**DECISION / INTENDED ARCHITECTURE:**

```text
┌───────────────────────────────────────────────┐
│                    USER                       │
│  Natural-language query + image(s) / scene   │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│               FRONTEND / GUI                  │
│  Gradio UI                                    │
│  - image upload(s)                            │
│  - modality selection / display               │
│  - query box                                  │
│  - analysis button                            │
│  - answer                                     │
│  - evidence                                   │
│  - confidence                                 │
│  - execution trace                            │
│  - report download                            │
│  - optional map / scene selector              │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                 BACKEND API                   │
│      Python orchestration/application layer   │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                SATQUERY AGENT                 │
│                                               │
│  1. Query understanding                       │
│  2. Input validation                          │
│  3. Requirement determination                │
│  4. Model/tool selection                     │
│  5. Parameter/config selection               │
│  6. Execution sequencing                     │
│  7. Result aggregation                        │
│  8. Confidence/evidence handling             │
│  9. Audit summary                             │
└───────────────┬───────────────────────────────┘
                │
        ┌───────┼──────────┬───────────┐
        ▼       ▼          ▼           ▼
     GeoChat  Grounding   Change    Optical/SAR
       VQA      model     model       workflow
        │       │          │           │
        └───────┴──────────┴───────────┘
                │
                ▼
┌───────────────────────────────────────────────┐
│                EVIDENCE LAYER                 │
│  - text answer                                │
│  - boxes / masks / overlays                   │
│  - derived visualizations                     │
│  - intermediate metrics                       │
│  - confidence                                 │
│  - model/tool metadata                        │
│  - execution trace                            │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
                USER-FACING RESULT
```

## 4.2 Frontend

**DECISION:** Gradio was selected as the rapid-prototype UI framework because the team is primarily Python-oriented and does not want to spend scarce hackathon time building a separate HTML/CSS/JavaScript frontend.

Planned UI elements:

- one/two image upload controls depending on workflow,
- natural-language query textbox,
- optional modality indicators,
- run/analyze button,
- answer panel,
- visual evidence panel,
- confidence panel,
- execution summary/audit panel,
- optional example queries,
- optional map/scene selector.

**OPEN:** Whether the final team wants a pure Gradio frontend or a richer frontend/backend split remains an implementation choice. Under severe time constraints, Gradio is the established MVP direction.

## 4.3 Backend

**DECISION:** Python is the default backend language.

The backend should separate responsibilities instead of putting all behavior in one UI file.

Recommended logical modules:

```text
backend/
├── agent.py                # orchestration/controller
├── router.py               # task classification/dispatch
├── validators.py           # input compatibility checks
├── registry.py             # model/tool registry
├── schemas.py              # typed request/result models
├── evidence.py             # evidence packaging/overlays
├── audit.py                # execution trace
└── config.py               # runtime configuration
```

**PROPOSAL:** Use typed request/response schemas and explicit model interfaces so specialist implementations can be swapped without rewriting the controller.

## 4.4 Database

**FACT:** No database has yet been selected or implemented.

**PROPOSAL:** A database is not necessary for the hackathon MVP unless persistent users, jobs, or history become a requirement.

Potential future choices include:

- SQLite for local/demo persistence,
- PostgreSQL for production/cloud deployment.

Do not add a database merely because “production systems need one.” Add it only when a demonstrated requirement exists.

## 4.5 Model/tool registry

**DECISION / PROPOSAL:** The architecture should expose specialist capabilities through a registry rather than hard-coding every specialist inside the controller.

Conceptually:

```python
REGISTRY = {
    "single_vqa": GeoChatVQA(...),
    "grounding": GroundingModel(...),
    "change": ChangeModel(...),
    "optical_sar": OpticalSARWorkflow(...),
}
```

Each registered component should declare at least:

- supported task,
- input modality,
- number of images,
- expected formats,
- whether geospatial metadata is required,
- whether co-registration is required,
- available parameters,
- output schema.

**PROPOSAL:** Make capability metadata machine-readable so the agent can validate compatibility before dispatch.

## 4.6 APIs / external services

**FACT:** The current real inference path does not depend on a paid hosted VLM API. GeoChat weights are loaded locally in Colab from Google Drive.

**UNKNOWN:** Exact external services to be used in the final application are not fixed.

Discussed / researched data sources include:

- Copernicus Data Space / Sentinel-1 / Sentinel-2 for development data,
- Bhoonidhi / NRSC for Indian Earth observation data,
- USGS EarthExplorer for Landsat.

**IMPORTANT:** These are data-source options, not proof of an implemented live data-retrieval API.

## 4.7 Authentication and authorization

**UNKNOWN:** No authentication/authorization requirement was established for the hackathon MVP.

**PROPOSAL:** Omit authentication from the MVP unless the submission/deployment environment requires it. Add authentication only if a persistent multi-user deployment is actually built.

## 4.8 Storage

Current working storage pattern:

```text
Local D: drive
  └── large BigEarthNet-S1 dataset

Google Drive
  └── GeoChat-7B checkpoint

Colab /content
  └── temporary code/runtime files
```

**FACT:** Colab runtime storage is ephemeral; Google Drive was used for persistence.

**DECISION:** Do not attempt to load the full 51 GiB BigEarthNet dataset into Colab as a normal runtime dataset. Process subsets/patches or use local storage/preprocessing as appropriate.

## 4.9 Deployment

**UNKNOWN / OPEN:** Final deployment environment has not been finalized.

The roadmap suggested Hugging Face Spaces for a free demo deployment. That is a possible UI/deployment path, but **large model hosting and GPU availability must be verified before committing**.

Do not assume a normal CPU Hugging Face Space can host GeoChat-7B at acceptable latency.

---

# 5. Important Design Decisions

## 5.1 Use GeoChat-7B as the primary remote-sensing VLM

**DECISION:** GeoChat-7B is the primary VLM currently chosen for the prototype.

**Alternative considered:** Qwen2-VL / Qwen2.5-VL 7B family.

**Why:** GeoChat is specifically designed/fine-tuned for remote-sensing tasks and supports capabilities such as VQA, captioning and grounding. Qwen is a strong general VLM with visual localization but is not inherently remote-sensing-specific.

**Tradeoff:** GeoChat is much less convenient to run than a current general VLM because its official codebase depends on an older software stack and its model is memory-heavy. The T4 solution required an isolated Python 3.10 environment and 4-bit loading.

**Status:** **FINAL for current prototype unless a concrete technical blocker appears.**

## 5.2 Use 4-bit loading, not local full-precision inference

**DECISION:** Use 4-bit loading for GeoChat-7B on the T4.

**Why:** Memory constraints.

**Tradeoff:** Quantization can affect quality; inference is not equivalent to original precision.

**Status:** **FINAL for current T4 prototype.**

## 5.3 Use Google Colab T4 for heavy model inference

**DECISION:** T4 was selected over a TPU for GeoChat inference.

**Why:** The workload is PyTorch/Hugging Face model inference, where a CUDA GPU is much more straightforward than adapting the workflow to a TPU.

**Status:** **FINAL for current prototype.**

## 5.4 Keep VS Code/laptop and Colab logically separated rather than depending on live VS Code ↔ Colab integration

**DECISION:** During the two-day emergency prototype period, the team decided not to make live VS Code-to-Colab runtime integration a critical dependency.

**Why:** Reduce failure points. VS Code/laptop can own project code, Git, preprocessing and UI work; Colab can own heavy VLM inference.

**Tradeoff:** Some copying/sync overhead exists.

**Status:** **Working development strategy; can change later.**

## 5.5 Do not fine-tune a giant model from scratch during the urgent prototype window

**DECISION:** No from-scratch training. Use pretrained/adapted open-source models.

**Why:** Time, compute and complexity.

**Status:** **FINAL for hackathon MVP.**

**IMPORTANT COMPLIANCE ISSUE:** The SIH statement still requires at least one VLM component adapted using BigEarthNet or open-source training data. GeoChat is already remote-sensing fine-tuned on open-source training data, but the team has **not** performed its own BigEarthNet fine-tuning. Claude must verify exactly what the SIH wording accepts and what evidence we should present rather than claiming “we fine-tuned GeoChat on BigEarthNet” when we did not.

## 5.6 Use Gradio for rapid UI

**DECISION:** Gradio is the current UI choice for the MVP.

**Alternative:** Full custom frontend such as React/TypeScript.

**Why:** Faster implementation; existing team is Python-heavy; less frontend overhead.

**Tradeoff:** Less control/polish than a fully custom frontend.

**Status:** **FINAL for current MVP unless UI requirements make it inadequate.**

## 5.7 Use specialist workflows instead of one generic VLM for everything

**DECISION:** The architecture must preserve task-specific routing and specialist model/tool slots.

**Why:** This is central to the agentic nature of the problem and remote-sensing constraints.

**Tradeoff:** More engineering effort than calling one model.

**Status:** **FINAL architectural principle.**

## 5.8 Use a simple rule-based router as an MVP, but not as the final conceptual definition of the agent

**DECISION:** A simple deterministic router is acceptable for initial implementation.

The roadmap suggested rules such as:

- two images + SAR → optical/SAR,
- two images → change detection,
- “highlight/where” → grounding,
- “describe/caption” → captioning,
- otherwise → VQA.

**IMPORTANT:** This is useful as an MVP implementation technique, but we do **not** want Claude to interpret “agentic” as merely an `if/else` statement forever.

**Status:** **MVP decision; final sophistication is open.**

## 5.9 Do not claim autonomous imagery retrieval unless implemented

**DECISION:** The prototype should be honest about whether imagery is uploaded/preloaded or dynamically retrieved.

**Why:** The map is not automatically the model input. Autonomous retrieval would require a real data-access subsystem.

**Status:** **FINAL honesty/architecture constraint.**

---

# 6. Ideas We Rejected

## 6.1 Load and use the full ~51 GiB BigEarthNet-S1 dataset in Colab

**REJECTED.**

**Why:** Too large for practical Colab runtime handling and unnecessary for a prototype. The dataset is useful for training/development and sampling, but the whole dataset should not be copied into the active inference runtime.

## 6.2 Make full VS Code ↔ Colab live integration a core dependency

**REJECTED for the urgent prototype.**

**Why:** Additional connectivity/runtime failure modes when the priority is getting the system demonstrably working.

## 6.3 Run normal GeoChat-7B locally on the RTX 4050 6 GB laptop GPU

**REJECTED as the primary inference path.**

**Why:** The full checkpoint is too large for straightforward local loading in normal precision. The working T4 + 4-bit Colab path is more practical.

## 6.4 Use a current modern Transformers environment directly with official GeoChat code

**REJECTED.**

**Why:** Compatibility errors occurred. The official GeoChat code expects an older dependency stack. The isolated Python 3.10 environment solved this.

## 6.5 Train a large remote-sensing VLM from scratch

**REJECTED.**

**Why:** Not feasible in the available time/compute and unnecessary for the prototype.

## 6.6 Build all four specialist models from scratch immediately

**REJECTED.**

The roadmap itself suggested starting with one general remote-sensing VLM and then expanding.

**Why:** Too much engineering for the available time.

## 6.7 Build a large 500+ line placeholder UI first and assume the main infrastructure is “50% done”

**REJECTED as a strategic approach.**

**Why:** A placeholder UI can prove frontend plumbing but does not prove the actual model, specialist workflows, evidence semantics, or benchmark compliance. We explicitly pivoted away from spending the limited time on a dummy UI before getting a real remote-sensing model working.

## 6.8 Treat the system as merely “LLM → API”

**REJECTED conceptually.**

**Why:** That would not demonstrate the intended orchestration, validation, specialist selection and evidence pipeline.

## 6.9 Treat a general-purpose VLM as automatically equivalent to a remote-sensing VLM

**REJECTED as the default.**

**Why:** The SIH task explicitly concerns domain adaptation and specialist remote-sensing tasks. GeoChat was chosen because it is remote-sensing-specific.

---

# 7. Known Problems / Concerns

## 7.1 Confirmed problems

### A. GeoChat dependency mismatch

**FACT:** Modern Colab packages were incompatible with the official GeoChat code. This was solved with a separate Python 3.10 environment and pinned older dependencies.

### B. Initial model loading exhausted system RAM

**FACT:** An earlier GeoChat loading attempt caused a Colab runtime crash after exhausting available RAM, even though 4-bit loading was requested.

**FACT:** A later run successfully loaded the model in 4-bit on the T4.

**CONCERN:** Model loading remains a resource-sensitive operation. Avoid unnecessary duplicate model loads and keep the heavy model in one long-lived process where possible.

### C. Inference initially failed because image encoding was omitted

**FACT:** The first inference test passed a PIL image forward without the required internal encoding step and failed with:

```text
AttributeError: 'Image' object has no attribute 'ndim'
```

The working sequence added:

```python
chat.encode_img(img_list)
```

**LESSON:** Respect official model wrapper APIs rather than assuming generic Hugging Face image handling.

### D. Generic TIFF filenames created metadata uncertainty

**FACT:** Copernicus downloads produced TIFF files with generic names such as `22.tiff`, `23.tiff`, `24.tiff`.

**CONCERN:** The band identity should be established from the download metadata/product structure, not guessed from the filename alone.

### E. SAR is easy to misinterpret visually

**FACT:** VV/VH imagery is noisy/speckled and does not have a natural RGB interpretation.

**CONCERN:** Model/tool prompts and visualizations must not make simplistic claims such as “bright = building” or “dark = water” without context.

## 7.2 Suspected / likely problems

### A. 6 GB laptop GPU is insufficient for straightforward local GeoChat deployment

**ASSUMPTION backed by practical experience:** Local inference of full GeoChat-7B on the RTX 4050 6 GB GPU is not the preferred path.

### B. Hugging Face Spaces may not be sufficient for the final model deployment by default

**CONCERN:** The previously suggested deployment route may lack sufficient persistent GPU/VRAM for an acceptable GeoChat service depending on the current hosting configuration.

Do not assume a free Space can host the final model simply because Gradio can run there.

### C. A simplistic text router may misclassify queries

**CONCERN:** Pure keyword rules are brittle.

Example:

```text
“Where did the settlement expand between the two dates?”
```

This is both temporal change and spatial grounding. The architecture may need multi-step planning rather than selecting exactly one task.

### D. Confidence scores can become fabricated numbers

**CONCERN:** Returning `0.85` simply because the code needs a confidence field is not legitimate confidence estimation.

Confidence should be:

- derived from model probabilities where available,
- based on calibrated heuristics,
- or explicitly described as an estimated/heuristic confidence.

Do not manufacture benchmark-like confidence scores.

## 7.3 Things not yet investigated

- Exact current Git repository architecture.
- Exact SIH scoring rubric / judge interpretation beyond the official problem statement.
- Exact public benchmark evaluation protocol required for every task.
- Hidden dataset file naming/API availability at runtime.
- Exact grounding output format expected by the hidden evaluation.
- Exact change-model checkpoint that will fit within available resources.
- Exact optical–SAR fusion model/tool that can be integrated within the deadline.
- Whether BigEarthNet adaptation can be meaningfully demonstrated within the remaining time.
- Production data retrieval APIs and quotas.
- Final deployment GPU/latency constraints.
- Multi-user concurrency requirements.
- Robust input co-registration validation for arbitrary uploads.

---

# 8. Assumptions

| Assumption | Status | Notes |
|---|---|---|
| GeoChat is a useful primary remote-sensing VLM for the prototype | **Verified enough for MVP** | Actual VQA inference worked. |
| One remote-sensing VLM can cover several initial capabilities via prompting | **Partially verified** | VQA verified; captioning/grounding still need end-to-end tests. |
| A deterministic router is sufficient for MVP orchestration | **Unverified** | Reasonable prototype approach, but may be too brittle for combined tasks. |
| Gradio is sufficient for the required GUI | **Unverified** | Likely practical; final usability remains to be tested. |
| T4 4-bit GeoChat inference is sufficiently fast for a live demo | **Partially verified** | Loading worked; full latency/throughput not benchmarked. |
| GeoChat's existing open-source remote-sensing adaptation can satisfy the SIH “adapted visual/VLM” requirement | **Unverified** | Must verify against exact SIH wording and evidence requirements. |
| BigEarthNet can be used for a meaningful adaptation/fine-tuning step under the remaining time | **Unverified** | Depends on time, compute, and exact expected task. |
| Free/low-cost deployment can host the full stack | **Unverified** | Especially uncertain for the large VLM. |
| Live satellite-data retrieval is required for the prototype | **Unknown** | Not established; supplied imagery can support the core workflow. |
| A database is needed for the MVP | **Unverified / probably no** | No explicit requirement has been established. |
| Every task should have exactly one specialist model | **False as an architectural assumption** | Some queries may require sequential or parallel specialists. |

---

# 9. Hackathon Constraints

## 9.1 Time constraint

**FACT:** At one point, on **2026-09-13**, the team stated that there were only approximately **two days left to make the full working prototype**.

**UNKNOWN:** The actual current time remaining as of the handoff is not established. Claude must not assume that the “two days” statement is still current; verify the current deadline/timeline if it affects prioritization.

## 9.2 Team capability

**FACT:** The team is primarily CSE/ML-oriented and has experience with Python, C/C++, PyTorch, NLP/ML and related coursework.

**FACT:** The project team is not treating advanced web development as its core strength; this influenced the choice of Gradio.

## 9.3 Hardware/resources

Known resources discussed:

- RTX 4050 laptop GPU with 6 GB VRAM,
- 16 GB system RAM on the laptop,
- Google Colab T4 (~15 GB VRAM),
- Google Drive with persistent storage,
- large local D: drive containing BigEarthNet-S1.

## 9.4 Runtime/deployment implications

- Heavy VLM inference currently happens on Colab T4.
- Local laptop is more appropriate for orchestration, preprocessing, UI and Git.
- Colab runtime is temporary.
- Google Drive stores the large GeoChat checkpoint persistently.
- Full BigEarthNet should remain off the inference runtime unless sampled.

## 9.5 Reliability requirements

**DECISION / PRIORITY:** The final demo must favor a smaller set of reliable capabilities over a large number of unstable features.

A polished VQA + one credible second capability with trace/evidence is more valuable than five broken “specialists.”

## 9.6 Demo requirements

The intended demo should make the agentic behavior visible:

- user asks different types of queries,
- system accepts different input configurations,
- agent routes differently,
- specialist executes,
- output contains evidence/trace.

## 9.7 API/service limitations

**FACT:** No paid hosted VLM API is currently necessary for the working GeoChat proof of concept.

**UNKNOWN:** Whether additional data sources, hosted models, or APIs will be required later.

---

# 10. Evaluation / Demonstration

## 10.1 What the judges need to see according to the problem requirements

**FACT:** The system should visibly demonstrate:

1. natural-language interaction;
2. input compatibility checks;
3. task/model/tool selection;
4. remote-sensing VQA;
5. captioning or grounding;
6. change analysis/change-VQA;
7. optical + SAR joint analysis;
8. evidence/visual outputs;
9. confidence;
10. auditable execution summary;
11. a usable GUI.

## 10.2 Highest-priority pieces for reliability

**PROPOSAL based on engineering constraints:**

### Tier 1 — Must be reliable

- GeoChat model loading and inference.
- Input validation.
- Deterministic/sensible routing.
- Clear answer format.
- At least one evidence visualization.
- Execution trace.
- GUI that can run end-to-end without manual intervention.

### Tier 2 — Must be credible

- Grounding.
- Change analysis.
- Optical + SAR processing.

### Tier 3 — Useful but can be simplified

- sophisticated memory,
- user authentication,
- persistent database,
- production-scale APIs,
- elaborate map stack,
- complex multi-agent architecture.

## 10.3 Evaluation caution

**IMPORTANT:** Do not claim that UI/orchestration is evaluated instead of model/task accuracy. The problem explicitly defines benchmark tasks and hidden evaluation data. UI/orchestration is necessary, but task correctness remains important.

## 10.4 Benchmark honesty

Do not present illustrative values as benchmark results.

A previous presentation mockup used example values such as:

- “2017 vs 2025”,
- “built-up +8%”,
- “cleared +5.2%”,
- “confidence 91%”.

**FACT:** These were presentation/demo examples only.

They must be labelled **illustrative/demo values** unless actually computed from an evaluation sample.

---

# 11. Open Questions

## Critical before implementation

1. **What exactly does the SIH submission/evaluation mean by the requirement that at least one visual/VLM component be “adapted using BigEarthNet or open-source training data”?**
   - Is using an already fine-tuned open-source remote-sensing VLM such as GeoChat sufficient?
   - Does the team need to perform its own adaptation step?
   - What evidence should be documented?

2. **Which real specialist should be used for change detection/change-VQA within our hardware/time limits?**

3. **How will grounding be implemented with an output format compatible with the expected task (boxes, masks, coordinates)?**

4. **What is the minimum viable optical + SAR fusion implementation that is technically honest and useful for the demo?**

5. **What is the final demo/deployment environment, especially GPU availability for GeoChat?**

6. **What is the exact current Git repository structure and which code is actually committed?**

7. **What benchmark samples can be run locally to verify VQA/captioning/grounding/change functionality?**

8. **What is the current actual remaining time before submission/demo?**

## Important

9. Should the agent use an LLM for routing, deterministic rules, or a hybrid approach?
10. Should a query be allowed to invoke multiple specialists sequentially?
11. How should confidence be computed/calibrated?
12. What evidence artifacts are generated for each task type?
13. Should the system support arbitrary GeoTIFF CRS/resolution combinations or only normalized inputs?
14. How should co-registration be validated?
15. Should the map itself retrieve imagery, or should image upload remain the demo path?
16. Which output schema will be common to all specialist models?

## Can decide later

17. Persistent database technology.
18. Authentication.
19. Production cloud architecture.
20. Custom frontend framework.
21. Long-term model-serving stack (vLLM, Triton, dedicated GPU server, etc.).
22. Long-term memory/user profiles.
23. Full automated satellite-data retrieval.

---

# 12. Recommended Development Order

The following deliberately separates **team decisions already made** from **new recommendations**.

## Phase 0 — Audit before changes

**DECISION:** Claude must inspect the current repository first.

**PROPOSAL:** Build an inventory of:

- files,
- imports,
- entry points,
- model code,
- UI code,
- configuration,
- tests,
- dependency files,
- current branches/status.

Do not refactor until this audit is complete.

## Phase 1 — Stabilize the known-working model path

**DECISION:** Keep the proven GeoChat + T4 + Python 3.10 + 4-bit path.

Tasks:

1. Move the working GeoChat inference logic into a clean reusable wrapper.
2. Load the model only once per process.
3. Define a stable VQA interface.
4. Add basic timing/error handling.
5. Test multiple images/questions.

## Phase 2 — Define shared interfaces

**PROPOSAL:** Before writing multiple specialists, define normalized data contracts:

```python
AnalysisRequest
AnalysisResult
Evidence
ExecutionStep
Confidence
```

A specialist should ideally return a consistent structure such as:

```python
{
    "answer": ...,
    "evidence": ...,
    "confidence": ...,
    "model": ...,
    "task": ...,
    "metadata": ...
}
```

Do not let each specialist invent a completely different result format.

## Phase 3 — Implement validation

**PROPOSAL:** Implement a dedicated validation layer before routing.

Validate:

- number of images,
- file format,
- modality,
- dimensionality,
- CRS if required,
- spatial compatibility,
- temporal pairing requirements,
- optical/SAR pairing.

Return structured validation errors rather than raw exceptions.

## Phase 4 — Implement routing

**DECISION:** Start with deterministic routing.

**PROPOSAL:** Make routing data-driven through a registry rather than a giant nested `if` block.

For example:

```text
query + inputs
     ↓
parse intent / requirements
     ↓
create candidate workflows
     ↓
filter by input compatibility
     ↓
select workflow
```

Later, if needed, an LLM can assist in interpreting intent, but deterministic validation should remain authoritative for hard constraints such as image count/modality.

## Phase 5 — Grounding

**PROPOSAL / PRIORITY:** Get one real grounding workflow working as soon as possible because it upgrades the system from “text VQA” to “text + spatial evidence.”

Validate:

- coordinates/bounding boxes,
- overlay rendering,
- mapping model coordinates back to displayed pixels,
- output schema.

## Phase 6 — Change workflow

**PROPOSAL:** Integrate a suitable lightweight/open-source change model or a credible change-analysis baseline.

Requirements:

- two temporal images,
- compatible dimensions/georeferencing,
- evidence visualization,
- change description,
- confidence/trace.

Do not train a large model from scratch.

## Phase 7 — Optical + SAR

**PROPOSAL:** First create a deterministic image/raster processing baseline before attempting a sophisticated learned fusion model if time is short.

Possible MVP:

```text
optical → relevant spectral/visual interpretation
SAR     → VV/VH feature extraction
            ↓
combined workflow
            ↓
answer + evidence
```

A learned dual-encoder fusion model remains a longer-term architecture option.

## Phase 8 — Evidence + audit layer

**DECISION:** Every workflow should produce an auditable execution summary.

Example:

```text
Task: temporal_change_analysis
Input: 2 GeoTIFF images
Modality: optical
Model: <change model>
Parameters: ...
Validation: passed
Steps:
  1. loaded t1
  2. loaded t2
  3. aligned/verified inputs
  4. change model executed
  5. evidence overlay generated
Confidence: ...
```

Do not expose hidden chain-of-thought. The audit trail should describe **actions, models, parameters and results**, not internal reasoning text.

## Phase 9 — Gradio integration

**DECISION:** Build the UI around the backend interfaces, not the other way around.

The UI should remain thin:

```text
UI event → AnalysisRequest → Agent → AnalysisResult → render
```

Avoid embedding model logic directly inside Gradio callbacks.

## Phase 10 — Evaluation harness

**PROPOSAL:** Build a small evaluation runner before polishing the UI.

For every task:

- input fixture,
- expected answer/label/geometry where available,
- model output,
- latency,
- confidence,
- qualitative notes.

Use the public benchmarks discussed by the problem statement where legally/technically available.

## Phase 11 — Demo hardening

**PROPOSAL:** Freeze a small number of reliable demo scenarios.

Prefer:

- 1 single-image VQA example,
- 1 grounding example,
- 1 change example,
- 1 optical+SAR example.

Each should be repeatable from a clean startup.

## Phase 12 — Deployment

**PROPOSAL:** Only after local end-to-end reliability is established should the team decide whether and how to deploy online.

Do not let deployment constraints destabilize the working local/T4 path prematurely.

---

# 13. Context for an AI Coding Agent

## 13.1 Project identity

You are assisting with **SIH26167 — SatQuery AI**, a Smart India Hackathon project intended to make remote-sensing analysis accessible through natural-language, agentic orchestration of specialist models/tools.

The team is building a system in which:

```text
Natural-language query + satellite imagery
          ↓
      SatQuery Agent
          ↓
 validate + interpret + select
          ↓
 specialist model(s)/tool(s)
          ↓
 answer + spatial evidence + confidence + execution trace
```

## 13.2 Current state you must understand

The team is **not starting from zero**, but the implementation is still early.

The strongest verified capability is:

- GeoChat-7B loaded in 4-bit on a Colab Tesla T4;
- a real remote-sensing VQA request successfully generated an answer.

The team has also explored Sentinel-2 and Sentinel-1/BigEarthNet GeoTIFF data and understands basic raster metadata, georeferencing, optical band stacking, and SAR VV/VH visualization.

However, the complete SatQuery application does not yet exist.

## 13.3 Decisions you should treat as established

Unless a concrete technical blocker is discovered, preserve these:

1. **GeoChat-7B is the primary remote-sensing VLM for the current prototype.**
2. **T4 + 4-bit inference is the current heavy-model path.**
3. **Python is the primary implementation language.**
4. **Gradio is the current rapid UI choice.**
5. **The agent is distinct from the LLM.**
6. **Specialist routing is a core architectural requirement.**
7. **Input validation is a first-class component.**
8. **Evidence + confidence + auditable execution summary are first-class outputs.**
9. **The map is UI/location context; do not claim it is the model input unless actual imagery retrieval is implemented.**
10. **Do not train a large model from scratch.**
11. **Do not make live VS Code ↔ Colab integration a hard dependency unless there is a clear benefit that outweighs its failure modes.**

## 13.4 Decisions that are still open

Do not assume these have been finalized:

- final change model,
- grounding model implementation details,
- optical–SAR fusion model,
- exact routing algorithm beyond MVP rules,
- database,
- authentication,
- deployment platform,
- live data retrieval,
- exact evidence schema,
- confidence methodology,
- exact BigEarthNet adaptation procedure.

## 13.5 Things requiring extra scrutiny

### A. SIH requirement compliance

The requirement concerning visual/VLM adaptation using BigEarthNet or open-source training data must be interpreted carefully. Do not fabricate training claims.

### B. Model resource usage

GeoChat is large and sensitive to environment/version compatibility. Avoid loading it multiple times. Prefer one model process/service and reuse it.

### C. Agentic architecture

Do not call a giant `if/else` function “the complete agent” and stop there. For MVP, deterministic routing is fine. Architect the interfaces so it can evolve.

### D. Confidence

Do not insert arbitrary numbers simply to populate a UI field.

### E. Evidence

Do not claim a textual answer is “evidence-grounded” unless the system can actually show what visual/spatial artifact supports it.

### F. Geospatial correctness

When using raster data:

- preserve CRS,
- preserve transform/resolution,
- validate alignment for pairs,
- distinguish pixel coordinates from geographic coordinates.

### G. Demo reliability

Prefer a small number of thoroughly tested workflows over a broad but fragile system.

## 13.6 What you MUST NOT assume

1. The existing Git repo represents the final architecture.
2. Planned modules already exist.
3. BigEarthNet fine-tuning has already happened.
4. Grounding already works.
5. Change detection already works.
6. Optical–SAR fusion already works.
7. Satellite imagery is automatically retrieved from the map.
8. A production database exists.
9. Authentication exists.
10. Hugging Face Spaces can host the complete final model stack without verification.
11. Any mock/demo numerical result is a real benchmark result.
12. A current model dependency stack can be substituted for GeoChat's known-compatible environment without testing.

## 13.7 How you should work in the repository

Before making significant code changes:

1. Inspect the entire repository tree.
2. Read all relevant README/config/dependency files.
3. Identify the actual entry point(s).
4. Identify which files are prototypes vs reusable modules.
5. Run the existing code/tests if possible.
6. Identify duplicate or dead code.
7. Produce a short architecture audit before making large refactors.

When implementing:

- make small, reviewable changes;
- preserve working GeoChat inference;
- avoid unnecessary dependency changes;
- avoid replacing the stack merely for stylistic reasons;
- separate UI, orchestration, model wrappers and geospatial processing;
- add tests for routing and validation;
- add structured logging/audit output;
- fail gracefully when an input is unsupported.

## 13.8 Preferred specialist interface

Use an abstract contract approximately like:

```python
class Specialist:
    name: str
    capabilities: set[str]

    def can_handle(self, request) -> bool:
        ...

    def analyze(self, request) -> "AnalysisResult":
        ...
```

And a normalized result:

```python
@dataclass
class AnalysisResult:
    task: str
    model: str
    answer: str
    confidence: float | None
    evidence: list
    metadata: dict
    execution_steps: list
```

This is a **PROPOSAL**, not an already-established code contract. Adapt it to the real repository after audit.

## 13.9 Architectural principle for tool/model selection

Use hard constraints for safety/correctness and flexible intelligence for interpretation.

For example:

```text
LLM / query parser
       ↓
proposed task(s)
       ↓
validator / capability registry
       ↓
only compatible specialists remain
       ↓
execution
```

This prevents the LLM from selecting a model that cannot accept the supplied imagery.

## 13.10 Research/code integrity

Do not report:

- fabricated benchmark scores,
- fabricated confidence values,
- fabricated training runs,
- fabricated dataset usage,
- unsupported model capabilities.

When a component is a demo baseline or heuristic, label it that way.

## 13.11 Product philosophy

SatQuery should feel like:

> **“Ask → Reason → Select → Analyze → Verify → Visualize.”**

But the system should earn that description technically. The agent must actually inspect inputs, select models/tools, execute them and expose the resulting evidence/trace.

---

# Claude's First Task

**Do not modify code yet.**

First perform a repository and architecture audit using this handoff as context.

Deliver:

1. **Repository inventory** — files, entry points, dependencies, current run path, tests, and what is actually implemented.
2. **Architecture comparison** — compare the current repository architecture against the intended SatQuery architecture above.
3. **Risk register** — identify the top architectural, model, dependency, geospatial, evaluation, and deployment risks.
4. **Requirement gap analysis** — map each SIH requirement to one of:
   - implemented,
   - partially implemented,
   - planned,
   - missing,
   - unknown.
5. **GeoChat integration audit** — confirm the safest way to turn the currently proven Colab inference code into a reusable specialist/backend component without breaking its working environment.
6. **Critical unknowns** — identify which unresolved questions must be answered before major implementation work.
7. **Recommended next 3 implementation steps** — keep them focused on the actual repository and the most important gaps; do not propose a wholesale rewrite unless the audit provides concrete evidence that it is necessary.

### Special instruction

The current Git repository is intentionally small and incomplete. **Do not infer that its existing structure is the correct structure for the final SatQuery system.** Treat it as an implementation starting point and preserve working code where practical, while proposing targeted architectural improvements where justified.

Only after completing this audit should you begin making code changes.
