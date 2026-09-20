# Round 1 submission kit: PPT, demo video, link

Audience: the team preparing the first-round SIH submission.

> If SIH provides an official PPT template, **use it**. The outline below maps our content onto the usual SIH idea-deck sections.

## 1. PPT outline (6 slides)
| # | Slide | Content | Source |
|---|---|---|---|
| 1 | Title | SIH26167 · SatQuery AI · theme · team name · web-app link | — |
| 2 | Proposed solution | Problem in one line (single-task RS tools need experts) · our answer: an **agent** that validates inputs, picks specialist tools, runs them, and returns evidence + confidence + an auditable trace · the 5 SIH query types it handles | `docs/architecture-audit-2026-09-16.md` §2 |
| 3 | Technical approach | Pipeline diagram (README "How it works") · stack: React + MapLibre client, FastAPI, Python, PyTorch, Falcon 0.7B RS-VLM, rasterio/NumPy/SciPy · input configurations → plan templates · show one real trace from the Details drawer | README, app screenshot |
| 4 | Feasibility & viability | Runs today on a 6 GB laptop GPU: measured 0.42 s median per task, 1.94 GB peak VRAM · risks + mitigations: domain shift to Cartosat/RISAT, adaptation requirement, SAR-VLM limits, hosting (a free HF Space now needs PRO, so Round 1 serves the map app through a tunnel from the laptop) | `docs/decisions.md` D-019–D-022, D-024 |
| 5 | Impact & benefits | Non-experts get answers from imagery (floods/water, urban growth, land cover) · auditable outputs build trust · the modular registry lets ISRO specialists be plugged in later | — |
| 6 | Roadmap & references | **After selection:** fine-tune on BigEarthNet.txt (we hold S1+S2 locally), benchmark evaluation (RSVQA, VRSBench, CDVQA), stronger change/fusion models, RISAT/Cartosat handling · references (below) | audit §13 |

**References to cite:**
- Falcon, arXiv:2503.11070
- BigEarthNet.txt, arXiv:2603.29630
- GeoChat, CVPR 2024
- EarthDial, CVPR 2025
- VRSBench, NeurIPS 2024
- CDVQA, IEEE TGRS 2022
- RSVQA, IEEE TGRS 2020

## 2. Demo video script (~3 minutes, record on the GPU laptop)

Record the **map-first web app** (`web/` served by `satquery/server.py`): it is the primary interface (D-024).

1. **(15 s) Hook.** "Ask a satellite image a question, and see exactly how the answer was produced." Open on the map.
2. **(30 s) Single image.** Sidebar **Help → demo scenario 2** (caption), then scenario 3 (grounding). The GeoTIFF lands
   in place on the basemap; the question is filled in, press send.
   - Read the answer card, then open **Details**: routing rule, every tool call with its permitted parameters, input checks.
   - Show the highlighted water overlay pinned to the image footprint.
3. **(25 s) VQA.** Scenario 1. Show the confidence and its *method* ("beam-search sequence probability, uncalibrated").
4. **(30 s) Area selection.** Sidebar **Select area → Circle**: press at the centre, drag out, release. The card states
   that the analysis was restricted to the drawn shape and that pixels outside it are excluded.
5. **(40 s) Change.** Scenario 4 (dated pair), optionally scenario 5 for the comparative phrasing.
   - Show the before/after overlay: red = VLM, yellow = deterministic map, and the **Temporal change** view.
   - Read the agreement figures **from the screen**: the change-map method changed on 2026-09-20 (D-026), so the old
     numbers no longer apply. See §5.
6. **(40 s) Optical + SAR.** Scenario 6.
   - Explain the colours (both sensors agree vs one sensor only).
   - Read the per-class agreement: water **IoU 0.97** across the two sensors.
   - Be honest that built-up scores IoU 0.00 here: Falcon finds no buildings at 10 m, and the SAR bright mask is a
     labelled heuristic, so in this coastal scene it is most likely rocky shoreline.
   - Scenario 7 (SAR alone) is a good 10 s follow-up: water 87.3%, mean co-pol backscatter -15.78 dB.
7. **(15 s) Safety.** Scenario 8 (mismatched grids). The agent **refuses** before planning, naming the grid and CRS mismatch.
8. **(15 s) Report.** Download the HTML report and scroll through the trace.
9. **(10 s) Close.** Show the link, then the roadmap.

Tips:
- Warm the model up with one query before recording.
- Hide the terminal; the app fills the window.
- Keep narration factual: use "heuristic" where the app says heuristic.

## 3. Web-app link checklist

> **A free HF Space is not available.** Hugging Face returns HTTP 402 for both `cpu-basic` Gradio and ZeroGPU on a free
> account: hosting either now requires PRO (see `docs/decisions.md` D-020). Round 1 serves the map-first app from the GPU
> laptop through a tunnel (D-024).

Run this on the GPU laptop, close to the judging window:
```powershell
cd web; npm install; npm run build; cd ..          # once: builds the client FastAPI serves
$env:SATQUERY_VLM_BACKEND="falcon"; $env:SATQUERY_DEVICE="cuda"
.venv\Scripts\python -m uvicorn satquery.server:app --port 8000     # terminal 1
cloudflared tunnel --url http://localhost:8000                      # terminal 2: prints the public URL
```
- [ ] `http://127.0.0.1:8000` shows the map app locally, and **Help → demo scenario** completes one full run.
- [ ] `cloudflared` printed a `https://<random>.trycloudflare.com` URL. **Each launch gives a new URL.**
- [ ] Opened that URL from a phone or another network to confirm it is reachable from outside.
- [ ] Ran one query through the public URL first, so judges do not hit the model load (~15 s).
- [ ] The laptop stays **online, awake and running both commands** for as long as the link must work.
- [ ] Current URL pasted into the PPT and the submission form.
- [ ] Stop both when the judging window closes; the tunnel exposes local port 8000 publicly while it runs.

Notes:
- Uploads through a public link are capped by `SATQUERY_MAX_UPLOAD_MB` (default 2048) and there is no authentication, so
  only run the tunnel during the judging window.
- **Fallback if the tunnel fails:** the Gradio UI still works and carries its own share link
  (`$env:SATQUERY_SHARE="1"; .venv\Scripts\python app.py`, public URL on port 7860, expires after a week). It is the older,
  non-map interface, so use it only if the map app cannot be served. Share links need `frpc` in `<HF_HOME>/gradio/frpc/`;
  endpoint protection may quarantine it as a tunneling tool.

## 4. Honest-claims checklist (see CLAUDE.md §7)
| We CAN say | We must NOT say (yet) |
|---|---|
| "Uses Falcon, a vision-language model pre-trained on remote-sensing data by its authors" | "We fine-tuned a model on BigEarthNet" (not done; it is on the roadmap) |
| "Agentic orchestration: validation → intent → plan → registered tools → trace" | "LLM-based reasoning agent" (routing is rule-based by design) |
| "Deterministic SAR and spectral tools, clearly labelled heuristic" | "Calibrated confidence" or any accuracy number we have not measured |
| "Accepts GeoTIFF/TIFF (and PNG/JPEG for benchmarks); checks grids and dates" | "Works on Cartosat/RISAT" (untested; domain shift is a known risk) |
| "Runs on a 6 GB laptop GPU: 6/6 gate tasks, 0.42 s median, 1.94 GB peak VRAM, measured 2026-09-17" | "Hosted on a free Hugging Face ZeroGPU Space" (hosting one now requires PRO; the link is a tunnel from our laptop) |
| "Optical and SAR agree on water with IoU 0.97 on a co-registered BigEarthNet pair" | "The optical-SAR built-up agreement works" (it scored IoU 0.00; Falcon finds no buildings at 10 m) |

## 5. Numbers to re-measure before recording (change-map correction, D-026)

The deterministic change map now scales both dates by one shared percentile range. Only the **bi-temporal** figures
moved; grounding, SAR and optical–SAR numbers in §2 are unaffected and were re-verified on 2026-09-20.

| Figure | Old (in earlier drafts) | Now | Status |
|---|---|---|---|
| Navi Mumbai, deterministic change map | 27.6% of the scene | **25.7%** | Re-measured 2026-09-20. Model-independent: the change map never calls the VLM, so this holds for any backend. |
| Navi Mumbai, Otsu separability | 0.674 | **0.728** | Re-measured 2026-09-20. Higher is better: less false change to separate. |
| Navi Mumbai, VLM change detection | 38.1% | **re-measure** | Needs a real Falcon run on the GPU laptop. |
| VLM ↔ change-map agreement IoU | 0.33 | **re-measure** | Depends on the VLM mask above. |

**Do this before recording:** run scenario 4 once on the GPU laptop with `SATQUERY_VLM_BACKEND=falcon`, read the two
"re-measure" figures off the result card, and put those values in the PPT. Do not reuse the old numbers, and do not
quote a figure that has not been measured on the machine you are demoing from (CLAUDE.md §7).
