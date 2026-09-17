# Round 1 submission kit: PPT, demo video, link

Audience: the team preparing the first-round SIH submission.

> If SIH provides an official PPT template, **use it**. The outline below maps our content onto the usual SIH idea-deck sections.

## 1. PPT outline (6 slides)
| # | Slide | Content | Source |
|---|---|---|---|
| 1 | Title | SIH26167 · SatQuery AI · theme · team name · web-app link | — |
| 2 | Proposed solution | Problem in one line (single-task RS tools need experts) · our answer: an **agent** that validates inputs, picks specialist tools, runs them, and returns evidence + confidence + an auditable trace · the 5 SIH query types it handles | `docs/architecture-audit-2026-09-16.md` §2 |
| 3 | Technical approach | Pipeline diagram (README "How it works") · stack: Python, Gradio, PyTorch, Falcon 0.7B RS-VLM, rasterio/NumPy/SciPy · input configurations → plan templates · show one real trace table | README, app screenshot |
| 4 | Feasibility & viability | Runs today on a 6 GB laptop GPU: measured 0.42 s median per task, 1.94 GB peak VRAM · risks + mitigations: domain shift to Cartosat/RISAT, adaptation requirement, SAR-VLM limits, hosting (a free HF Space now needs PRO, so Round 1 serves a Gradio share link) | `docs/decisions.md` D-019–D-022 |
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
1. **(15 s) Hook.** "Ask a satellite image a question, and see exactly how the answer was produced."
2. **(30 s) Single image.** Load scenario 2 (caption), then scenario 3 (grounding).
   - Point out the **routing rule** and the **plan table**.
   - Show the highlighted overlay. Grounding reports water at 80.1% of the coast scene.
3. **(25 s) VQA.** Scenario 1. Show the confidence and its *method* ("beam-search sequence probability, uncalibrated").
4. **(40 s) Change.** Scenario 4 (dated pair), optionally scenario 5 for the comparative phrasing.
   - Show the before/after overlay: red = VLM, yellow = deterministic map.
   - Point out the agreement score: VLM 38.1%, deterministic 27.6%, IoU 0.33.
5. **(40 s) Optical + SAR.** Scenario 6.
   - Explain the colours (both sensors agree vs one sensor only).
   - Read the per-class agreement: water **IoU 0.97** across the two sensors.
   - Be honest that built-up scores IoU 0.00 here: Falcon finds no buildings at 10 m, and the SAR bright mask is a
     labelled heuristic, so in this coastal scene it is most likely rocky shoreline.
   - Scenario 7 (SAR alone) is a good 10 s follow-up: water 87.3%, mean co-pol backscatter -15.78 dB.
6. **(15 s) Safety.** Scenario 8 (mismatched grids). The agent **refuses** before planning, naming the grid and CRS mismatch.
7. **(15 s) Report.** Download the HTML report and scroll through the trace.
8. **(10 s) Close.** Show the link, then the roadmap.

Tips:
- Warm the model up with one query before recording.
- Hide the terminal.
- Keep narration factual: use "heuristic" where the app says heuristic.

## 3. Web-app link checklist

> **A free HF Space is not available.** Hugging Face returns HTTP 402 for both `cpu-basic` Gradio and ZeroGPU on a free
> account: hosting either now requires PRO (see `docs/decisions.md` D-020). Round 1 uses a Gradio share link instead.

Run this on the GPU laptop, close to the judging window:
```powershell
$env:SATQUERY_VLM_BACKEND="falcon"; $env:SATQUERY_DEVICE="cuda"; $env:SATQUERY_SHARE="1"
.venv\Scripts\python app.py
```
- [ ] The console prints `Running on public URL: https://<id>.gradio.live`. **Each launch gives a new URL.**
- [ ] Opened that URL from a phone or another network to confirm it is reachable from outside.
- [ ] Ran one query first to warm the model, so judges do not hit the ~15 s load.
- [ ] The laptop stays **online, awake and running the app** for as long as the link must work. Closing the app kills the link.
- [ ] Link expires **1 week** after launch (reported by Gradio at startup).
- [ ] Current URL pasted into the PPT and the submission form.
- [ ] Stop the app when the judging window closes; the tunnel exposes local port 7860 publicly while it runs.

If the share link fails to start, check that `frpc` is present in `<HF_HOME>/gradio/frpc/`. Endpoint protection may quarantine
it as a tunneling tool; verify its SHA-256 against the value pinned in `gradio/tunneling.py` before restoring it.

## 4. Honest-claims checklist (see CLAUDE.md §7)
| We CAN say | We must NOT say (yet) |
|---|---|
| "Uses Falcon, a vision-language model pre-trained on remote-sensing data by its authors" | "We fine-tuned a model on BigEarthNet" (not done; it is on the roadmap) |
| "Agentic orchestration: validation → intent → plan → registered tools → trace" | "LLM-based reasoning agent" (routing is rule-based by design) |
| "Deterministic SAR and spectral tools, clearly labelled heuristic" | "Calibrated confidence" or any accuracy number we have not measured |
| "Accepts GeoTIFF/TIFF (and PNG/JPEG for benchmarks); checks grids and dates" | "Works on Cartosat/RISAT" (untested; domain shift is a known risk) |
| "Runs on a 6 GB laptop GPU: 6/6 gate tasks, 0.42 s median, 1.94 GB peak VRAM, measured 2026-09-17" | "Hosted on a free Hugging Face ZeroGPU Space" (hosting one now requires PRO; the link is a Gradio share tunnel from our laptop) |
| "Optical and SAR agree on water with IoU 0.97 on a co-registered BigEarthNet pair" | "The optical-SAR built-up agreement works" (it scored IoU 0.00; Falcon finds no buildings at 10 m) |
