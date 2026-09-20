# Remote-sensing adaptation plan (SIH R5)

Audience: the team member who will run the adaptation. Nothing here has been trained yet; this is the plan and the
feasibility case. Claims are labelled **FACT** (measured or read from the repo), **ESTIMATE** (reasoned, to be
measured) and **RISK**.

## 1. What the requirement actually says

From `Problem_Statement.odt`:

> **Remote-sensing adaptation:** At least one visual or vision-language component must be fine-tuned or otherwise
> adapted using BigEarthNet.txt or the any open source training data.

and, under Expected Solution:

> A generic LLM or VLM without remote-sensing adaptation will not satisfy the requirements.

Read carefully, this asks for **one** component, **adapted by us**, on BigEarthNet.txt **or any open-source data**. It
does not demand a large model, a full fine-tune, or a benchmark score. What it does demand is that the adaptation is
real and demonstrable, because the observable execution trace is what gets evaluated.

**Where we stand (FACT).** Falcon is remote-sensing pre-trained *by its authors*, not by us (D-021). Our honest-claims
checklist already forbids saying we fine-tuned anything (`docs/round1-submission-kit.md` §4). So R5 is currently the
one mandatory item with no implementation.

## 2. What would qualify

| Option | Qualifies? | Cost | Verdict |
|---|---|---|---|
| **A. LoRA fine-tune of Falcon on BigEarthNet.txt VQA pairs** | Yes: a vision-language component, adapted by us, on the named dataset | ~2–3 h GPU | **Recommended primary** |
| **B. Train our own multi-label land-cover classifier on BigEarthNet patches** (ResNet-18, S2 4-band or S1 VV/VH) | Yes: a visual component adapted on open data | ~30–60 min GPU | **Recommended fallback**, and a good second component if time allows |
| C. Reuse `BIFOLD-BigEarthNetv2-0/resnet18-s1` as-is (`specialists/s1_classifier.py`) | **No** — somebody else trained it | none | Does not satisfy R5 |
| D. Full fine-tune of Falcon (0.7B) | Yes | Will not fit 6 GB (see §7) | Out of scope for Round 1 |

Option A is the strongest answer because it adapts the *same* component the demo runs on, and the adapted model then
shows up by name in every execution trace.

## 3. Model to start from

**Falcon-Single-Instruction-Large, 0.7B** — the model already integrated (`satquery/specialists/falcon.py`, D-021).

- **FACT:** 0.7B parameters, `model.safetensors` is 3,350,641,628 bytes (fp32 on disk), pinned to
  `transformers==4.49.0` because its remote code has no `GenerationMixin`.
- **FACT:** inference measured at 0.42 s median/task and **1.94 GB peak reserved VRAM** on the RTX 4050 (fp16).
- **ASSUMPTION:** it follows the Florence-2 design (DaViT vision tower + encoder–decoder text model), which is what its
  processor/generate signature in our wrapper implies. This decides which modules LoRA attaches to, so **verify it in
  the §10 spike** rather than trusting it.

## 4. Dataset

**BigEarthNet.txt**, which the problem statement names, and which we already hold.

**FACT — it is a VQA dataset.** Each row in `demo/examples/*/text.json` looks like:

```json
{"patch_id": "S2A_MSIL2A_20170613T101031_N9999_R022_T34VER_26_59",
 "s1_name": "S1B_IW_GRDH_1SDV_20170613T160455_34VER_26_59",
 "input": "Would you confirm that any marine water borders upon a mixed forest?",
 "output": "yes", "type": "binary", "category": "adjacency", "split": "test",
 "country": "Finland", "season": "Summer"}
```

That is exactly the shape Falcon's VQA task needs: an image, a question, a short answer. It also carries an official
`split`, so a held-out evaluation needs no invented split, and a `category`, so results can be reported per question
type instead of as one opaque number.

**FACT — the extraction already exists.** `scripts/build_round1_subset.py` reads `metadata.parquet`, selects patches
by class, and writes per patch: `s2_bgrn.tif` (B02/B03/B04/B08), `s1_vv_vh.tif`, `preview.png` and the matching
`text.json` rows, plus a `manifest.json`. It never scans the whole dataset (D-019). Building the training subset is
the same command with a larger `--per-class` and `--split train`.

### Size

| Split | Patches | QA rows (ESTIMATE, ~3–10 rows/patch) | Disk (ESTIMATE) |
|---|---|---|---|
| train | 2,000 | ~6,000–15,000 | ~500 MB |
| validation | 300 | ~900–3,000 | ~75 MB |
| test (held out, untouched during training) | 500 | ~1,500–5,000 | ~125 MB |

Small on purpose. The goal is a credible, documented adaptation, not a leaderboard.

### Preprocessing

1. `build_round1_subset.py --split train --per-class N` → patch folders + `text.json`.
2. Render each patch to a 3-channel uint8 RGB the VLM can take: B04/B03/B02 with the same percentile stretch the app
   uses (`imaging.stretch`), so training input matches what `render_rgb` feeds at inference. **This matters:** train on
   what the app will actually show the model.
3. Build a JSONL of `{image_path, question, answer, category, split}`, one line per QA row.
4. **Balance the yes/no rows** (see RISK in §12), and cap rows per patch so a few chatty patches do not dominate.
5. Keep the patch-id → split mapping in the manifest, committed, so the evaluation is reproducible.

## 5. Adaptation method

**LoRA** (not QLoRA, not a full fine-tune).

- Quantization is unnecessary: fp16 weights are only ~1.7 GB, so QLoRA's extra complexity buys nothing here.
- Start with **LoRA on the text decoder's attention projections (q/k/v/o), vision tower frozen**, `r=8`, `alpha=16`,
  dropout 0.05. Freezing the vision tower is what keeps activations small enough for 6 GB.
- Batch size 1 with **gradient accumulation 8–16**, lr 1e-4 cosine, 1–2 epochs, fp16 autocast.
- Enable **gradient checkpointing** only if the VRAM probe demands it (it costs ~30% speed).

## 6. Storage

| Item | Size |
|---|---|
| Training subset (images + JSONL) | ~700 MB |
| Falcon weights in the HF cache | ~3.4 GB (FACT: already downloaded on the GPU laptop) |
| LoRA adapter output | ~10–30 MB (ESTIMATE, r=8) |
| Logs, eval artifacts | < 5 MB |

~5 GB free disk is plenty. The 110 GB dataset is **not** needed on the training machine — only the extracted subset.

## 7. VRAM: is 6 GB enough?

| Component | fp16 ESTIMATE |
|---|---|
| Model weights | ~1.7 GB |
| LoRA params + AdamW states (LoRA only) | < 100 MB |
| Activations, batch 1, vision tower frozen | ~2–3 GB |
| CUDA context + fragmentation | ~0.6 GB |
| **Peak** | **~4.5–5.5 GB** |

**Verdict: 6 GB is realistically sufficient for LoRA at batch 1**, with roughly the same headroom the inference gate
measured (1.94 GB peak against 4.95 GB free). It is **not** sufficient for a full fine-tune: AdamW on 0.7B in fp16
needs ~8–9 GB for weights, gradients and optimizer states alone.

This is the one number that must be measured before committing to a long run — see the §10 spike.

## 8. Training time (ESTIMATE)

Basis: measured inference is 0.42 s/task at beams=3; a greedy forward is cheaper, a training step (forward+backward)
is roughly 2× a greedy forward.

| Setup | Step | 6,000 samples × 2 epochs |
|---|---|---|
| RTX 4050, batch 1 | ~0.6–0.9 s | **~2–3 h** |
| Colab T4, batch 4 | ~1.5–2.5 s/batch | ~1.5–2.5 h |

Either way it is an overnight-or-an-afternoon job, not a multi-day one.

## 9. Integration with the current pipeline

Deliberately minimal — no architecture change, no new tool interface:

1. Add `SATQUERY_FALCON_ADAPTER` (a local path or HF repo id) to `satquery/settings.py`.
2. In `FalconVLM.load()`, after the base model loads, apply the PEFT adapter when that setting is set.
3. Report it in the trace: set `model_id` to `"<base> + <adapter>"` so **every execution trace names the adapted
   model**. This is what demonstrates R5 to an evaluator, and it costs one string.
4. Nothing else changes: the registry, the tool signatures, `analyze()` and the UI all stay as they are. With the
   setting unset, behaviour is exactly today's.

## 10. Order of work (stop at the first thing that fails)

1. **Spike, ~30 min, no training.** Load Falcon, print its module names, attach LoRA, run **one** forward+backward, and
   record `torch.cuda.max_memory_reserved()`. This answers the architecture assumption (§3), the PEFT compatibility
   risk (§12) and the VRAM estimate (§7) in one go. **Decide here whether to stay on the 4050 or move to Colab.**
2. **Build the subset** with the existing script; commit the manifest, not the images.
3. **Baseline evaluation** of the unmodified model on the held-out rows — before any training, or there is nothing to
   compare against.
4. **Train** 1 epoch, check the loss curve and re-evaluate; then a second epoch only if it helped.
5. **Evaluate and write up.** Re-run the 8 demo scenarios to confirm nothing regressed qualitatively.

## 11. Evaluation and evidence to keep

**Evaluation.** Exact-match accuracy on held-out BigEarthNet.txt rows (patches never seen in training), reported
**before vs after** and **per `category`**, plus the yes/no base rate so a "always answer yes" degenerate result is
visible rather than hidden. No benchmark claims beyond what we measure (CLAUDE.md §7).

**Evidence for the submission** — everything except the weights goes in the repo:

| Artifact | Where |
|---|---|
| Training script + config | `experiments/adaptation/` |
| Data manifest (patch ids, splits, row counts) | `experiments/adaptation/manifest.json` |
| Loss curve / training log (CSV) | `experiments/adaptation/results/` |
| Before/after eval JSON + short report | `experiments/adaptation/results/` |
| Exact commands and environment (torch, transformers, peft versions) | the report |
| **Adapter weights** | **Not committed** (CLAUDE.md §11): push to the HF Hub, or keep on the laptop and record the SHA-256 |

## 12. Risks

| Risk | Mitigation |
|---|---|
| PEFT may not attach cleanly to Falcon's `trust_remote_code` model, and `transformers` is pinned to 4.49 | The §10 spike settles it in 30 minutes, before any data work |
| Falcon's training recipe (how labels are fed to an encoder–decoder) is not in our wrapper | Their inference path is known and working; derive the label format from the processor, and verify the loss actually falls on a 20-sample overfit test |
| Weight provenance: the re-upload is not hash-verified (D-021) | Already recorded; unchanged by this work |
| **Answer collapse**: BigEarthNet.txt has many binary rows, so the model can learn to always say "yes" | Balance yes/no, report per-category accuracy and the base rate |
| Adapting on 120×120 10 m Sentinel-2 may not transfer to Cartosat 0.65 m | State it plainly: this adapts to BigEarthNet, which is what R5 asks for. Do not claim it improves Cartosat performance |
| Time pressure before Round 1 | Fallback B (§2) is ~1 h end to end and cannot fail for architectural reasons |

## 13. The smallest credible implementation

If time collapses, this is the floor that still satisfies R5 honestly:

- 2,000 train patches → ~6,000 balanced QA pairs; 500 held-out patches for evaluation.
- LoRA `r=8` on the decoder, vision tower frozen, 1 epoch, batch 1 + grad accum 8.
- ~2–3 h on the RTX 4050; adapter ~20 MB.
- Before/after exact-match on ~1,000 held-out rows, reported per category.
- Wired in behind `SATQUERY_FALCON_ADAPTER`, so the trace names the adapted model.
- Written up in `experiments/adaptation/results/`.

If even that is at risk, do **fallback B** instead: train the ResNet-18 land-cover classifier on BigEarthNet patches
(~1 h, trivially within 6 GB), register it as a domain-gated tool (D-015 already anticipates this), and report its
per-class F1. It is a smaller claim, but it is a true one.

## 14. GPU: RTX 4050 vs Colab

| | RTX 4050 (6 GB) | Colab (free) |
|---|---|---|
| VRAM | 6 GB total, ~4.95 GB free measured (FACT) | T4 16 GB when allocated; varies by session |
| Fits LoRA batch 1 | Yes (ESTIMATE §7) | Yes, and allows batch 4–8 |
| Data | The BigEarthNet source is already on a team laptop (D-019); no upload | Must upload the ~700 MB subset to Drive |
| Model | Falcon already in the local HF cache (FACT) | Re-downloads ~3.4 GB per session |
| Interruptions | None | Idle disconnects and session caps; a long run can be lost |
| Known setup pitfalls | Developer Mode must be on, or `huggingface_hub` fails with `WinError 1314` (D-021) | Runtime resets lose the environment; checkpoint to Drive |

**Recommendation: train on the RTX 4050.** The data and the model are already there, the run is short enough that
session limits are the bigger risk, and §7 says it fits. Keep Colab as the fallback for one specific trigger: if the
§10 spike measures peak VRAM above ~5.2 GB, move to Colab rather than fighting the memory.

**Setup needed on the RTX 4050 laptop** (no security changes, nothing destructive):

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[models,dev]"   # torch 2.8 cu128, transformers 4.49
uv pip install --python .venv peft
# Windows: turn on Developer Mode first, or huggingface_hub cannot link cache blobs (D-021)
```

**One logistics item to confirm:** D-019 records the 110 GB BigEarthNet on "a team laptop", while the GPU is the
friend's RTX 4050. If they are different machines, run `build_round1_subset.py` on the data machine and copy only the
~700 MB subset across. Nothing else about the plan changes.
