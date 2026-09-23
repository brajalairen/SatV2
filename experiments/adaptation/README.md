# SIH R5: LoRA adaptation of Falcon on BigEarthNet.txt

Self-contained, like `experiments/model_feasibility/`: the app never imports this directory. The one
thing that crosses over is `satquery.imaging`, reused by `build_dataset.py` so training images are
rendered by exactly the code path that renders them at inference.

**Nothing here claims a result that is not in `results/`.** See CLAUDE.md §7.

## What this adapts, and what it does not

Falcon is remote-sensing pre-trained *by its authors* (D-021), which does not satisfy SIH R5. This
trains a LoRA adapter **we** fit, on **BigEarthNet.txt**, the dataset the problem statement names.

Scope limits, stated plainly:

- **Binary (yes/no) VQA rows only.** Bounding-box rows need Falcon's `<bin>` coordinate grammar, a
  separate and riskier target; binary rows are where exact-match is unambiguous.
- **No land-cover class balancing.** `metadata.parquet` is not on this machine, so patches are selected
  from BigEarthNet.txt's own `split` column and stratified by question `category` instead.
- Adapts to **Sentinel-2, 120x120 px at 10 m**. This says nothing about Cartosat-2S at 0.65 m, and no
  such claim may be made.

## Files

| File | Purpose |
|---|---|
| `falcon_lora.py` | shared loading (same flash_attn patch as the app), LoRA attach, JSONL/IO helpers |
| `build_dataset.py` | patch selection, RGB render via `satquery.imaging`, balanced JSONL + manifest |
| `train_lora.py` | LoRA training, per-step CSV log, adapter + config output |
| `evaluate.py` | exact-match per category, before/after; identical code for both |
| `results/` | committed evidence: manifest, logs, evaluations, report |

## Reproduce

Requires the `models` extra plus `peft`. Install `peft`/`accelerate` with `--no-deps`, or they pull
`transformers` past 4.50 and break Falcon's remote code (D-021):

```powershell
uv pip install --python .venv -e ".[models,data,dev]"
uv pip install --python .venv --no-deps "peft==0.17.1" "accelerate==1.10.1"
```

```powershell
# 1. dataset (~97 MB into gitignored data/; the manifest is committed)
.venv\Scripts\python experiments\adaptation\build_dataset.py --out data\adaptation

# 2. sanity check: the loss must collapse on 20 samples before spending an epoch
.venv\Scripts\python experiments\adaptation\train_lora.py --overfit 20 --epochs 25 --lr 5e-4 `
    --grad-accum 4 --out runs\adaptation\overfit

# 3. baseline BEFORE training, or there is nothing to compare against
.venv\Scripts\python experiments\adaptation\evaluate.py --split test `
    --out experiments\adaptation\results\eval_before.json

# 4. train
.venv\Scripts\python experiments\adaptation\train_lora.py --out runs\adaptation\adapter

# 5. after
.venv\Scripts\python experiments\adaptation\evaluate.py --split test --adapter runs\adaptation\adapter `
    --out experiments\adaptation\results\eval_after.json
```

## Using the adapter in the app

```powershell
$env:SATQUERY_VLM_BACKEND="falcon"; $env:SATQUERY_FALCON_ADAPTER="runs\adaptation\adapter"
```

`model_id` then reads `"<base> + <adapter>"`, so **every execution trace names the adapted model**.
Unset, behaviour is identical to the base model and `peft` is never imported.

## Data and weights are not committed

`data/` and `runs/` are gitignored (CLAUDE.md §11). The manifest records the source parquet's SHA-256
and the exact selection settings, so the dataset can be rebuilt; the adapter's own SHA-256 is recorded
in `results/report.md`.
