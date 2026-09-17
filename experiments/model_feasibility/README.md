# Model feasibility test: Falcon 0.7B on a 6 GB laptop GPU

**Question answered:** can we run a remote-sensing VLM locally for the Round 1 demo?

- **Isolated.** It has its own virtual environment. It imports nothing from the app, and nothing imports it.
- **Temporary.** Delete this folder once a model decision is made.

**Why Falcon and not GeoChat-7B:** see `docs/decisions.md` (GeoChat-7B at 4-bit needs about 6.4–6.8 GB VRAM against ~5.2 GB free on the target laptop).

## What it does
1. Downloads Falcon's own sample images from its GitHub repo into `samples/`.
2. Downloads the model (~3.4 GB, fp32 weights) into the normal Hugging Face cache.
3. Runs 6 tasks with the prompts from Falcon's README:
   - classification
   - VQA
   - captioning
   - box detection
   - segmentation
   - bi-temporal change detection
4. Records load time, latency per task, peak process RAM, and peak VRAM (both torch-reserved and whole-GPU).
5. Writes `results/<run>/report.json` plus one overlay PNG per task, then prints a PASS/FAIL gate.

The gate passes when all of these hold:
- at least 4 of 6 tasks produce output
- on CUDA, peak reserved VRAM is 5.0 GB or less
- on CUDA, median latency is 5 s or less

**The gate does not judge answer quality.** Open the overlay PNGs to check that.

## Model source (read this)
The official `TianHuiLab/Falcon-Single-Instruction-Large` repo is **gated with manual approval**. It also stores its files in a subfolder, which breaks standard loading (see HF discussion #2).

We use the public re-upload **`mehmetbayik/Falcon-Single-Instruction-Large`**:
- Its code, config and tokenizer files are **hash-identical** to the official repo.
- Its `model.safetensors` has the **identical byte size** (3,350,641,628). The official hash is hidden behind the gate, so the weights themselves are not hash-verified.
- To use the official weights once access is granted: `--model-id TianHuiLab/Falcon-Single-Instruction-Large`. That also needs `huggingface-cli login` and subfolder handling, which is not implemented here.

## Setup (Windows PowerShell, run inside this folder)
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -r requirements.txt
```
No `uv`? Use `py -3.12 -m venv .venv`, then `.venv\Scripts\pip install ...` with the same arguments.

**Windows path-length trap:** `transformers` installs a file with a 73-character name. If the path to this folder is longer than ~130 characters, Windows' 260-character limit silently breaks the install. You then get `ImportError: cannot import name 'dummy_essentia_and_librosa...'`. Fix it by cloning the repo to a short path (e.g. `D:\sq\`) or by enabling Windows long paths.

## Run
Close other GPU-heavy apps first (browsers with video, games), because the gate measures free VRAM.
```powershell
.venv\Scripts\python run_feasibility.py --device cuda                  # main test (~5 min after downloads)
.venv\Scripts\python run_feasibility.py --device cuda --num-beams 1    # faster decoding variant
.venv\Scripts\python run_feasibility.py --device cpu --tasks IMG_CAP IMG_VQA   # CPU fallback timing
```

## If something fails
| Symptom | Try |
|---|---|
| Error mentioning `GenerationMixin` / `generate` | `uv pip install --python .venv transformers==4.41.2` (Falcon's original pin), then re-run |
| `flash_attn` import error | The script already patches this. Report the full traceback. |
| CUDA out of memory | Close other GPU apps; re-run with `--num-beams 1` |
| `torch.cuda.is_available()` is False | The CUDA wheel was not installed: repeat the torch install line |

## What to send back
The whole `results/` folder (JSON + PNGs, a few MB) and the console output.
