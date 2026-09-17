"""Isolated feasibility test: can Falcon (0.7B remote-sensing VLM) run on this machine?

Self-contained on purpose: imports nothing from the SatQuery app, and nothing imports it.
Measures load time, peak VRAM/RAM and per-task latency on Falcon's official sample images,
saves raw outputs plus overlay images, and writes one JSON report to ./results/.

Usage (see README.md):
    python run_feasibility.py --device cuda
    python run_feasibility.py --device cpu --tasks IMG_CAP IMG_VQA
"""

import argparse
import json
import os
import platform
import re
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import torch
import transformers
from PIL import Image, ImageDraw
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

HERE = Path(__file__).resolve().parent
SAMPLES_BASE = "https://raw.githubusercontent.com/TianHuiLab/Falcon/main/image_samples/"

# Prompts and sample images are taken verbatim from Falcon's README examples.
TASKS = {
    "IMG_CLS": ("Classify the image.\nUse one or a few words.",
                ["IMG_CLS/[IMG_CLS]_003_AID_3525_river_192_ori.png"]),
    "IMG_VQA": ("Is the number of roads equal to the number of residential areas?",
                ["IMG_VQA/[IMG_VQA]_007_HRBEN_5965_1335_ori.png"]),
    "IMG_CAP": ("Describe the image.",
                ["IMG_CAP/[IMG_CAP]_010_RSICD_208_church_56_ori.png"]),
    "REG_DET_HBB": ("Detect all stadium in the image.\nUse horizontal bounding boxes.",
                    ["REG_DET_HBB/[REG_DET_HBB]_004_DIOR_5212_12735_ori.png"]),
    "PIX_SEG": ("Segment out road in the image.",
                ["PIX_SEG/[PIX_SEG]_034_GEONRW_376_5755_rgb-ori.png"]),
    "PIX_CHG": ("Find changes in the two images.",
                ["PIX_CHG/[PIX_CHG]_199_WHU-CD_28911_590_ori.png",
                 "PIX_CHG/[PIX_CHG]_199_WHU-CD_28911_590_post.png"]),
}

# Decision gate for a first-round demo on a 6 GB laptop GPU.
GATE_MAX_PEAK_VRAM_GB = 5.0
GATE_MAX_MEDIAN_LATENCY_S_CUDA = 5.0
GATE_MIN_TASKS_WITH_OUTPUT = 4


def patched_get_imports(filename):
    """Falcon's remote code imports flash_attn behind an `if`; transformers still demands it."""
    return [imp for imp in get_imports(filename) if imp != "flash_attn"]


class PeakSampler:
    """Samples process RSS and whole-GPU used memory (includes other processes) in the background."""

    def __init__(self, interval_s=0.1):
        self.interval_s = interval_s
        self.peak_rss = 0
        self.peak_gpu_used = 0
        self._stop = threading.Event()
        self._nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception:
            pass
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        proc = psutil.Process()
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, proc.memory_info().rss)
            if self._nvml:
                pynvml, handle = self._nvml
                self.peak_gpu_used = max(self.peak_gpu_used, pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            time.sleep(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def gb(n_bytes):
    return round(n_bytes / 1024**3, 3)


def environment_report(device):
    vm = psutil.virtual_memory()
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "ram_total_gb": gb(vm.total),
        "ram_available_gb_at_start": gb(vm.available),
        "device": device,
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report.update(gpu=torch.cuda.get_device_name(0), vram_total_gb=gb(total),
                      vram_free_gb_at_start=gb(free), torch_cuda=torch.version.cuda)
    return report


def ensure_samples(samples_dir):
    samples_dir.mkdir(parents=True, exist_ok=True)
    for _, rel_paths in TASKS.values():
        for rel in rel_paths:
            target = samples_dir / rel
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            url = SAMPLES_BASE + urllib.parse.quote(rel)
            print(f"downloading sample {rel}")
            urllib.request.urlretrieve(url, target)


def load_task_image(samples_dir, rel_paths):
    images = [Image.open(samples_dir / rel).convert("RGB") for rel in rel_paths]
    if len(images) == 1:
        return images[0]
    # Bi-temporal composite exactly as in Falcon's inference.py: t1 top-left, t2 bottom-right.
    img1, img2 = np.array(images[0]), np.array(images[1])
    blank = np.zeros(img1.shape)
    left = np.concatenate((img1, blank), axis=0)
    right = np.concatenate((blank, img2), axis=0)
    return Image.fromarray(np.uint8(np.concatenate((left, right), axis=1)))


def dequantize(bins, size, n_bins=1000):
    """Falcon coordinates are 1000x1000 bins; convert bin indices to pixel centres."""
    w, h = size
    return [((b + 0.5) * (w if i % 2 == 0 else h) / n_bins) for i, b in enumerate(bins)]


def parse_output(task, text, size):
    if task == "REG_DET_HBB":
        boxes = re.findall(r"<(\d+)><(\d+)><(\d+)><(\d+)>", text)
        return {"boxes_px": [dequantize([int(v) for v in box], size) for box in boxes]}
    if task in ("PIX_SEG", "PIX_CHG"):
        polygons = []
        for instance in re.findall(r"<poly>(.*?)</poly>", text):
            for part in instance.split("<sep>"):
                values = [int(v) for v in re.findall(r"<(\d+)>", part)]
                if len(values) >= 6:
                    polygons.append(dequantize(values[: len(values) // 2 * 2], size))
        return {"polygons_px": polygons}
    return {"answer": re.sub(r"</?s>|<pad>", "", text).strip()}


def save_overlay(image, parsed, path):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for box in parsed.get("boxes_px", []):
        draw.rectangle(box, outline=(255, 0, 0), width=3)
    for poly in parsed.get("polygons_px", []):
        draw.polygon(list(zip(poly[0::2], poly[1::2])), outline=(255, 0, 0), width=3)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="mehmetbayik/Falcon-Single-Instruction-Large")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default=None, help="default: fp16 on cuda, fp32 on cpu")
    parser.add_argument("--num-beams", type=int, default=3, help="Falcon's reference setting is 3")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--samples-dir", type=Path, default=HERE / "samples")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but torch.cuda.is_available() is False (CPU-only torch wheel?).")
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype or ("fp16" if args.device == "cuda" else "fp32")]
    run_name = f"{args.model_id.split('/')[-1]}_{args.device}_{str(dtype).split('.')[-1]}_beams{args.num_beams}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"run": run_name, "model_id": args.model_id, "environment": environment_report(args.device),
              "settings": {"dtype": str(dtype), "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens}}
    print(json.dumps(report["environment"], indent=2))
    ensure_samples(args.samples_dir)

    with PeakSampler() as sampler:
        t0 = time.perf_counter()
        local_path = snapshot_download(args.model_id)
        report["download_or_cache_s"] = round(time.perf_counter() - t0, 1)

        t0 = time.perf_counter()
        with patch("transformers.dynamic_module_utils.get_imports", patched_get_imports):
            model = AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=True, torch_dtype=dtype)
            processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
        model = model.to(args.device).eval()
        if args.device == "cuda":
            torch.cuda.synchronize()
        report["load_s"] = round(time.perf_counter() - t0, 1)
        print(f"model loaded in {report['load_s']} s")

        results = []
        for i, task in enumerate(args.tasks):
            prompt, rel_paths = TASKS[task]
            image = load_task_image(args.samples_dir, rel_paths)
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            entry = {"task": task, "prompt": prompt, "image_size": list(image.size)}
            try:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=inputs["input_ids"].to(args.device),
                        pixel_values=inputs["pixel_values"].to(args.device, dtype),
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                        do_sample=False,
                    )
                if args.device == "cuda":
                    torch.cuda.synchronize()
                entry["latency_s"] = round(time.perf_counter() - t0, 2)
                text = processor.batch_decode(generated, skip_special_tokens=False)[0]
                entry.update(generated_tokens=int(generated.shape[-1]), raw_output=text,
                             parsed=parse_output(task, text, image.size))
                overlay = out_dir / f"{task}.png"
                save_overlay(image, entry["parsed"], overlay)
                entry["overlay"] = str(overlay)
            except Exception as error:  # record and continue: one failing task must not hide the others
                entry["error"] = f"{type(error).__name__}: {error}"
            entry["first_task_includes_warmup"] = i == 0
            results.append(entry)
            print(f"{task:12s} {entry.get('latency_s', 'ERR'):>6} s  {str(entry.get('parsed', entry.get('error')))[:120]}")

    report["tasks"] = results
    report["peak_process_ram_gb"] = gb(sampler.peak_rss)
    report["peak_gpu_used_gb_all_processes"] = gb(sampler.peak_gpu_used) if sampler.peak_gpu_used else None
    if args.device == "cuda":
        report["torch_peak_allocated_gb"] = gb(torch.cuda.max_memory_allocated())
        report["torch_peak_reserved_gb"] = gb(torch.cuda.max_memory_reserved())

    ok = [r for r in results if "error" not in r and any(r["parsed"].values())]
    timed = [r["latency_s"] for r in results[1:] if "latency_s" in r] or [r["latency_s"] for r in ok]
    median_latency = statistics.median(timed) if timed else None
    gate = {"tasks_with_output": len(ok), "median_latency_s_excl_warmup": median_latency,
            "passes_output_gate": len(ok) >= GATE_MIN_TASKS_WITH_OUTPUT}
    if args.device == "cuda":
        peak_vram = report["torch_peak_reserved_gb"]
        gate["passes_vram_gate"] = peak_vram <= GATE_MAX_PEAK_VRAM_GB
        gate["passes_latency_gate"] = median_latency is not None and median_latency <= GATE_MAX_MEDIAN_LATENCY_S_CUDA
    gate["verdict"] = "PASS" if all(v for k, v in gate.items() if k.startswith("passes_")) else "FAIL"
    gate["note"] = "Also inspect the overlay PNGs: the gate checks feasibility, not answer quality."
    report["gate"] = gate

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(gate, indent=2))
    print(f"report written to {report_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
