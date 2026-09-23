"""LoRA adaptation of Falcon on BigEarthNet.txt binary VQA rows (SIH R5).

Config is the one measured feasible on the RTX 4050 on 2026-09-21: LoRA r=8, alpha=16,
dropout 0.05, the 96 decoder attention projections, vision tower frozen, batch 1 with gradient
accumulation, fp16 base + fp32 LoRA params. Measured peak reserved VRAM 2.09 GB of 6.44 GB.

Run the overfit check first; if the loss does not collapse on 20 samples, stop and diagnose
rather than burning an epoch:
  python experiments/adaptation/train_lora.py --overfit 20 --out runs/overfit

Then the real run:
  python experiments/adaptation/train_lora.py --out runs/adapter
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from falcon_lora import MODEL_ID, attach_lora, load_base, load_rgb, read_jsonl  # noqa: E402


def encode(processor, rgb, question: str, answer: str, device, dtype):
    inputs = processor(text=question.strip(), images=Image.fromarray(rgb), return_tensors="pt")
    labels = processor.tokenizer(str(answer).strip(), return_tensors="pt").input_ids
    return (inputs["input_ids"].to(device), inputs["pixel_values"].to(device, dtype), labels.to(device))


def evaluate_loss(model, processor, rows, data_root, device, dtype, limit=None):
    """Mean loss over held-out rows: the signal for whether a second epoch is worth running."""
    rows = rows[:limit] if limit else rows
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for row in rows:
            input_ids, pixel_values, labels = encode(
                processor, load_rgb(data_root, row), row["question"], row["answer"], device, dtype)
            total += float(model(input_ids=input_ids, pixel_values=pixel_values, labels=labels).loss)
    model.train()
    return round(total / max(len(rows), 1), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/adaptation"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, help="use only the first N training rows")
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--overfit", type=int,
                        help="sanity check: train on N samples for several epochs and expect the loss to collapse")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.data / "train.jsonl")
    val_rows = read_jsonl(args.data / "validation.jsonl") if (args.data / "validation.jsonl").exists() else []
    if args.overfit:
        train_rows, val_rows = train_rows[: args.overfit], []
        args.epochs = max(args.epochs, 8)  # the point is to see the loss fall on a tiny set
    elif args.limit:
        train_rows = train_rows[: args.limit]
    print(f"train rows {len(train_rows):,}  validation rows {len(val_rows):,}", flush=True)

    model, processor, device, dtype = load_base(args.model_id, args.device)
    model, targets = attach_lora(model, args.rank, args.alpha, args.dropout)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA on {len(targets)} modules, {trainable:,} trainable params", flush=True)

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    steps = math.ceil(len(train_rows) * args.epochs / args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    log_path = args.out / "train_log.csv"
    log = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log)
    writer.writerow(["epoch", "sample", "loss", "lr", "seconds"])

    start = time.perf_counter()
    history = []
    seen = 0
    for epoch in range(args.epochs):
        random.shuffle(train_rows)
        running = 0.0
        for i, row in enumerate(train_rows):
            input_ids, pixel_values, labels = encode(
                processor, load_rgb(args.data, row), row["question"], row["answer"], device, dtype)
            loss = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels).loss
            (loss / args.grad_accum).backward()
            running += float(loss.detach())
            seen += 1
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(train_rows):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (i + 1) % 50 == 0 or i + 1 == len(train_rows):
                mean = running / min(50, (i % 50) + 1 if (i + 1) % 50 else 50)
                writer.writerow([epoch, i + 1, round(mean, 4),
                                 round(scheduler.get_last_lr()[0], 8), round(time.perf_counter() - start, 1)])
                log.flush()
                print(f"epoch {epoch} {i + 1:,}/{len(train_rows):,}  loss {mean:.4f}  "
                      f"{(time.perf_counter() - start) / seen:.2f}s/sample", flush=True)
                running = 0.0
        entry = {"epoch": epoch, "train_samples": len(train_rows)}
        if val_rows:
            entry["val_loss"] = evaluate_loss(model, processor, val_rows, args.data, device, dtype, args.val_limit)
            print(f"epoch {epoch} validation loss {entry['val_loss']}", flush=True)
        history.append(entry)
    log.close()

    model.save_pretrained(str(args.out))
    config = {
        "model_id": args.model_id,
        "method": "LoRA (peft)",
        "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout,
        "target_modules": targets,
        "target_module_count": len(targets),
        "vision_tower": "frozen",
        "trainable_params": trainable,
        "batch_size": 1, "grad_accum": args.grad_accum, "effective_batch": args.grad_accum,
        "lr": args.lr, "scheduler": "cosine", "epochs": args.epochs,
        "dtype": "fp16 base, fp32 LoRA params",
        "seed": args.seed,
        "train_rows": len(train_rows),
        "history": history,
        "minutes": round((time.perf_counter() - start) / 60, 1),
        "torch": torch.__version__,
    }
    if device == "cuda":
        config["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
        config["gpu"] = torch.cuda.get_device_name(0)
    (args.out / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in config.items() if k != "target_modules"}, indent=2))
    print(f"\nadapter: {args.out}\nlog: {log_path}")


if __name__ == "__main__":
    main()
