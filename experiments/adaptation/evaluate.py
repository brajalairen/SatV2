"""Evaluate Falcon on held-out BigEarthNet.txt binary VQA rows, with or without the LoRA adapter.

The SAME code runs before and after training, so the two numbers are comparable. Reports
exact-match overall and per category, plus the yes/no base rate and the model's own answer
distribution, so an "always yes" collapse is visible rather than hidden (docs/adaptation-plan.md §12).

Usage:
  python experiments/adaptation/evaluate.py --split test --out results/eval_before.json
  python experiments/adaptation/evaluate.py --split test --adapter runs/adapter --out results/eval_after.json
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from falcon_lora import MODEL_ID, apply_adapter, load_base, load_rgb, normalise, read_jsonl  # noqa: E402


def generate(model, processor, device, dtype, rgb, question: str, num_beams: int, max_new_tokens: int) -> str:
    import torch
    from PIL import Image

    inputs = processor(text=question.strip(), images=Image.fromarray(rgb), return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs["input_ids"].to(device),
            pixel_values=inputs["pixel_values"].to(device, dtype),
            max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False,
        )
    text = processor.batch_decode(output, skip_special_tokens=True)[0]
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/adaptation"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--adapter", type=Path, help="LoRA adapter directory; omit for the base model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="evaluate only the first N rows")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)  # binary answers are one word
    args = parser.parse_args()

    rows = read_jsonl(args.data / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    print(f"{args.split}: {len(rows):,} rows", flush=True)

    model, processor, device, dtype = load_base(args.model_id, args.device)
    if args.adapter:
        model = apply_adapter(model, str(args.adapter))
        print(f"adapter applied: {args.adapter}", flush=True)

    correct = 0
    by_category: dict[str, list[int]] = defaultdict(list)
    predictions = Counter()
    references = Counter()
    samples = []
    start = time.perf_counter()
    for i, row in enumerate(rows):
        rgb = load_rgb(args.data, row)
        raw = generate(model, processor, device, dtype, rgb, row["question"],
                       args.num_beams, args.max_new_tokens)
        hit = int(normalise(raw) == normalise(row["answer"]))
        correct += hit
        by_category[row["category"]].append(hit)
        predictions[normalise(raw)[:20] or "(empty)"] += 1
        references[normalise(row["answer"])] += 1
        if i < 10:
            samples.append({"question": row["question"], "reference": row["answer"],
                            "prediction": raw, "category": row["category"], "correct": bool(hit)})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1:,}/{len(rows):,}  running exact-match {correct / (i + 1):.4f}", flush=True)
    elapsed = time.perf_counter() - start

    total = len(rows) or 1
    report = {
        "model_id": args.model_id,
        "adapter": str(args.adapter) if args.adapter else None,
        "split": args.split,
        "rows": len(rows),
        "num_beams": args.num_beams,
        "exact_match": round(correct / total, 4),
        "correct": correct,
        "per_category": {c: {"n": len(v), "exact_match": round(sum(v) / len(v), 4)}
                         for c, v in sorted(by_category.items())},
        "reference_answer_distribution": dict(references),
        "yes_base_rate": round(references.get("yes", 0) / total, 4),
        "prediction_distribution": dict(predictions.most_common(10)),
        "seconds": round(elapsed, 1),
        "seconds_per_row": round(elapsed / total, 3),
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in
                      ("exact_match", "per_category", "yes_base_rate", "prediction_distribution")}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
