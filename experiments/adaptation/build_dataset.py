"""Build the BigEarthNet.txt VQA training set for the SIH R5 LoRA adaptation.

Selects patches from BigEarthNet.txt.parquet (9.55M rows), renders each to RGB with the SAME
path the app uses at inference (satquery.imaging.load_image -> render_rgb), and writes one
JSONL row per question.

Scope (docs/adaptation-plan.md, as amended 2026-09-21):
- type == "binary" only. Bounding-box rows need Falcon's <bin> coordinate grammar, a separate
  and riskier target; binary rows are where exact-match is meaningful.
- Patch selection comes from BigEarthNet.txt's own `split` column, NOT metadata.parquet, which
  is not on this machine. Land-cover class balance is therefore not controlled: state that.
- yes/no is balanced per split, and the base rate is recorded, so an "always yes" collapse is visible.

Splits are disjoint by patch_id: a patch contributes to exactly one split.

Usage:
  python experiments/adaptation/build_dataset.py --out data/adaptation
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from satquery.imaging import load_image, render_rgb  # noqa: E402

PARQUET = Path(r"D:\Hackathon\satquery remote sensing\SatQuery\BigEarthNet.txt\BigEarthNet.txt.parquet")
S2_DIR = Path(r"D:\Hackathon\Overtime\Big EarthNet\BigEarthNet-S2\BigEarthNet-S2")
S1_DIR = Path(r"D:\Hackathon\Overtime\Big EarthNet\BigEarthNet-S1\BigEarthNet-S1")

# B04/B03/B02 = red/green/blue. satquery.imaging maps a 3-band TIFF to red, green, blue in order.
RGB_BANDS = ["B04", "B03", "B02"]
COLUMNS = ["patch_id", "s1_name", "input", "output", "type", "category", "split", "country", "season"]


def index_products(root: Path) -> dict[str, list[str]]:
    """Map a 10-char prefix to product folder names, so patch lookup needs no recursive scan."""
    index: dict[str, list[str]] = {}
    for entry in root.iterdir():
        if entry.is_dir():
            index.setdefault(entry.name[:10], []).append(entry.name)
    return index


def find_patch_dir(root: Path, name: str, index: dict[str, list[str]]) -> Path | None:
    for product in index.get(name[:10], []):
        if name.startswith(product) and (root / product / name).is_dir():
            return root / product / name
    return None


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def render_patch(patch_dir: Path, patch_id: str, out_path: Path) -> bool:
    """Write an RGB PNG using the app's own render path, so training input matches inference.

    Stacks B04/B03/B02 into one in-memory GeoTIFF-equivalent by reading each band, then hands the
    result to render_rgb, which applies the single shared percentile stretch across all three bands.
    """
    import rasterio

    bands, profile = [], None
    for band in RGB_BANDS:
        band_path = patch_dir / f"{patch_id}_{band}.tif"
        if not band_path.exists():
            return False
        with rasterio.open(band_path) as src:
            bands.append(src.read(1))
            profile = profile or src.profile
    profile.update(count=3, dtype=bands[0].dtype)

    tmp = out_path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(np.stack(bands))
        for i, name in enumerate(["red", "green", "blue"]):
            dst.set_band_description(i + 1, name)
    try:
        rgb = render_rgb(load_image(tmp, modality="optical"))
        Image.fromarray(rgb).save(out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def collect_rows(parquet: Path, splits: set[str], row_groups: list[int] | None) -> dict[str, list[dict]]:
    """Read binary rows grouped by split. Reads row groups one at a time: never the whole file."""
    handle = pq.ParquetFile(parquet)
    groups = row_groups if row_groups is not None else range(handle.metadata.num_row_groups)
    by_split: dict[str, list[dict]] = defaultdict(list)
    for g in groups:
        table = handle.read_row_group(g, columns=COLUMNS).to_pydict()
        for i in range(len(table["patch_id"])):
            if table["type"][i] != "binary" or table["split"][i] not in splits:
                continue
            by_split[table["split"][i]].append({k: table[k][i] for k in COLUMNS})
    return by_split


def balance_yes_no(rows: list[dict], rng: random.Random) -> list[dict]:
    """Downsample the majority answer so the base rate is 50%, hiding no 'always yes' win."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row["output"]).strip().lower()].append(row)
    if len(buckets) < 2:
        return rows
    n = min(len(v) for v in buckets.values())
    out: list[dict] = []
    for answers in buckets.values():
        rng.shuffle(answers)
        out.extend(answers[:n])
    rng.shuffle(out)
    return out


def select(rows: list[dict], patch_budget: int, rows_per_patch: int, rng: random.Random,
           exclude: set[str]) -> list[dict]:
    """Pick patches stratified by category, then cap rows per patch and balance yes/no."""
    by_patch: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["patch_id"] not in exclude:
            by_patch[row["patch_id"]].append(row)

    # Stratify: order patches by their dominant category, round-robin across categories.
    by_category: dict[str, list[str]] = defaultdict(list)
    for patch_id, patch_rows in by_patch.items():
        dominant = Counter(r["category"] for r in patch_rows).most_common(1)[0][0]
        by_category[dominant].append(patch_id)
    for ids in by_category.values():
        rng.shuffle(ids)

    chosen: list[str] = []
    categories = sorted(by_category)
    while len(chosen) < patch_budget and any(by_category[c] for c in categories):
        for category in categories:
            if by_category[category] and len(chosen) < patch_budget:
                chosen.append(by_category[category].pop())

    picked: list[dict] = []
    for patch_id in chosen:
        patch_rows = by_patch[patch_id][:]
        rng.shuffle(patch_rows)
        picked.extend(patch_rows[:rows_per_patch])
    return balance_yes_no(picked, rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--s2-dir", type=Path, default=S2_DIR)
    parser.add_argument("--s1-dir", type=Path, default=S1_DIR)
    parser.add_argument("--out", type=Path, default=Path("data/adaptation"))
    parser.add_argument("--train-patches", type=int, default=2000)
    parser.add_argument("--val-patches", type=int, default=300)
    parser.add_argument("--test-patches", type=int, default=500)
    parser.add_argument("--rows-per-patch", type=int, default=4)
    parser.add_argument("--row-groups", type=int, nargs="*",
                        help="read only these parquet row groups (default: all 78)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.parquet.name} ...", flush=True)
    by_split = collect_rows(args.parquet, {"train", "validation", "test"}, args.row_groups)
    for split, rows in sorted(by_split.items()):
        print(f"  {split:11s} {len(rows):8,} binary rows, {len({r['patch_id'] for r in rows}):7,} patches")

    budgets = {"train": args.train_patches, "validation": args.val_patches, "test": args.test_patches}
    s2_index, s1_index = index_products(args.s2_dir), index_products(args.s1_dir)

    used_patches: set[str] = set()
    manifest = {
        "dataset": "BigEarthNet.txt (BIFOLD-BigEarthNetv2-0), CDLA-Permissive-1.0",
        "parquet": str(args.parquet),
        "parquet_sha256": file_sha256(args.parquet),
        "image_source": "BigEarthNet v2.0 S2 L2A, bands B04/B03/B02",
        "render": "satquery.imaging.load_image -> render_rgb (one shared 2-98 percentile stretch)",
        "row_type_filter": "binary",
        "selection": "stratified by category from BigEarthNet.txt `split`; metadata.parquet unavailable, "
                     "so land-cover class balance is NOT controlled",
        "yes_no": "balanced per split by downsampling the majority answer",
        "seed": args.seed,
        "rows_per_patch_cap": args.rows_per_patch,
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        rows = by_split.get(split, [])
        if not rows:
            print(f"WARNING: no rows for split {split}")
            continue
        picked = select(rows, budgets[split], args.rows_per_patch, rng, exclude=used_patches)

        images_dir = args.out / "images" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        written: list[dict] = []
        rendered: dict[str, Path] = {}
        skipped = 0
        for row in picked:
            patch_id = row["patch_id"]
            if patch_id not in rendered:
                patch_dir = find_patch_dir(args.s2_dir, patch_id, s2_index)
                out_png = images_dir / f"{patch_id}.png"
                if patch_dir is None or not (out_png.exists() or render_patch(patch_dir, patch_id, out_png)):
                    rendered[patch_id] = None
                else:
                    rendered[patch_id] = out_png
            image_path = rendered[patch_id]
            if image_path is None:
                skipped += 1
                continue
            written.append({
                "image_path": str(image_path.relative_to(args.out)).replace("\\", "/"),
                "question": row["input"],
                "answer": str(row["output"]).strip(),
                "category": row["category"],
                "patch_id": patch_id,
                "s1_name": row["s1_name"],
                "country": row["country"],
                "split": split,
            })
            if len(written) % 500 == 0:
                print(f"  {split}: {len(written):,} rows written", flush=True)

        used_patches.update(r["patch_id"] for r in written)
        jsonl = args.out / f"{split}.jsonl"
        with open(jsonl, "w", encoding="utf-8") as handle:
            for row in written:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        answers = Counter(r["answer"].lower() for r in written)
        total = sum(answers.values()) or 1
        manifest["splits"][split] = {
            "rows": len(written),
            "patches": len({r["patch_id"] for r in written}),
            "skipped_rows_missing_imagery": skipped,
            "answers": dict(answers),
            "yes_base_rate": round(answers.get("yes", 0) / total, 4),
            "by_category": dict(Counter(r["category"] for r in written)),
            "jsonl": jsonl.name,
        }
        print(f"{split}: {len(written):,} rows, {manifest['splits'][split]['patches']:,} patches, "
              f"yes rate {manifest['splits'][split]['yes_base_rate']}, skipped {skipped}")

    # Disjointness is the claim the evaluation rests on, so assert it rather than trusting the flow.
    seen: dict[str, str] = {}
    for split in manifest["splits"]:
        for line in (args.out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            patch_id = json.loads(line)["patch_id"]
            if seen.setdefault(patch_id, split) != split:
                raise SystemExit(f"patch {patch_id} appears in both {seen[patch_id]} and {split}")
    manifest["splits_disjoint_by_patch_id"] = True

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
