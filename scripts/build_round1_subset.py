"""Build the small Round 1 demo subset from the local BigEarthNet v2 S1+S2 dataset (~110 GB), per D-019.

Reads metadata.parquet and ONLY the selected patches. It never scans or loads the whole dataset.
For each patch it writes into <out>/<patch_id>/:
  s2_bgrn.tif    B02, B03, B04, B08 (10 m) with band descriptions
  s1_vv_vh.tif   VV, VH with band descriptions
  preview.png    true-colour stretch
  text.json      matching BigEarthNet.txt annotations (if --text-parquet is given)
plus <out>/manifest.json.

Usage (needs: pip install pandas pyarrow):
  python scripts/build_round1_subset.py --bigearthnet-root D:/BigEarthNet --out data/subsets/round1 \
      --text-parquet data/bigearthnet/txt/BigEarthNet.txt.parquet
Folder layout assumed (BigEarthNet v2.0, UNVERIFIED locally; override with --s1-dir/--s2-dir):
  <root>/metadata.parquet, <root>/BigEarthNet-S2/<tile>/<patch_id>/<patch_id>_B02.tif,
  <root>/BigEarthNet-S1/<product>/<s1_name>/<s1_name>_VV.tif
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from PIL import Image

DEFAULT_CLASSES = ["Urban fabric", "Inland waters", "Arable land", "Broad-leaved forest", "Industrial or commercial units"]
S2_BANDS = ["B02", "B03", "B04", "B08"]
TEXT_ID_CANDIDATES = ["patch_id", "s2_patch_id", "s2v2_name", "image_id", "id", "name"]


def find_patch_dir(root: Path, patch_name: str, product_index: dict[str, list[str]]) -> Path | None:
    """Find <root>/<product>/<patch_name> by matching product-folder prefixes (no recursive scan)."""
    for product in product_index.get(patch_name[:10], []):
        if patch_name.startswith(product) and (root / product / patch_name).is_dir():
            return root / product / patch_name
    return None


def index_products(root: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for entry in root.iterdir():
        if entry.is_dir():
            index.setdefault(entry.name[:10], []).append(entry.name)
    return index


def stack_bands(paths: list[Path], names: list[str], out_path: Path) -> dict:
    arrays, profile = [], None
    for path in paths:
        with rasterio.open(path) as src:
            arrays.append(src.read(1))
            profile = profile or src.profile
    profile.update(count=len(arrays), dtype=arrays[0].dtype)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.stack(arrays))
        for i, name in enumerate(names):
            dst.set_band_description(i + 1, name)
    return {"crs": str(profile["crs"]), "transform": list(profile["transform"])[:6], "shape": list(arrays[0].shape)}


def preview(s2_path: Path, out_path: Path):
    with rasterio.open(s2_path) as src:
        bands = src.read([3, 2, 1]).astype(np.float32)
    lo, hi = np.percentile(bands, [2, 98])
    rgb = np.clip((bands - lo) / (hi - lo + 1e-6), 0, 1)
    Image.fromarray((np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)).resize((360, 360), Image.NEAREST).save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bigearthnet-root", type=Path, required=True)
    parser.add_argument("--s1-dir", type=Path)
    parser.add_argument("--s2-dir", type=Path)
    parser.add_argument("--metadata", type=Path, help="default: <root>/metadata.parquet")
    parser.add_argument("--patch-ids", nargs="+", help="export these S2 patch ids instead of selecting by class; "
                                                       "needs --text-parquet and no metadata.parquet")
    parser.add_argument("--text-parquet", type=Path, help="BigEarthNet.txt.parquet (optional)")
    parser.add_argument("--text-id-column", help="column in the text parquet holding the S2 patch id")
    parser.add_argument("--out", type=Path, default=Path("data/subsets/round1"))
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    root = args.bigearthnet_root
    s1_dir, s2_dir = args.s1_dir or root / "BigEarthNet-S1", args.s2_dir or root / "BigEarthNet-S2"

    if args.patch_ids:
        # Explicit patches. BigEarthNet.txt.parquet carries the authoritative patch_id -> s1_name mapping,
        # so metadata.parquet is not needed here. Class labels are NOT available on this path.
        if not args.text_parquet:
            parser.error("--patch-ids requires --text-parquet, which supplies s1_name")
        wanted = list(dict.fromkeys(args.patch_ids))
        pairs = pd.read_parquet(args.text_parquet, columns=["patch_id", "s1_name", "country", "season"],
                                filters=[("patch_id", "in", wanted)])
        pairs = pairs.drop_duplicates(subset="patch_id")
        missing = set(wanted) - set(pairs["patch_id"])
        if missing:
            print(f"WARNING: not found in the text parquet: {sorted(missing)}")
        chosen = {}
        for _, row in pairs.iterrows():
            row = row.copy()
            row["labels"] = []  # unavailable without metadata.parquet
            chosen[row["patch_id"]] = row
    else:
        meta = pd.read_parquet(args.metadata or root / "metadata.parquet")
        print(f"metadata: {len(meta)} rows, columns: {list(meta.columns)}")
        if "split" in meta.columns:
            meta = meta[meta["split"] == args.split]
        for flag in ("contains_seasonal_snow", "contains_cloud_or_shadow"):
            if flag in meta.columns:
                meta = meta[~meta[flag].astype(bool)]

        rng = random.Random(args.seed)
        chosen = {}
        for cls in args.classes:
            rows = meta[meta["labels"].apply(lambda labels: cls in list(labels))]
            for _, row in rows.sample(n=min(args.per_class, len(rows)), random_state=rng.randint(0, 10**6)).iterrows():
                chosen.setdefault(row["patch_id"], row)
    print(f"selected {len(chosen)} patches")

    text = None
    if args.text_parquet:
        # Resolve the id column from the schema, then filter while reading: the full file is ~9.5M rows,
        # and CLAUDE.md 11 forbids whole-dataset loads.
        names = list(pq.read_schema(args.text_parquet).names)
        id_column = args.text_id_column or next((c for c in TEXT_ID_CANDIDATES if c in names), None)
        print(f"BigEarthNet.txt columns: {names}; patch id column: {id_column}")
        if id_column:
            text = pd.read_parquet(args.text_parquet, filters=[(id_column, "in", list(chosen))])
            print(f"BigEarthNet.txt rows for the selected patches: {len(text)}")
        else:
            print("WARNING: no patch id column found; re-run with --text-id-column")

    s1_index, s2_index = index_products(s1_dir), index_products(s2_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for patch_id, row in chosen.items():
        s2_patch = find_patch_dir(s2_dir, patch_id, s2_index)
        s1_patch = find_patch_dir(s1_dir, row["s1_name"], s1_index) if "s1_name" in row else None
        if s2_patch is None or s1_patch is None:
            print(f"skip {patch_id}: S2 found={s2_patch is not None}, S1 found={s1_patch is not None}")
            continue
        target = args.out / patch_id
        target.mkdir(exist_ok=True)
        s2_grid = stack_bands([s2_patch / f"{patch_id}_{b}.tif" for b in S2_BANDS], S2_BANDS, target / "s2_bgrn.tif")
        s1_grid = stack_bands([s1_patch / f"{row['s1_name']}_{p}.tif" for p in ("VV", "VH")], ["VV", "VH"], target / "s1_vv_vh.tif")
        preview(target / "s2_bgrn.tif", target / "preview.png")
        entry = {"patch_id": patch_id, "labels": list(row["labels"]), "country": row.get("country"),
                 "s1_name": row["s1_name"], "same_grid": s1_grid == s2_grid, "s2_grid": s2_grid}
        if text is not None:
            id_column = args.text_id_column or next(c for c in TEXT_ID_CANDIDATES if c in text.columns)
            rows = text[text[id_column] == patch_id].to_dict("records")
            (target / "text.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
            entry["text_annotations"] = len(rows)
        manifest.append(entry)
        print(f"ok {patch_id} labels={entry['labels']} same_grid={entry['same_grid']}")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"wrote {len(manifest)} patches to {args.out}")


if __name__ == "__main__":
    main()
