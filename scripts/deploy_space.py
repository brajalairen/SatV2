"""Publish the web app to a Hugging Face ZeroGPU Space (D-020).

Stages only what the app needs (app.py, satquery/, requirements.txt, demo/examples/, and a Space README),
then uploads it with huggingface_hub, which stores binary files (PNG/TIFF) via Xet/LFS as the Hub requires.

One-time setup:
  1. `huggingface-cli login` with a WRITE token from an account that is 30+ days old with a verified email.
  2. Run this script; it creates the Space if it does not exist.
  3. In the Space settings, set Hardware to "ZeroGPU" (free accounts may host up to 2 ZeroGPU Spaces).
Usage:
  python scripts/deploy_space.py --space-id <username>/satquery-ai
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "build" / "space"
SPACE_README = """---
title: SatQuery AI
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
license: apache-2.0
short_description: Agentic vision-language assistant for satellite imagery (SIH26167)
---

# SatQuery AI: SIH26167 prototype

Ask natural-language questions about optical, SAR, and bi-temporal satellite images. The agent validates the inputs,
plans which specialist tools to run, executes them, and returns an answer with visual evidence, a confidence
estimate (with its method), a full execution trace, and a downloadable report.

Round 1 prototype. Heuristic components and confidences are labelled as such in the app.
"""


def stage() -> Path:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    shutil.copy2(ROOT / "app.py", STAGING / "app.py")
    shutil.copy2(ROOT / "requirements.txt", STAGING / "requirements.txt")
    shutil.copytree(ROOT / "satquery", STAGING / "satquery", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    examples = ROOT / "demo" / "examples"
    if examples.exists():
        shutil.copytree(examples, STAGING / "demo" / "examples")
    (STAGING / "README.md").write_text(SPACE_README, encoding="utf-8")
    return STAGING


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space-id", required=True, help="e.g. username/satquery-ai")
    parser.add_argument("--dry-run", action="store_true", help="stage files only, do not upload")
    args = parser.parse_args()

    folder = stage()
    files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    print(f"staged {len(files)} files in {folder}")
    if args.dry_run:
        print("\n".join(files))
        return
    api = HfApi()
    api.create_repo(args.space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(repo_id=args.space_id, repo_type="space", folder_path=str(folder),
                      commit_message="Deploy SatQuery AI", delete_patterns=["satquery/**", "demo/**"])
    print(f"uploaded: https://huggingface.co/spaces/{args.space_id}")
    print("If this is the first deploy: open Settings -> Hardware -> select ZeroGPU.")


if __name__ == "__main__":
    main()
