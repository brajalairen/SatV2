"""Headless entry point (D-013). Examples:

    python -m satquery ask --image scene.tif --modality optical --query "Describe the land cover."
    python -m satquery ask --image t1.tif --modality optical --image t2.tif --modality optical \
        --dates 2019-01-01 2023-01-01 --query "What changed between these two dates?"
"""

import argparse
import sys

from satquery.api import analyze
from satquery.schemas import AnalysisRequest, ImageInput


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="satquery", description="SatQuery AI command-line interface")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="analyse 1-2 images with a natural-language query")
    ask.add_argument("--image", action="append", required=True, help="image path (repeat for a pair)")
    ask.add_argument("--modality", action="append", choices=["optical", "sar"], help="one per --image (default optical)")
    ask.add_argument("--dates", nargs="*", default=[], help="acquisition dates in image order (YYYY-MM-DD)")
    ask.add_argument("--query", required=True)
    ask.add_argument("--task", choices=["vqa", "caption", "grounding", "change_analysis", "cross_modal_analysis"])
    args = parser.parse_args(argv)

    modalities = args.modality or []
    images = [ImageInput(path=path, modality=modalities[i] if i < len(modalities) else "optical",
                         acquired=args.dates[i] if i < len(args.dates) else None) for i, path in enumerate(args.image)]
    response = analyze(AnalysisRequest(query=args.query, images=images, forced_task=args.task))

    print(f"status : {response.status}")
    print(f"task   : {response.task}  ({response.trace.intent.matched_rule if response.trace.intent else '-'})")
    print(f"answer : {response.answer}")
    if response.confidence:
        print(f"confidence: {response.confidence.value} [{response.confidence.method}]")
    for step in response.trace.steps:
        print(f"  {step.step_id} {step.tool:26s} {step.status:7s} {step.duration_s:6.2f}s {step.error or ''}")
    for issue in response.trace.validation:
        print(f"  [{issue.severity}] {issue.code}: {issue.message}")
    print(f"report : {response.report_html}")
    return {"ok": 0, "partial": 0, "invalid_input": 2}.get(response.status, 1)


if __name__ == "__main__":
    sys.exit(main())
