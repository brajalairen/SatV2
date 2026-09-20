"""The web client's wire types (web/src/state/types.ts) mirror the Python models by hand. This keeps
the two from drifting apart: every interface must list exactly the fields its model sends."""

import re
from pathlib import Path

import pytest

from satquery import schemas, server

TYPES_TS = Path(__file__).resolve().parent.parent / "web" / "src" / "state" / "types.ts"

MIRRORED = {
    "ImageSummary": schemas.ImageSummary,
    "ValidationIssue": schemas.ValidationIssue,
    "Intent": schemas.Intent,
    "PlanStep": schemas.PlanStep,
    "Evidence": schemas.Evidence,
    "Confidence": schemas.Confidence,
    "StepResult": schemas.StepResult,
    "ExecutionTrace": schemas.ExecutionTrace,
    "AnalysisResponse": schemas.AnalysisResponse,
    "OverlayLayer": server.OverlayLayer,
    "AreaScope": server.AreaScope,
    "AnalyzeResult": server.AnalyzeResult,
    "UploadInfo": server.UploadInfo,
    "Example": server.Example,
    "Health": server.Health,
}


def interface_fields(source: str, name: str) -> set[str]:
    block = re.search(rf"export interface {name} \{{\n(.*?)\n\}}", source, re.S)
    assert block, f"interface {name} not found in types.ts"
    # Top-level members are indented two spaces; nested object types stay on one line.
    return set(re.findall(r"^  (\w+)\??:", block.group(1), re.M))


@pytest.mark.parametrize("name", sorted(MIRRORED))
def test_web_types_match_the_python_models(name):
    source = TYPES_TS.read_text(encoding="utf-8")
    assert interface_fields(source, name) == set(MIRRORED[name].model_fields), f"{name} differs between types.ts and Python"
