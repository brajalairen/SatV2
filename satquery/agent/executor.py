"""Runs plan steps through the tool registry, enforcing permitted parameters and recording every step."""

import json
import time

from satquery.schemas import PlanStep, StepResult
from satquery.specialists.tools import REGISTRY, ToolContext


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def execute(plan: list[PlanStep], ctx: ToolContext) -> list[StepResult]:
    results = []
    for step in plan:
        started = time.perf_counter()
        tool = REGISTRY.get(step.tool)
        try:
            if tool is None:
                raise KeyError(f"tool '{step.tool}' is not in the registry")
            params = tool.params_model(**step.params)  # rejects parameters the tool does not permit
            out = tool.run(ctx, step.image_indices, params, step.step_id)
            ctx.artifacts[step.step_id] = out
            results.append(StepResult(
                step_id=step.step_id, tool=step.tool, model=out.model,
                status="skipped" if out.skipped_reason else "ok", params=params.model_dump(),
                outputs=_jsonable(out.outputs), evidence=out.evidence, confidence=out.confidence,
                error=out.skipped_reason, duration_s=round(time.perf_counter() - started, 3)))
        except Exception as error:  # one failing step must not hide the others
            results.append(StepResult(step_id=step.step_id, tool=step.tool, status="failed", params=step.params,
                                      error=f"{type(error).__name__}: {error}",
                                      duration_s=round(time.perf_counter() - started, 3)))
    return results
