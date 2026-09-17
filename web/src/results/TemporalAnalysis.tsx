/** Before/after view for a change analysis. Every number here is read out of the execution trace:
 *  nothing is interpolated, and when the pipeline reported no figure the row is simply absent. */

import { useAppStore } from "../state/useAppStore";
import type { StepResult } from "../state/types";

function findStep(steps: StepResult[], tool: string): StepResult | undefined {
  return steps.find((step) => step.tool === tool && step.status === "ok");
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function TemporalAnalysis() {
  const response = useAppStore((s) => s.result?.response);
  if (!response) return null;

  const [before, after] = response.trace.images;
  const steps = response.trace.steps;
  const compare = findStep(steps, "change.compare_areas");
  const changed = findStep(steps, "vlm.change") ?? findStep(steps, "change.map");
  const changedFraction = asNumber(changed?.outputs.fraction);

  const beforePercent = asNumber(compare?.outputs.before_percent);
  const afterPercent = asNumber(compare?.outputs.after_percent);
  const delta = asNumber(compare?.outputs.change_percentage_points);
  const target = typeof compare?.outputs.target === "string" ? compare.outputs.target : null;
  const measure = typeof compare?.outputs.measure === "string" ? compare.outputs.measure : null;
  const verdict = typeof compare?.outputs.verdict === "string" ? compare.outputs.verdict : null;

  return (
    <div className="pb-1">
      <div className="flex items-center justify-between gap-2 text-[11px] text-muted">
        <span>{before?.acquired ?? "before"}</span>
        <span className="h-px flex-1 bg-line" aria-hidden="true" />
        <span>{after?.acquired ?? "after"}</span>
      </div>

      {changedFraction !== null && (
        <div className="mt-3">
          <div className="flex items-baseline justify-between">
            <span className="text-[11px] font-medium tracking-wide text-muted uppercase">Scene changed</span>
            <span className="text-[15px] font-semibold text-ink">{(changedFraction * 100).toFixed(1)}%</span>
          </div>
          <Bar value={changedFraction} />
        </div>
      )}

      {verdict === "inconclusive" && (
        <p className="mt-3 rounded-[var(--radius-sm)] bg-sunken px-2.5 py-2 text-[11px] leading-relaxed text-muted">
          {target ? `No ${target} was detected in either image, ` : "Nothing was detected in either image, "}
          so the direction of change could not be determined. It may not be resolvable at this resolution.
        </p>
      )}

      {beforePercent !== null && afterPercent !== null && verdict !== "inconclusive" && (
        <div className="mt-4">
          <p className="mb-2 text-[11px] font-medium tracking-wide text-muted uppercase">
            {target ? `${target} coverage` : "Coverage"}
          </p>
          <Row label={before?.acquired ?? "Before"} percent={beforePercent} />
          <Row label={after?.acquired ?? "After"} percent={afterPercent} />
          {delta !== null && (
            <p className="mt-2 text-[12px] text-ink">
              <span className={delta > 0 ? "text-ok" : delta < 0 ? "text-danger" : undefined}>
                {delta > 0 ? "+" : ""}
                {delta.toFixed(1)} percentage points
              </span>
              {verdict && <span className="text-muted"> ({verdict})</span>}
            </p>
          )}
          {measure && <p className="mt-1 text-[11px] leading-relaxed text-faint">Measured as {measure}.</p>}
        </div>
      )}

      {changedFraction === null && beforePercent === null && (
        <p className="mt-3 text-[12px] leading-relaxed text-muted">
          This run did not produce comparable area figures. The written answer and the execution trace under Details
          hold everything the agent reported.
        </p>
      )}

      <p className="mt-4 border-t border-line pt-2.5 text-[10px] leading-relaxed text-faint">
        Figures come from the tools listed in the trace. Several use fixed thresholds and are heuristic.
      </p>
    </div>
  );
}

function Row({ label, percent }: { label: string; percent: number }) {
  return (
    <div className="mb-2 last:mb-0">
      <div className="flex items-baseline justify-between text-[12px]">
        <span className="text-muted">{label}</span>
        <span className="font-medium text-ink">{percent.toFixed(1)}%</span>
      </div>
      <Bar value={percent / 100} />
    </div>
  );
}

function Bar({ value }: { value: number }) {
  const width = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-sunken">
      <div className="h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${width}%` }} />
    </div>
  );
}
