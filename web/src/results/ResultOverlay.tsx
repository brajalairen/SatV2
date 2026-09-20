/** The result card. It mounts only once an analysis has finished: selecting an area, or adding an
 *  image, shows nothing here. */

import { useState } from "react";
import { AlertTriangle, Ban, Check, TriangleAlert, X } from "lucide-react";
import { useShallow } from "zustand/react/shallow";
import { useAppStore } from "../state/useAppStore";
import { cx, IconButton, Surface } from "../ui/primitives";
import { ResultActions } from "./ResultActions";
import { DetailsDrawer } from "./DetailsDrawer";
import { TemporalAnalysis } from "./TemporalAnalysis";
import type { ResponseStatus } from "../state/types";

const STATUS: Record<ResponseStatus, { icon: typeof Check; tone: string; label: string }> = {
  ok: { icon: Check, tone: "text-ok", label: "Complete" },
  partial: { icon: TriangleAlert, tone: "text-warn", label: "Partial: some steps failed" },
  invalid_input: { icon: Ban, tone: "text-danger", label: "Input rejected" },
  error: { icon: AlertTriangle, tone: "text-danger", label: "Failed" },
};

export type ResultView = "answer" | "temporal";

export function ResultOverlay() {
  const result = useAppStore((s) => s.result);
  const clearResult = useAppStore((s) => s.clearResult);
  const detailsOpen = useAppStore((s) => s.detailsOpen);
  const [view, setView] = useState<ResultView>("answer");

  if (!result) return null;

  const { response } = result;
  const status = STATUS[response.status];
  const StatusIcon = status.icon;

  return (
    <>
      <div className="pointer-events-none absolute top-4 right-4 bottom-28 z-20 flex w-[360px] max-w-[calc(100vw-2rem)] flex-col items-end gap-2 max-sm:top-auto max-sm:right-4 max-sm:left-4 max-sm:bottom-32 max-sm:w-auto">
        {/* The compass sits top-right, so the card starts below it. */}
        <div className="h-11 shrink-0" aria-hidden="true" />

        <Surface raised className="pointer-events-auto flex max-h-full min-h-0 w-full flex-col overflow-hidden">
          <header className="flex shrink-0 items-start gap-2 px-4 pt-3.5 pb-2">
            <StatusIcon className={cx("mt-0.5 h-4 w-4 shrink-0", status.tone)} strokeWidth={2} />
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-semibold tracking-wide text-muted uppercase">
                {taskLabel(response.task) ?? status.label}
              </p>
              {response.status !== "ok" && <p className={cx("text-[11px]", status.tone)}>{status.label}</p>}
            </div>
            <IconButton label="Dismiss result" side="left" onClick={clearResult} className="-mt-1 -mr-1.5 h-7 w-7">
              <X className="h-4 w-4" strokeWidth={2} />
            </IconButton>
          </header>

          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 pb-1">
            {view === "answer" ? (
              <>
                <p className="text-[13px] leading-relaxed text-ink">{response.answer}</p>
                <AreaLine />
                {response.confidence && <ConfidenceLine />}
                <OverlayToggles />
              </>
            ) : (
              <TemporalAnalysis />
            )}
          </div>

          <ResultActions view={view} onViewChange={setView} />
        </Surface>
      </div>

      {detailsOpen && <DetailsDrawer />}
    </>
  );
}

/** States what the drawn area did. A selected area must never be silently ignored, so when the
 *  crop could not be applied the card says so and gives the reason the server reported. */
function AreaLine() {
  const area = useAppStore((s) => s.result?.area);
  if (!area) return null;

  return area.applied ? (
    <p className="mt-2 text-[11px] leading-relaxed text-muted">
      {area.masked
        ? `Restricted to the shape you drew: pixels outside it are excluded from every figure (its box is ${area.width}x${area.height} px of ${area.source_width}x${area.source_height}).`
        : `Restricted to your selected area: ${area.width}x${area.height} px of ${area.source_width}x${area.source_height}.`}
    </p>
  ) : (
    <p className="mt-2 text-[11px] leading-relaxed text-warn">
      Ran on the whole image - {area.reason}.
    </p>
  );
}

/** Confidence is always shown with the method that produced it: these values are uncalibrated. */
function ConfidenceLine() {
  const confidence = useAppStore((s) => s.result?.response.confidence);
  if (!confidence) return null;

  return (
    <div className="mt-3 rounded-[var(--radius-sm)] bg-sunken px-2.5 py-2">
      <p className="flex items-baseline gap-1.5">
        <span className="text-[11px] font-medium tracking-wide text-muted uppercase">Confidence</span>
        <span className="text-[13px] font-semibold text-ink">
          {confidence.value === null ? "n/a" : confidence.value.toFixed(2)}
        </span>
        {!confidence.calibrated && <span className="text-[10px] text-faint">uncalibrated</span>}
      </p>
      <p className="mt-1 text-[11px] leading-relaxed text-faint">{confidence.method}</p>
      {confidence.note && <p className="mt-1 text-[11px] leading-relaxed text-faint">{confidence.note}</p>}
    </div>
  );
}

/** Lets the user turn individual evidence overlays off without losing the result. */
function OverlayToggles() {
  const layers = useAppStore(useShallow((s) => s.result?.overlay_layers ?? []));
  const hidden = useAppStore((s) => s.hiddenOverlays);
  const toggleOverlay = useAppStore((s) => s.toggleOverlay);

  if (!layers.length) return null;

  return (
    <div className="mt-3">
      <p className="mb-1.5 text-[11px] font-medium tracking-wide text-muted uppercase">On the map</p>
      <ul className="space-y-1">
        {layers.map((layer) => (
          <li key={layer.url}>
            <label className="flex cursor-pointer items-start gap-2 text-[11px] leading-snug text-muted">
              <input
                type="checkbox"
                checked={!hidden.has(layer.url)}
                onChange={() => toggleOverlay(layer.url)}
                className="mt-0.5 h-3 w-3 shrink-0 accent-[rgb(var(--accent))]"
              />
              <span>
                {layer.label}
                {layer.approximate && <span className="text-faint"> (placement approximate)</span>}
              </span>
            </label>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function taskLabel(task: string | null): string | null {
  if (!task) return null;
  return {
    vqa: "Answer",
    caption: "Description",
    grounding: "Highlighted region",
    change_analysis: "Change analysis",
    cross_modal_analysis: "Optical + SAR analysis",
  }[task] ?? task;
}
