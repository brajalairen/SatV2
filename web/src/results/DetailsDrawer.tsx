/** The evaluator-facing detail, one click behind the result card.
 *
 *  SIH scores the observable execution trace (selected task, tools, permitted parameters, outputs),
 *  so it is all here in full: the routing rule, every step, input validation, the visual evidence
 *  and both report downloads. Hidden by default, never hard to find. */

import { Download, X } from "lucide-react";
import { useAppStore } from "../state/useAppStore";
import { cx, IconButton, Surface } from "../ui/primitives";
import { taskLabel } from "./ResultOverlay";
import type { StepStatus } from "../state/types";

const STATUS_TONE: Record<StepStatus, string> = {
  ok: "text-ok",
  failed: "text-danger",
  skipped: "text-faint",
};

export function DetailsDrawer() {
  const result = useAppStore((s) => s.result);
  const setDetailsOpen = useAppStore((s) => s.setDetailsOpen);
  if (!result) return null;

  const { response } = result;
  const { trace } = response;
  const overlays = response.evidence.filter((item) => item.kind === "overlay" && item.file);

  return (
    <div className="pointer-events-none absolute inset-0 z-40 flex justify-end">
      {/* Dimming the map here would fight the map-first principle, so only the panel is opaque. */}
      <Surface
        raised
        role="dialog"
        aria-label="Analysis details"
        className="pointer-events-auto m-4 flex w-[420px] max-w-[calc(100vw-2rem)] flex-col overflow-hidden"
      >
        <header className="flex shrink-0 items-center justify-between border-b border-line px-4 py-3">
          <div>
            <h2 className="text-[13px] font-semibold text-ink">Execution trace</h2>
            <p className="mt-0.5 text-[11px] text-faint">
              {trace.run_id} - {trace.total_duration_s.toFixed(2)}s
            </p>
          </div>
          <IconButton label="Close details" side="left" onClick={() => setDetailsOpen(false)} className="-mr-1.5 h-7 w-7">
            <X className="h-4 w-4" strokeWidth={2} />
          </IconButton>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
          <Block title="Routing">
            <Pair label="Question" value={trace.query} />
            <Pair label="Input configuration" value={trace.input_config ?? "-"} mono />
            <Pair label="Task" value={taskLabel(response.task) ?? "-"} />
            {trace.intent && <Pair label="Matched rule" value={trace.intent.matched_rule} />}
            {trace.intent?.target && <Pair label="Target" value={trace.intent.target} />}
          </Block>

          <Block title={`Steps (${trace.steps.length})`}>
            <ol className="space-y-2">
              {trace.steps.map((step) => (
                <li key={step.step_id} className="rounded-[var(--radius-sm)] bg-sunken px-2.5 py-2">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-mono text-[11px] font-medium text-ink">{step.tool}</span>
                    <span className={cx("text-[10px] font-semibold uppercase", STATUS_TONE[step.status])}>
                      {step.status}
                    </span>
                  </div>
                  <p className="mt-0.5 text-[10px] text-faint">
                    {step.model ?? "deterministic"} - {step.duration_s.toFixed(2)}s
                  </p>
                  {Object.keys(step.params).length > 0 && (
                    <pre className="mt-1.5 font-mono text-[10px] leading-relaxed break-all whitespace-pre-wrap text-muted">
                      {JSON.stringify(step.params)}
                    </pre>
                  )}
                  {step.error && <p className="mt-1 text-[11px] text-danger">{step.error}</p>}
                </li>
              ))}
            </ol>
          </Block>

          <Block title={`Input checks (${trace.validation.length})`}>
            {trace.validation.length === 0 ? (
              <p className="text-[12px] text-muted">No issues found.</p>
            ) : (
              <ul className="space-y-1.5">
                {trace.validation.map((issue, index) => (
                  <li key={`${issue.code}-${index}`} className="text-[11px] leading-relaxed">
                    <span className={cx("font-semibold", issue.severity === "error" ? "text-danger" : "text-warn")}>
                      {issue.severity}
                    </span>{" "}
                    <span className="font-mono text-muted">{issue.code}</span>
                    <span className="text-muted"> - {issue.message}</span>
                  </li>
                ))}
              </ul>
            )}
          </Block>

          <Block title="Inputs">
            <ul className="space-y-1.5">
              {trace.images.map((image) => (
                <li key={image.index} className="text-[11px] leading-relaxed text-muted">
                  <span className="font-medium text-ink">{image.name}</span> - {image.modality}, {image.width}x
                  {image.height} px, bands {image.bands.join(", ")}, CRS {image.crs ?? "none"}
                  {image.acquired && `, ${image.acquired}`}
                  {image.decimation > 1 && `, read at 1/${image.decimation.toFixed(1)}`}
                </li>
              ))}
            </ul>
          </Block>

          {overlays.length > 0 && (
            <Block title="Visual evidence">
              <div className="space-y-3">
                {overlays.map((item) => (
                  <figure key={item.file}>
                    <img
                      src={item.file!}
                      alt={item.label}
                      className="w-full rounded-[var(--radius-sm)] border border-line"
                    />
                    <figcaption className="mt-1 text-[10px] leading-relaxed text-faint">{item.label}</figcaption>
                  </figure>
                ))}
              </div>
            </Block>
          )}
        </div>

        <footer className="flex shrink-0 gap-2 border-t border-line px-4 py-2.5">
          {response.report_html && <ReportLink href={response.report_html} label="HTML report" />}
          {response.report_json && <ReportLink href={response.report_json} label="JSON trace" />}
        </footer>
      </Surface>
    </div>
  );
}

function ReportLink({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener"
      className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] px-2 py-1 text-[12px] font-medium text-muted transition-colors hover:bg-hover hover:text-ink"
    >
      <Download className="h-3.5 w-3.5" strokeWidth={1.75} />
      {label}
    </a>
  );
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-b border-line px-4 py-3 last:border-b-0">
      <h3 className="mb-2 text-[11px] font-semibold tracking-wide text-muted uppercase">{title}</h3>
      {children}
    </section>
  );
}

function Pair({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <p className="mb-1.5 flex gap-2 text-[11px] leading-relaxed last:mb-0">
      <span className="w-[108px] shrink-0 text-faint">{label}</span>
      <span className={cx("min-w-0 flex-1 text-ink", mono && "font-mono")}>{value}</span>
    </p>
  );
}
