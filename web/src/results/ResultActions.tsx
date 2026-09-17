/** Actions on a result. Each one is enabled only when it can genuinely do something; where it
 *  cannot, the tooltip says why rather than the button silently doing nothing. */

import { Clock, Crosshair, Download, FileText } from "lucide-react";
import { useAppStore, temporalReadiness } from "../state/useAppStore";
import { useMap, fitBounds } from "../map/MapView";
import { Button, Tooltip } from "../ui/primitives";
import type { ResultView } from "./ResultOverlay";

export function ResultActions({ view, onViewChange }: { view: ResultView; onViewChange: (view: ResultView) => void }) {
  const result = useAppStore((s) => s.result);
  const layers = useAppStore((s) => s.layers);
  const detailsOpen = useAppStore((s) => s.detailsOpen);
  const setDetailsOpen = useAppStore((s) => s.setDetailsOpen);
  const map = useMap();

  if (!result) return null;

  const { response } = result;
  const temporal = temporalReadiness(layers);
  // A change task has already run the bi-temporal comparison, so its numbers are there to show.
  const temporalAvailable = response.task === "change_analysis" && temporal.ready;
  const placed = result.overlay_layers[0];
  const footprint = response.trace.images.find((image) => image.bounds_wgs84)?.bounds_wgs84;

  return (
    <footer className="shrink-0 border-t border-line px-2 py-2">
      <div className="flex flex-wrap items-center gap-1">
        <Tooltip label={footprint ? "Frame the analysed area" : "No georeferenced image to zoom to"} side="top">
          <Button
            size="sm"
            disabled={!footprint}
            onClick={() => map && footprint && fitBounds(map, footprint)}
            className="text-muted"
          >
            <Crosshair className="h-3.5 w-3.5" strokeWidth={1.75} />
            View on map
          </Button>
        </Tooltip>

        <Button size="sm" onClick={() => setDetailsOpen(!detailsOpen)} className="text-muted" aria-pressed={detailsOpen}>
          <FileText className="h-3.5 w-3.5" strokeWidth={1.75} />
          Details
        </Button>

        <Tooltip
          label={
            temporalAvailable
              ? "Show the before/after comparison"
              : response.task !== "change_analysis"
                ? "Ask a change question across two dates to compare"
                : (temporal.reason ?? "Not available")
          }
          side="top"
        >
          <Button
            size="sm"
            disabled={!temporalAvailable}
            aria-pressed={view === "temporal"}
            onClick={() => onViewChange(view === "temporal" ? "answer" : "temporal")}
            className="text-muted"
          >
            <Clock className="h-3.5 w-3.5" strokeWidth={1.75} />
            Temporal change
          </Button>
        </Tooltip>

        {response.report_html && (
          <Button
            size="sm"
            className="ml-auto text-muted"
            onClick={() => window.open(response.report_html!, "_blank", "noopener")}
          >
            <Download className="h-3.5 w-3.5" strokeWidth={1.75} />
            Report
          </Button>
        )}
      </div>

      {!placed && result.response.evidence.some((e) => e.kind === "overlay") && (
        <p className="px-1 pt-1.5 text-[10px] leading-relaxed text-faint">
          Evidence images are in Details. They are not drawn on the map because they do not match a single
          georeferenced grid.
        </p>
      )}
    </footer>
  );
}
