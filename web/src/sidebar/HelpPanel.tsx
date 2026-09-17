/** Help. Doubles as the quickest way in: the demo scenarios load real georeferenced imagery that
 *  ships with the repository, so a first-time user can see a full result without finding a file. */

import { useEffect, useState } from "react";
import { Play } from "lucide-react";
import { api, ApiError } from "../state/api";
import { useAppStore } from "../state/useAppStore";
import { useMap, fitBounds } from "../map/MapView";
import { Spinner, useToast } from "../ui/primitives";
import type { Example } from "../state/types";

export function HelpPanel() {
  const [examples, setExamples] = useState<Example[]>([]);
  const [loading, setLoading] = useState<number | null>(null);
  const addLayers = useAppStore((s) => s.addLayers);
  const modelIsFake = useAppStore((s) => s.modelIsFake);
  const closeSidebar = useAppStore((s) => s.closeSidebar);
  const setPendingQuery = useAppStore((s) => s.setPendingQuery);
  const map = useMap();
  const toast = useToast();

  useEffect(() => {
    api.examples().then(setExamples).catch(() => setExamples([]));
  }, []);

  const runExample = async (example: Example) => {
    setLoading(example.index);
    try {
      const uploads = await api.loadExample(example.index);
      addLayers(uploads);
      const bounds = uploads.find((u) => u.summary.bounds_wgs84)?.summary.bounds_wgs84;
      if (map && bounds) fitBounds(map, bounds);
      // Hand the map back to the user and load the question, ready to send.
      setPendingQuery(example.query);
      closeSidebar();
      toast(`${example.label} loaded. Press send to run it.`);
    } catch (error) {
      toast(error instanceof ApiError ? error.message : "Could not load that scenario.");
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="p-4 text-[12px] leading-relaxed text-muted">
      <Step n={1} title="Add an image">
        Use <strong className="font-medium text-ink">+</strong> in the command bar to add a GeoTIFF. Georeferenced files
        appear in place on the map; others open in a viewer instead.
      </Step>
      <Step n={2} title="Ask a question">
        Type in plain language, for example <em>&ldquo;Highlight the water body in this image.&rdquo;</em> The agent
        picks the tools, runs them and shows what it did.
      </Step>
      <Step n={3} title="Read the result">
        A card appears with the answer and a confidence value. <strong className="font-medium text-ink">Details</strong>{" "}
        opens the full execution trace and the downloadable report.
      </Step>

      {examples.length > 0 && (
        <>
          <h3 className="mt-5 mb-2 text-[11px] font-semibold tracking-wide text-ink uppercase">Try a demo scenario</h3>
          <ul className="-mx-1">
            {examples.map((example) => (
              <li key={example.index}>
                <button
                  type="button"
                  disabled={loading !== null}
                  onClick={() => runExample(example)}
                  className="flex w-full items-start gap-2 rounded-[var(--radius-sm)] px-3 py-2 text-left transition-colors hover:bg-hover disabled:opacity-50"
                >
                  {loading === example.index ? (
                    <Spinner className="mt-0.5 shrink-0 text-accent" />
                  ) : (
                    <Play className="mt-0.5 h-3.5 w-3.5 shrink-0 text-faint" strokeWidth={1.75} />
                  )}
                  <span>
                    <span className="block text-[12px] font-medium text-ink">{example.label}</span>
                    <span className="mt-0.5 block text-[11px] text-faint">{example.query}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </>
      )}

      <div className="mt-5 border-t border-line pt-3 text-[11px] leading-relaxed text-faint">
        <p>
          <strong className="font-medium text-muted">What this build does not do.</strong> It does not search or
          download satellite imagery. There is no imagery catalogue, so filtering by source, resolution or cloud cover
          is switched off. Analysis runs on the files you provide.
        </p>
        <p className="mt-2">
          Confidence values are uncalibrated and every one states the method that produced it. Several tools are
          heuristic and are labelled as such in the trace.
        </p>
        {modelIsFake && (
          <p className="mt-2 text-warn">
            This server is running the labelled stand-in model, not the real one. Answers are placeholders.
          </p>
        )}
      </div>
    </div>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="mb-3.5 flex gap-2.5">
      <span className="mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-sunken text-[11px] font-semibold text-muted">
        {n}
      </span>
      <div>
        <p className="text-[12px] font-medium text-ink">{title}</p>
        <p className="mt-0.5">{children}</p>
      </div>
    </div>
  );
}
