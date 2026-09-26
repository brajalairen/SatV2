/** The bottom command bar: add something, ask about it, optionally by voice. */

import { useEffect, useRef, useState } from "react";

/** Seconds since this mounted, i.e. since the analysis started. */
function Elapsed() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, []);
  return <span className="tabular-nums">{seconds} s</span>;
}
import { ArrowUp, Plus, SquareDashed } from "lucide-react";
import { PROGRESS_LABELS, nextAnalysisSource, useAppStore } from "../state/useAppStore";
import { api } from "../state/api";
import { Spinner, Surface, cx, IconButton } from "../ui/primitives";
import { UploadMenu } from "./UploadMenu";
import { VoiceInput } from "./VoiceInput";

export function AICommandBar() {
  const [query, setQuery] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const textarea = useRef<HTMLTextAreaElement>(null);

  const layers = useAppStore((s) => s.layers);
  const pending = useAppStore((s) => s.pending);
  const stage = useAppStore((s) => s.stage);
  const result = useAppStore((s) => s.result);
  const error = useAppStore((s) => s.error);
  const aoi = useAppStore((s) => s.aoi);
  const openSection = useAppStore((s) => s.openSection);
  const runAnalysis = useAppStore((s) => s.runAnalysis);
  const cancelAnalysis = useAppStore((s) => s.cancelAnalysis);
  const pendingQuery = useAppStore((s) => s.pendingQuery);
  const setPendingQuery = useAppStore((s) => s.setPendingQuery);

  // The same routing the store uses, so what this bar says matches what the question will run on.
  const source = nextAnalysisSource(layers, aoi);
  const images = source.kind === "images" ? source.images : [];
  const hasImages = images.length > 0;
  // A drawn rectangle with none of the user's own imagery under it retrieves its own.
  const canFetchForArea = source.kind === "fetch";
  const ready = hasImages || canFetchForArea;
  // A circle or polygon with nothing under it: retrieval takes a box, so say so rather than convert.
  const areaShapeUnsupported = source.kind === "none" && Boolean(aoi);

  useEffect(() => {
    api.exampleQueries().then(setSuggestions).catch(() => setSuggestions([]));
  }, []);

  useEffect(() => {
    if (pendingQuery === null) return;
    setQuery(pendingQuery);
    setPendingQuery(null);
    textarea.current?.focus();
  }, [pendingQuery, setPendingQuery]);

  // Grow with the text, up to a few lines, then scroll.
  useEffect(() => {
    const element = textarea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 120)}px`;
  }, [query]);

  const submit = () => {
    const trimmed = query.trim();
    if (!trimmed || pending) return;
    runAnalysis(trimmed);
    setQuery("");
  };

  const showSuggestions = ready && !result && !pending && !query.trim();

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-4 z-30 flex flex-col items-center gap-2 px-4">
      {showSuggestions && (
        <div className="pointer-events-auto flex max-w-[720px] flex-wrap justify-center gap-1.5">
          {suggestions.slice(0, 3).map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => {
                setQuery(suggestion);
                textarea.current?.focus();
              }}
              className="max-w-full truncate rounded-full border border-line bg-surface px-3 py-1.5 text-[12px] text-muted shadow-[var(--shadow-sm)] transition-colors hover:bg-hover hover:text-ink"
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}

      {canFetchForArea && !error && !pending && !result && (
        <Surface className="pointer-events-auto flex max-w-[720px] items-start gap-2.5 px-3 py-2.5">
          <SquareDashed className="mt-0.5 h-4 w-4 shrink-0 text-faint" strokeWidth={1.75} />
          <p className="text-[12px] leading-relaxed text-muted">
            Area selected. Ask a question and Sentinel-2 imagery will be fetched for it from{" "}
            <span className="text-ink">Copernicus Data Space</span>, or add your own GeoTIFF with{" "}
            <strong className="font-medium text-ink">+</strong>.
          </p>
        </Surface>
      )}

      {areaShapeUnsupported && !error && (
        <Surface className="pointer-events-auto flex max-w-[720px] items-start gap-2.5 px-3 py-2.5">
          <SquareDashed className="mt-0.5 h-4 w-4 shrink-0 text-faint" strokeWidth={1.75} />
          <p className="text-[12px] leading-relaxed text-muted">
            Fetching imagery supports rectangles only. Draw a rectangle over this area,{" "}
            <button
              type="button"
              onClick={() => openSection("help")}
              className="font-medium text-accent underline-offset-2 hover:underline"
            >
              try a demo scenario
            </button>{" "}
            or add a GeoTIFF that covers the shape with <strong className="font-medium text-ink">+</strong>.
          </p>
        </Surface>
      )}

      {error && (
        <Surface className="pointer-events-auto max-w-[720px] border-danger/40 px-3 py-2 text-[12px] text-danger">
          {error}
        </Surface>
      )}

      {/* A real model can take a minute: show which step is running, and let it be abandoned.
          Retrieval adds two steps before the analysis, so one generic spinner would hide them. */}
      {pending && (
        <Surface role="status" className="pointer-events-auto flex items-center gap-2.5 py-1.5 pr-1.5 pl-3 text-[12px] text-muted">
          <Spinner className="h-3.5 w-3.5 text-accent" />
          <span>
            {stage ? PROGRESS_LABELS[stage] : "Analysing"} <Elapsed />
          </span>
          <button
            type="button"
            onClick={cancelAnalysis}
            className="rounded-[var(--radius-sm)] px-2 py-1 font-medium text-ink transition-colors hover:bg-hover"
          >
            Cancel
          </button>
        </Surface>
      )}

      <div className="pointer-events-auto relative w-full max-w-[720px]">
        <UploadMenu open={menuOpen} onClose={() => setMenuOpen(false)} />

        <Surface raised className="flex items-end gap-1 p-1.5">
          <IconButton
            label="Add data"
            side="top"
            active={menuOpen}
            onClick={() => setMenuOpen((value) => !value)}
            aria-expanded={menuOpen}
          >
            <Plus className="h-[18px] w-[18px]" strokeWidth={2} />
          </IconButton>

          <textarea
            ref={textarea}
            rows={1}
            value={query}
            disabled={pending}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder={
              hasImages
                ? aoi
                  ? "Ask about the selected area..."
                  : "Ask anything about these images..."
                : canFetchForArea
                  ? "Ask about this area, and imagery will be fetched for it..."
                  : "Add an image or select an area to get started..."
            }
            aria-label="Ask a question about your imagery"
            className="max-h-[120px] flex-1 resize-none self-center bg-transparent px-2 py-2 text-[14px] leading-snug text-ink placeholder:text-faint focus:outline-none disabled:opacity-60"
          />

          <VoiceInput
            disabled={pending}
            onTranscript={(text) => {
              setQuery((current) => (current ? `${current} ${text}` : text));
              textarea.current?.focus();
            }}
          />

          <IconButton
            label={pending ? "Analysing" : "Send"}
            side="top"
            disabled={!query.trim() || pending}
            onClick={submit}
            className={cx(
              "transition-colors",
              query.trim() && !pending && "bg-accent text-accent-ink hover:brightness-110",
            )}
          >
            {pending ? <Spinner /> : <ArrowUp className="h-[18px] w-[18px]" strokeWidth={2.25} />}
          </IconButton>
        </Surface>

        {/* What the next question runs on: never a hidden choice. With no image loaded, the
            selected area fetches its own imagery, so there is no file name to name yet. */}
        {ready && (
          <p className="mt-1.5 truncate text-center text-[11px] text-faint">
            {!hasImages
              ? "Fetching Sentinel-2 imagery for the selected area"
              : `${aoi ? "Analysing the selected area of " : "Analysing "}${
                  images.length === 1 ? images[0]!.name : `${images[0]!.name} and ${images[1]!.name}`
                }`}
          </p>
        )}
      </div>
    </div>
  );
}
