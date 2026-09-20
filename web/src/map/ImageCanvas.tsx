/** Viewer for imagery that cannot go on a map.
 *
 *  SIH requires plain TIFF without a CRS, and PNG/JPEG for the benchmark sets. None of those can be
 *  placed geographically, so rather than guessing a location they open here: a floating panel over
 *  the map, with the same command bar and the same result card. */

import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, ImageOff, Minus, Plus, X } from "lucide-react";
import { useAppStore } from "../state/useAppStore";
import { Button, IconButton, Surface } from "../ui/primitives";

export function ImageCanvas() {
  const layers = useAppStore((s) => s.layers);
  const removeLayer = useAppStore((s) => s.removeLayer);
  const [index, setIndex] = useState(0);
  const [zoom, setZoom] = useState(1);

  const offMap = layers.filter((layer) => !layer.mappable);
  const newest = offMap[offMap.length - 1]?.id ?? null;
  // Hiding the viewer hides it for the images already there. A newly added one opens it again, on
  // itself, rather than arriving unseen.
  const [dismissedAt, setDismissedAt] = useState<string | null>(null);
  useEffect(() => {
    if (newest) setIndex(offMap.length - 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [newest]);

  if (!offMap.length || dismissedAt === newest) return null;

  const current = offMap[Math.min(index, offMap.length - 1)]!;

  return (
    <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center p-4">
      <Surface
        raised
        className="pointer-events-auto flex max-h-[min(70vh,560px)] w-[min(520px,100%)] flex-col overflow-hidden"
      >
        <header className="flex shrink-0 items-center gap-2 border-b border-line px-3 py-2">
          <ImageOff className="h-3.5 w-3.5 shrink-0 text-faint" strokeWidth={1.75} />
          <div className="min-w-0 flex-1">
            <p className="truncate text-[12px] font-medium text-ink">{current.name}</p>
            <p className="text-[10px] text-faint">
              No georeferencing - shown as an image. It analyses normally.
            </p>
          </div>
          <IconButton
            label="Hide viewer"
            side="left"
            onClick={() => setDismissedAt(newest)}
            className="h-7 w-7"
          >
            <X className="h-4 w-4" strokeWidth={2} />
          </IconButton>
        </header>

        <div className="min-h-0 flex-1 overflow-auto bg-sunken p-3">
          <img
            src={current.preview_url}
            alt={`${current.name}, rendered as analysed`}
            style={{ width: `${zoom * 100}%` }}
            className="mx-auto block rounded-[var(--radius-sm)] border border-line"
          />
        </div>

        <footer className="flex shrink-0 items-center gap-1 border-t border-line px-2 py-1.5">
          {offMap.length > 1 && (
            <>
              <IconButton
                label="Previous image"
                side="top"
                disabled={index === 0}
                onClick={() => setIndex((value) => value - 1)}
                className="h-7 w-7"
              >
                <ChevronLeft className="h-4 w-4" />
              </IconButton>
              <span className="text-[11px] text-faint">
                {index + 1} / {offMap.length}
              </span>
              <IconButton
                label="Next image"
                side="top"
                disabled={index >= offMap.length - 1}
                onClick={() => setIndex((value) => value + 1)}
                className="h-7 w-7"
              >
                <ChevronRight className="h-4 w-4" />
              </IconButton>
            </>
          )}

          <div className="ml-auto flex items-center gap-1">
            <IconButton
              label="Zoom out"
              side="top"
              disabled={zoom <= 0.5}
              onClick={() => setZoom((value) => Math.max(0.5, value - 0.25))}
              className="h-7 w-7"
            >
              <Minus className="h-3.5 w-3.5" />
            </IconButton>
            <span className="w-9 text-center text-[11px] text-faint">{Math.round(zoom * 100)}%</span>
            <IconButton
              label="Zoom in"
              side="top"
              disabled={zoom >= 4}
              onClick={() => setZoom((value) => Math.min(4, value + 0.25))}
              className="h-7 w-7"
            >
              <Plus className="h-3.5 w-3.5" />
            </IconButton>
            <Button size="sm" className="text-muted" onClick={() => removeLayer(current.id)}>
              Remove
            </Button>
          </div>
        </footer>
      </Surface>
    </div>
  );
}
