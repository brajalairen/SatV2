/** Map controls: a compass top-right, zoom bottom-right. Subtle by design — they sit over the map
 *  and should read as part of it rather than as a toolbar. */

import { useEffect, useState } from "react";
import { Minus, Plus } from "lucide-react";
import { useMap } from "./MapView";
import { IconButton, Surface, Tooltip } from "../ui/primitives";

export function CompassControl() {
  const map = useMap();
  const [bearing, setBearing] = useState(0);
  const [pitch, setPitch] = useState(0);

  useEffect(() => {
    if (!map) return;
    const update = () => {
      setBearing(map.getBearing());
      setPitch(map.getPitch());
    };
    update();
    map.on("rotate", update);
    map.on("pitch", update);
    return () => {
      map.off("rotate", update);
      map.off("pitch", update);
    };
  }, [map]);

  const oriented = Math.abs(bearing) < 0.5 && pitch < 0.5;
  const label = oriented ? "Map is north-up" : "Reset orientation to north";

  return (
    <Tooltip label={label} side="left">
      <button
        type="button"
        aria-label={label}
        onClick={() => map?.easeTo({ bearing: 0, pitch: 0, duration: 400 })}
        className="flex h-10 w-10 items-center justify-center rounded-full border border-line bg-surface text-ink shadow-[var(--shadow-md)] transition-colors hover:bg-hover"
      >
        <svg viewBox="0 0 24 24" className="h-5 w-5" style={{ transform: `rotate(${-bearing}deg)` }} aria-hidden="true">
          {/* North half is filled, south half hollow: the usual compass-needle convention. */}
          <path d="M12 3.5 L15.4 13 L12 11.2 Z" fill="currentColor" />
          <path d="M12 20.5 L8.6 11 L12 12.8 Z" fill="currentColor" opacity="0.32" />
          <path d="M12 3.5 L15.4 13 L12 11.2 L8.6 13 Z" fill="none" stroke="currentColor" strokeWidth="0.9" strokeLinejoin="round" />
        </svg>
      </button>
    </Tooltip>
  );
}

export function ZoomControls() {
  const map = useMap();
  return (
    <Surface className="flex flex-col overflow-hidden p-0">
      <IconButton label="Zoom in" side="left" onClick={() => map?.zoomIn({ duration: 200 })} className="rounded-none">
        <Plus className="h-4 w-4" strokeWidth={2} />
      </IconButton>
      <span className="mx-1.5 h-px bg-line" />
      <IconButton label="Zoom out" side="left" onClick={() => map?.zoomOut({ duration: 200 })} className="rounded-none">
        <Minus className="h-4 w-4" strokeWidth={2} />
      </IconButton>
    </Surface>
  );
}
