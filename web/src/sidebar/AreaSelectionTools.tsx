/** Area selection. Rectangle is the default tool; the rest are there for people who want them. */

import { Circle, Hexagon, MapPin, Square, Trash2 } from "lucide-react";
import { useState } from "react";
import { useAppStore, type DrawMode } from "../state/useAppStore";
import { useMap, fitBounds } from "../map/MapView";
import { Button, cx, inputClass, Surface } from "../ui/primitives";

const TOOLS: { mode: Exclude<DrawMode, null>; label: string; icon: typeof Square; hint: string }[] = [
  { mode: "rectangle", label: "Rectangle", icon: Square, hint: "Drag a box, or click two opposite corners" },
  { mode: "polygon", label: "Polygon", icon: Hexagon, hint: "Click each corner, then click the first again" },
  { mode: "circle", label: "Circle", icon: Circle, hint: "Drag out from the centre, or click centre then edge" },
  { mode: "point", label: "Point", icon: MapPin, hint: "Click a single location" },
];

export function AreaSelectionTools() {
  const drawMode = useAppStore((s) => s.drawMode);
  const setDrawMode = useAppStore((s) => s.setDrawMode);
  const aoi = useAppStore((s) => s.aoi);
  const setAoi = useAppStore((s) => s.setAoi);
  const saveArea = useAppStore((s) => s.saveArea);
  const map = useMap();
  const [name, setName] = useState("");

  const active = TOOLS.find((tool) => tool.mode === drawMode);

  return (
    <div className="p-4">
      <div className="grid grid-cols-2 gap-2">
        {TOOLS.map((tool) => (
          <button
            key={tool.mode}
            type="button"
            aria-pressed={drawMode === tool.mode}
            onClick={() => setDrawMode(drawMode === tool.mode ? null : tool.mode)}
            className={cx(
              "flex flex-col items-center gap-1.5 rounded-[var(--radius-sm)] border px-2 py-3 transition-colors",
              drawMode === tool.mode
                ? "border-accent bg-accent-soft text-accent"
                : "border-line text-muted hover:border-line-strong hover:bg-hover hover:text-ink",
            )}
          >
            <tool.icon className="h-[18px] w-[18px]" strokeWidth={1.75} />
            <span className="text-[12px] font-medium">{tool.label}</span>
          </button>
        ))}
      </div>

      <p className="mt-3 min-h-[32px] text-[11px] leading-relaxed text-faint">
        {active ? `${active.hint}. Press Escape to cancel.` : "Pick a tool, then draw on the map."}
      </p>

      {aoi && (
        <Surface className="mt-1 p-3 shadow-none">
          <div className="flex items-start justify-between gap-2">
            <div>
              <p className="text-[12px] font-medium text-ink">Area selected</p>
              <p className="mt-0.5 text-[11px] text-faint">{describeBounds(aoi.bounds)}</p>
            </div>
            <Button
              size="sm"
              aria-label="Clear selected area"
              onClick={() => setAoi(null)}
              className="-mt-0.5 -mr-1 text-muted"
            >
              <Trash2 className="h-3.5 w-3.5" strokeWidth={1.75} />
            </Button>
          </div>

          <div className="mt-3 flex gap-1.5">
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Name this area"
              aria-label="Name this area"
              className={cx(inputClass, "flex-1")}
              onKeyDown={(event) => {
                if (event.key !== "Enter") return;
                saveArea(name);
                setName("");
              }}
            />
            <Button
              variant="outline"
              size="sm"
              className="h-auto"
              onClick={() => {
                saveArea(name);
                setName("");
              }}
            >
              Save
            </Button>
          </div>

          <Button
            variant="quiet"
            size="sm"
            className="mt-2 w-full text-muted"
            onClick={() => map && fitBounds(map, aoi.bounds)}
          >
            Zoom to area
          </Button>
        </Surface>
      )}

      <p className="mt-4 border-t border-line pt-3 text-[11px] leading-relaxed text-faint">
        A selected area narrows the analysis to the part of your images inside it. It cannot fetch new
        imagery: this build has no imagery catalogue, so an area over empty map has nothing to analyse.
      </p>
    </div>
  );
}

export function describeBounds([west, south, east, north]: [number, number, number, number]): string {
  const spanKm = (east - west) * 111 * Math.cos(((north + south) / 2) * (Math.PI / 180));
  const heightKm = (north - south) * 111;
  const size = spanKm < 1 || heightKm < 1
    ? `${Math.round(spanKm * 1000)} x ${Math.round(heightKm * 1000)} m`
    : `${spanKm.toFixed(1)} x ${heightKm.toFixed(1)} km`;
  return `${size} - centred ${((north + south) / 2).toFixed(3)}, ${((east + west) / 2).toFixed(3)}`;
}
