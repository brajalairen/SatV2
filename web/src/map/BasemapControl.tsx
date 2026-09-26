/** Basemap picker, above the zoom buttons. It changes how the map looks and nothing else: the drawn
 *  area, the layers, the question and any result live in the store, and MapView re-adds every overlay
 *  once the new style has loaded. Analysis never sees the basemap. */

import { useCallback, useState } from "react";
import { MapIcon } from "lucide-react";
import { BASEMAPS } from "./basemap";
import { useAppStore } from "../state/useAppStore";
import { cx, IconButton, MenuItem, Popover, Surface } from "../ui/primitives";

export function BasemapControl() {
  const basemap = useAppStore((s) => s.basemap);
  const setBasemap = useAppStore((s) => s.setBasemap);
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);

  return (
    <div className="relative">
      <Surface className="overflow-hidden p-0">
        <IconButton
          label="Basemap"
          side="left"
          active={open}
          showTooltip={!open}
          aria-haspopup="menu"
          aria-expanded={open}
          // The menu closes on any press outside itself, and this button is outside it. While the menu
          // is open, keeping the press from reaching the document stops it closing a moment before
          // this click would toggle it straight back open.
          onPointerDown={(event) => open && event.stopPropagation()}
          onClick={() => setOpen((value) => !value)}
          className="rounded-none"
        >
          <MapIcon className="h-4 w-4" strokeWidth={2} />
        </IconButton>
      </Surface>

      {/* Opens to the left and downward, beside the zoom buttons. The result card sits above in this
          same column and grows downward with its content, so a menu opening upward would meet a long
          card; opening downward, it can only reach the card if the card already reaches this button. */}
      <Popover open={open} onClose={close} className="top-0 right-full mr-2 w-52">
        <div role="menu" aria-label="Basemap">
          <p className="px-3 pt-1.5 pb-1 text-[11px] font-semibold tracking-wide text-muted uppercase">Basemap</p>
          {BASEMAPS.map((option) => {
            const selected = option.id === basemap;
            return (
              <MenuItem
                key={option.id}
                checked={selected}
                icon={<RadioMark selected={selected} />}
                onClick={() => {
                  setBasemap(option.id);
                  close();
                }}
              >
                {option.label}
              </MenuItem>
            );
          })}
          <p className="mt-1 border-t border-line px-3 pt-2 pb-1 text-[10px] leading-relaxed text-faint">
            Display only. Analysis uses the imagery you add, or Sentinel-2 fetched for your area.
          </p>
        </div>
      </Popover>
    </div>
  );
}

function RadioMark({ selected }: { selected: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={cx(
        "flex h-3.5 w-3.5 items-center justify-center rounded-full border",
        selected ? "border-accent" : "border-line",
      )}
    >
      {selected && <span className="h-1.5 w-1.5 rounded-full bg-accent" />}
    </span>
  );
}
