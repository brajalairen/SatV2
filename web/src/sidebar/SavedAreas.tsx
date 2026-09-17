/** Saved areas, kept in this browser's local storage. Nothing is sent anywhere. */

import { Bookmark, Trash2 } from "lucide-react";
import { useAppStore } from "../state/useAppStore";
import { useMap, fitBounds } from "../map/MapView";
import { Button } from "../ui/primitives";
import { EmptyState } from "./Sidebar";
import { describeBounds } from "./AreaSelectionTools";

export function SavedAreas() {
  const savedAreas = useAppStore((s) => s.savedAreas);
  const removeSavedArea = useAppStore((s) => s.removeSavedArea);
  const setAoi = useAppStore((s) => s.setAoi);
  const map = useMap();

  if (!savedAreas.length) {
    return (
      <EmptyState icon={Bookmark}>
        No saved areas yet. Draw an area under <strong className="font-medium text-ink">Select area</strong> and give it
        a name to keep it here.
      </EmptyState>
    );
  }

  return (
    <div>
      <ul>
        {savedAreas.map((area) => (
          <li key={area.id} className="flex items-start gap-2 border-b border-line px-4 py-3 last:border-b-0">
            <button
              type="button"
              className="min-w-0 flex-1 text-left"
              onClick={() => {
                setAoi({ feature: area.geometry, bounds: area.bounds });
                if (map) fitBounds(map, area.bounds);
              }}
            >
              <span className="block truncate text-[12px] font-medium text-ink">{area.name}</span>
              <span className="mt-0.5 block text-[11px] text-faint">{describeBounds(area.bounds)}</span>
            </button>
            <Button
              size="sm"
              aria-label={`Delete ${area.name}`}
              onClick={() => removeSavedArea(area.id)}
              className="w-7 shrink-0 px-0 text-muted"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </li>
        ))}
      </ul>
      <p className="px-4 py-3 text-[11px] leading-relaxed text-faint">
        Saved in this browser only. Clearing site data removes them.
      </p>
    </div>
  );
}
