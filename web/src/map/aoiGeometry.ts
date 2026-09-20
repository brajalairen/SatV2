/** Geometry of a drawn area, kept apart from the map so it can be reasoned about (and tested)
 *  without a live MapLibre instance. `AoiLayer` holds the drawing itself. */

import type { TerraDrawMouseEvent } from "terra-draw";
import type { Aoi, DrawMode } from "../state/useAppStore";

/** Where the pointer went down, in the form Terra Draw's events use. */
export type Press = Pick<TerraDrawMouseEvent, "lng" | "lat" | "containerX" | "containerY">;

/** The tools offered, and the Terra Draw mode each one runs. All enclose an area (D-025). */
export const MODE_NAMES: Record<Exclude<DrawMode, null>, string> = {
  rectangle: "rectangle",
  polygon: "polygon",
  circle: "circle",
};

/** Bounding box of any drawn geometry, used to frame the camera and to save the area. */
export function featureBounds(feature: GeoJSON.Feature): [number, number, number, number] {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;

  const visit = (coordinates: unknown): void => {
    if (!Array.isArray(coordinates)) return;
    if (typeof coordinates[0] === "number" && typeof coordinates[1] === "number") {
      const [lon, lat] = coordinates as [number, number];
      west = Math.min(west, lon);
      east = Math.max(east, lon);
      south = Math.min(south, lat);
      north = Math.max(north, lat);
      return;
    }
    coordinates.forEach(visit);
  };

  if (feature.geometry && "coordinates" in feature.geometry) visit(feature.geometry.coordinates);
  return [west, south, east, north];
}

/** The area a finished shape describes, or null when it encloses none and should be discarded. */
export function drawnArea(feature: GeoJSON.Feature): Aoi | null {
  const bounds = featureBounds(feature);
  if (!bounds.every(Number.isFinite)) return null;
  const [west, south, east, north] = bounds;
  if (west === east || south === north) return null; // a drag that ended where it began
  return { feature, bounds };
}

/** Terra Draw's event with the position the pointer actually went down at.
 *
 *  Terra Draw starts a dragged shape at the first pointer move past its drag threshold, which after
 *  a quick flick or a touch can be well away from where the user pressed. */
export function anchoredEvent(event: TerraDrawMouseEvent, press: Press | null): TerraDrawMouseEvent {
  return press ? { ...event, ...press } : event;
}
