/** Basemap styles. OpenFreeMap serves Positron without an API key; if it is unreachable the map
 *  falls back to a plain background so the app still works offline with only the rasters visible. */

import type { StyleSpecification } from "maplibre-gl";

const OPENFREEMAP = {
  light: "https://tiles.openfreemap.org/styles/positron",
  dark: "https://tiles.openfreemap.org/styles/dark",
} as const;

export const basemapUrl = (theme: "light" | "dark"): string => OPENFREEMAP[theme];

/** Used when the tile service cannot be reached: no imagery, just a neutral canvas. */
export const offlineStyle = (theme: "light" | "dark"): StyleSpecification => ({
  version: 8,
  sources: {},
  layers: [
    {
      id: "background",
      type: "background",
      paint: { "background-color": theme === "dark" ? "#1b1d22" : "#eef0f3" },
    },
  ],
});

export const INITIAL_VIEW = { center: [20, 30] as [number, number], zoom: 1.6 };
