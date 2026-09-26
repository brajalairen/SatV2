/** Basemap styles: what the map looks like, never what is analysed.
 *
 *  Standard is OpenFreeMap (Positron in the light theme, Dark in the dark theme), served without an
 *  API key. Satellite is Esri World Imagery, also keyless; Hybrid lays Esri's roads and place-name
 *  reference layers over it. If a style cannot be fetched, the map falls back to a plain background so
 *  the app still works offline with only the rasters visible.
 *
 *  A basemap is display only. Analysis runs on the imagery the user adds, or on the Sentinel-2 scene
 *  fetched for a drawn area, and switching basemaps changes neither.
 *
 *  There is no Terrain option: no reliable terrain source exists in this build, and a broken option
 *  would be worse than none. */

import type { StyleSpecification } from "maplibre-gl";

export type BasemapId = "standard" | "satellite" | "hybrid";

export const BASEMAPS: { id: BasemapId; label: string }[] = [
  { id: "standard", label: "Standard" },
  { id: "satellite", label: "Satellite" },
  { id: "hybrid", label: "Hybrid" },
];

export const isBasemapId = (value: unknown): value is BasemapId => BASEMAPS.some((basemap) => basemap.id === value);

const OPENFREEMAP = {
  light: "https://tiles.openfreemap.org/styles/positron",
  dark: "https://tiles.openfreemap.org/styles/dark",
} as const;

export const basemapUrl = (theme: "light" | "dark"): string => OPENFREEMAP[theme];

// Esri tile services address tiles as {z}/{y}/{x} (row before column), not the usual {z}/{x}/{y}.
const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services";
const esriTiles = (service: string) => `${ESRI}/${service}/MapServer/tile/{z}/{y}/{x}`;

// Each service's own copyright text, read from its ?f=json metadata on 2026-09-26.
const IMAGERY_ATTRIBUTION = "Imagery: Esri, Vantor, Earthstar Geographics, and the GIS User Community";
const REFERENCE_ATTRIBUTION = "Roads and labels: Esri, HERE, Garmin, © OpenStreetMap contributors, and the GIS User Community";

/** Tiles exist deeper than this in cities, but not everywhere: past z19 Esri can answer with a
 *  "map data not yet available" placeholder. MapLibre over-zooms the z19 tile instead. */
const ESRI_MAX_ZOOM = 19;

function imageryStyle(withReference: boolean): StyleSpecification {
  const raster = (service: string, attribution: string) => ({
    type: "raster" as const,
    tiles: [esriTiles(service)],
    tileSize: 256,
    maxzoom: ESRI_MAX_ZOOM,
    attribution,
  });
  return {
    version: 8,
    sources: {
      imagery: raster("World_Imagery", IMAGERY_ATTRIBUTION),
      ...(withReference
        ? {
            roads: raster("Reference/World_Transportation", REFERENCE_ATTRIBUTION),
            places: raster("Reference/World_Boundaries_and_Places", REFERENCE_ATTRIBUTION),
          }
        : {}),
    },
    layers: [
      // Seen only while tiles load, or where there are none.
      { id: "basemap-background", type: "background", paint: { "background-color": "#1b1d22" } },
      { id: "basemap-imagery", type: "raster", source: "imagery" },
      ...(withReference
        ? [
            { id: "basemap-roads", type: "raster" as const, source: "roads" },
            { id: "basemap-places", type: "raster" as const, source: "places" },
          ]
        : []),
    ],
  };
}

/** The style to load for a basemap. Standard keeps following the theme, exactly as before. */
export function basemapStyle(basemap: BasemapId, theme: "light" | "dark"): string | StyleSpecification {
  if (basemap === "satellite") return imageryStyle(false);
  if (basemap === "hybrid") return imageryStyle(true);
  return basemapUrl(theme);
}

/** Which style a (basemap, theme) pair loads. Imagery does not change with the theme, so switching the
 *  theme over Satellite or Hybrid must not reload the map. */
export const styleKey = (basemap: BasemapId, theme: "light" | "dark"): string =>
  basemap === "standard" ? `standard-${theme}` : basemap;

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
