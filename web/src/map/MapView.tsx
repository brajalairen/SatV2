/** The map. It owns the MapLibre instance and reconciles three kinds of layer against the store:
 *  the uploaded rasters, the evidence overlays from a result, and the drawn area of interest. */

import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { Map as MapLibreMap, type GeoJSONSource, type ErrorEvent } from "maplibre-gl";
import { basemapUrl, INITIAL_VIEW, offlineStyle } from "./basemap";
import { useShallow } from "zustand/react/shallow";
import { useAppStore, visibleOverlays, type Layer } from "../state/useAppStore";
import type { OverlayLayer } from "../state/types";

interface MapHandle {
  map: MapLibreMap | null;
  /** MapLibre rejects addSource/addLayer until the style has loaded, so anything that touches
   *  sources (Terra Draw, the layer effects) must wait for this. Plain UI does not. */
  styleReady: boolean;
}

const MapContext = createContext<MapHandle>({ map: null, styleReady: false });
export const useMap = () => useContext(MapContext).map;
export const useMapReady = () => useContext(MapContext).styleReady;

/** Corner order the API sends (TL, TR, BR, BL) is exactly what an image source expects.
 *  The server always emits four, which the tuple type states for MapLibre. */
type Corners = [[number, number], [number, number], [number, number], [number, number]];

const rasterSourceId = (id: string) => `raster-${id}`;
const rasterLayerId = (id: string) => `raster-layer-${id}`;
const overlaySourceId = (url: string) => `overlay-${hash(url)}`;
const overlayLayerId = (url: string) => `overlay-layer-${hash(url)}`;

function hash(value: string): string {
  let result = 0;
  for (let i = 0; i < value.length; i += 1) result = (result * 31 + value.charCodeAt(i)) | 0;
  return Math.abs(result).toString(36);
}

export function MapView({ children }: { children?: ReactNode }) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const [map, setMap] = useState<MapLibreMap | null>(null);
  const [styleReady, setStyleReady] = useState(false);

  const theme = useAppStore((s) => s.theme);
  const layers = useAppStore((s) => s.layers);
  // useShallow: visibleOverlays builds a new array each call, which plain reference
  // equality would treat as a change on every render.
  const overlays = useAppStore(useShallow(visibleOverlays));
  const aoi = useAppStore((s) => s.aoi);

  // --- create the map once
  useEffect(() => {
    if (!container.current) return;
    const instance = new MapLibreMap({
      container: container.current,
      style: basemapUrl(theme),
      center: INITIAL_VIEW.center,
      zoom: INITIAL_VIEW.zoom,
      attributionControl: { compact: true },
      renderWorldCopies: false, // stops a raster being drawn twice across the antimeridian
    });
    instance.on("style.load", () => setStyleReady(true));
    instance.on("error", (event: ErrorEvent) => {
      // A failed tile fetch must not take the app down: fall back to a plain canvas.
      if (String(event.error?.message ?? "").includes("style")) instance.setStyle(offlineStyle(theme));
    });
    mapRef.current = instance;
    setMap(instance);
    return () => {
      instance.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- swap the basemap with the theme, then re-add everything the style dropped
  useEffect(() => {
    if (!map) return;
    setStyleReady(false);
    map.setStyle(basemapUrl(theme));
  }, [map, theme]);

  // --- uploaded rasters
  useEffect(() => {
    if (!map || !styleReady) return;
    syncRasters(map, layers);
  }, [map, styleReady, layers]);

  // --- evidence overlays, always drawn above the rasters
  useEffect(() => {
    if (!map || !styleReady) return;
    syncOverlays(map, overlays);
  }, [map, styleReady, overlays]);

  // --- area of interest, kept above the rasters and overlays so the selection stays visible
  useEffect(() => {
    if (!map || !styleReady) return;
    syncAoi(map, aoi?.feature ?? null);
  }, [map, styleReady, aoi, layers, overlays]);

  return (
    <MapContext.Provider value={{ map, styleReady }}>
      {/* Sized directly, not via `absolute inset-0`: MapLibre's unlayered CSS forces
          position:relative on this element and would collapse it to zero height. */}
      <div ref={container} className="h-full w-full" aria-label="Map" role="application" />
      {/* The interface does not wait for the basemap: if tiles are slow or unreachable, the
          controls still work and the layer effects above simply run once the style is ready. */}
      {map ? children : null}
    </MapContext.Provider>
  );
}

/* ------------------------------------------------------------------ reconciliation */

function syncRasters(map: MapLibreMap, layers: Layer[]) {
  const wanted = new Set(layers.filter((l) => l.mappable).map((l) => rasterSourceId(l.id)));
  removeStale(map, "raster-", wanted);

  layers
    .filter((layer) => layer.mappable && layer.summary.corners_wgs84)
    .forEach((layer) => {
      const sourceId = rasterSourceId(layer.id);
      const layerId = rasterLayerId(layer.id);
      const coordinates = layer.summary.corners_wgs84 as unknown as Corners;

      if (!map.getSource(sourceId)) {
        map.addSource(sourceId, { type: "image", url: layer.preview_url, coordinates });
        map.addLayer({ id: layerId, type: "raster", source: sourceId, paint: { "raster-fade-duration": 200 } });
      }
      map.setPaintProperty(layerId, "raster-opacity", layer.visible ? layer.opacity : 0);
      // Store order is draw order: moving a layer to the top each pass keeps them stacked correctly.
      map.moveLayer(layerId);
    });
}

function syncOverlays(map: MapLibreMap, overlays: OverlayLayer[]) {
  const wanted = new Set(overlays.map((o) => overlaySourceId(o.url)));
  removeStale(map, "overlay-", wanted);

  overlays.forEach((overlay) => {
    const sourceId = overlaySourceId(overlay.url);
    const layerId = overlayLayerId(overlay.url);
    if (!map.getSource(sourceId)) {
      map.addSource(sourceId, { type: "image", url: overlay.url, coordinates: overlay.corners_wgs84 as unknown as Corners });
      map.addLayer({ id: layerId, type: "raster", source: sourceId, paint: { "raster-opacity": 0.95 } });
    }
    map.moveLayer(layerId);
  });
}

const AOI_SOURCE = "aoi";
const AOI_LAYERS = ["aoi-fill", "aoi-line", "aoi-point"];

function syncAoi(map: MapLibreMap, feature: GeoJSON.Feature | null) {
  const empty: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };
  const data: GeoJSON.FeatureCollection = feature
    ? { type: "FeatureCollection", features: [feature] }
    : empty;

  const source = map.getSource(AOI_SOURCE);
  if (source && "setData" in source) {
    (source as GeoJSONSource).setData(data);
    AOI_LAYERS.forEach((id) => map.getLayer(id) && map.moveLayer(id));
    return;
  }
  map.addSource(AOI_SOURCE, { type: "geojson", data });
  map.addLayer({
    id: "aoi-fill",
    type: "fill",
    source: AOI_SOURCE,
    filter: ["!=", ["geometry-type"], "Point"],
    paint: { "fill-color": "#2563eb", "fill-opacity": 0.08 },
  });
  map.addLayer({
    id: "aoi-line",
    type: "line",
    source: AOI_SOURCE,
    filter: ["!=", ["geometry-type"], "Point"],
    paint: { "line-color": "#2563eb", "line-width": 1.75, "line-dasharray": [2, 1.5] },
  });
  map.addLayer({
    id: "aoi-point",
    type: "circle",
    source: AOI_SOURCE,
    filter: ["==", ["geometry-type"], "Point"],
    paint: {
      "circle-radius": 6,
      "circle-color": "#2563eb",
      "circle-stroke-width": 2,
      "circle-stroke-color": "#ffffff",
    },
  });
}

/** Drop sources (and their layers) whose prefix matches but that the store no longer wants. */
function removeStale(map: MapLibreMap, prefix: string, wanted: Set<string>) {
  const style = map.getStyle();
  if (!style?.sources) return;
  Object.keys(style.sources)
    .filter((id) => id.startsWith(prefix) && !wanted.has(id))
    .forEach((sourceId) => {
      (style.layers ?? [])
        .filter((layer) => "source" in layer && layer.source === sourceId)
        .forEach((layer) => map.getLayer(layer.id) && map.removeLayer(layer.id));
      if (map.getSource(sourceId)) map.removeSource(sourceId);
    });
}

/* ------------------------------------------------------------------ camera helpers */

export function fitBounds(map: MapLibreMap, bounds: [number, number, number, number], padding = 80) {
  const [west, south, east, north] = bounds;
  map.fitBounds(
    [
      [west, south],
      [east, north],
    ],
    { padding, duration: 700, maxZoom: 16 },
  );
}
