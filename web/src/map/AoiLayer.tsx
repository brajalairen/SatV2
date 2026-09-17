/** Area-of-interest drawing, via Terra Draw. The drawn feature lives in the store; MapView renders
 *  it, so Terra Draw's own styling is kept invisible and there is only one source of truth. */

import { useEffect, useRef } from "react";
import { TerraDraw, TerraDrawCircleMode, TerraDrawPointMode, TerraDrawPolygonMode, TerraDrawRectangleMode } from "terra-draw";
import { TerraDrawMapLibreGLAdapter } from "terra-draw-maplibre-gl-adapter";
import { useMap, useMapReady } from "./MapView";
import { useAppStore, type DrawMode } from "../state/useAppStore";

/** Terra Draw draws nothing of its own: MapView owns the AOI styling, so there is one source of
 *  truth for how a selected area looks. The style keys differ per mode, hence two objects. */
const INVISIBLE_POLYGON = { fillOpacity: 0, outlineWidth: 0, outlineOpacity: 0 } as const;
const INVISIBLE_POINT = { pointWidth: 0, pointOpacity: 0, pointOutlineWidth: 0 } as const;

const MODE_NAMES: Record<Exclude<DrawMode, null>, string> = {
  rectangle: "rectangle",
  polygon: "polygon",
  circle: "circle",
  point: "point",
};

export function AoiLayer() {
  const map = useMap();
  const styleReady = useMapReady();
  const drawRef = useRef<TerraDraw | null>(null);
  const drawMode = useAppStore((s) => s.drawMode);
  const aoi = useAppStore((s) => s.aoi);
  const setAoi = useAppStore((s) => s.setAoi);
  const setDrawMode = useAppStore((s) => s.setDrawMode);

  // --- attach Terra Draw to the map
  useEffect(() => {
    if (!map || !styleReady) return; // Terra Draw adds its own sources, so the style must be up
    const draw = new TerraDraw({
      adapter: new TerraDrawMapLibreGLAdapter({ map }),
      modes: [
        // "click-move-or-drag": Terra Draw defaults to click-move, so a dragged box would pan the
        // map instead of drawing. Accepting both means either gesture does what the user expects.
        new TerraDrawRectangleMode({ styles: INVISIBLE_POLYGON, drawInteraction: "click-move-or-drag" }),
        new TerraDrawPolygonMode({ styles: { ...INVISIBLE_POLYGON, ...INVISIBLE_POINT } }),
        new TerraDrawCircleMode({ styles: INVISIBLE_POLYGON, drawInteraction: "click-move-or-drag" }),
        new TerraDrawPointMode({ styles: INVISIBLE_POINT }),
      ],
    });
    draw.start();
    draw.setMode("static");
    drawRef.current = draw;

    draw.on("finish", (id, context) => {
      if (context.action !== "draw") return; // ignore edits and drags of an existing shape
      const drawn = draw.getSnapshotFeature(id);
      if (!drawn) return;
      const feature = drawn as GeoJSON.Feature;
      setAoi({ feature, bounds: featureBounds(feature) });
      // One area at a time: the store now holds it, so Terra Draw's copy goes away.
      draw.clear();
      setDrawMode(null);
    });

    return () => {
      draw.stop();
      drawRef.current = null;
    };
  }, [map, styleReady, setAoi, setDrawMode]);

  // --- follow the selected tool
  useEffect(() => {
    const draw = drawRef.current;
    if (!draw) return;
    draw.setMode(drawMode ? MODE_NAMES[drawMode] : "static");
  }, [drawMode]);

  // --- an area cleared from elsewhere must also clear any half-drawn shape
  useEffect(() => {
    if (!aoi) drawRef.current?.clear();
  }, [aoi]);

  // --- Escape cancels drawing
  useEffect(() => {
    if (!drawMode) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      drawRef.current?.clear();
      setDrawMode(null);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [drawMode, setDrawMode]);

  return null;
}

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
