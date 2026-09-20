/** Area-of-interest drawing, via Terra Draw. Terra Draw owns only the shape being drawn; once it is
 *  finished the feature moves to the store and MapView renders it, so a selected area has one
 *  source of truth.
 *
 *  Rectangle and circle are press-drag-release, like a selection tool in any drawing program: the
 *  shape follows the cursor while the button is held and is finished on release. Polygon is
 *  click-per-corner, having no natural drag. Every tool encloses an area (D-025). */

import { useEffect, useRef } from "react";
import {
  TerraDraw,
  TerraDrawCircleMode,
  TerraDrawPolygonMode,
  TerraDrawRectangleMode,
  type TerraDrawMouseEvent,
} from "terra-draw";
import { TerraDrawMapLibreGLAdapter } from "terra-draw-maplibre-gl-adapter";
import type { Map as MapLibreMap } from "maplibre-gl";
import { useMap, useMapReady } from "./MapView";
import { useAppStore } from "../state/useAppStore";
import { anchoredEvent, drawnArea, featureBounds, MODE_NAMES, type Press } from "./aoiGeometry";

export { featureBounds };

/** Terra Draw starts a dragged shape at the first pointer move past its drag threshold, which after
 *  a quick flick or a touch can be well away from where the user pressed. These start it exactly at
 *  the press instead: the rectangle's first corner, the circle's centre. */
class PressAnchoredRectangleMode extends TerraDrawRectangleMode {
  press: Press | null = null;
  override onDragStart(event: TerraDrawMouseEvent, setMapDraggability: (enabled: boolean) => void) {
    super.onDragStart(anchoredEvent(event, this.press), setMapDraggability);
  }
}

class PressAnchoredCircleMode extends TerraDrawCircleMode {
  press: Press | null = null;
  override onDragStart(event: TerraDrawMouseEvent, setMapDraggability: (enabled: boolean) => void) {
    super.onDragStart(anchoredEvent(event, this.press), setMapDraggability);
  }
}

/** Terra Draw's default coordinate precision: 9 decimal places, about 0.1 mm. */
const round9 = (value: number) => Math.round(value * 1e9) / 1e9;

/** The press position, computed the way the MapLibre adapter computes its own event positions. */
function pressAt(map: MapLibreMap, event: PointerEvent): Press {
  const { left, top } = map.getContainer().getBoundingClientRect();
  const containerX = event.clientX - left;
  const containerY = event.clientY - top;
  const { lng, lat } = map.unproject([containerX, containerY]);
  return { lng: round9(lng), lat: round9(lat), containerX, containerY };
}

/** The in-progress shape, in the blue of a finished area (MapView) but a touch stronger, so it reads
 *  as live while it follows the cursor. */
const ACCENT = "#2563eb";
const DRAWING = { fillColor: ACCENT, fillOpacity: 0.12, outlineColor: ACCENT, outlineOpacity: 1, outlineWidth: 2 } as const;


export function AoiLayer() {
  const map = useMap();
  const styleReady = useMapReady();
  const drawRef = useRef<TerraDraw | null>(null);
  /** True between the first point of a new shape and its finish or cancellation. */
  const drawingRef = useRef(false);
  const drawMode = useAppStore((s) => s.drawMode);
  const aoi = useAppStore((s) => s.aoi);
  const setAoi = useAppStore((s) => s.setAoi);
  const setDrawMode = useAppStore((s) => s.setDrawMode);

  // --- attach Terra Draw to the map
  useEffect(() => {
    if (!map || !styleReady) return; // Terra Draw adds its own sources, so the style must be up
    // "click-drag": press to anchor, drag to size, release to finish. Terra Draw stops the map
    // panning as soon as the pointer moves, so the drag draws instead of moving the map.
    const rectangle = new PressAnchoredRectangleMode({ styles: DRAWING, drawInteraction: "click-drag" });
    const circle = new PressAnchoredCircleMode({ styles: DRAWING, drawInteraction: "click-drag" });
    const draw = new TerraDraw({
      adapter: new TerraDrawMapLibreGLAdapter({ map }),
      modes: [
        rectangle,
        new TerraDrawPolygonMode({ styles: { ...DRAWING, closingPointColor: ACCENT, closingPointOutlineColor: "#ffffff" } }),
        circle,
      ],
    });
    draw.start();

    // Terra Draw listens on the canvas, so the press is recorded there too.
    const canvas = map.getCanvas();
    const onPointerDown = (event: PointerEvent) => {
      if (!event.isPrimary) return;
      rectangle.press = circle.press = pressAt(map, event);
    };
    canvas.addEventListener("pointerdown", onPointerDown);
    // A theme switch rebuilds this instance: resume whichever tool is selected.
    const current = useAppStore.getState().drawMode;
    draw.setMode(current ? MODE_NAMES[current] : "static");
    drawRef.current = draw;

    // A new shape replaces the selected area the moment it starts, as a selection tool does, so the
    // old outline never lingers beside the one being drawn.
    draw.on("change", (_ids, type) => {
      if (type !== "create") return;
      drawingRef.current = true;
      if (useAppStore.getState().aoi) setAoi(null);
    });

    draw.on("finish", (id, context) => {
      if (context.action !== "draw") return; // ignore edits and drags of an existing shape
      drawingRef.current = false;
      const drawn = draw.getSnapshotFeature(id);
      // One area at a time: the store takes the finished shape, so Terra Draw's copy goes away.
      draw.clear();
      if (!drawn) return;
      const area = drawnArea(drawn as GeoJSON.Feature);
      if (!area) return; // encloses nothing: stay in the tool so they can draw again
      setAoi(area);
      setDrawMode(null);
    });

    return () => {
      canvas.removeEventListener("pointerdown", onPointerDown);
      try {
        draw.stop(); // detaches its pointer listeners first, then removes its layers
      } catch {
        // A basemap switch rebuilds the style, which has already removed Terra Draw's layers.
      }
      drawRef.current = null;
      drawingRef.current = false;
    };
  }, [map, styleReady, setAoi, setDrawMode]);

  // --- follow the selected tool. Switching tools abandons any half-drawn shape.
  useEffect(() => {
    const draw = drawRef.current;
    if (!draw) return;
    drawingRef.current = false;
    draw.setMode(drawMode ? MODE_NAMES[drawMode] : "static");
  }, [drawMode]);

  // --- an area cleared from elsewhere must also clear any leftover shape, but not the new shape
  //     whose start is what cleared it
  useEffect(() => {
    if (!aoi && !drawingRef.current) drawRef.current?.clear();
  }, [aoi]);

  // --- Escape cancels drawing and leaves no area behind, even mid-drag
  useEffect(() => {
    if (!drawMode) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      drawingRef.current = false;
      drawRef.current?.clear();
      setDrawMode(null);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [drawMode, setDrawMode]);

  return null;
}

