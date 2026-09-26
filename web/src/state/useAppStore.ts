/** Application state. One store, because almost every panel reacts to the same few facts:
 *  which rasters are loaded, what area is selected, and whether an analysis has produced a result. */

import { create } from "zustand";
import { api, ApiError } from "./api";
import type { AnalyzeResult, Modality, OverlayLayer, SceneMetadata, TaskType, UploadInfo } from "./types";

/** Area-enclosing shapes only: a point cannot restrict an analysis (D-025). */
export type DrawMode = "rectangle" | "polygon" | "circle" | null;
export type SidebarSection = "search" | "select" | "layers" | "saved" | "help" | null;

/** A loaded raster plus its display state. `mappable` decides map layer vs off-map viewer. */
export interface Layer extends UploadInfo {
  visible: boolean;
  opacity: number;
}

export interface SavedArea {
  id: string;
  name: string;
  createdAt: string;
  geometry: GeoJSON.Feature;
  bounds: [number, number, number, number];
}

export interface Aoi {
  feature: GeoJSON.Feature;
  bounds: [number, number, number, number];
}

/** What the app is doing while `pending` is true, so the user sees the real step, not one spinner. */
export type ProgressStage = "searching" | "preparing" | "analysing";

export const PROGRESS_LABELS: Record<ProgressStage, string> = {
  searching: "Finding suitable satellite imagery",
  preparing: "Preparing Sentinel-2 imagery",
  analysing: "Analysing the selected area",
};

interface AppState {
  // --- inputs
  layers: Layer[];
  aoi: Aoi | null;
  drawMode: DrawMode;
  savedAreas: SavedArea[];

  // --- analysis
  pending: boolean;
  /** Which step is running. null when idle; drives the progress label instead of one spinner. */
  stage: ProgressStage | null;
  result: AnalyzeResult | null;
  error: string | null;
  /** Provenance of the scene retrieved for this result, when imagery was fetched rather than uploaded. */
  scene: SceneMetadata | null;
  /** Overlays the user has switched off; a result may carry several. */
  hiddenOverlays: Set<string>;

  // --- interface
  sidebarOpen: boolean;
  section: SidebarSection;
  /** Handed to the command bar so loading a demo scenario also fills in its question. */
  pendingQuery: string | null;
  detailsOpen: boolean;
  theme: "light" | "dark";
  modelIsFake: boolean | null;

  // --- actions
  addLayers: (uploads: UploadInfo[]) => void;
  removeLayer: (id: string) => void;
  updateLayer: (id: string, patch: Partial<Pick<Layer, "visible" | "opacity" | "modality" | "acquired">>) => void;
  reorderLayer: (id: string, direction: -1 | 1) => void;

  setAoi: (aoi: Aoi | null) => void;
  setDrawMode: (mode: DrawMode) => void;
  saveArea: (name: string) => void;
  removeSavedArea: (id: string) => void;

  runAnalysis: (query: string, forcedTask?: TaskType) => Promise<void>;
  /** Stops waiting for the running analysis. The server finishes its current run regardless. */
  cancelAnalysis: () => void;
  clearResult: () => void;
  toggleOverlay: (url: string) => void;

  openSection: (section: SidebarSection) => void;
  closeSidebar: () => void;
  setPendingQuery: (query: string | null) => void;
  setDetailsOpen: (open: boolean) => void;
  toggleTheme: () => void;
  setModelIsFake: (value: boolean) => void;
  setError: (message: string | null) => void;
}

const SAVED_AREAS_KEY = "satquery.savedAreas";
const THEME_KEY = "satquery.theme";

/** Longest wait for one analysis. A real model on CPU takes about a minute for the longest plans.
 *  Retrieval runs inside the same budget, and a cold Copernicus scene can take minutes to prepare. */
const ANALYSIS_TIMEOUT_MS = 6 * 60 * 1000;
/** Long enough for a progress stage to paint before the next one replaces it. */
const MIN_STAGE_PAINT_MS = 350;
/** Aborting stops the wait only: the server thread finishes its run, and the model lock queues the
 *  next question behind it. The messages say so rather than implying the work was undone. */
const STOPPED_WAITING = "Stopped waiting for this analysis. If the server was still working, your next question starts once it finishes.";
const TIMED_OUT = "No answer after 6 minutes, so the app stopped waiting. Fetching a new satellite scene can take a few minutes; the server may still be working, so try again shortly.";
/** The analysis in flight, if any. Outside the store: it is a handle, not state to render. */
let inflight: AbortController | null = null;

/** Browser storage is a per-viewer convenience here: it must never break the app when unavailable. */
function readSavedAreas(): SavedArea[] {
  try {
    const raw = localStorage.getItem(SAVED_AREAS_KEY);
    const saved = raw ? (JSON.parse(raw) as SavedArea[]) : [];
    // Points saved before D-025 can no longer restrict anything, so they are not offered.
    return saved.filter((area) => area.geometry?.geometry?.type !== "Point");
  } catch {
    return [];
  }
}

function writeSavedAreas(areas: SavedArea[]): void {
  try {
    localStorage.setItem(SAVED_AREAS_KEY, JSON.stringify(areas));
  } catch {
    /* private mode or blocked storage: the areas simply do not persist */
  }
}

function readTheme(): "light" | "dark" {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* fall through to the system preference */
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function applyTheme(theme: "light" | "dark"): void {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* ignore */
  }
}

/** The drawn shape to analyse, when it encloses an area. */
export function areaGeometry(aoi: Aoi | null): GeoJSON.Polygon | GeoJSON.MultiPolygon | null {
  const geometry = aoi?.feature.geometry;
  return geometry && (geometry.type === "Polygon" || geometry.type === "MultiPolygon") ? geometry : null;
}

/**
 * Is the drawn area an axis-aligned rectangle? Imagery retrieval takes a bounding box, so for a
 * circle or polygon the fetched raster would cover more ground than the user drew. Rather than
 * convert the shape silently, the store says so and offers the box explicitly.
 *
 * Mirrors `geo.is_bounding_box` on the server: every corner sits on the bounding box, so the ring
 * traces the box itself. The drawn shape carries no record of which tool made it.
 */
export function isRectangle(aoi: Aoi | null, tolerance = 1e-9): boolean {
  const geometry = areaGeometry(aoi);
  if (!geometry || geometry.type !== "Polygon" || !aoi) return false;
  const ring = geometry.coordinates[0];
  if (!Array.isArray(ring) || ring.length < 4 || ring.length > 5) return false;

  const [west, south, east, north] = aoi.bounds;
  return ring.every((position) => {
    const lon = position?.[0];
    const lat = position?.[1];
    if (typeof lon !== "number" || typeof lat !== "number") return false;
    const onVerticalEdge = Math.abs(lon - west) <= tolerance || Math.abs(lon - east) <= tolerance;
    const onHorizontalEdge = Math.abs(lat - south) <= tolerance || Math.abs(lat - north) <= tolerance;
    return onVerticalEdge && onHorizontalEdge;
  });
}

/**
 * The pipeline takes one or two images. When more are loaded we send the two most recently added,
 * oldest acquisition first, so a bi-temporal pair is ordered before/after rather than by upload
 * order. The UI states which images a question will use, so this is never a hidden choice.
 */
export function selectAnalysisImages(layers: Layer[]): Layer[] {
  const chosen = layers.slice(-2);
  if (chosen.length < 2) return chosen;
  const [first, second] = chosen as [Layer, Layer];
  if (first.acquired && second.acquired && first.acquired > second.acquired) return [second, first];
  return chosen;
}

export const useAppStore = create<AppState>((set, get) => ({
  layers: [],
  aoi: null,
  drawMode: null,
  savedAreas: readSavedAreas(),

  pending: false,
  stage: null,
  result: null,
  scene: null,
  error: null,
  hiddenOverlays: new Set(),

  sidebarOpen: false,
  section: null,
  pendingQuery: null,
  detailsOpen: false,
  theme: readTheme(),
  modelIsFake: null,

  addLayers: (uploads) =>
    set((state) => ({
      layers: [...state.layers, ...uploads.map((u) => ({ ...u, visible: true, opacity: 1 }))].slice(-8),
      error: null,
    })),

  removeLayer: (id) =>
    set((state) => {
      const layers = state.layers.filter((l) => l.id !== id);
      // A result describes the images it ran on; dropping one of those makes it stale. Matched by
      // upload id: names repeat, and a drawn area analyses a cropped copy under another name.
      const stale = state.result?.upload_ids.includes(id) ?? false;
      return { layers, result: stale ? null : state.result, detailsOpen: stale ? false : state.detailsOpen };
    }),

  updateLayer: (id, patch) =>
    set((state) => ({ layers: state.layers.map((l) => (l.id === id ? { ...l, ...patch } : l)) })),

  reorderLayer: (id, direction) =>
    set((state) => {
      const index = state.layers.findIndex((l) => l.id === id);
      const target = index + direction;
      if (index < 0 || target < 0 || target >= state.layers.length) return state;
      const layers = [...state.layers];
      const [moved] = layers.splice(index, 1);
      layers.splice(target, 0, moved!);
      return { layers };
    }),

  setAoi: (aoi) => set({ aoi }),

  setDrawMode: (drawMode) => set({ drawMode }),

  saveArea: (name) => {
    const { aoi, savedAreas } = get();
    if (!aoi) return;
    const area: SavedArea = {
      id: crypto.randomUUID(),
      name: name.trim() || `Area ${savedAreas.length + 1}`,
      createdAt: new Date().toISOString(),
      geometry: aoi.feature,
      bounds: aoi.bounds,
    };
    const next = [area, ...savedAreas];
    writeSavedAreas(next);
    set({ savedAreas: next });
  },

  removeSavedArea: (id) => {
    const next = get().savedAreas.filter((a) => a.id !== id);
    writeSavedAreas(next);
    set({ savedAreas: next });
  },

  runAnalysis: async (query, forcedTask) => {
    const { layers, aoi } = get();
    const uploaded = selectAnalysisImages(layers);

    // Uploaded or preloaded imagery always wins: the fallback path is untouched by retrieval.
    // Only when there is nothing loaded does a drawn area fetch imagery for itself.
    if (!uploaded.length && !aoi) {
      set({ error: "Add an image first, or select an area on the map, then ask a question about it." });
      return;
    }
    if (!uploaded.length && aoi && !isRectangle(aoi)) {
      set({
        error:
          "Fetching imagery currently supports rectangles only. Draw a rectangle over this area, " +
          "or add a GeoTIFF covering the shape you drew.",
      });
      return;
    }

    const controller = new AbortController();
    inflight = controller;
    const timer = setTimeout(() => controller.abort("timeout"), ANALYSIS_TIMEOUT_MS);
    const retrieving = !uploaded.length;
    set({
      pending: true,
      stage: retrieving ? "searching" : "analysing",
      error: null,
      result: null,
      scene: null,
      detailsOpen: false,
      hiddenOverlays: new Set(),
    });

    // Which step failed decides the message the user sees, so the two are never conflated.
    let stage: ProgressStage = retrieving ? "searching" : "analysing";
    try {
      let images = uploaded.map((l) => ({ upload_id: l.id, modality: l.modality, acquired: l.acquired }));
      let scene: SceneMetadata | null = null;

      if (retrieving && aoi) {
        // All retrieval logic lives on the server; this only carries the request across.
        const fetched = await api.fetchImagery(query, aoi.bounds, { signal: controller.signal });
        if (controller.signal.aborted) return;

        stage = "preparing";
        set({ stage });
        scene = fetched.metadata;
        images = [{ upload_id: fetched.upload.id, modality: fetched.upload.modality, acquired: fetched.upload.acquired }];
        // The retrieved raster joins the map like any other layer, beside the basemap, not as it.
        get().addLayers([fetched.upload]);
        // Yield so this stage actually paints. React batches state set in one synchronous block, so
        // without a turn of the event loop "Preparing Sentinel-2 imagery" is replaced by "Analysing"
        // before the browser ever renders it, and the user sees the step skipped.
        await new Promise((resolve) => setTimeout(resolve, MIN_STAGE_PAINT_MS));
        if (controller.signal.aborted) return;
      }

      stage = "analysing";
      set({ stage });
      const result = await api.analyze(
        query,
        images,
        // A drawn area restricts the analysis to the part of each image inside it: the shape itself
        // for a rectangle, circle or polygon; the box alone for anything else. A fetched scene is
        // already cut to the area, so re-cropping it would only lose edge pixels to rounding.
        retrieving
          ? { forcedTask, signal: controller.signal }
          : { aoiBbox: aoi?.bounds ?? null, aoiGeometry: areaGeometry(aoi), forcedTask, signal: controller.signal },
      );
      set({ result, scene, pending: false, stage: null });
    } catch (error) {
      const message = controller.signal.aborted
        ? controller.signal.reason === "timeout"
          ? TIMED_OUT
          : STOPPED_WAITING
        : error instanceof ApiError
          ? // The server explains retrieval failures in terms a user can act on ("no scene under
            // the cloud limit", "the area is too large"), so its wording is kept verbatim.
            stage === "analysing"
            ? error.message
            : `Could not get imagery for this area. ${error.message}`
          : stage === "analysing"
            ? "The analysis failed unexpectedly."
            : "Could not get imagery for this area.";
      set({ error: message, pending: false, stage: null });
    } finally {
      clearTimeout(timer);
      if (inflight === controller) inflight = null;
    }
  },

  cancelAnalysis: () => inflight?.abort("cancelled"),

  clearResult: () =>
    set({ result: null, scene: null, detailsOpen: false, error: null, hiddenOverlays: new Set() }),

  toggleOverlay: (url) =>
    set((state) => {
      const hidden = new Set(state.hiddenOverlays);
      hidden.has(url) ? hidden.delete(url) : hidden.add(url);
      return { hiddenOverlays: hidden };
    }),

  openSection: (section) =>
    set((state) =>
      state.sidebarOpen && state.section === section
        ? { sidebarOpen: false, section: null }
        : { sidebarOpen: true, section },
    ),

  closeSidebar: () => set({ sidebarOpen: false, section: null }),

  setPendingQuery: (pendingQuery) => set({ pendingQuery }),

  setDetailsOpen: (detailsOpen) => set({ detailsOpen }),

  toggleTheme: () =>
    set((state) => {
      const theme = state.theme === "dark" ? "light" : "dark";
      applyTheme(theme);
      return { theme };
    }),

  setModelIsFake: (modelIsFake) => set({ modelIsFake }),

  setError: (error) => set({ error }),
}));

/** Overlays from the current result that the map should draw. */
export function visibleOverlays(state: AppState): OverlayLayer[] {
  return (state.result?.overlay_layers ?? []).filter((layer) => !state.hiddenOverlays.has(layer.url));
}

export function layerLabel(modality: Modality): string {
  return modality === "sar" ? "SAR" : "Optical";
}

/** Bi-temporal actions need two images with distinct acquisition dates; the pipeline cannot infer them. */
export function temporalReadiness(layers: Layer[]): { ready: boolean; reason: string | null } {
  const images = selectAnalysisImages(layers);
  if (images.length < 2) return { ready: false, reason: "Add a second image to compare dates." };
  const [before, after] = images as [Layer, Layer];
  if (!before.acquired || !after.acquired) return { ready: false, reason: "Set an acquisition date on both images." };
  if (before.acquired === after.acquired) return { ready: false, reason: "Both images share the same date." };
  return { ready: true, reason: null };
}
