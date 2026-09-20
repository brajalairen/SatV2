/** Application state. One store, because almost every panel reacts to the same few facts:
 *  which rasters are loaded, what area is selected, and whether an analysis has produced a result. */

import { create } from "zustand";
import { api, ApiError } from "./api";
import type { AnalyzeResult, Modality, OverlayLayer, TaskType, UploadInfo } from "./types";

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

interface AppState {
  // --- inputs
  layers: Layer[];
  aoi: Aoi | null;
  drawMode: DrawMode;
  savedAreas: SavedArea[];

  // --- analysis
  pending: boolean;
  result: AnalyzeResult | null;
  error: string | null;
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

/** Longest wait for one analysis. A real model on CPU takes about a minute for the longest plans. */
const ANALYSIS_TIMEOUT_MS = 3 * 60 * 1000;
/** Aborting stops the wait only: the server thread finishes its run, and the model lock queues the
 *  next question behind it. The messages say so rather than implying the work was undone. */
const STOPPED_WAITING = "Stopped waiting for this analysis. If the server was still working, your next question starts once it finishes.";
const TIMED_OUT = "No answer after 3 minutes, so the app stopped waiting. The server may still be busy; try again shortly.";
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
  result: null,
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
    const images = selectAnalysisImages(layers);
    if (!images.length) {
      set({
        error: aoi
          ? "A selected area on its own has nothing to analyse: this build has no imagery catalogue. Add a GeoTIFF covering the area, or try a demo scenario from Help."
          : "Add an image first, then ask a question about it.",
      });
      return;
    }
    const controller = new AbortController();
    inflight = controller;
    const timer = setTimeout(() => controller.abort("timeout"), ANALYSIS_TIMEOUT_MS);
    set({ pending: true, error: null, result: null, detailsOpen: false, hiddenOverlays: new Set() });
    try {
      const result = await api.analyze(
        query,
        images.map((l) => ({ upload_id: l.id, modality: l.modality, acquired: l.acquired })),
        // A drawn area restricts the analysis to the part of each image inside it: the shape itself
        // for a rectangle, circle or polygon; the box alone for anything else.
        { aoiBbox: aoi?.bounds ?? null, aoiGeometry: areaGeometry(aoi), forcedTask, signal: controller.signal },
      );
      set({ result, pending: false });
    } catch (error) {
      const message = controller.signal.aborted
        ? controller.signal.reason === "timeout"
          ? TIMED_OUT
          : STOPPED_WAITING
        : error instanceof ApiError
          ? error.message
          : "The analysis failed unexpectedly.";
      set({ error: message, pending: false });
    } finally {
      clearTimeout(timer);
      if (inflight === controller) inflight = null;
    }
  },

  cancelAnalysis: () => inflight?.abort("cancelled"),

  clearResult: () => set({ result: null, detailsOpen: false, error: null, hiddenOverlays: new Set() }),

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
