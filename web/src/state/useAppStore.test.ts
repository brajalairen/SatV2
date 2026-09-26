/** Store behaviour that the browser checks cover end to end: what an analysis is sent, which result
 *  a removed layer invalidates, and what cancelling does. The API module is mocked. */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "./api";
import {
  areaGeometry,
  isRectangle,
  nextAnalysisSource,
  readBasemap,
  selectAnalysisImages,
  temporalReadiness,
  useAppStore,
  type Aoi,
  type Layer,
} from "./useAppStore";
import type { AnalyzeResult, UploadInfo } from "./types";

const analyze = vi.fn();
const fetchImagery = vi.fn();
vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      analyze: (...args: unknown[]) => analyze(...args),
      fetchImagery: (...args: unknown[]) => fetchImagery(...args),
    },
  };
});

function upload(id: string, extra: Partial<UploadInfo> = {}): Layer {
  return {
    id,
    name: `${id}.tif`,
    modality: "optical",
    acquired: null,
    preview_url: `/api/uploads/${id}/preview.png`,
    mappable: true,
    visible: true,
    opacity: 1,
    summary: {
      index: 0, name: `${id}.tif`, modality: "optical", width: 100, height: 100, bands: ["red"],
      crs: "EPSG:32643", acquired: null, decimation: 1, bounds_wgs84: [10, 50, 11, 51],
      corners_wgs84: [[10, 51], [11, 51], [11, 50], [10, 50]], georeference_note: null,
    },
    ...extra,
  } as Layer;
}

function result(uploadIds: string[]): AnalyzeResult {
  return {
    upload_ids: uploadIds,
    overlay_layers: [],
    area: null,
    response: {
      status: "ok", task: "caption", answer: "an answer", evidence: [], confidence: null,
      report_html: null, report_json: null,
      trace: {
        run_id: "r", created_at: "", query: "q", input_config: "single_optical", images: [],
        validation: [], intent: null, plan: [], steps: [], total_duration_s: 0,
      },
    },
  } as AnalyzeResult;
}

/** A rectangle: every corner sits on the bounding box, so imagery can be fetched for it. */
/** A single-date /api/fetch-imagery answer, in the shape the server sends it. */
function singleFetch(scene: UploadInfo, metadata: Record<string, unknown>) {
  return { mode: "single", upload: scene, metadata, images: [{ role: "single", upload: scene, metadata }],
           temporal: null, cached: false };
}

const polygonAoi = (): Aoi => ({
  feature: {
    type: "Feature", properties: {},
    geometry: { type: "Polygon", coordinates: [[[10, 50], [11, 50], [11, 51], [10, 51], [10, 50]]] },
  },
  bounds: [10, 50, 11, 51],
});

/** A triangle: encloses an area, but its corners do not trace its bounding box. */
const triangleAoi = (): Aoi => ({
  feature: {
    type: "Feature", properties: {},
    geometry: { type: "Polygon", coordinates: [[[10, 50], [11, 50], [10.5, 51], [10, 50]]] },
  },
  bounds: [10, 50, 11, 51],
});

/** An area saved in a browser before the point tool was removed (D-025). */
const pointAoi = (): Aoi => ({
  feature: { type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [10.5, 50.5] } },
  bounds: [10.5, 50.5, 10.5, 50.5],
});

/** The options the store passed to api.analyze on its most recent call. */
function analyseOptions(): { aoiBbox: unknown; aoiGeometry: unknown } {
  const call = analyze.mock.calls.at(-1);
  if (!call) throw new Error("api.analyze was not called");
  return call[2] as { aoiBbox: unknown; aoiGeometry: unknown };
}

beforeEach(() => {
  analyze.mockReset();
  fetchImagery.mockReset();
  analyze.mockResolvedValue(result(["a"]));
  useAppStore.setState({
    layers: [], aoi: null, result: null, error: null, pending: false, drawMode: null,
    stage: null, scenes: [], basemap: "standard",
  });
  localStorage.removeItem("satquery.basemap");
});

describe("choosing what to analyse", () => {
  it("takes the two most recent images", () => {
    const chosen = selectAnalysisImages([upload("a"), upload("b"), upload("c")]);
    expect(chosen.map((l) => l.id)).toEqual(["b", "c"]);
  });

  it("orders a dated pair oldest first, whatever order they were added", () => {
    const chosen = selectAnalysisImages([upload("new", { acquired: "2023-01-01" }), upload("old", { acquired: "2019-01-01" })]);
    expect(chosen.map((l) => l.id)).toEqual(["old", "new"]);
  });

  it("reports why a temporal comparison is not available", () => {
    expect(temporalReadiness([upload("a")]).reason).toMatch(/second image/);
    expect(temporalReadiness([upload("a"), upload("b")]).reason).toMatch(/acquisition date/);
    const same = [upload("a", { acquired: "2020-01-01" }), upload("b", { acquired: "2020-01-01" })];
    expect(temporalReadiness(same).reason).toMatch(/same date/);
    const ok = [upload("a", { acquired: "2019-01-01" }), upload("b", { acquired: "2023-01-01" })];
    expect(temporalReadiness(ok)).toEqual({ ready: true, reason: null });
  });
});

describe("the drawn area that reaches the backend", () => {
  it("sends a polygon as the shape itself, alongside its box", async () => {
    useAppStore.setState({ layers: [upload("a")], aoi: polygonAoi() });
    await useAppStore.getState().runAnalysis("what is here?");

    expect(analyseOptions().aoiBbox).toEqual([10, 50, 11, 51]);
    expect(analyseOptions().aoiGeometry).toMatchObject({ type: "Polygon" });
  });

  it("never sends a point as a shape: it encloses no area (D-025)", () => {
    expect(areaGeometry(pointAoi())).toBeNull();
    expect(areaGeometry(polygonAoi())).toMatchObject({ type: "Polygon" });
    expect(areaGeometry(null)).toBeNull();
  });

  it("sends no area at all when none is drawn", async () => {
    useAppStore.setState({ layers: [upload("a")] });
    await useAppStore.getState().runAnalysis("what is here?");

    expect(analyseOptions().aoiBbox).toBeNull();
    expect(analyseOptions().aoiGeometry).toBeNull();
  });

  it("asks for nothing when neither an image nor an area is present", async () => {
    useAppStore.setState({ layers: [], aoi: null });
    await useAppStore.getState().runAnalysis("what is here?");

    expect(analyze).not.toHaveBeenCalled();
    expect(fetchImagery).not.toHaveBeenCalled();
    expect(useAppStore.getState().error).toMatch(/add an image|select an area/i);
  });
});

describe("the result and the layers it describes", () => {
  it("keeps the result when an unrelated layer is removed", () => {
    useAppStore.setState({ layers: [upload("a"), upload("b")], result: result(["b"]) });
    useAppStore.getState().removeLayer("a");

    expect(useAppStore.getState().result).not.toBeNull();
    expect(useAppStore.getState().layers.map((l) => l.id)).toEqual(["b"]);
  });

  it("clears the result when a layer it ran on is removed", () => {
    useAppStore.setState({ layers: [upload("a"), upload("b")], result: result(["a", "b"]), detailsOpen: true });
    useAppStore.getState().removeLayer("b");

    expect(useAppStore.getState().result).toBeNull();
    expect(useAppStore.getState().detailsOpen).toBe(false);
  });

  it("matches by upload id, not by name: a crop is analysed under another name", () => {
    // Two files can share a name, and a drawn area analyses "<name>-area.tif".
    const layers = [upload("a", { name: "scene.tif" }), upload("b", { name: "scene.tif" })];
    useAppStore.setState({ layers, result: result(["a"]) });
    useAppStore.getState().removeLayer("b");

    expect(useAppStore.getState().result).not.toBeNull();
  });
});

describe("waiting for an analysis", () => {
  it("reports a failure from the server", async () => {
    analyze.mockRejectedValue(new ApiError("unknown upload 'x'", 404));
    useAppStore.setState({ layers: [upload("a")] });
    await useAppStore.getState().runAnalysis("what is here?");

    expect(useAppStore.getState().pending).toBe(false);
    expect(useAppStore.getState().error).toBe("unknown upload 'x'");
  });

  it("cancelling stops the wait and says the server may still be working", async () => {
    analyze.mockImplementation((_q: string, _i: unknown, options: { signal: AbortSignal }) =>
      new Promise((_resolve, reject) => options.signal.addEventListener("abort", () => reject(options.signal.reason))));
    useAppStore.setState({ layers: [upload("a")] });

    const running = useAppStore.getState().runAnalysis("what is here?");
    expect(useAppStore.getState().pending).toBe(true);
    useAppStore.getState().cancelAnalysis();
    await running;

    expect(useAppStore.getState().pending).toBe(false);
    expect(useAppStore.getState().result).toBeNull();
    expect(useAppStore.getState().error).toMatch(/Stopped waiting/);
  });
});

describe("areas saved in the browser", () => {
  it("drops points saved before the point tool was removed (D-025)", async () => {
    localStorage.setItem(
      "satquery.savedAreas",
      JSON.stringify([
        { id: "1", name: "a point", createdAt: "", geometry: pointAoi().feature, bounds: [10.5, 50.5, 10.5, 50.5] },
        { id: "2", name: "a box", createdAt: "", geometry: polygonAoi().feature, bounds: [10, 50, 11, 51] },
      ]),
    );
    vi.resetModules();
    const fresh = await import("./useAppStore");

    expect(fresh.useAppStore.getState().savedAreas.map((area) => area.name)).toEqual(["a box"]);
  });
});


describe("fetching imagery for a drawn area", () => {
  /** What /api/fetch-imagery answers with: an ordinary upload plus the scene's provenance. */
  function fetched(id = "fetched-1") {
    return singleFetch(
      { ...upload(id), name: "Sentinel-2 L2A 2026-09-05", acquired: "2026-09-05" },
      {
        provider: "Copernicus Data Space Ecosystem", collection: "sentinel-2-l2a",
        satellite: "Sentinel-2", product_level: "L2A", acquired: "2026-09-05",
        acquired_datetime: "2026-09-05T05:53:55Z", cloud_cover: 41.96,
        bbox_wgs84: [10, 50, 11, 51] as [number, number, number, number],
        crs: "EPSG:4326", resolution_m: 10,
        bands: ["B02 (blue)"], width: 1052, height: 1106, scene_id: "S2A_TEST",
        attribution: "Contains modified Copernicus Sentinel data", cached: false,
        alternatives_considered: 7,
      },
    );
  }

  it("recognises a rectangle, and only a rectangle", () => {
    expect(isRectangle(polygonAoi())).toBe(true);
    expect(isRectangle(triangleAoi())).toBe(false);
    expect(isRectangle(pointAoi())).toBe(false);
    expect(isRectangle(null)).toBe(false);
  });

  it("retrieves imagery, then analyses it through the existing path", async () => {
    fetchImagery.mockResolvedValue(fetched());
    analyze.mockResolvedValue(result(["fetched-1"]));
    useAppStore.setState({ layers: [], aoi: polygonAoi() });

    await useAppStore.getState().runAnalysis("are there water bodies here?");

    expect(fetchImagery).toHaveBeenCalledTimes(1);
    expect(fetchImagery.mock.calls[0]?.[1]).toEqual([10, 50, 11, 51]);
    // the retrieved scene is analysed as an ordinary upload
    expect(analyze.mock.calls[0]?.[1]).toEqual([
      { upload_id: "fetched-1", modality: "optical", acquired: "2026-09-05" },
    ]);
    expect(useAppStore.getState().result).not.toBeNull();
    expect(useAppStore.getState().pending).toBe(false);
    expect(useAppStore.getState().stage).toBeNull();
  });

  it("keeps the scene's provenance for the result card", async () => {
    fetchImagery.mockResolvedValue(fetched());
    useAppStore.setState({ layers: [], aoi: polygonAoi() });

    await useAppStore.getState().runAnalysis("what is here?");

    const scenes = useAppStore.getState().scenes;
    expect(scenes).toHaveLength(1);
    const scene = scenes[0];
    expect(scene?.satellite).toBe("Sentinel-2");
    expect(scene?.acquired).toBe("2026-09-05");
    expect(scene?.cloud_cover).toBe(41.96);
  });

  it("adds the retrieved raster to the map as a layer of its own", async () => {
    fetchImagery.mockResolvedValue(fetched());
    useAppStore.setState({ layers: [], aoi: polygonAoi() });

    await useAppStore.getState().runAnalysis("what is here?");

    expect(useAppStore.getState().layers.map((l) => l.id)).toEqual(["fetched-1"]);
  });

  it("does not fetch when imagery is already loaded: upload stays the fallback", async () => {
    useAppStore.setState({ layers: [upload("a")], aoi: polygonAoi() });
    await useAppStore.getState().runAnalysis("what is here?");

    expect(fetchImagery).not.toHaveBeenCalled();
    expect(analyseOptions().aoiBbox).toEqual([10, 50, 11, 51]);
  });

  it("refuses a circle or polygon rather than silently using its box", async () => {
    useAppStore.setState({ layers: [], aoi: triangleAoi() });
    await useAppStore.getState().runAnalysis("what is here?");

    expect(fetchImagery).not.toHaveBeenCalled();
    expect(useAppStore.getState().error).toMatch(/rectangle/i);
  });

  it("clears the previous scene when a new analysis starts", async () => {
    useAppStore.setState({ scenes: [fetched().metadata as never], layers: [upload("a")] });
    await useAppStore.getState().runAnalysis("what is here?");
    expect(useAppStore.getState().scenes).toEqual([]);
  });
});

describe("when retrieval or analysis fails", () => {
  const rectangle = () => useAppStore.setState({ layers: [], aoi: polygonAoi() });

  it.each([
    ["no_imagery_found", 404, "No Sentinel-2 L2A scene matched this area within the search window."],
    ["aoi_too_large", 413, "The selected area is about 4,626,540 km2, above the 400 km2 limit."],
    ["credentials_missing", 503, "Copernicus credentials are not configured."],
    ["only_one_acquisition", 404, "Only one suitable Sentinel-2 acquisition was found for this area and time window."],
  ])("keeps the server's own wording for %s", async (code, status, message) => {
    fetchImagery.mockRejectedValue(new ApiError(message, status, code));
    rectangle();

    await useAppStore.getState().runAnalysis("what is here?");

    const error = useAppStore.getState().error ?? "";
    expect(error).toContain(message);
    expect(error).not.toMatch(/failed unexpectedly/);
    expect(useAppStore.getState().pending).toBe(false);
    expect(useAppStore.getState().stage).toBeNull();
  });

  it("says retrieval failed, not that the analysis did", async () => {
    fetchImagery.mockRejectedValue(new ApiError("Copernicus timed out.", 504, "provider_timeout"));
    rectangle();

    await useAppStore.getState().runAnalysis("what is here?");

    expect(useAppStore.getState().error).toMatch(/could not get imagery/i);
    expect(analyze).not.toHaveBeenCalled();
  });

  it("reports an analysis failure without blaming retrieval", async () => {
    fetchImagery.mockResolvedValue(
      singleFetch(upload("fetched-1"), { satellite: "Sentinel-2", acquired: "2026-09-05", cloud_cover: 1 }));
    analyze.mockRejectedValue(new ApiError("The image could not be read.", 400));
    rectangle();

    await useAppStore.getState().runAnalysis("what is here?");

    const error = useAppStore.getState().error ?? "";
    expect(error).toContain("The image could not be read.");
    expect(error).not.toMatch(/could not get imagery/i);
  });

  it("does not leave the UI stuck when retrieval is cancelled", async () => {
    let reject: (error: unknown) => void = () => {};
    fetchImagery.mockReturnValue(new Promise((_resolve, r) => { reject = r; }));
    rectangle();

    const running = useAppStore.getState().runAnalysis("what is here?");
    expect(useAppStore.getState().pending).toBe(true);
    expect(useAppStore.getState().stage).toBe("searching");

    useAppStore.getState().cancelAnalysis();
    reject(new DOMException("aborted", "AbortError"));
    await running;

    expect(useAppStore.getState().pending).toBe(false);
    expect(useAppStore.getState().stage).toBeNull();
    expect(useAppStore.getState().error).toMatch(/stopped waiting/i);
  });
});


describe("choosing a basemap", () => {
  it("changes the basemap and nothing else, and makes no request", () => {
    const scenes = [{ satellite: "Sentinel-2", acquired: "2026-09-19", cloud_cover: 0.02 }] as never;
    useAppStore.setState({
      layers: [upload("a")],
      aoi: polygonAoi(),
      result: result(["a"]),
      scenes,
      pendingQuery: "are there water bodies here?",
      hiddenOverlays: new Set(["/api/runs/r/input.png"]),
      drawMode: "rectangle",
    });
    const before = useAppStore.getState();

    for (const basemap of ["satellite", "hybrid", "standard"] as const) {
      useAppStore.getState().setBasemap(basemap);
      const after = useAppStore.getState();
      expect(after.basemap).toBe(basemap);
      // the same objects, not copies: nothing was recomputed or reset
      expect(after.aoi).toBe(before.aoi);
      expect(after.layers).toBe(before.layers);
      expect(after.result).toBe(before.result);
      expect(after.scenes).toBe(before.scenes);
      expect(after.pendingQuery).toBe(before.pendingQuery);
      expect(after.hiddenOverlays).toBe(before.hiddenOverlays);
      expect(after.drawMode).toBe(before.drawMode);
      expect(after.error).toBe(before.error);
      expect(after.pending).toBe(false);
    }
    expect(fetchImagery).not.toHaveBeenCalled();
    expect(analyze).not.toHaveBeenCalled();
  });

  it("does not change what a later analysis fetches or sends", async () => {
    fetchImagery.mockResolvedValue(
      singleFetch(upload("fetched-1"), { satellite: "Sentinel-2", acquired: "2026-09-19", cloud_cover: 0.02 }));
    useAppStore.setState({ layers: [], aoi: polygonAoi() });
    useAppStore.getState().setBasemap("hybrid");

    await useAppStore.getState().runAnalysis("are there water bodies here?");

    // retrieval is asked for the drawn area and nothing about the basemap
    expect(fetchImagery.mock.calls[0]?.[1]).toEqual([10, 50, 11, 51]);
    expect(JSON.stringify(fetchImagery.mock.calls[0])).not.toMatch(/hybrid|satellite|basemap/i);
    expect(JSON.stringify(analyze.mock.calls[0])).not.toMatch(/hybrid|satellite|basemap/i);
  });

  it("remembers the choice in this browser, and ignores a value it does not offer", () => {
    useAppStore.getState().setBasemap("hybrid");
    expect(localStorage.getItem("satquery.basemap")).toBe("hybrid");
    expect(readBasemap()).toBe("hybrid");

    localStorage.setItem("satquery.basemap", "terrain");
    expect(readBasemap()).toBe("standard");
  });

  it("still switches when browser storage is blocked", () => {
    const blocked = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("storage blocked");
    });
    useAppStore.getState().setBasemap("satellite");
    expect(useAppStore.getState().basemap).toBe("satellite");
    blocked.mockRestore();
  });
});


describe("comparing two dates", () => {
  function scene(role: "before" | "after", id: string, acquired: string, cloud: number) {
    return {
      role,
      upload: { ...upload(id), name: `Sentinel-2 L2A ${acquired} (${role})`, acquired },
      metadata: { satellite: "Sentinel-2", product_level: "L2A", acquired, cloud_cover: cloud, scene_id: id },
    };
  }

  /** A temporal /api/fetch-imagery answer: two real acquisitions, oldest first. */
  function temporalFetch() {
    const before = scene("before", "jul", "2026-07-05", 1.5);
    const after = scene("after", "sep", "2026-09-19", 0.02);
    return {
      mode: "temporal", upload: after.upload, metadata: after.metadata, images: [before, after], cached: false,
      temporal: {
        basis: "relative period", explanation: "first vs last third", days_apart: 76,
        before_window: { start: "2026-06-30", end: "2026-07-29", label: "2026-06-30 to 2026-07-29" },
        after_window: { start: "2026-08-29", end: "2026-09-27", label: "2026-08-29 to 2026-09-27" },
      },
    };
  }

  const unsubscribe: (() => void)[] = [];
  afterEach(() => unsubscribe.splice(0).forEach((stop) => stop()));

  function recordStages(): string[] {
    const stages: string[] = [];
    unsubscribe.push(useAppStore.subscribe((state) => {
      if (state.stage && stages.at(-1) !== state.stage) stages.push(state.stage);
    }));
    return stages;
  }

  it("sends both retrieved scenes to the existing analysis, earlier first", async () => {
    fetchImagery.mockResolvedValue(temporalFetch());
    useAppStore.setState({ layers: [], aoi: polygonAoi() });

    await useAppStore.getState().runAnalysis("What changed here over the last 3 months?");

    expect(analyze.mock.calls[0]?.[1]).toEqual([
      { upload_id: "jul", modality: "optical", acquired: "2026-07-05" },
      { upload_id: "sep", modality: "optical", acquired: "2026-09-19" },
    ]);
  });

  it("keeps both provenance records, with their real dates and cloud cover", async () => {
    fetchImagery.mockResolvedValue(temporalFetch());
    useAppStore.setState({ layers: [], aoi: polygonAoi() });

    await useAppStore.getState().runAnalysis("What changed here?");

    const scenes = useAppStore.getState().scenes;
    expect(scenes.map((s) => s.acquired)).toEqual(["2026-07-05", "2026-09-19"]);
    expect(scenes.map((s) => s.cloud_cover)).toEqual([1.5, 0.02]);
  });

  it("puts both scenes on the map, marked as retrieved", async () => {
    fetchImagery.mockResolvedValue(temporalFetch());
    useAppStore.setState({ layers: [], aoi: polygonAoi() });

    await useAppStore.getState().runAnalysis("What changed here?");

    const layers = useAppStore.getState().layers;
    expect(layers.map((l) => l.id)).toEqual(["jul", "sep"]);
    expect(layers.every((l) => l.fetched)).toBe(true);
  });

  it("shows the comparison stages it really goes through", async () => {
    fetchImagery.mockResolvedValue(temporalFetch());
    useAppStore.setState({ layers: [], aoi: polygonAoi() });
    const stages = recordStages();

    await useAppStore.getState().runAnalysis("What changed here?");

    expect(stages).toEqual(["searching", "comparing", "analysing-change"]);
  });

  it("keeps the single-date stages for a single-date question", async () => {
    fetchImagery.mockResolvedValue(fetchedSingle());
    useAppStore.setState({ layers: [], aoi: polygonAoi() });
    const stages = recordStages();

    await useAppStore.getState().runAnalysis("Are there water bodies here?");

    expect(stages).toEqual(["searching", "preparing", "analysing"]);
  });

  function fetchedSingle() {
    return singleFetch({ ...upload("one"), acquired: "2026-09-19" },
                       { satellite: "Sentinel-2", acquired: "2026-09-19", cloud_cover: 0.02 });
  }
});

describe("what the next question runs on", () => {
  const retrieved = (id: string) => ({ ...upload(id), fetched: true });

  it("a retrieved scene never blocks a new retrieval for the drawn rectangle", async () => {
    // after a single-date answer, a question about change must reach two-date retrieval
    fetchImagery.mockResolvedValue(singleFetch(upload("new"), { satellite: "Sentinel-2", acquired: "2026-09-19" }));
    useAppStore.setState({ layers: [retrieved("old")], aoi: polygonAoi() });

    expect(nextAnalysisSource(useAppStore.getState().layers, polygonAoi())).toEqual({ kind: "fetch" });
    await useAppStore.getState().runAnalysis("What changed here?");

    expect(fetchImagery).toHaveBeenCalledTimes(1);
  });

  it("the user's own imagery still wins over retrieval", async () => {
    useAppStore.setState({ layers: [retrieved("old"), upload("mine")], aoi: polygonAoi() });

    await useAppStore.getState().runAnalysis("what is here?");

    expect(fetchImagery).not.toHaveBeenCalled();
    expect(analyze.mock.calls[0]?.[1]).toEqual([{ upload_id: "mine", modality: "optical", acquired: null }]);
  });

  it("retrieved scenes are analysed directly once no rectangle is left", () => {
    const source = nextAnalysisSource([retrieved("old")], null);
    expect(source.kind === "images" && source.images.map((l) => l.id)).toEqual(["old"]);
  });

  it("a new retrieval replaces the layers of the previous one", () => {
    useAppStore.setState({ layers: [upload("mine"), retrieved("old")] });
    useAppStore.getState().setFetchedLayers([upload("jul"), upload("sep")]);

    expect(useAppStore.getState().layers.map((l) => l.id)).toEqual(["mine", "jul", "sep"]);
  });
});
