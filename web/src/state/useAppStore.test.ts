/** Store behaviour that the browser checks cover end to end: what an analysis is sent, which result
 *  a removed layer invalidates, and what cancelling does. The API module is mocked. */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "./api";
import { areaGeometry, selectAnalysisImages, temporalReadiness, useAppStore, type Aoi, type Layer } from "./useAppStore";
import type { AnalyzeResult, UploadInfo } from "./types";

const analyze = vi.fn();
vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, analyze: (...args: unknown[]) => analyze(...args) } };
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

const polygonAoi = (): Aoi => ({
  feature: {
    type: "Feature", properties: {},
    geometry: { type: "Polygon", coordinates: [[[10, 50], [11, 50], [11, 51], [10, 51], [10, 50]]] },
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
  analyze.mockResolvedValue(result(["a"]));
  useAppStore.setState({ layers: [], aoi: null, result: null, error: null, pending: false, drawMode: null });
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

  it("refuses to analyse an area with no imagery under it", async () => {
    useAppStore.setState({ layers: [], aoi: polygonAoi() });
    await useAppStore.getState().runAnalysis("what is here?");

    expect(analyze).not.toHaveBeenCalled();
    expect(useAppStore.getState().error).toMatch(/no imagery catalogue/);
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
