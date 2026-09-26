/** Typed client for satquery/server.py. Relative URLs: Vite proxies them in dev, FastAPI serves
 *  them directly in production. */

import type {
  AnalyzeResult,
  Example,
  FetchImageryResult,
  Health,
  Modality,
  TaskType,
  UploadInfo,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    /** Machine-readable cause from the imagery layer (e.g. "aoi_too_large"), when the server sent one. */
    readonly code: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    if (init?.signal?.aborted) throw error; // cancelled or timed out by the caller, not unreachable
    throw new ApiError("Cannot reach the analysis server. Is it running?", 0);
  }
  if (!response.ok) {
    const { message, code } = await readError(response);
    throw new ApiError(message, response.status, code);
  }
  return (await response.json()) as T;
}

/** FastAPI puts a string in `detail` for HTTPException and a list for validation errors.
 *  Imagery retrieval instead answers {code, message}, which carries the reason a user needs
 *  ("no scene under the cloud limit", "the area is too large"), so that wins when present. */
async function readError(response: Response): Promise<{ message: string; code: string | null }> {
  try {
    const body = await response.json();
    if (typeof body?.message === "string" && body.message) {
      return { message: body.message, code: typeof body?.code === "string" ? body.code : null };
    }
    const detail = body?.detail;
    if (typeof detail === "string") return { message: detail, code: null };
    if (Array.isArray(detail) && detail.length) {
      return { message: detail.map((d) => d?.msg ?? String(d)).join("; "), code: null };
    }
  } catch {
    /* fall through to the status text */
  }
  return { message: response.statusText || `Request failed (${response.status})`, code: null };
}

export const api = {
  health: () => request<Health>("/api/health"),

  examples: () => request<Example[]>("/api/examples"),

  exampleQueries: () => request<string[]>("/api/example-queries"),

  loadExample: (index: number) => request<UploadInfo[]>(`/api/examples/${index}/load`, { method: "POST" }),

  /** Without a modality the server reads it from the file's band descriptions (VV/VH -> SAR), and
   *  says how it decided; a SAR file sent as "optical" would otherwise pair up as a date comparison. */
  upload: (file: File, modality?: Modality | null, acquired?: string | null) => {
    const form = new FormData();
    form.append("file", file);
    if (modality) form.append("modality", modality);
    if (acquired) form.append("acquired", acquired);
    return request<UploadInfo>("/api/uploads", { method: "POST", body: form });
  },

  /**
   * Retrieve satellite imagery for a drawn area, so a question can be asked without uploading a
   * GeoTIFF. All the retrieval logic lives on the server: this only carries the request across.
   * The result is an ordinary upload, which then goes through `analyze` unchanged.
   */
  fetchImagery: (
    query: string,
    aoiBbox: [number, number, number, number],
    options: { daysBack?: number | null; maxCloud?: number | null; signal?: AbortSignal } = {},
  ) =>
    request<FetchImageryResult>("/api/fetch-imagery", {
      method: "POST",
      signal: options.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        aoi_bbox: aoiBbox,
        days_back: options.daysBack ?? null,
        max_cloud: options.maxCloud ?? null,
      }),
    }),

  analyze: (
    query: string,
    images: { upload_id: string; modality?: Modality; acquired?: string | null }[],
    options: {
      aoiBbox?: [number, number, number, number] | null;
      /** The drawn shape itself, so a circle or polygon is analysed as drawn, not as its box. */
      aoiGeometry?: GeoJSON.Polygon | GeoJSON.MultiPolygon | null;
      forcedTask?: TaskType;
      signal?: AbortSignal;
    } = {},
  ) =>
    request<AnalyzeResult>("/api/analyze", {
      method: "POST",
      signal: options.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        images,
        forced_task: options.forcedTask ?? null,
        aoi_bbox: options.aoiBbox ?? null,
        aoi_geometry: options.aoiGeometry ?? null,
      }),
    }),
};

/**
 * Place search. Nominatim moves the camera and nothing more: it says where a place is, and makes
 * no claim about imagery being available there.
 */
export interface Place {
  name: string;
  lon: number;
  lat: number;
  bbox: [number, number, number, number] | null;
}

export async function searchPlaces(query: string, signal?: AbortSignal): Promise<Place[]> {
  const url = `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=5&q=${encodeURIComponent(query)}`;
  const response = await fetch(url, { signal, headers: { Accept: "application/json" } });
  if (!response.ok) throw new ApiError("Place search is unavailable.", response.status);
  const rows = (await response.json()) as {
    display_name: string;
    lon: string;
    lat: string;
    boundingbox?: [string, string, string, string];
  }[];
  return rows.map((row) => ({
    name: row.display_name,
    lon: Number(row.lon),
    lat: Number(row.lat),
    // Nominatim orders its box [south, north, west, east]; the map wants [w, s, e, n].
    bbox: row.boundingbox
      ? [
          Number(row.boundingbox[2]),
          Number(row.boundingbox[0]),
          Number(row.boundingbox[3]),
          Number(row.boundingbox[1]),
        ]
      : null,
  }));
}
