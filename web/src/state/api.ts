/** Typed client for satquery/server.py. Relative URLs: Vite proxies them in dev, FastAPI serves
 *  them directly in production. */

import type { AnalyzeResult, Example, Health, Modality, TaskType, UploadInfo } from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
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
    throw new ApiError(await readError(response), response.status);
  }
  return (await response.json()) as T;
}

/** FastAPI puts a string in `detail` for HTTPException and a list for validation errors. */
async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length) return detail.map((d) => d?.msg ?? String(d)).join("; ");
  } catch {
    /* fall through to the status text */
  }
  return response.statusText || `Request failed (${response.status})`;
}

export const api = {
  health: () => request<Health>("/api/health"),

  examples: () => request<Example[]>("/api/examples"),

  exampleQueries: () => request<string[]>("/api/example-queries"),

  loadExample: (index: number) => request<UploadInfo[]>(`/api/examples/${index}/load`, { method: "POST" }),

  upload: (file: File, modality: Modality, acquired?: string | null) => {
    const form = new FormData();
    form.append("file", file);
    form.append("modality", modality);
    if (acquired) form.append("acquired", acquired);
    return request<UploadInfo>("/api/uploads", { method: "POST", body: form });
  },

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
