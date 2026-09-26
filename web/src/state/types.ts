/**
 * Wire types, mirroring satquery/schemas.py and the Pydantic models in satquery/server.py.
 * Field names match the Python side exactly; change them together.
 */

export type Modality = "optical" | "sar";
export type InputConfig = "single_optical" | "single_sar" | "pair_cross_modal" | "pair_bitemporal";
export type TaskType = "vqa" | "caption" | "grounding" | "change_analysis" | "cross_modal_analysis";
export type Severity = "error" | "warning";
export type StepStatus = "ok" | "failed" | "skipped";
export type ResponseStatus = "ok" | "partial" | "invalid_input" | "error";

export interface ImageSummary {
  index: number;
  name: string;
  modality: Modality;
  width: number;
  height: number;
  bands: string[];
  crs: string | null;
  acquired: string | null;
  decimation: number;
  /** null whenever the image carries no CRS + transform: it cannot be placed on the map. */
  bounds_wgs84: [number, number, number, number] | null;
  corners_wgs84: [number, number][] | null;
  georeference_note: string | null;
}

export interface ValidationIssue {
  code: string;
  severity: Severity;
  message: string;
  image_index: number | null;
}

export interface Intent {
  task: TaskType;
  target: string | null;
  comparative: boolean;
  matched_rule: string;
}

export interface PlanStep {
  step_id: string;
  tool: string;
  image_indices: number[];
  params: Record<string, unknown>;
  purpose: string;
}

export interface Evidence {
  kind: "bbox" | "mask" | "overlay" | "metric";
  label: string;
  image_index: number | null;
  bbox: [number, number, number, number] | null;
  fraction: number | null;
  value: number | string | null;
  /** A /api/runs/... URL once the server has rewritten it. */
  file: string | null;
  source_step: string;
}

export interface Confidence {
  value: number | null;
  /** How the number was produced. Always shown: these values are uncalibrated. */
  method: string;
  calibrated: boolean;
  note: string | null;
}

export interface StepResult {
  step_id: string;
  tool: string;
  model: string | null;
  status: StepStatus;
  params: Record<string, unknown>;
  outputs: Record<string, unknown>;
  evidence: Evidence[];
  confidence: Confidence | null;
  error: string | null;
  duration_s: number;
}

export interface ExecutionTrace {
  run_id: string;
  created_at: string;
  query: string;
  input_config: InputConfig | null;
  images: ImageSummary[];
  validation: ValidationIssue[];
  intent: Intent | null;
  plan: PlanStep[];
  steps: StepResult[];
  total_duration_s: number;
}

export interface AnalysisResponse {
  status: ResponseStatus;
  task: TaskType | null;
  answer: string;
  evidence: Evidence[];
  confidence: Confidence | null;
  trace: ExecutionTrace;
  report_html: string | null;
  report_json: string | null;
}

export interface OverlayLayer {
  url: string;
  label: string;
  image_index: number;
  corners_wgs84: [number, number][];
  /** The source grid is rotated, so the four-corner placement is an approximation. */
  approximate: boolean;
}

/** Whether a drawn area actually narrowed the analysis. Absent when no area was sent. */
export interface AreaScope {
  applied: boolean;
  reason: string | null;
  width: number | null;
  height: number | null;
  source_width: number | null;
  source_height: number | null;
  /** Pixels outside a drawn circle or polygon were excluded, not merely cropped to its box. */
  masked: boolean;
}

export interface AnalyzeResult {
  response: AnalysisResponse;
  overlay_layers: OverlayLayer[];
  area: AreaScope | null;
  /** The uploads this result ran on, in input order. */
  upload_ids: string[];
}

export interface UploadInfo {
  id: string;
  name: string;
  modality: Modality;
  acquired: string | null;
  summary: ImageSummary;
  preview_url: string;
  /** false for PNG/JPEG and CRS-less TIFFs, which are shown off-map instead. */
  mappable: boolean;
  /** How `modality` was decided at upload (e.g. "from band descriptions VV, VH"). Absent for demo
   *  scenarios and retrieved scenes, whose source fixes it. */
  modality_basis?: string | null;
}

export interface Example {
  index: number;
  label: string;
  query: string;
  images: { path: string; modality?: Modality; acquired?: string }[];
}

export interface Health {
  status: "ok";
  vlm_backend: string;
  device: string;
  /** true when a labelled stand-in is answering instead of the real model. */
  model_is_fake: boolean;
  /** Imagery source name when retrieval is configured, else null. Never a credential. */
  imagery_provider: string | null;
  /** Whether imagery can be fetched for a drawn area instead of uploading a GeoTIFF. */
  imagery_available: boolean;
}

/** Provenance for one retrieved scene. Mirrors satquery.providers.SceneMetadata. */
export interface SceneMetadata {
  provider: string;
  collection: string;
  satellite: string;
  product_level: string;
  acquired: string;
  acquired_datetime: string;
  cloud_cover: number | null;
  bbox_wgs84: [number, number, number, number];
  crs: string;
  resolution_m: number;
  bands: string[];
  width: number;
  height: number;
  scene_id: string | null;
  attribution: string;
  cached: boolean;
  alternatives_considered: number;
  /** How the raster was produced (Process API, one day only, reflectance, grid). */
  processing: string;
  /** "sar" for a Sentinel-1 scene retrieved to pair with an optical one. */
  modality?: Modality;
}

/** One retrieved scene, registered as an upload. Mirrors satquery.server.FetchedScene. */
export interface FetchedScene {
  role: "single" | "before" | "after" | "optical" | "sar";
  upload: UploadInfo;
  metadata: SceneMetadata;
}

/** A period searched for one scene of a comparison. Never an acquisition date. */
export interface ComparisonWindow {
  start: string;
  end: string;
  label: string;
}

/** How the two dates of a comparison were chosen. Mirrors satquery.server.TemporalInfo. */
export interface TemporalInfo {
  basis: string;
  explanation: string;
  before_window: ComparisonWindow;
  after_window: ComparisonWindow;
  days_apart: number;
}

/** How the SAR scene was matched to the optical one. Mirrors satquery.server.CrossModalInfo. */
export interface CrossModalInfo {
  days_apart: number;
  max_days_apart: number;
  explanation: string;
}

export interface FetchImageryResult {
  /** "temporal" when the question needs two dates; "cross_modal" when it asks for optical and SAR together. */
  mode: "single" | "temporal" | "cross_modal";
  /** The most recent scene (the optical one of a sensor pair), kept for single-date clients. */
  upload: UploadInfo;
  metadata: SceneMetadata;
  /** Every scene retrieved: oldest first, or optical then SAR. */
  images: FetchedScene[];
  temporal: TemporalInfo | null;
  cross_modal?: CrossModalInfo | null;
  cached: boolean;
}

/** Structured retrieval failure. `code` distinguishes the cause; never contains credentials. */
export interface RetrievalProblem {
  code: string;
  message: string;
  detail?: string;
}
