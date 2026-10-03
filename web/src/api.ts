export type AssetType = "character" | "prop" | "tile" | "background";
export type View = "side" | "topdown" | "iso";
export type Backend = "local" | "openrouter";

export interface JobSpec {
  prompt: string;
  name: string;
  asset_type: AssetType;
  style: string;
  view: View;
  tile_size: number;
  frame_size: number;
  background_width: number;
  background_height: number;
  parallax_layers: number;
  directions: 1 | 4 | 8;
  iso_depth: number;
  tile_chunk: number | null;
  variants: number;
  candidates: number;
  seed: number | null;
  backend: Backend;
  image_model: string | null;
  enhance_prompt: boolean;
  text_model: string | null;
  vision_model: string | null;
  animate: {
    enabled: boolean;
    action: string;
    custom_motion: string;
    frames: number;
    fps: number;
    loop: boolean;
    backend: Backend;
    model: string | null;
  };
  output: {
    spritesheet: boolean;
    atlas: boolean;
    tilesheet: boolean;
    padding: number;
    extrude: number;
    max_atlas_size: number;
  };
}

export interface Options {
  asset_types: AssetType[];
  views: View[];
  actions: string[];
  styles: Record<string, string>;
  image_profiles: Record<string, string>;
  default_image_profile: string;
  video_profiles: Record<string, string>;
  default_video_profile: string;
  ml_available: boolean;
  defaults: Omit<JobSpec, "prompt">;
}

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface JobRow {
  id: string;
  created: number;
  status: JobStatus;
  progress: number;
  stage: string;
  message: string;
  spec: JobSpec;
  error?: string | null;
}

export interface LoaderEntry {
  loader: "atlas" | "multiatlas" | "spritesheet" | "tileset" | "image";
  key: string;
  texture?: string;
  json?: string;
  textures?: string[];
  frameConfig?: { frameWidth: number; frameHeight: number; margin: number; spacing: number };
  tileWidth?: number;
  tileHeight?: number;
  margin?: number;
  spacing?: number;
  layer?: string;
  scrollFactor?: number;
}

export interface Manifest {
  job_id: string;
  seed: number;
  subject: string;
  cost_usd: number;
  seconds: number;
  log: string[];
  result: {
    kind: "sprites" | "tiles" | "background";
    entries: LoaderEntry[];
    frames?: string[];
    anims?: { key: string; frames: string[] }[];
    anim_files?: string[];
    frame_size?: number;
    tile_width?: number;
    tile_height?: number;
    orientation?: "orthogonal" | "isometric";
    iso_depth?: number;
    chunk?: number;
    variants?: number;
    width?: number;
    height?: number;
  };
}

export interface Job extends JobRow {
  result: Manifest | null;
}

export interface Settings {
  openrouter_key_set: boolean;
  openrouter_key_masked: string | null;
  openrouter_key_source: "file" | "env" | null;
  prefs: Partial<Record<"image_model" | "text_model" | "vision_model" | "video_model", string>>;
}

export interface ORModel {
  id: string;
  name: string;
  description?: string;
  pricing: Record<string, string>;
}

export type ModelKind = "image" | "text" | "vision" | "video";

export type LocalModelStatus = "missing" | "queued" | "downloading" | "ready" | "error";

export interface LocalModel {
  id: string;
  label: string;
  kind: "image" | "video" | "matting";
  size_gb: number | null;
  default: boolean;
  gated: boolean;
  status: LocalModelStatus;
  error: string | null;
  downloaded_bytes: number | null;
}

export interface LocalModels {
  ml_available: boolean;
  models: LocalModel[];
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  options: () => req<Options>("/api/options"),
  health: () => req<{ device: string; ml_available: boolean; current_job: string | null }>("/api/health"),
  settings: () => req<Settings>("/api/settings"),
  saveSettings: (body: { openrouter_api_key?: string; prefs?: Settings["prefs"] }) =>
    req<Settings>("/api/settings", { method: "PUT", body: JSON.stringify(body) }),
  testKey: () => req<{ ok: boolean; label?: string; usage?: number; limit?: number | null }>("/api/settings/test", { method: "POST" }),
  models: (kind: ModelKind, refresh = false) =>
    req<{ models: ORModel[] }>(`/api/openrouter/models?kind=${kind}${refresh ? "&refresh=true" : ""}`).then((r) => r.models),
  localModels: () => req<LocalModels>("/api/local-models"),
  downloadModels: (ids: string[]) =>
    req<{ models: LocalModel[] }>("/api/local-models/download", { method: "POST", body: JSON.stringify({ ids }) }),
  jobs: () => req<JobRow[]>("/api/jobs"),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
  createJob: (spec: JobSpec) => req<{ id: string }>("/api/jobs", { method: "POST", body: JSON.stringify(spec) }),
  cancelJob: (id: string) => req<{ cancelled: boolean }>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  deleteJob: (id: string) => req<{ deleted: boolean }>(`/api/jobs/${id}`, { method: "DELETE" }),
};

export const assetUrl = (jobId: string, file: string) => `/outputs/${jobId}/assets/${file}`;

export function formatPrice(p: Record<string, string>): string {
  const img = p.image_output ?? p.image;
  if (img) return `$${Number(img).toPrecision(2)}/img unit`;
  if (p.prompt) return `$${(Number(p.prompt) * 1e6).toFixed(2)}/M in`;
  const sku = Object.entries(p)[0];
  if (sku) return `${sku[1]} ${sku[0].replace(/_/g, " ")}`;
  return "";
}
