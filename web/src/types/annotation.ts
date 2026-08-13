export type TokenUsage = {
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  cached_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
};

export type ModelProvider = "dashscope" | "ollama";

export type ModelProviderSummary = {
  provider: ModelProvider;
  dashscopeConfigured: boolean;
  ollamaConfigured: boolean;
  ollamaBaseUrl: string | null;
  ollamaVisionModel: string | null;
};

export type LabelMeShape = {
  label: string;
  score?: number;
  points: number[][];
  shape_type: "rectangle" | string;
  group_id?: null;
  description?: string | null;
  difficult?: boolean;
  flags?: Record<string, unknown>;
  attributes?: Record<string, unknown>;
  kie_linking?: unknown[];
};

export type LabelMeAnnotation = {
  version?: string;
  flags?: Record<string, unknown>;
  checked?: boolean;
  imagePath: string;
  imageData?: null;
  imageWidth: number;
  imageHeight: number;
  description?: string;
  shapes: LabelMeShape[];
};

export type FolderImage = {
  path: string;
  name: string;
  annotationPath: string;
  hasAnnotation: boolean;
  width: number | null;
  height: number | null;
  size: number;
  status?: "idle" | "processing" | "done" | "failed" | "skipped";
};

export type FolderScanResponse = {
  inputDir: string;
  images: FolderImage[];
  labelCounts: Record<string, number>;
};

export type ServerFolderEntry = {
  name: string;
  path: string;
};

export type ServerFolderBrowseResponse = {
  roots: ServerFolderEntry[];
  currentPath: string | null;
  parentPath: string | null;
  directories: ServerFolderEntry[];
};

export type ServerFolderValidationResponse = {
  resolvedPath: string;
  exists: boolean;
  isDirectory: boolean;
  readable: boolean;
  writable: boolean;
  imageCount?: number;
};

export type CategoryGuide = {
  label: string;
  visual_cues: string[];
  ocr_cues: string[];
  negative_cues: string[];
  ambiguity_notes: string[];
};

export type JobSummary = {
  total: number;
  processed: number;
  annotated: number;
  skipped: number;
  empty: number;
  failed: number;
  total_seconds: number;
  image_seconds: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  total_cached_tokens: number;
  total_cache_creation_input_tokens: number;
  token_usage_images: number;
  label_counts: Record<string, number>;
};

export type JobEvent = {
  type: string;
  image_path?: string;
  output_path?: string;
  status?: string;
  seconds?: number;
  shapes?: number;
  token_usage?: TokenUsage | null;
  error?: string | null;
  processed?: number;
  annotated?: number;
  empty?: number;
  failed?: number;
  skipped?: number;
  total?: number;
  labels?: string[];
  guides?: CategoryGuide[];
  input_dir?: string;
  output_dir?: string;
  summary?: JobSummary;
};

export type JobState = {
  id: string;
  status: "queued" | "running" | "finished" | "failed" | "cancelled";
  events: JobEvent[];
  summary?: JobSummary;
  error?: string;
};
