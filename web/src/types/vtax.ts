export const VTAX_STAGES = [
  "definition",
  "data",
  "cleaning",
  "annotation",
  "review",
  "dataset",
  "configuration",
  "training",
  "export",
] as const;

export type VtaxStageName = (typeof VTAX_STAGES)[number];

export const WORKFLOW_STAGES = [
  "data",
  "cleaning",
  "annotation",
  "dataset",
  "training",
  "export",
] as const satisfies readonly VtaxStageName[];

export type VtaxStageStatus =
  | "locked"
  | "ready"
  | "imported"
  | "queued"
  | "running"
  | "review_required"
  | "completed"
  | "failed"
  | "stale"
  | "cancelled";

export type VtaxStage = {
  project_id: string;
  name: VtaxStageName;
  position: number;
  status: VtaxStageStatus;
  revision: number;
  config: Record<string, unknown>;
  summary: Record<string, unknown>;
  active_job_id: string | null;
  warning: string | null;
  updated_at: string;
};

export type VtaxArtifact = {
  id: string;
  project_id: string;
  stage: VtaxStageName;
  kind: string;
  path: string;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type VtaxProject = {
  id: string;
  name: string;
  description: string;
  task_type: "detection";
  labels: string[];
  start_stage: VtaxStageName;
  workspace_path: string;
  created_at: string;
  updated_at: string;
  stages: VtaxStage[];
  artifacts: VtaxArtifact[];
};

export type VtaxJob = {
  id: string;
  project_id: string;
  stage: VtaxStageName;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted";
  command: string[];
  config: Record<string, unknown>;
  progress: Record<string, unknown>;
  error: string | null;
  log_path: string;
  pid: number | null;
  exit_code: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

export type VtaxHealth = {
  status: "ok" | "degraded";
  workspace_root: string;
  database: boolean;
  gpu: boolean;
  environments: Record<string, boolean>;
  models: Record<string, boolean>;
  disk_free_bytes: number;
  providers: Record<string, boolean>;
};

export const STAGE_META: Record<VtaxStageName, { title: string; short: string; description: string }> = {
  definition: { title: "类别定义", short: "类别", description: "定义检测任务和类别顺序" },
  data: { title: "数据选择", short: "数据", description: "登记新数据、回放数据或已有帧" },
  cleaning: { title: "视频清洗", short: "清洗", description: "抽帧、质量过滤、去重和聚类" },
  annotation: { title: "大模型预标注", short: "预标注", description: "生成检测框并在同一页面完成人工审核" },
  review: { title: "人工审核", short: "审核", description: "逐图核验、调整并确认标注" },
  dataset: { title: "选择数据集", short: "数据集", description: "组合新数据与历史回放数据并生成 YOLO 数据集" },
  configuration: { title: "训练配置", short: "配置", description: "选择 YOLO26 规模和训练参数" },
  training: { title: "模型训练", short: "训练", description: "选择预训练权重、配置参数并实时查看训练进度" },
  export: { title: "模型导出", short: "导出", description: "将上一阶段 best.pt 导出为不同精度的 ONNX" },
};
