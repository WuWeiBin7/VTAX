"use client";

import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronRight,
  CircleStop,
  Database,
  Download,
  Film,
  Folder,
  Play,
  RefreshCw,
  Settings2,
  SlidersHorizontal,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import { AnnotationWorkspace } from "@/components/annotation-workspace";
import { VideoSourceSelector } from "@/components/video-source-selector";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { completeStage, runStage, vtaxRequest } from "@/lib/vtax-client";
import {
  STAGE_META,
  WORKFLOW_STAGES,
  type VtaxArtifact,
  type VtaxJob,
  type VtaxProject,
  type VtaxStageName,
type VtaxStageStatus,
} from "@/types/vtax";

type LabeledDataset = {
  id: string;
  name: string;
  path: string;
  project_id?: string | null;
  labels: string[];
  image_count: number;
  annotated_count: number;
  updated_at: string;
};

type ModelWeight = {
  id: string;
  name: string;
  path: string;
  source: "project" | "base";
  project_id?: string | null;
  project_name?: string | null;
  kind: string;
  bytes: number;
  updated_at?: string | null;
};

export function ProjectWorkspace({ projectId, stageName }: { projectId: string; stageName: VtaxStageName }) {
  const router = useRouter();
  const [project, setProject] = useState<VtaxProject | null>(null);
  const [job, setJob] = useState<VtaxJob | null>(null);
  const [events, setEvents] = useState<Array<Record<string, unknown>>>([]);
  const [form, setForm] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const stage = project?.stages.find((item) => item.name === stageName);
  const reviewStage = project?.stages.find((item) => item.name === "review");
  const configurationStage = project?.stages.find((item) => item.name === "configuration");
  const artifacts = useMemo(() => project?.artifacts.filter((item) => item.stage === stageName) ?? [], [project, stageName]);

  useEffect(() => {
    void loadProject();
    return () => eventSourceRef.current?.close();
    // Project reload is keyed to the route and selected stage.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, stageName]);

  useEffect(() => {
    if (!stage?.active_job_id) return;
    void loadJob(stage.active_job_id);
    subscribe(stage.active_job_id);
    // The stage's persistent job id is the subscription key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage?.active_job_id]);

  useEffect(() => {
    if (!job?.id || !["queued", "running"].includes(job.status)) return;
    let disposed = false;
    const refresh = async () => {
      try {
        const latest = await vtaxRequest<{ job: VtaxJob | null; events: Array<Record<string, unknown>> }>(
          `/projects/${projectId}/stages/${stageName}/jobs/latest`,
        );
        if (disposed || latest.job?.id !== job.id) return;
        setJob(latest.job);
        setEvents(latest.events);
        if (!["queued", "running"].includes(latest.job.status)) void loadProject();
      } catch {
        // SSE is the fast path; polling is an independent recovery path.
      }
    };
    void refresh();
    const timer = window.setInterval(refresh, 1000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
    // The timer is intentionally restarted only when the active job changes state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id, job?.status, projectId, stageName]);

  async function loadProject() {
    try {
      const value = await vtaxRequest<VtaxProject>(`/projects/${projectId}`);
      setProject(value);
      const current = value.stages.find((item) => item.name === stageName);
      const values = stageName === "data" && current?.summary ? current.summary : current?.config;
      setForm(values ? stringifyForm(values) : {});
      if (isRunnable(stageName)) await loadLatestStageJob();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "加载项目失败");
    }
  }

  async function loadLatestStageJob() {
    try {
      const latest = await vtaxRequest<{ job: VtaxJob | null; events: Array<Record<string, unknown>> }>(
        `/projects/${projectId}/stages/${stageName}/jobs/latest`,
      );
      setJob(latest.job);
      setEvents(latest.events);
      if (latest.job && ["queued", "running"].includes(latest.job.status)) subscribe(latest.job.id);
    } catch {
      setJob(null);
      setEvents([]);
    }
  }

  async function loadJob(jobId: string) {
    try {
      setJob(await vtaxRequest<VtaxJob>(`/jobs/${jobId}`));
    } catch {
      // Project status still remains the source of truth.
    }
  }

  function subscribe(jobId: string) {
    eventSourceRef.current?.close();
    const source = new EventSource(`/api/vtax/api/v1/jobs/${jobId}/events`);
    eventSourceRef.current = source;
    source.onmessage = (message) => {
      const payload = JSON.parse(message.data) as Record<string, unknown>;
      setEvents((current) => [...current.slice(-499), payload]);
      if (payload.type === "process_exit" || payload.type === "job_error" || payload.type === "job_cancelled") {
        void loadJob(jobId);
        void loadProject();
      }
    };
    source.onerror = () => {
      void loadJob(jobId);
      void loadProject();
    };
  }

  async function startJob(configOverride?: Record<string, unknown>) {
    setBusy(true);
    setError(null);
    try {
      const config = configOverride ?? stageConfig(stageName, form);
      if (stageName === "training") {
        await completeStage(projectId, "configuration", config);
      }
      const next = await runStage(projectId, stageName, config);
      setJob(next);
      setEvents([]);
      subscribe(next.id);
      await loadProject();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "启动任务失败");
    } finally {
      setBusy(false);
    }
  }

  async function finishProject() {
    setBusy(true);
    setError(null);
    try {
      await vtaxRequest(`/projects/${projectId}/finish`, { method: "POST" });
      router.push("/");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "完成项目并清理中间数据失败");
    } finally {
      setBusy(false);
    }
  }

  async function stopJob() {
    if (!job) return;
    setBusy(true);
    try {
      setJob(await vtaxRequest<VtaxJob>(`/jobs/${job.id}/cancel`, { method: "POST" }));
      await loadProject();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "停止任务失败");
    } finally {
      setBusy(false);
    }
  }

  async function finish(metadata: Record<string, unknown> = {}, force = false) {
    setBusy(true);
    setError(null);
    try {
      setProject(await completeStage(projectId, stageName, metadata, force));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "完成阶段失败");
    } finally {
      setBusy(false);
    }
  }

  async function finishCombinedReview(folderPath: string, labels: string[]) {
    setBusy(true);
    setError(null);
    try {
      const patched = await vtaxRequest<VtaxProject>(`/projects/${projectId}`, {
        method: "PATCH",
        body: JSON.stringify({ labels }),
      });
      setProject(patched);
      await completeStage(projectId, "annotation", { annotation_dir: folderPath, labels });
      await vtaxRequest<VtaxProject>(`/projects/${projectId}/stages/review/import`, {
        method: "POST",
        body: JSON.stringify({ artifact_type: "labelme", path: folderPath, metadata: { source: "combined_annotation_review" } }),
      });
      const completed = await completeStage(projectId, "review", {}, true);
      setProject(completed);
      router.push(`/projects/${projectId}/dataset`);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "完成审核失败";
      setError(message);
      throw new Error(message);
    } finally {
      setBusy(false);
    }
  }

  if (!project || !stage) {
    return <div className="grid min-h-dvh place-items-center bg-[#eef1ea]"><div className="text-center"><RefreshCw className="mx-auto mb-3 size-7 animate-spin text-[#0d9488]" /><p className="text-sm text-slate-500">加载项目工作区…</p>{error && <p className="mt-2 text-red-600">{error}</p>}</div></div>;
  }

  const progress = progressPercent(job, events);
  const displayedStatus = stageName === "annotation" ? combinedAnnotationStatus(stage.status, reviewStage?.status) : stageName === "training" ? combinedTrainingStatus(configurationStage?.status, stage.status) : stage.status;
  const nextStage = WORKFLOW_STAGES[WORKFLOW_STAGES.indexOf(stageName as (typeof WORKFLOW_STAGES)[number]) + 1];
  const canContinue = stageName !== "annotation" && Boolean(nextStage) && ["completed", "imported"].includes(displayedStatus);
  return (
    <div className="grid h-dvh min-w-0 grid-rows-[auto_minmax(0,1fr)] overflow-hidden bg-[#eef1ea] text-slate-900">
      <header className="border-b border-slate-300 bg-[#f7f7f2] px-4 py-3 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3"><Button asChild variant="ghost" size="icon"><Link href="/"><ArrowLeft className="size-4" /></Link></Button><div><div className="flex items-center gap-2"><h1 className="font-semibold">{project.name}</h1><Badge variant="outline">目标检测</Badge></div><p className="text-xs text-slate-500">{project.labels.join(" · ")} · {project.workspace_path}</p></div></div>
          <StatusBadge status={displayedStatus} />
        </div>
      </header>

      <div className="grid min-h-0 lg:grid-cols-[220px_minmax(0,1fr)]">
        <aside className="border-r border-slate-300 bg-[#f2f3ed] p-3">
          <div className="mb-3 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">工作流阶段</div>
          <nav className="space-y-1">
            {WORKFLOW_STAGES.map((name) => project.stages.find((item) => item.name === name)!).filter(Boolean).map((item, index) => {
              const active = item.name === stageName;
              const itemStatus = item.name === "annotation" ? combinedAnnotationStatus(item.status, reviewStage?.status) : item.name === "training" ? combinedTrainingStatus(configurationStage?.status, item.status) : item.status;
              return (
                <Link key={item.name} href={`/projects/${project.id}/${item.name}`} className={`flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm transition ${active ? "bg-[#0d9488] text-white shadow-sm" : "text-slate-700 hover:bg-white"}`}>
                  <StageDot status={itemStatus} active={active} />
                  <span className="min-w-0 flex-1 truncate">{index + 1}. {STAGE_META[item.name].short}</span>
                  {itemStatus === "running" ? <RefreshCw className="size-3.5 animate-spin" /> : <ChevronRight className="size-3.5 opacity-50" />}
                </Link>
              );
            })}
          </nav>
        </aside>

        <main className={`min-h-0 min-w-0 p-4 xl:p-5 ${stageName === "annotation" ? "overflow-hidden" : "overflow-auto"}`}>
          <div className={`mx-auto max-w-[1280px] ${stageName === "annotation" ? "grid h-full min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-3" : "space-y-4"}`}>
            <div className="flex flex-wrap items-start justify-between gap-3"><div><div className="text-xs font-medium text-[#0d9488]">阶段 {WORKFLOW_STAGES.indexOf(stageName as (typeof WORKFLOW_STAGES)[number]) + 1}/{WORKFLOW_STAGES.length}</div><h2 className="text-2xl font-semibold tracking-tight">{STAGE_META[stageName].title}</h2><p className="mt-1 text-sm text-slate-500">{STAGE_META[stageName].description}</p></div><div className="flex items-center gap-2">{stage.warning && <Badge variant="outline" className="border-amber-300 bg-amber-50 text-amber-800"><AlertTriangle className="mr-1 size-3" />{stage.warning}</Badge>}{canContinue && nextStage ? <Button className="bg-[#0d9488] hover:bg-[#0f766e]" onClick={() => router.push(`/projects/${project.id}/${nextStage}`)}>下一阶段：{STAGE_META[nextStage].short}<ArrowRight className="ml-2 size-4" /></Button> : null}</div></div>

            {stageName === "annotation" ? <CombinedAnnotationStage project={project} busy={busy} onCompleteReview={finishCombinedReview} /> : (
              <>
                <StageForm stageName={stageName} project={project} form={form} setForm={setForm} onComplete={finish} onRun={startJob} onFinishProject={finishProject} job={job} busy={busy} status={stage.status} />
                {isRunnable(stageName) && stageName !== "export" && <JobPanel stageName={stageName} job={job} events={events} progress={progress} busy={busy} onRun={startJob} onStop={stopJob} />}
              </>
            )}

            {error && stageName !== "annotation" && <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div>}
            {stageName !== "annotation" && stageName !== "export" && <ArtifactPanel artifacts={artifacts} />}
          </div>
        </main>
      </div>
    </div>
  );
}

function StageForm({ stageName, project, form, setForm, onComplete, onRun, onFinishProject, job, busy, status }: {
  stageName: VtaxStageName;
  project: VtaxProject;
  form: Record<string, string>;
  setForm: (value: Record<string, string>) => void;
  onComplete: (metadata?: Record<string, unknown>) => void;
  onRun: (config?: Record<string, unknown>) => void;
  onFinishProject: () => void;
  job: VtaxJob | null;
  busy: boolean;
  status: VtaxStageStatus;
}) {
  const set = (key: string, value: string) => setForm({ ...form, [key]: value });
  if (stageName === "data") {
    return <VideoSourceSelector initialFolders={parseStringArray(form.source_directories)} initialSelectedVideos={parseStringArray(form.selected_videos)} busy={busy} onSave={onComplete} />;
  }
  if (stageName === "cleaning") {
    return <CleaningStage project={project} form={form} set={set} onRun={onRun} busy={busy} status={status} />;
  }
  if (stageName === "dataset") {
    return <DatasetSelectionStage project={project} form={form} setForm={setForm} onRun={onRun} busy={busy} status={status} />;
  }
  if (stageName === "training") {
    return <TrainingStage project={project} form={form} setForm={setForm} onRun={onRun} busy={busy} status={status} />;
  }
  if (stageName === "export") {
    return <ExportStage project={project} form={form} setForm={setForm} onRun={onRun} onFinishProject={onFinishProject} job={job} busy={busy} />;
  }
  return null;
}

function CombinedAnnotationStage({ project, busy, onCompleteReview }: {
  project: VtaxProject;
  busy: boolean;
  onCompleteReview: (folderPath: string, labels: string[]) => Promise<void>;
}) {
  const cleaned = [...project.artifacts].reverse().find((artifact) => artifact.stage === "cleaning" && artifact.kind === "cleaned_images");
  if (!cleaned) return <Card className="border-amber-300 bg-amber-50 p-5 text-amber-900"><AlertTriangle className="mb-2 size-5" /><p className="font-medium">尚未找到清洗后的图像帧</p><p className="mt-1 text-sm">请先完成视频清洗阶段，再进入大模型预标注。</p></Card>;
  return <Card className="h-full min-h-0 min-w-0 overflow-hidden border-slate-300 bg-white shadow-sm"><AnnotationWorkspace projectId={project.id} initialFolderPath={cleaned.path} initialLabels={project.labels} reviewMode integratedMode completionBusy={busy} onCompleteReview={onCompleteReview} /></Card>;
}

function CleaningStage({ project, form, set, onRun, busy, status }: {
  project: VtaxProject;
  form: Record<string, string>;
  set: (key: string, value: string) => void;
  onRun: () => void;
  busy: boolean;
  status: VtaxStageStatus;
}) {
  const dataStage = project.stages.find((stage) => stage.name === "data");
  const sourceDirectories = stringArray(dataStage?.summary.source_directories);
  const selectedVideos = stringArray(dataStage?.summary.selected_videos);
  const folders = selectedVideosByFolder(sourceDirectories, selectedVideos);
  const fallbackArtifact = [...project.artifacts].reverse().find((artifact) => artifact.stage === "data" && ["videos", "frames"].includes(artifact.kind));
  const hasInput = selectedVideos.length > 0 || Boolean(fallbackArtifact);
  const running = ["queued", "running"].includes(status);
  const locked = status === "locked";

  return <Card className="overflow-hidden border-slate-300 bg-white/95 shadow-sm">
    <div className="border-b border-slate-200 bg-gradient-to-r from-teal-50 to-white px-5 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3"><div className="grid size-10 shrink-0 place-items-center rounded-xl bg-[#0d9488] text-white"><SlidersHorizontal className="size-5" /></div><div><h3 className="font-semibold">视频清洗任务</h3><p className="mt-0.5 text-sm text-slate-500">对数据选择阶段确认的视频执行抽帧、质量过滤、DINOv3 去重聚类和代表帧筛选。</p></div></div>
        <Badge className="bg-[#0d9488] text-white"><Film className="mr-1 size-3" />{selectedVideos.length} 个视频</Badge>
      </div>
    </div>

    <div className="space-y-5 p-5">
      <section>
        <div className="mb-3 flex items-center justify-between"><div><h4 className="text-sm font-semibold">本次清洗范围</h4><p className="text-xs text-slate-500">视频来源由数据选择阶段锁定，本页不再重复选择路径。</p></div></div>
        <div className="space-y-2">
          {folders.length ? folders.map((folder) => <div key={folder.path} className="flex items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3"><div className="grid size-9 shrink-0 place-items-center rounded-lg bg-white text-[#0d9488] shadow-sm"><Folder className="size-4" /></div><div className="min-w-0 flex-1"><div className="break-all text-sm font-medium text-slate-800">{folder.path}</div><div className="mt-0.5 text-xs text-slate-500">已加入清洗任务</div></div><Badge variant="outline" className="shrink-0 border-teal-200 bg-white text-teal-700">{folder.count} 个视频</Badge></div>) : fallbackArtifact ? <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3"><Folder className="size-5 shrink-0 text-[#0d9488]" /><div className="min-w-0 flex-1"><div className="break-all text-sm font-medium">{fallbackArtifact.path}</div><div className="text-xs text-slate-500">已有上游 {fallbackArtifact.kind === "frames" ? "帧目录" : "视频选择目录"}</div></div></div> : <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-4 text-sm text-amber-800">尚未选择视频。请先返回“数据选择”阶段保存需要清洗的视频。</div>}
        </div>
      </section>

      <section className="rounded-xl border border-slate-200 p-4">
        <div className="mb-4"><h4 className="text-sm font-semibold">清洗参数</h4><p className="text-xs text-slate-500">使用推荐值即可启动；每次运行都会创建新的清洗产物。</p></div>
        <div className="grid gap-4 sm:grid-cols-3"><NumberField label="抽帧间隔（秒）" value={form.interval ?? "0.5"} onChange={(value) => set("interval", value)} /><NumberField label="目标代表帧数" value={form.target_count ?? "1000"} onChange={(value) => set("target_count", value)} /><NumberField label="DINO Batch" value={form.batch_size ?? "16"} onChange={(value) => set("batch_size", value)} /></div>
        <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-500"><span className="rounded-full bg-slate-100 px-3 py-1">视频抽帧</span><span>→</span><span className="rounded-full bg-slate-100 px-3 py-1">质量过滤</span><span>→</span><span className="rounded-full bg-slate-100 px-3 py-1">DINOv3 去重聚类</span><span>→</span><span className="rounded-full bg-slate-100 px-3 py-1">代表帧</span></div>
      </section>

      <div className="flex flex-wrap items-center gap-3"><Button size="lg" className="bg-[#0d9488] px-6 hover:bg-[#0f766e]" onClick={() => onRun()} disabled={busy || running || locked || !hasInput}><Play className="mr-2 size-4" />{running ? "清洗运行中…" : busy ? "提交中…" : "开始清洗"}</Button>{locked && <span className="text-sm text-amber-700">请先完成数据选择阶段。</span>}{!locked && !hasInput && <span className="text-sm text-amber-700">需要先选择至少一个视频。</span>}</div>
    </div>
  </Card>;
}

function DatasetSelectionStage({ project, form, setForm, onRun, busy, status }: {
  project: VtaxProject;
  form: Record<string, string>;
  setForm: (value: Record<string, string>) => void;
  onRun: (config?: Record<string, unknown>) => void;
  busy: boolean;
  status: VtaxStageStatus;
}) {
  const [datasets, setDatasets] = useState<LabeledDataset[]>([]);
  const [selectedReplay, setSelectedReplay] = useState<string[]>(parseStringArray(form.replay_paths));
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const latestLabeled = [...project.artifacts].reverse().find(
    (artifact) => artifact.kind === "labeled_dataset" && artifact.stage === "review",
  );
  const newDataPath = form.new_data_path || latestLabeled?.path || "";
  const seed = form.seed || "42";
  const train = form.train_ratio || "80";
  const val = form.val_ratio || "15";
  const test = form.test_ratio || "5";

  useEffect(() => {
    let active = true;
    void vtaxRequest<{ datasets: LabeledDataset[] }>("/labeled-datasets")
      .then((payload) => {
        if (!active) return;
        setDatasets(payload.datasets);
        if (!newDataPath) {
          const current = payload.datasets.find((dataset) => dataset.project_id === project.id);
          if (current) setForm({ ...form, new_data_path: current.path });
        }
      })
      .catch((reason: unknown) => {
        if (active) setLoadError(reason instanceof Error ? reason.message : "加载已标注数据集失败");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
    // Dataset discovery is tied to the project and latest labeled artifact.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id, latestLabeled?.path]);

  const newDataset = datasets.find((dataset) => dataset.path === newDataPath);
  const replayDatasets = datasets.filter((dataset) => dataset.path !== newDataPath);
  const replayCandidates = replayDatasets
    .filter((dataset) => selectedReplay.includes(dataset.path))
    .reduce((total, dataset) => total + dataset.image_count, 0);
  const replayTarget = Math.min(replayCandidates, Math.round((newDataset?.image_count ?? 0) * 0.4));
  const ratios = [Number(train), Number(val), Number(test)];
  const ratiosValid = ratios.every((value) => Number.isFinite(value) && value >= 0) && Math.abs(ratios.reduce((a, b) => a + b, 0) - 100) < 0.001;
  const running = ["queued", "running"].includes(status);

  function update(key: string, value: string) {
    setForm({ ...form, [key]: value });
  }

  function toggleReplay(path: string, checked: boolean) {
    const next = checked ? [...selectedReplay, path] : selectedReplay.filter((value) => value !== path);
    const unique = [...new Set(next)];
    setSelectedReplay(unique);
    setForm({ ...form, replay_paths: JSON.stringify(unique) });
  }

  function launch() {
    if (!newDataPath || !ratiosValid) return;
    const config = {
      new_data_path: newDataPath,
      replay_paths: selectedReplay,
      replay_ratio: 0.4,
      seed: Number(seed || 42),
      ratios: ratios.map((value) => value / 100),
    };
    setForm({
      ...form,
      new_data_path: newDataPath,
      replay_paths: JSON.stringify(selectedReplay),
      replay_ratio: "0.4",
      seed,
      train_ratio: train,
      val_ratio: val,
      test_ratio: test,
    });
    onRun(config);
  }

  return <Card className="overflow-hidden border-slate-300 bg-white/95 shadow-sm">
    <div className="border-b border-slate-200 bg-gradient-to-r from-teal-50 to-white px-5 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div><h3 className="font-semibold">选择数据集</h3><p className="mt-1 text-sm text-slate-500">以上一阶段固化的数据为新数据，可从历史标注项目中选择回放数据，再统一转换为 YOLO。</p></div>
        <Badge className="bg-[#0d9488] text-white">{datasets.length} 个已标注数据集</Badge>
      </div>
    </div>
    <div className="space-y-5 p-5">
      {loading ? <div className="flex items-center gap-2 text-sm text-slate-500"><RefreshCw className="size-4 animate-spin" />正在读取 LabeledDataset…</div> : null}
      {loadError ? <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{loadError}</div> : null}

      <section>
        <div className="mb-2 flex items-center justify-between"><div><h4 className="text-sm font-semibold">新数据</h4><p className="text-xs text-slate-500">默认使用本项目刚刚完成审核的数据集。</p></div><Badge variant="outline">固定</Badge></div>
        {newDataset ? <DatasetRow dataset={newDataset} selected label="本项目新数据" /> : <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">尚未找到本项目的已标注数据集。请返回预标注页面点击“完成审核”，系统会将结果固化到 LabeledDataset。</div>}
      </section>

      <section>
        <div className="mb-2 flex flex-wrap items-end justify-between gap-3"><div><h4 className="text-sm font-semibold">回放数据</h4><p className="text-xs text-slate-500">可选择 0 到多个历史项目；系统从所选项目整体随机抽取相当于新数据 40% 的图像。</p></div><div className="text-xs text-slate-500">候选 {replayCandidates} 张 · 计划抽取 <span className="font-semibold text-teal-700">{replayTarget} 张</span></div></div>
        <div className="grid gap-2 lg:grid-cols-2">
          {replayDatasets.length ? replayDatasets.map((dataset) => {
            const compatible = !dataset.labels.length || dataset.labels.join("\u0000") === project.labels.join("\u0000");
            const checked = selectedReplay.includes(dataset.path);
            return <label key={dataset.path} className={`block rounded-xl border p-3 transition ${checked ? "border-teal-400 bg-teal-50" : "border-slate-200 bg-slate-50"} ${compatible ? "cursor-pointer" : "cursor-not-allowed opacity-60"}`}>
              <div className="flex items-start gap-3"><input className="mt-1 size-4 accent-teal-600" type="checkbox" checked={checked} disabled={!compatible || running} onChange={(event) => toggleReplay(dataset.path, event.target.checked)} /><div className="min-w-0 flex-1"><DatasetRow dataset={dataset} compact />{!compatible ? <p className="mt-1 text-xs text-amber-700">类别与当前项目不一致，不能作为回放数据。</p> : null}</div></div>
            </label>;
          }) : <div className="rounded-xl border border-dashed border-slate-300 p-4 text-sm text-slate-500">暂无其他已标注数据集，可以只使用新数据启动。</div>}
        </div>
      </section>

      <section className="rounded-xl border border-slate-200 bg-slate-50 p-4">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          <NumberField label="随机种子" value={seed} onChange={(value) => update("seed", value)} />
          <NumberField label="训练集 %" value={train} onChange={(value) => update("train_ratio", value)} />
          <NumberField label="验证集 %" value={val} onChange={(value) => update("val_ratio", value)} />
          <NumberField label="测试集 %" value={test} onChange={(value) => update("test_ratio", value)} />
          <ReadOnly label="回放抽取比例" value="新数据的 40%" />
        </div>
        {!ratiosValid ? <p className="mt-2 text-sm text-red-600">训练、验证、测试比例之和必须为 100%。</p> : null}
      </section>

      <div className="flex flex-wrap items-center gap-3"><Button size="lg" className="bg-[#0d9488] px-6 hover:bg-[#0f766e]" onClick={launch} disabled={busy || running || !newDataset || !ratiosValid}><Play className="mr-2 size-4" />{running ? "正在生成 YOLO 数据集…" : busy ? "提交中…" : "启动本阶段"}</Button><span className="text-xs text-slate-500">新数据 {newDataset?.image_count ?? 0} 张 + 回放 {replayTarget} 张，之后按设定比例随机切分。</span></div>
    </div>
  </Card>;
}

function DatasetRow({ dataset, selected = false, label, compact = false }: { dataset: LabeledDataset; selected?: boolean; label?: string; compact?: boolean }) {
  return <div className={`min-w-0 ${compact ? "" : "rounded-xl border border-slate-200 bg-slate-50 p-4"}`}>
    <div className="flex flex-wrap items-center gap-2"><span className="font-medium text-slate-800">{dataset.name}</span>{label ? <Badge className="bg-teal-600 text-white">{label}</Badge> : null}{selected ? <Check className="size-4 text-teal-600" /> : null}</div>
    <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500"><span>{dataset.image_count} 张图像</span><span>{dataset.annotated_count} 个 JSON</span><span>{dataset.labels.length ? dataset.labels.join("、") : "类别未知"}</span></div>
    {!compact ? <div className="mt-1 truncate text-xs text-slate-400" title={dataset.path}>{dataset.path}</div> : null}
  </div>;
}

function TrainingStage({ project, form, setForm, onRun, busy, status }: {
  project: VtaxProject;
  form: Record<string, string>;
  setForm: (value: Record<string, string>) => void;
  onRun: (config?: Record<string, unknown>) => void;
  busy: boolean;
  status: VtaxStageStatus;
}) {
  const [weights, setWeights] = useState<ModelWeight[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const modelPath = form.model || "";
  const selected = weights.find((weight) => weight.path === modelPath);
  const dataset = [...project.artifacts].reverse().find((artifact) => artifact.kind === "dataset_yaml");
  const running = ["queued", "running"].includes(status);

  useEffect(() => {
    let active = true;
    void vtaxRequest<{ weights: ModelWeight[] }>("/weights")
      .then((payload) => {
        if (!active) return;
        setWeights(payload.weights);
        if (!modelPath) {
          const recommended = payload.weights.find((weight) => weight.id === "base:yolo26m") ?? payload.weights[0];
          if (recommended) setForm({ ...form, model: recommended.path, amp: form.amp || "false" });
        }
      })
      .catch((reason: unknown) => {
        if (active) setLoadError(reason instanceof Error ? reason.message : "加载模型权重失败");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
    // Weight registry is loaded once for the selected project.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id]);

  function set(key: string, value: string) {
    setForm({ ...form, [key]: value });
  }

  function launch() {
    if (!modelPath || !dataset) return;
    onRun({
      model: modelPath,
      epochs: Number(form.epochs || 80), imgsz: Number(form.imgsz || 960),
      batch: Number(form.batch || 8), workers: Number(form.workers || 8),
      patience: Number(form.patience || 25), device: form.device || "0",
      optimizer: "auto", seed: 42, amp: form.amp === "true",
    });
  }

  return <Card className="overflow-hidden border-slate-300 bg-white/95 shadow-sm">
    <div className="border-b border-slate-200 bg-gradient-to-r from-teal-50 to-white px-5 py-4"><div className="flex items-start gap-3"><div className="grid size-10 place-items-center rounded-xl bg-[#0d9488] text-white"><Settings2 className="size-5" /></div><div><h3 className="font-semibold">模型训练</h3><p className="mt-1 text-sm text-slate-500">选择系统基础权重或所有项目永久保存的训练权重，并直接启动 YOLO26。</p></div></div></div>
    <div className="space-y-5 p-5">
      {loading ? <div className="flex items-center gap-2 text-sm text-slate-500"><RefreshCw className="size-4 animate-spin" />正在读取 Weight…</div> : null}
      {loadError ? <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{loadError}</div> : null}
      <div className="grid gap-4 lg:grid-cols-[minmax(300px,1fr)_minmax(300px,1fr)]">
        <div className="space-y-1.5"><Label>预训练权重</Label><Select value={modelPath} onValueChange={(value) => set("model", value)}><SelectTrigger className="w-full"><SelectValue placeholder="选择一个预训练权重" /></SelectTrigger><SelectContent>{weights.map((weight) => <SelectItem key={weight.id} value={weight.path}>{weight.name}{weight.project_name ? ` · ${weight.project_name}` : ""}</SelectItem>)}</SelectContent></Select></div>
        <div className="rounded-xl border border-teal-200 bg-teal-50 px-4 py-3"><div className="text-xs font-medium text-teal-700">当前使用的预训练权重</div><div className="mt-1 font-semibold text-slate-900">{selected?.name ?? "尚未选择"}</div><div className="mt-1 truncate text-xs text-slate-500" title={modelPath}>{modelPath || "-"}</div></div>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <NumberField label="Epoch" value={form.epochs || "80"} onChange={(value) => set("epochs", value)} />
        <NumberField label="图像尺寸" value={form.imgsz || "960"} onChange={(value) => set("imgsz", value)} />
        <NumberField label="Batch" value={form.batch || "8"} onChange={(value) => set("batch", value)} />
        <NumberField label="Workers" value={form.workers || "8"} onChange={(value) => set("workers", value)} />
        <NumberField label="Patience" value={form.patience || "25"} onChange={(value) => set("patience", value)} />
        <TextField label="设备" value={form.device || "0"} onChange={(value) => set("device", value)} />
        <SelectField label="混合精度 AMP" value={form.amp || "false"} onChange={(value) => set("amp", value)} options={[["false", "关闭（默认 / FP32）"], ["true", "开启 AMP"]]} />
        <ReadOnly label="训练数据集" value={dataset ? dataset.path.split("/").at(-2) || "上一阶段 YOLO 数据集" : "尚未生成数据集"} />
      </div>
      <div className="flex flex-wrap items-center gap-3"><Button size="lg" className="bg-[#0d9488] px-6 hover:bg-[#0f766e]" onClick={launch} disabled={busy || running || !modelPath || !dataset}><Play className="mr-2 size-4" />{running ? "模型训练中…" : busy ? "提交中…" : "开始模型训练"}</Button><span className="text-xs text-slate-500">数据集自动使用上一阶段产物；训练日志和 CMD 指标显示在下方任务监控。</span></div>
    </div>
  </Card>;
}

function ExportStage({ project, form, setForm, onRun, onFinishProject, job, busy }: {
  project: VtaxProject;
  form: Record<string, string>;
  setForm: (value: Record<string, string>) => void;
  onRun: (config?: Record<string, unknown>) => void;
  onFinishProject: () => void;
  job: VtaxJob | null;
  busy: boolean;
}) {
  const best = [...project.artifacts].reverse().find((artifact) => artifact.kind === "best_pt");
  const exported = [...project.artifacts].reverse().find((artifact) => artifact.kind === "onnx");
  const weights = form.weights || best?.path || "";
  const precision = form.precision || "fp32";
  const running = job?.status === "queued" || job?.status === "running";
  const succeeded = job?.status === "completed" && Boolean(exported);

  function set(key: string, value: string) {
    setForm({ ...form, [key]: value });
  }

  function launch() {
    if (!weights) return;
    const config = { weights, precision, imgsz: Number(form.imgsz || 960), device: form.device || "0" };
    setForm({ ...form, weights, precision, imgsz: form.imgsz || "960", device: form.device || "0" });
    onRun(config);
  }

  return <Card className="overflow-hidden border-slate-300 bg-white/95 shadow-sm">
    <div className="border-b border-slate-200 bg-gradient-to-r from-teal-50 to-white px-5 py-4"><div><h3 className="font-semibold">ONNX 模型导出</h3><p className="mt-1 text-sm text-slate-500">默认使用上一阶段永久保存的 best.pt，选择精度后导出到工作目录 ONNX。</p></div></div>
    <div className="space-y-5 p-5">
      <div className="rounded-xl border border-teal-200 bg-teal-50 p-4"><div className="text-xs font-medium text-teal-700">上一阶段 best.pt</div><div className="mt-1 font-semibold">{best?.path.split("/").at(-1) ?? "尚未生成"}</div><div className="mt-1 break-all text-xs text-slate-500">{weights || "请先完成模型训练"}</div></div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="space-y-1.5"><Label>ONNX 精度</Label><Select value={precision} onValueChange={(value) => set("precision", value)}><SelectTrigger className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="fp32">FP32 · 兼容性最高</SelectItem><SelectItem value="fp16">FP16 · 体积更小</SelectItem><SelectItem value="int8">INT8 · 动态量化</SelectItem></SelectContent></Select></div>
        <NumberField label="导出图像尺寸" value={form.imgsz || "960"} onChange={(value) => set("imgsz", value)} />
        <TextField label="设备" value={form.device || "0"} onChange={(value) => set("device", value)} />
        <ReadOnly label="永久目录" value="/root/WWB/VTAX/ONNX" />
      </div>
      <div className="flex flex-wrap items-center gap-3"><Button size="lg" className="bg-[#0d9488] px-6 hover:bg-[#0f766e]" onClick={launch} disabled={busy || running || !weights}><Play className="mr-2 size-4" />{running ? "正在导出…" : "导出"}</Button>{job?.status === "failed" ? <span className="text-sm text-red-600">{job.error || "导出失败"}</span> : null}</div>
      {succeeded ? <div className="rounded-xl border border-emerald-300 bg-emerald-50 p-4"><div className="flex flex-wrap items-center justify-between gap-3"><div><div className="flex items-center gap-2 font-semibold text-emerald-800"><Check className="size-5" />模型导出成功</div><div className="mt-1 break-all text-sm text-emerald-700">{exported?.path}</div><div className="mt-1 text-xs text-slate-500">点击完成退出后，将清理项目工作区中的帧、清洗结果、训练数据、运行目录和日志，保留 LabeledDataset、Weight、ONNX、类别指南及原始视频。</div></div><Button className="bg-emerald-600 hover:bg-emerald-700" onClick={onFinishProject} disabled={busy}><Check className="mr-2 size-4" />{busy ? "清理中…" : "完成退出"}</Button></div></div> : null}
    </div>
  </Card>;
}

function JobPanel({ stageName, job, events, progress, busy, onRun, onStop }: { stageName: VtaxStageName; job: VtaxJob | null; events: Array<Record<string, unknown>>; progress: number; busy: boolean; onRun: (config?: Record<string, unknown>) => void; onStop: () => void }) {
  const running = job?.status === "running" || job?.status === "queued";
  return <Card className="border-slate-300 bg-[#101827] p-4 text-slate-100 shadow-sm">
    <div className="mb-3 flex flex-wrap items-center justify-between gap-3"><div><div className="flex items-center gap-2"><Activity className="size-4 text-teal-400" /><h3 className="font-medium">任务监控</h3>{job && <Badge variant="outline" className="border-slate-600 text-slate-300">{job.status}</Badge>}</div><p className="mt-1 text-xs text-slate-400">{job ? `Job ${job.id.slice(0, 12)} · ${job.log_path}` : "尚未运行"}</p></div><div className="flex gap-2">{running ? <Button variant="destructive" size="sm" onClick={onStop} disabled={busy}><CircleStop className="mr-1 size-4" />停止</Button> : job && <Button size="sm" variant="secondary" onClick={() => onRun()}><RefreshCw className="mr-1 size-4" />创建新运行</Button>}</div></div>
    <Progress value={progress} className="mb-3" />
    {stageName === "cleaning" ? <CleaningProgress job={job} events={events} /> : null}
    {stageName === "training" ? <TrainingProgress job={job} events={events} /> : null}
    {job?.status === "completed" && Object.keys(job.progress).length ? <JobSummary stageName={stageName} summary={job.progress} /> : null}
    <div className="max-h-80 min-h-48 overflow-auto rounded-md border border-slate-700 bg-black/30 p-3 font-mono text-xs leading-5 text-slate-300">{events.length ? events.map((event, index) => <div key={`${String(event.sequence ?? index)}-${index}`}><span className="text-teal-400">[{String(event.type ?? "log")}]</span> {eventText(event)}</div>) : <div className="text-slate-500">实时标准输出和结构化进度会显示在这里。</div>}</div>
    {job?.error && <p className="mt-3 text-sm text-red-300">{job.error}</p>}
  </Card>;
}

function TrainingProgress({ job, events }: { job: VtaxJob | null; events: Array<Record<string, unknown>> }) {
  const latest = [...events].reverse().find((event) => ["training_progress", "training_epoch"].includes(String(event.type))) ?? job?.progress ?? {};
  const metrics = [...events].reverse().find((event) => event.type === "training_metrics") ?? latest;
  const epoch = Number(latest.epoch ?? latest.current ?? 0);
  const epochs = Number(latest.epochs ?? latest.total ?? job?.config.epochs ?? 0);
  const fields = [
    ["训练轮次", epochs ? `${epoch}/${epochs}` : "等待首轮"],
    ["Box Loss", metricValue(latest.box_loss)], ["Cls Loss", metricValue(latest.cls_loss)],
    ["DFL Loss", metricValue(latest.dfl_loss)], ["Precision", metricValue(metrics.precision)],
    ["Recall", metricValue(metrics.recall)], ["mAP50", metricValue(metrics.map50)],
    ["mAP50-95", metricValue(metrics.map50_95)], ["GPU 显存", latest.gpu_memory_gb != null ? `${latest.gpu_memory_gb} GB` : "-"],
  ];
  return <div className="mb-3 grid gap-2 sm:grid-cols-3 xl:grid-cols-5">{fields.map(([label, value]) => <div key={label} className="rounded-md border border-slate-700 bg-slate-900/60 px-3 py-2"><div className="text-[11px] text-slate-500">{label}</div><div className="mt-1 font-mono text-sm text-cyan-200">{value}</div></div>)}</div>;
}

function JobSummary({ stageName, summary }: { stageName: VtaxStageName; summary: Record<string, unknown> }) {
  return <div className="mb-3 rounded-md border border-emerald-700 bg-emerald-950/35 p-3"><div className="mb-2 flex items-center gap-2 text-sm font-medium text-emerald-200"><Check className="size-4" />{stageName === "training" ? "训练完成，最终产物与信息" : "清洗完成，最终统计"}</div><pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all text-xs leading-5 text-emerald-100/80">{JSON.stringify(summary, null, 2)}</pre></div>;
}

function CleaningProgress({ job, events }: { job: VtaxJob | null; events: Array<Record<string, unknown>> }) {
  const phases = [
    ["extract", "视频抽帧"], ["discover", "帧发现"], ["quality", "质量过滤"],
    ["features", "DINOv3 特征"], ["dedup", "去重"], ["cluster", "聚类"],
    ["select", "代表帧选择"], ["materialize", "固化结果"], ["report", "生成报告"],
  ] as const;
  const order = new Map<string, number>(phases.map(([key], index) => [key, index]));
  const latest = [...events].reverse().find((event) => event.type === "cleaning_progress");
  const current = order.get(String(latest?.phase ?? "")) ?? -1;
  const completed = job?.status === "completed";
  return <div className="mb-3 grid gap-2 sm:grid-cols-3 xl:grid-cols-5">
    {phases.map(([key, label], index) => {
      const done = completed || index < current;
      const active = !completed && index === current;
      return <div key={key} className={`rounded-md border px-3 py-2 text-xs ${done ? "border-teal-700 bg-teal-950/50 text-teal-200" : active ? "border-cyan-400 bg-cyan-950/50 text-cyan-100" : "border-slate-700 bg-slate-900/50 text-slate-500"}`}><div className="flex items-center gap-2">{done ? <Check className="size-3.5" /> : active ? <RefreshCw className="size-3.5 animate-spin" /> : <span className="size-3.5 rounded-full border border-current" />}<span>{label}</span></div>{active && latest?.message ? <div className="mt-1 truncate opacity-75" title={String(latest.message)}>{String(latest.message)}</div> : null}</div>;
    })}
  </div>;
}

function ArtifactPanel({ artifacts }: { artifacts: VtaxArtifact[] }) {
  if (!artifacts.length) return null;
  return <Card className="border-slate-300 bg-white p-4"><div className="mb-3 flex items-center gap-2"><Database className="size-4 text-[#0d9488]" /><h3 className="font-medium">本阶段产物</h3></div><div className="space-y-2">{artifacts.map((artifact) => <div key={artifact.id} className="flex items-center justify-between gap-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm"><div className="min-w-0"><Badge variant="outline" className="mr-2">{artifact.kind}</Badge><span className="break-all text-slate-600">{artifact.path}</span></div>{["best_pt", "onnx", "last_pt", "dataset_yaml"].includes(artifact.kind) && <Button asChild size="sm" variant="ghost"><a href={`/api/vtax/api/v1/artifacts/${artifact.id}/download`}><Download className="size-4" /></a></Button>}</div>)}</div></Card>;
}

function TextField({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (value: string) => void; placeholder?: string }) { return <div className="space-y-1.5"><Label>{label}</Label><Input value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} /></div>; }
function NumberField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) { return <div className="space-y-1.5"><Label>{label}</Label><Input type="number" value={value} onChange={(event) => onChange(event.target.value)} /></div>; }
function ReadOnly({ label, value }: { label: string; value: string }) { return <div className="space-y-1.5"><Label>{label}</Label><div className="min-h-9 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">{value}</div></div>; }
function SelectField({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: string[][] }) { return <div className="space-y-1.5"><Label>{label}</Label><Select value={value} onValueChange={onChange}><SelectTrigger className="w-full"><SelectValue /></SelectTrigger><SelectContent>{options.map(([key, text]) => <SelectItem key={key} value={key}>{text}</SelectItem>)}</SelectContent></Select></div>; }
function StatusBadge({ status }: { status: VtaxStageStatus }) { const labels: Record<VtaxStageStatus, string> = { locked: "未解锁", ready: "可开始", imported: "已导入", queued: "排队中", running: "运行中", review_required: "待审核", completed: "已完成", failed: "失败", stale: "需重跑", cancelled: "已取消" }; return <Badge className={statusColor(status)}>{labels[status]}</Badge>; }
function StageDot({ status, active }: { status: VtaxStageStatus; active: boolean }) { return <span className={`grid size-5 shrink-0 place-items-center rounded-full border text-[10px] ${active ? "border-white/50 bg-white/15" : ["completed", "imported"].includes(status) ? "border-emerald-300 bg-emerald-100 text-emerald-700" : "border-slate-300 bg-white text-slate-400"}`}>{["completed", "imported"].includes(status) ? <Check className="size-3" /> : ""}</span>; }

function statusColor(status: VtaxStageStatus) { if (["completed", "imported"].includes(status)) return "bg-emerald-600 text-white"; if (["failed", "stale", "cancelled"].includes(status)) return "bg-amber-600 text-white"; if (["running", "queued"].includes(status)) return "bg-blue-600 text-white"; if (status === "review_required") return "bg-[#f28c18] text-white"; return "bg-slate-600 text-white"; }
function isRunnable(stage: VtaxStageName) { return ["cleaning", "dataset", "training", "export"].includes(stage); }
function combinedAnnotationStatus(annotation: VtaxStageStatus, review?: VtaxStageStatus): VtaxStageStatus { if (["ready", "queued", "running", "failed", "stale", "cancelled"].includes(annotation)) return annotation; if (review && review !== "locked") return review; return annotation; }
function combinedTrainingStatus(configuration: VtaxStageStatus | undefined, training: VtaxStageStatus): VtaxStageStatus { if (training !== "locked") return training; if (["ready", "stale", "failed", "cancelled"].includes(configuration ?? "locked")) return configuration as VtaxStageStatus; if (["completed", "imported"].includes(configuration ?? "locked")) return "ready"; return training; }
function stringifyForm(config: Record<string, unknown>) { return Object.fromEntries(Object.entries(config).map(([key, value]) => [key, Array.isArray(value) && ["source_directories", "selected_videos", "replay_paths"].includes(key) ? JSON.stringify(value) : Array.isArray(value) ? value.join(",") : value == null ? "" : String(value)])); }
function parseStringArray(value?: string) { if (!value) return []; try { const parsed = JSON.parse(value) as unknown; return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : []; } catch { return []; } }
function stringArray(value: unknown): string[] { return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : []; }
function selectedVideosByFolder(sourceDirectories: string[], selectedVideos: string[]) { const counts = new Map(sourceDirectories.map((path) => [path, 0])); const ordered = [...sourceDirectories].sort((left, right) => right.length - left.length); for (const video of selectedVideos) { const folder = ordered.find((path) => video === path || video.startsWith(path.endsWith("/") ? path : `${path}/`)); if (folder) counts.set(folder, (counts.get(folder) ?? 0) + 1); } return sourceDirectories.map((path) => ({ path, count: counts.get(path) ?? 0 })); }
function stageConfig(stage: VtaxStageName, form: Record<string, string>): Record<string, unknown> { const compact = Object.fromEntries(Object.entries(form).filter(([, value]) => value !== "")); if (stage === "cleaning") { const automatic = Object.fromEntries(Object.entries(compact).filter(([key]) => !["input_path", "max_images"].includes(key))); return { ...automatic, interval: Number(form.interval || 0.5), target_count: Number(form.target_count || 1000), batch_size: Number(form.batch_size || 16) }; } if (stage === "dataset") return { new_data_path: form.new_data_path, replay_paths: parseStringArray(form.replay_paths), replay_ratio: 0.4, seed: Number(form.seed || 42), ratios: [Number(form.train_ratio || 80) / 100, Number(form.val_ratio || 15) / 100, Number(form.test_ratio || 5) / 100] }; return compact; }
function eventText(event: Record<string, unknown>) { if (event.message) return String(event.message); if (event.error) return String(event.error); if (event.summary) return JSON.stringify(event.summary); const ignored = new Set(["type", "sequence", "created_at"]); return Object.entries(event).filter(([key]) => !ignored.has(key)).map(([key, value]) => `${key}=${typeof value === "object" ? JSON.stringify(value) : String(value)}`).join(" "); }
function metricValue(value: unknown) { const number = Number(value); return Number.isFinite(number) ? number.toFixed(4) : "-"; }
function progressPercent(job: VtaxJob | null, events: Array<Record<string, unknown>>) { const last = [...events].reverse().find((event) => event.percent != null || event.current != null || event.processed != null); const percent = Number(last?.percent ?? job?.progress.percent ?? NaN); if (Number.isFinite(percent)) return Math.min(100, Math.max(0, Math.round(percent))); const current = Number(last?.current ?? last?.processed ?? job?.progress.current ?? 0); const total = Number(last?.total ?? job?.progress.total ?? 0); if (total > 0) return Math.min(100, Math.round((current / total) * 100)); if (job?.status === "completed") return 100; return job?.status === "running" ? 8 : 0; }
