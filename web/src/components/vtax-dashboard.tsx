"use client";

import { Activity, ArrowRight, Factory, FolderKanban, Plus, ShieldCheck, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { createProject, vtaxRequest } from "@/lib/vtax-client";
import { STAGE_META, WORKFLOW_STAGES, type VtaxHealth, type VtaxProject } from "@/types/vtax";

export function VtaxDashboard() {
  const router = useRouter();
  const [projects, setProjects] = useState<VtaxProject[]>([]);
  const [health, setHealth] = useState<VtaxHealth | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [labels, setLabels] = useState("hand");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void Promise.all([
      vtaxRequest<VtaxProject[]>("/projects").then(setProjects),
      vtaxRequest<VtaxHealth>("/system/health").then(setHealth),
    ]).catch((reason) => setError(reason instanceof Error ? reason.message : "无法连接 VTAX 服务"));
  }, []);

  async function submit() {
    const parsedLabels = labels.split(/[,，\s]+/).map((item) => item.trim()).filter(Boolean);
    if (!name.trim() || !parsedLabels.length) {
      setError("请填写项目名称和至少一个检测类别。");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const project = await createProject({ name: name.trim(), description, labels: parsedLabels, task_type: "detection", start_stage: "data" });
      router.push(`/projects/${project.id}/data`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "创建项目失败");
    } finally {
      setBusy(false);
    }
  }

  async function deleteProject(project: VtaxProject) {
    if (!window.confirm(`确认删除项目“${project.name}”？\n\n项目数据库记录和中间工作区会被删除；LabeledDataset、Weight、ONNX 等永久产物会保留。`)) return;
    setBusy(true);
    setError(null);
    try {
      await vtaxRequest(`/projects/${project.id}`, { method: "DELETE" });
      setProjects((current) => current.filter((item) => item.id !== project.id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "删除项目失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-dvh bg-[#eef1ea] text-slate-900">
      <header className="border-b border-slate-300 bg-[#f7f7f2]/95 px-6 py-4 shadow-sm">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="grid size-11 place-items-center rounded-xl bg-[#0d9488] text-white shadow-sm"><Factory /></div>
            <div><h1 className="text-xl font-semibold tracking-tight">VTAX 视频训标一体化</h1><p className="text-xs text-slate-500">从工厂视频到可部署模型的一站式工作台</p></div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className={health?.gpu ? "border-emerald-300 bg-emerald-50 text-emerald-700" : "border-amber-300 bg-amber-50 text-amber-700"}>
              <Activity className="mr-1 size-3" />GPU {health?.gpu ? "就绪" : "待检查"}
            </Badge>
            <Badge variant="outline" className={health?.status === "ok" ? "border-emerald-300 bg-emerald-50 text-emerald-700" : "border-amber-300 bg-amber-50 text-amber-700"}>
              <ShieldCheck className="mr-1 size-3" />{health?.status === "ok" ? "系统正常" : "环境不完整"}
            </Badge>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1500px] gap-5 p-5 xl:grid-cols-[420px_minmax(0,1fr)]">
        <Card className="h-fit border-slate-300 bg-white/90 p-5 shadow-sm">
          <div className="mb-5 flex items-center gap-2"><Plus className="size-5 text-[#0d9488]" /><h2 className="font-semibold">新建检测项目</h2></div>
          <div className="space-y-4">
            <Field label="项目名称"><Input value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：010 工位手部检测" /></Field>
            <Field label="项目描述"><Input value={description} onChange={(event) => setDescription(event.target.value)} placeholder="可选：场景、工位和目标说明" /></Field>
            <Field label="任务类型"><Input value="目标检测 / rectangle" readOnly className="bg-slate-50 text-slate-600" /></Field>
            <Field label="检测类别（逗号或空格分隔）"><Input value={labels} onChange={(event) => setLabels(event.target.value)} placeholder="hand, glove" /></Field>
            {error && <p className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</p>}
            <Button className="w-full bg-[#0d9488] hover:bg-[#0f766e]" onClick={submit} disabled={busy}>{busy ? "创建中…" : "创建并进入工作流"}<ArrowRight className="ml-2 size-4" /></Button>
          </div>
        </Card>

        <section className="space-y-4">
          <div className="flex items-center justify-between"><div><h2 className="text-lg font-semibold">项目工作区</h2><p className="text-sm text-slate-500">所有阶段状态、日志和模型产物都会持久保存</p></div><Badge variant="secondary">{projects.length} 个项目</Badge></div>
          {projects.length ? (
            <div className="grid gap-3 lg:grid-cols-2">
              {projects.map((project) => {
                const visibleStages = WORKFLOW_STAGES.map((name) => project.stages.find((stage) => stage.name === name)!).filter(Boolean);
                const activeRaw = project.stages.find((stage) => ["ready", "running", "review_required", "failed", "stale"].includes(stage.status));
                const activeName = activeRaw?.name === "review" ? "annotation" : activeRaw?.name === "configuration" ? "training" : activeRaw?.name;
                const active = visibleStages.find((stage) => stage.name === activeName) ?? visibleStages.at(-1)!;
                const review = project.stages.find((stage) => stage.name === "review");
                const completed = visibleStages.filter((stage) => ["completed", "imported"].includes(stage.name === "annotation" ? review?.status ?? stage.status : stage.status)).length;
                return (
                  <div key={project.id} role="button" tabIndex={0} onClick={() => router.push(`/projects/${project.id}/${active.name}`)} onKeyDown={(event) => { if (event.key === "Enter") router.push(`/projects/${project.id}/${active.name}`); }} className="group cursor-pointer rounded-xl border border-slate-300 bg-white/90 p-5 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-[#0d9488] hover:shadow-md">
                    <div className="flex items-start justify-between gap-3"><div><h3 className="font-semibold">{project.name}</h3><p className="mt-1 line-clamp-2 text-sm text-slate-500">{project.description || "目标检测项目"}</p></div><div className="flex items-center gap-1"><Button type="button" variant="ghost" size="icon" className="size-8 text-slate-400 opacity-70 hover:bg-red-50 hover:text-red-600 group-hover:opacity-100" disabled={busy} title="删除项目（永久产物会保留）" onClick={(event) => { event.stopPropagation(); void deleteProject(project); }}><Trash2 className="size-4" /></Button><FolderKanban className="size-5 text-[#0d9488]" /></div></div>
                    <div className="mt-4 flex flex-wrap gap-1">{project.labels.map((label) => <Badge key={label} variant="outline">{label}</Badge>)}</div>
                    <div className="mt-5 flex items-center justify-between text-xs text-slate-500"><span>当前：{STAGE_META[active.name].title}</span><span>{completed}/{WORKFLOW_STAGES.length} 阶段</span></div>
                    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-200"><div className="h-full rounded-full bg-[#0d9488]" style={{ width: `${(completed / WORKFLOW_STAGES.length) * 100}%` }} /></div>
                  </div>
                );
              })}
            </div>
          ) : (
            <Card className="grid min-h-72 place-items-center border-dashed border-slate-300 bg-white/60 p-8 text-center"><div><FolderKanban className="mx-auto mb-3 size-10 text-slate-400" /><p className="font-medium">还没有项目</p><p className="mt-1 text-sm text-slate-500">从左侧创建一个项目，或选择任意阶段导入现有产物。</p></div></Card>
          )}
        </section>
      </div>
    </main>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="space-y-1.5"><Label>{label}</Label>{children}</div>;
}
