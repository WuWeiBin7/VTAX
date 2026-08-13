"use client";

import { Check, Film, FolderPlus, Play, RefreshCw, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";

import { ServerFolderPicker } from "@/components/server-folder-picker";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

export type VideoFileEntry = {
  path: string;
  name: string;
  directory: string;
  bytes: number;
  format: string;
};

export function VideoSourceSelector({
  initialFolders,
  initialSelectedVideos,
  busy,
  onSave,
}: {
  initialFolders: string[];
  initialSelectedVideos: string[];
  busy: boolean;
  onSave: (selection: { source_directories: string[]; selected_videos: string[] }) => void;
}) {
  const [folders, setFolders] = useState(initialFolders);
  const [videos, setVideos] = useState<VideoFileEntry[]>([]);
  const [selected, setSelected] = useState(() => new Set(initialSelectedVideos));
  const [previewPath, setPreviewPath] = useState(initialSelectedVideos[0] ?? "");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [hasLoadedVideos, setHasLoadedVideos] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedVideos = useMemo(
    () => videos.filter((video) => selected.has(video.path)),
    [selected, videos],
  );
  const savedSelectedCount = selected.size;
  const previewVideo = videos.find((video) => video.path === previewPath) ?? videos[0];

  function addFolder(folderPath: string) {
    setFolders((current) => current.includes(folderPath) ? current : [...current, folderPath]);
    setPickerOpen(false);
    setError(null);
  }

  function removeFolder(folderPath: string) {
    setFolders((current) => current.filter((item) => item !== folderPath));
    setSelected((current) => new Set([...current].filter((path) => !isWithinFolder(path, folderPath))));
  }

  async function chooseVideos() {
    if (!folders.length) {
      setError("请先添加至少一个视频文件夹。");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/vtax/api/v1/files/videos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths: folders }),
      });
      const payload = (await response.json()) as {
        videos?: VideoFileEntry[];
        total?: number;
        truncated?: boolean;
        detail?: string;
      };
      if (!response.ok) throw new Error(payload.detail ?? "读取视频文件失败");
      const discovered = payload.videos ?? [];
      setVideos(discovered);
      setSelected((current) => {
        const available = new Set(discovered.map((video) => video.path));
        const retained = [...current].filter((path) => available.has(path));
        return new Set(!hasLoadedVideos && !current.size ? discovered.map((video) => video.path) : retained);
      });
      setHasLoadedVideos(true);
      setPreviewPath(discovered[0]?.path ?? "");
      setSelectorOpen(true);
      if (payload.truncated) setError("视频数量超过 1000，仅显示前 1000 个文件。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "读取视频文件失败");
    } finally {
      setLoading(false);
    }
  }

  function toggleVideo(path: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }

  function save() {
    const selectedPaths = videos.length ? selectedVideos.map((video) => video.path) : [...selected];
    if (!folders.length || !selectedPaths.length) {
      setError("请添加视频文件夹，并至少选择一个进入清洗阶段的视频。");
      return;
    }
    onSave({ source_directories: folders, selected_videos: selectedPaths });
  }

  return (
    <>
      <Card className="border-slate-300 bg-white/95 p-5 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <div className="grid size-9 shrink-0 place-items-center rounded-lg bg-teal-50 text-[#0d9488]"><Film className="size-5" /></div>
            <div><h3 className="font-semibold">视频数据源</h3><p className="text-sm text-slate-500">添加多个文件夹，再选择本次需要进入视频清洗的文件。</p></div>
          </div>
          <Badge variant="outline" className="border-teal-200 bg-teal-50 text-teal-700">已选 {savedSelectedCount} 个视频</Badge>
        </div>

        <div className="mt-5 rounded-xl border border-slate-200 bg-slate-50/70 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div><p className="text-sm font-medium">视频文件夹</p><p className="text-xs text-slate-500">支持保存多个服务器或 WSL 挂载目录。</p></div>
            <div className="flex flex-wrap gap-2">
              <Button variant="secondary" onClick={() => setPickerOpen(true)}><FolderPlus className="mr-2 size-4" />添加文件夹</Button>
              <Button className="bg-[#0d9488] hover:bg-[#0f766e]" onClick={() => void chooseVideos()} disabled={loading || !folders.length}>
                {loading ? <RefreshCw className="mr-2 size-4 animate-spin" /> : <Play className="mr-2 size-4" />}选择视频
              </Button>
            </div>
          </div>
          <div className="mt-3 space-y-2">
            {folders.length ? folders.map((folder) => (
              <div key={folder} className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white px-3 py-2">
                <Film className="size-4 shrink-0 text-[#0d9488]" />
                <span className="min-w-0 flex-1 break-all text-sm text-slate-700">{folder}</span>
                <Button variant="ghost" size="icon" onClick={() => removeFolder(folder)} aria-label={`移除 ${folder}`}><Trash2 className="size-4 text-slate-500" /></Button>
              </div>
            )) : <div className="rounded-lg border border-dashed border-slate-300 bg-white px-4 py-8 text-center text-sm text-slate-500">尚未添加视频文件夹</div>}
          </div>
        </div>

        {error && <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">{error}</div>}
        <div className="mt-5 flex flex-wrap items-center gap-3">
          <Button className="bg-[#0d9488] hover:bg-[#0f766e]" onClick={save} disabled={busy}><Check className="mr-2 size-4" />{busy ? "保存中…" : "保存并用于视频清洗"}</Button>
          <span className="text-xs text-slate-500">只会把已勾选的视频加入下一阶段，原始文件保持只读。</span>
        </div>
      </Card>

      <ServerFolderPicker
        open={pickerOpen}
        purpose="video"
        title="添加视频文件夹"
        description="选择包含视频的目录；可以重复添加多个目录。"
        initialPath={folders.at(-1)}
        onClose={() => setPickerOpen(false)}
        onSelect={(path) => addFolder(path)}
      />

      {selectorOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-4 backdrop-blur-sm">
          <div className="flex h-[min(90dvh,820px)] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-slate-300 bg-[#f8faf5] shadow-2xl">
            <div className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-slate-300 px-5 py-4">
              <div><h2 className="text-lg font-semibold">选择进入清洗阶段的视频</h2><p className="text-sm text-slate-500">共发现 {videos.length} 个视频，已选择 {selected.size} 个。</p></div>
              <div className="flex gap-2"><Button variant="secondary" onClick={() => setSelected(new Set(videos.map((video) => video.path)))}>全选</Button><Button variant="secondary" onClick={() => setSelected(new Set())}>全不选</Button><Button variant="ghost" size="icon" onClick={() => setSelectorOpen(false)}><X className="size-5" /></Button></div>
            </div>
            <div className="grid min-h-0 flex-1 lg:grid-cols-[380px_minmax(0,1fr)]">
              <ScrollArea className="min-h-0 border-r border-slate-300 bg-white">
                <div className="space-y-2 p-3">
                  {videos.length ? videos.map((video) => (
                    <div key={video.path} className={`flex cursor-pointer items-start gap-3 rounded-xl border p-3 transition ${previewVideo?.path === video.path ? "border-[#0d9488] bg-teal-50" : "border-slate-200 bg-white hover:border-teal-300"}`} onClick={() => setPreviewPath(video.path)}>
                      <input type="checkbox" checked={selected.has(video.path)} onChange={() => toggleVideo(video.path)} onClick={(event) => event.stopPropagation()} className="mt-1 size-4 accent-[#0d9488]" aria-label={`选择 ${video.name}`} />
                      <div className="min-w-0"><div className="truncate text-sm font-medium text-slate-900">{video.name}</div><div className="mt-1 truncate text-xs text-slate-500">{video.directory}</div><div className="mt-1 text-xs text-slate-400">{video.format.toUpperCase()} · {formatBytes(video.bytes)}</div></div>
                    </div>
                  )) : <div className="rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500">所选文件夹中没有支持的视频。</div>}
                </div>
              </ScrollArea>
              <div className="flex min-h-0 flex-col bg-slate-950 p-4">
                {previewVideo ? <><div className="mb-3 min-w-0 text-sm text-slate-200"><div className="truncate font-medium">{previewVideo.name}</div><div className="truncate text-xs text-slate-400">{previewVideo.path}</div></div><video key={previewVideo.path} controls preload="metadata" className="min-h-0 w-full flex-1 rounded-xl bg-black object-contain" src={videoUrl(previewVideo.path)}>当前浏览器无法预览此视频格式。</video></> : <div className="grid h-full place-items-center text-sm text-slate-400">选择左侧视频进行预览</div>}
              </div>
            </div>
            <div className="flex shrink-0 items-center justify-between gap-3 border-t border-slate-300 bg-white px-5 py-4"><span className="text-sm text-slate-500">已选择 {selected.size}/{videos.length}</span><Button className="bg-[#0d9488] hover:bg-[#0f766e]" onClick={() => setSelectorOpen(false)}><Check className="mr-2 size-4" />确认选择</Button></div>
          </div>
        </div>
      )}
    </>
  );
}

function videoUrl(path: string) {
  return `/api/vtax/api/v1/files/video?path=${encodeURIComponent(path)}`;
}

function isWithinFolder(filePath: string, folderPath: string) {
  const normalized = folderPath.endsWith("/") ? folderPath : `${folderPath}/`;
  return filePath === folderPath || filePath.startsWith(normalized);
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = value;
  let index = -1;
  do { size /= 1024; index += 1; } while (size >= 1024 && index < units.length - 1);
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[index]}`;
}
