"use client";

import { ArrowUp, Folder, FolderOpen, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import type {
  ServerFolderBrowseResponse,
  ServerFolderValidationResponse,
} from "@/types/annotation";

const RECENT_LIMIT = 8;

export function ServerFolderPicker({
  open,
  purpose,
  title,
  description,
  initialPath,
  onClose,
  onSelect,
}: {
  open: boolean;
  purpose: "annotation" | "video";
  title: string;
  description: string;
  initialPath?: string;
  onClose: () => void;
  onSelect: (path: string, details: ServerFolderValidationResponse) => void;
}) {
  const [browseState, setBrowseState] = useState<ServerFolderBrowseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [manualPath, setManualPath] = useState("");
  const [recentPaths, setRecentPaths] = useState<string[]>([]);

  const currentPath = browseState?.currentPath ?? "";

  useEffect(() => {
    if (!open) return;
    const startPath = initialPath?.trim() ?? "";
    // Dialog-local fields are reset whenever the picker is opened for a new path.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setManualPath(startPath);
    setRecentPaths(loadRecentFolders(purpose));
    void loadFolders(startPath || undefined);
  }, [open, purpose, initialPath]);

  const visibleEntries = useMemo(() => {
    if (!browseState) return [];
    if (browseState.currentPath) return browseState.directories;
    return browseState.roots;
  }, [browseState]);

  if (!open) return null;

  async function loadFolders(targetPath?: string) {
    setLoading(true);
    setError(null);
    try {
      const query = targetPath ? `?path=${encodeURIComponent(targetPath)}` : "";
      const response = await fetch(`/api/server-folders${query}`, { cache: "no-store" });
      const payload = (await response.json()) as ServerFolderBrowseResponse & { error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? "Failed to browse folders");
      }
      setBrowseState(payload);
      setManualPath(payload.currentPath ?? "");
    } catch (browseError) {
      setError(browseError instanceof Error ? browseError.message : "Failed to browse folders");
    } finally {
      setLoading(false);
    }
  }

  async function validateAndSelect(targetPath: string) {
    const trimmed = targetPath.trim();
    if (!trimmed) {
      setError("Please choose a server folder first.");
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const response = await fetch("/api/server-folders", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ path: trimmed, purpose }),
      });
      const payload = (await response.json()) as ServerFolderValidationResponse & { error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? "Failed to validate folder");
      }
      pushRecentFolder(purpose, payload.resolvedPath);
      onSelect(payload.resolvedPath, payload);
    } catch (validationError) {
      setError(validationError instanceof Error ? validationError.message : "Failed to validate folder");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4 backdrop-blur-sm">
      <div className="flex h-[min(88dvh,760px)] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-slate-300 bg-[#f8faf5] shadow-2xl">
        <div className="shrink-0 border-b border-slate-300 px-5 py-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-950">{title}</h2>
              <p className="mt-1 text-sm text-slate-600">{description}</p>
            </div>
            <Button variant="secondary" onClick={onClose}>
              取消
            </Button>
          </div>
        </div>

        <div className="grid min-h-0 flex-1 gap-4 p-5 lg:grid-cols-[minmax(0,1fr)_280px]">
          <div className="flex min-h-0 flex-col gap-3">
            <div className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-200 bg-white p-3">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => void loadFolders(browseState?.parentPath ?? undefined)}
                disabled={loading || !browseState?.parentPath}
              >
                <ArrowUp className="mr-2 h-4 w-4" />
                上一级
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => void loadFolders(currentPath || undefined)}
                disabled={loading}
              >
                <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
                刷新
              </Button>
              <Input
                value={manualPath}
                onChange={(event) => setManualPath(event.target.value)}
                placeholder="服务器文件夹路径"
                className="min-w-[220px] flex-1"
              />
              <Button variant="outline" size="sm" onClick={() => void loadFolders(manualPath)} disabled={loading}>
                打开
              </Button>
              <Button
                className="bg-[#0d9488] text-white hover:bg-[#0f766e]"
                size="sm"
                onClick={() => void validateAndSelect(manualPath || currentPath)}
                disabled={submitting}
              >
                <FolderOpen className="mr-2 h-4 w-4" />
                {submitting ? "选择中…" : "选择当前文件夹"}
              </Button>
            </div>

            <Card className="min-h-0 flex-1 overflow-hidden border-slate-200 bg-white">
              <div className="border-b border-slate-200 px-4 py-3 text-sm font-medium text-slate-900">
                {browseState?.currentPath ? `当前位置：${browseState.currentPath}` : "选择根目录"}
              </div>
              <ScrollArea className="min-h-0 h-full">
                <div className="space-y-2 p-3">
                  {visibleEntries.length ? (
                    visibleEntries.map((entry) => (
                      <button
                        key={entry.path}
                        type="button"
                        onClick={() => void loadFolders(entry.path)}
                        className="flex w-full items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 text-left transition hover:border-[#0d9488] hover:bg-[#eefcf8]"
                      >
                        <Folder className="h-4 w-4 shrink-0 text-[#0d9488]" />
                        <div className="min-w-0">
                          <div className="truncate font-medium text-slate-900">{entry.name}</div>
                          <div className="truncate text-xs text-slate-500">{entry.path}</div>
                        </div>
                      </button>
                    ))
                  ) : (
                    <div className="rounded-xl border border-dashed border-slate-200 px-4 py-6 text-center text-sm text-slate-500">
                      {loading ? "正在读取文件夹…" : "当前位置没有子文件夹。"}
                    </div>
                  )}
                </div>
              </ScrollArea>
            </Card>

            {error ? <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div> : null}
          </div>

          <Card className="flex min-h-0 flex-col border-slate-200 bg-white p-4">
            <div className="text-sm font-semibold text-slate-900">最近使用</div>
            <div className="mt-1 text-xs text-slate-500">
              最近选择的服务器文件夹。
            </div>
            <ScrollArea className="mt-3 min-h-0 flex-1">
              <div className="space-y-2">
                {recentPaths.length ? (
                  recentPaths.map((recentPath) => (
                    <button
                      key={recentPath}
                      type="button"
                      onClick={() => {
                        setManualPath(recentPath);
                        void loadFolders(recentPath);
                      }}
                      className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 text-left text-sm transition hover:border-[#0d9488] hover:bg-[#eefcf8]"
                    >
                      <div className="break-all font-medium text-slate-900">{recentPath}</div>
                    </button>
                  ))
                ) : (
                  <div className="rounded-xl border border-dashed border-slate-200 px-3 py-5 text-sm text-slate-500">
                    暂无最近使用的文件夹。
                  </div>
                )}
              </div>
            </ScrollArea>
          </Card>
        </div>
      </div>
    </div>
  );
}

function loadRecentFolders(purpose: "annotation" | "video") {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(storageKey(purpose));
    const payload = raw ? (JSON.parse(raw) as string[]) : [];
    return Array.isArray(payload) ? payload.filter((item) => typeof item === "string" && isWorkspacePath(item)) : [];
  } catch {
    return [];
  }
}

function pushRecentFolder(purpose: "annotation" | "video", folderPath: string) {
  if (typeof window === "undefined") return;
  if (!isWorkspacePath(folderPath)) return;
  const current = loadRecentFolders(purpose);
  const next = [folderPath, ...current.filter((item) => item !== folderPath)].slice(0, RECENT_LIMIT);
  window.localStorage.setItem(storageKey(purpose), JSON.stringify(next));
}

function storageKey(purpose: "annotation" | "video") {
  return `server-folder-recent:${purpose}`;
}

function isWorkspacePath(folderPath: string) {
  const normalized = folderPath.replaceAll("\\", "/").replace(/\/+$/, "");
  return normalized === "/root/WWB/VTAX/workspaces" || normalized.startsWith("/root/WWB/VTAX/workspaces/");
}
