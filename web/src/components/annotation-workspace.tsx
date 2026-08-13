"use client";

import {
  ArrowLeft,
  ArrowRight,
  Bot,
  Check,
  CircleStop,
  ImageIcon,
  MousePointer2,
  Plus,
  Play,
  RefreshCw,
  Save,
  Search,
  Square,
  Trash2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent, ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import type {
  CategoryGuide,
  FolderImage,
  FolderScanResponse,
  JobEvent,
  JobState,
  LabelMeAnnotation,
  LabelMeShape,
  ModelProvider,
  ModelProviderSummary,
} from "@/types/annotation";

const MIN_BOX_SIZE = 4;

const imageUrl = (path: string) => `/api/files/image?path=${encodeURIComponent(path)}`;

type BBox = {
  left: number;
  top: number;
  width: number;
  height: number;
};

type DragState = {
  shapeIndex: number;
  mode: "move" | "nw" | "ne" | "sw" | "se";
  startX: number;
  startY: number;
  startBox: BBox;
};

type ImageSize = {
  width: number;
  height: number;
};

type DefaultGuideResponse = {
  exists: boolean;
  source?: "global" | "project";
  labels: string[];
  guides: CategoryGuide[];
  missing_labels?: string[];
  path: string;
  updated_at?: string | null;
};

export function AnnotationWorkspace({
  projectId,
  initialFolderPath,
  initialLabels,
  reviewMode = false,
  integratedMode = false,
  completionBusy = false,
  onCompleteReview,
}: {
  projectId: string;
  initialFolderPath?: string;
  initialLabels?: string[];
  reviewMode?: boolean;
  integratedMode?: boolean;
  completionBusy?: boolean;
  onCompleteReview?: (folderPath: string, labels: string[]) => Promise<void>;
}) {
  const [folderPath, setFolderPath] = useState(initialFolderPath ?? "");
  const [labelsText, setLabelsText] = useState(initialLabels?.join(", ") ?? "");
  const [modelProvider, setModelProvider] = useState<ModelProvider>("dashscope");
  const [modelProviderSummary, setModelProviderSummary] = useState<ModelProviderSummary | null>(null);
  const [images, setImages] = useState<FolderImage[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [annotation, setAnnotation] = useState<LabelMeAnnotation | null>(null);
  const [visibleShapes, setVisibleShapes] = useState<Record<number, boolean>>({});
  const [selectedShapeIndex, setSelectedShapeIndex] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [isRunning, setIsRunning] = useState(false);
  const [isGeneratingGuides, setIsGeneratingGuides] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isCompletingReview, setIsCompletingReview] = useState(false);
  const [isDirty, setIsDirty] = useState(false);
  const [job, setJob] = useState<JobState | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [guideDraft, setGuideDraft] = useState<CategoryGuide[]>([]);
  const [guideMode, setGuideMode] = useState<"default" | "temporary">("default");
  const [isGuideModalOpen, setIsGuideModalOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [naturalSize, setNaturalSize] = useState<ImageSize>({ width: 0, height: 0 });
  const [dragState, setDragState] = useState<DragState | null>(null);

  const canvasRef = useRef<HTMLDivElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const annotationRequestRef = useRef(0);

  const labels = useMemo(() => parseLabels(labelsText), [labelsText]);
  const selectedImage = images[selectedIndex];
  const filteredImages = useMemo(() => filterImages(images, searchQuery), [images, searchQuery]);
  const renderSize = useMemo(
    () => ({
      width: naturalSize.width * zoom,
      height: naturalSize.height * zoom,
    }),
    [naturalSize, zoom],
  );
  const progress = job?.summary?.total
    ? (job.summary.processed + job.summary.skipped + job.summary.failed) / job.summary.total
    : jobProgressFromEvents(events);

  useEffect(() => {
    if (!selectedImage) {
      // Selection is external state owned by the scanned folder result.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setAnnotation(null);
      setSelectedShapeIndex(null);
      return;
    }
    void loadAnnotation(selectedImage.path);
    // Loading is intentionally tied to the selected image path only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedImage?.path]);

  useEffect(() => {
    if (!selectedImage) return;
    // Canvas measurements must be invalidated when the image element changes.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setNaturalSize({ width: 0, height: 0 });
    setZoom(1);
    // Reset behavior is keyed to the active image path.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedImage?.path]);

  useEffect(() => {
    if (!dragState) return;
    const onPointerMove = (event: PointerEvent) => updateDraggedShape(event);
    const onPointerUp = () => setDragState(null);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    };
    // The drag handler depends on the latest mutable drag/editing state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragState, annotation, renderSize, naturalSize]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTypingTarget(event.target)) return;
      if (event.key === "Delete" || event.key === "Backspace") {
        if (selectedShapeIndex !== null) {
          event.preventDefault();
          deleteSelectedShape();
        }
      }
      if (event.key.toLowerCase() === "a") selectImageByIndex(selectedIndex - 1);
      if (event.key.toLowerCase() === "d") selectImageByIndex(selectedIndex + 1);
      if (event.key.toLowerCase() === "r") addShape();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // Keyboard shortcuts should refresh when the active image/edit state changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIndex, selectedShapeIndex, annotation, labels, naturalSize]);

  useEffect(() => {
    return () => eventSourceRef.current?.close();
  }, []);

  useEffect(() => {
    if (!job?.id || !isRunning) return;
    let disposed = false;
    const refresh = async () => {
      try {
        const response = await fetch(`/api/jobs/${job.id}`, { cache: "no-store" });
        if (!response.ok) return;
        const snapshot = (await response.json()) as JobState;
        if (disposed) return;
        setJob(snapshot);
        setEvents(snapshot.events ?? []);
        applyImageEvents(snapshot.events ?? []);
        if (["finished", "failed", "cancelled"].includes(snapshot.status)) {
          setIsRunning(false);
          void performScan(folderPath, true);
        }
      } catch {
        // The live stream remains available when a polling request is interrupted.
      }
    };
    void refresh();
    const timer = window.setInterval(refresh, 1000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
    // Polling is a deliberate fallback for proxies that buffer SSE responses.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id, isRunning, folderPath]);

  useEffect(() => {
    void loadModelProvider();
  }, []);

  useEffect(() => {
    if (!initialFolderPath) return;
    void performScan(initialFolderPath);
    // Initial scan is intentionally keyed only to the externally supplied folder.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialFolderPath]);

  async function loadModelProvider() {
    try {
      const response = await fetch("/api/config/model-provider");
      const payload = (await response.json()) as ModelProviderSummary & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "Failed to read model provider config");
      setModelProviderSummary(payload);
    } catch {
      setModelProvider("dashscope");
    }
  }

  async function performScan(targetFolder: string, preserveSelection = false) {
    setError(null);
    try {
      const response = await fetch("/api/folders/scan", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ inputDir: targetFolder }),
      });
      const payload = (await response.json()) as FolderScanResponse & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "扫描失败");
      const selectedPath = preserveSelection ? selectedImage?.path : undefined;
      setImages(payload.images);
      setSelectedIndex(Math.max(0, selectedPath ? payload.images.findIndex((image) => image.path === selectedPath) : 0));
      if (!preserveSelection) setSearchQuery("");
      setFolderPath(targetFolder);
    } catch (scanError) {
      setError(scanError instanceof Error ? scanError.message : "扫描失败");
    }
  }

  async function loadAnnotation(path: string) {
    const requestId = ++annotationRequestRef.current;
    setError(null);
    setIsDirty(false);
    setAnnotation(null);
    setVisibleShapes({});
    setSelectedShapeIndex(null);
    try {
      const response = await fetch(`/api/files/annotation?path=${encodeURIComponent(path)}`);
      const payload = (await response.json()) as { annotation: LabelMeAnnotation | null; error?: string };
      if (!response.ok) throw new Error(payload.error ?? "读取标注失败");
      if (requestId !== annotationRequestRef.current) return;
      setAnnotation(normalizeAnnotationImageSize(payload.annotation, naturalSize));
      setVisibleShapes(toVisibleShapeMap(payload.annotation?.shapes ?? []));
    } catch (annotationError) {
      if (requestId !== annotationRequestRef.current) return;
      setError(annotationError instanceof Error ? annotationError.message : "读取标注失败");
    }
  }

  async function saveAnnotation() {
    if (!selectedImage || !annotation) return;
    await saveAnnotationValue(annotation);
  }

  async function saveAnnotationValue(nextAnnotation: LabelMeAnnotation) {
    if (!selectedImage) return;
    setIsSaving(true);
    setError(null);
    try {
      const response = await fetch("/api/files/annotation", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ imagePath: selectedImage.path, annotation: nextAnnotation }),
      });
      const payload = (await response.json()) as { annotation: LabelMeAnnotation; error?: string };
      if (!response.ok) throw new Error(payload.error ?? "保存失败");
      setAnnotation(payload.annotation);
      setIsDirty(false);
      setImages((current) =>
        current.map((image, index) =>
          index === selectedIndex ? { ...image, hasAnnotation: true, status: "done" } : image,
        ),
      );
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "保存失败");
    } finally {
      setIsSaving(false);
    }
  }

  async function markCurrentReviewed() {
    const base = ensureAnnotation();
    if (!base) {
      setError("请先选择一张图像。");
      return;
    }
    const reviewed = { ...base, checked: true };
    setAnnotation(reviewed);
    await saveAnnotationValue(reviewed);
  }

  async function completeReview() {
    if (!onCompleteReview || !folderPath || !images.length) {
      setError("没有可提交审核的标注图像。");
      return;
    }
    setIsCompletingReview(true);
    setError(null);
    try {
      if (isDirty) await saveAnnotation();
      await onCompleteReview(folderPath, labels);
    } catch (reviewError) {
      setError(reviewError instanceof Error ? reviewError.message : "完成审核失败");
    } finally {
      setIsCompletingReview(false);
    }
  }

  async function startJobFromDefault() {
    if (!folderPath || labels.length === 0) {
      setError("请填写图像文件夹路径和至少一个类别词");
      return;
    }
    setError(null);
    setIsGeneratingGuides(true);
    try {
      const patchResponse = await fetch(`/api/vtax/api/v1/projects/${projectId}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ labels }),
      });
      const patchPayload = (await patchResponse.json()) as { detail?: string };
      if (!patchResponse.ok) throw new Error(patchPayload.detail ?? "保存项目类别失败");

      const response = await fetch(`/api/vtax/api/v1/projects/${projectId}/category-guides`, {
        cache: "no-store",
      });
      const payload = (await response.json()) as DefaultGuideResponse & { detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "读取默认类别指南失败");
      if (payload.exists) {
        setGuideDraft(normalizeGuideDraft(payload.guides, labels));
        setGuideMode("default");
      } else {
        const researched = await fetchResearchedGuides();
        setGuideDraft(researched);
        setGuideMode("temporary");
      }
      setIsGuideModalOpen(true);
    } catch (guideError) {
      setError(guideError instanceof Error ? guideError.message : "类别指南加载失败");
    } finally {
      setIsGeneratingGuides(false);
    }
  }

  async function fetchResearchedGuides() {
    const response = await fetch("/api/jobs/guides", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ inputDir: folderPath, outputDir: folderPath, labels }),
    });
    const payload = (await response.json()) as { guides?: CategoryGuide[]; error?: string };
    if (!response.ok) throw new Error(payload.error ?? "类别指南生成失败");
    return normalizeGuideDraft(payload.guides ?? [], labels);
  }

  async function saveDefaultGuides(guides: CategoryGuide[]) {
    const response = await fetch(`/api/vtax/api/v1/projects/${projectId}/category-guides`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ labels, guides }),
    });
    const payload = (await response.json()) as { detail?: string };
    if (!response.ok) throw new Error(payload.detail ?? "保存默认类别指南失败");
  }

  async function saveProjectGuides(guides: CategoryGuide[]) {
    const response = await fetch(`/api/vtax/api/v1/projects/${projectId}/project-category-guides`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ labels, guides }),
    });
    const payload = (await response.json()) as { detail?: string };
    if (!response.ok) throw new Error(payload.detail ?? "保存项目类别指南失败");
  }

  async function confirmGuidesAndStart() {
    const guides = normalizeGuideDraft(guideDraft, labels);
    setIsGeneratingGuides(true);
    setError(null);
    try {
      if (guideMode === "default") await saveDefaultGuides(guides);
      else await saveProjectGuides(guides);
      await startJobWithGuides(guides);
    } catch (guideError) {
      setError(guideError instanceof Error ? guideError.message : "启动标注失败");
    } finally {
      setIsGeneratingGuides(false);
    }
  }

  async function researchTemporaryGuides() {
    if (!folderPath || labels.length === 0) {
      setError("请填写文件夹路径和至少一个类别词");
      return;
    }
    setError(null);
    setIsGeneratingGuides(true);
    try {
      const response = await fetch("/api/jobs/guides", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          inputDir: folderPath,
          outputDir: folderPath,
          labels,
        }),
      });
      const payload = (await response.json()) as { guides?: CategoryGuide[]; error?: string };
      if (!response.ok) throw new Error(payload.error ?? "类别指南生成失败");
      setGuideDraft(normalizeGuideDraft(payload.guides ?? [], labels));
      setGuideMode("temporary");
      setIsGuideModalOpen(true);
    } catch (guideError) {
      setError(guideError instanceof Error ? guideError.message : "类别指南生成失败");
    } finally {
      setIsGeneratingGuides(false);
    }
  }

  async function startJobWithGuides(guides: CategoryGuide[]) {
    if (!folderPath || labels.length === 0) {
      setError("请填写文件夹路径和至少一个类别词");
      return;
    }
    setError(null);
    setEvents([]);
    setJob(null);
    setIsRunning(true);
    setIsGuideModalOpen(false);
    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          inputDir: folderPath,
          outputDir: folderPath,
          labels,
          boxType: "rectangle",
          overwrite: false,
          provider: modelProvider,
          coordinateSystem: "normalized-1000",
          guides,
        }),
      });
      const payload = (await response.json()) as JobState & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "任务创建失败");
      setJob(payload);
      subscribeToJob(payload.id);
    } catch (jobError) {
      setIsRunning(false);
      setError(jobError instanceof Error ? jobError.message : "任务创建失败");
    }
  }

  function subscribeToJob(id: string) {
    eventSourceRef.current?.close();
    const source = new EventSource(`/api/jobs/${id}/events`);
    eventSourceRef.current = source;
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as JobEvent & { job?: JobState };
      if (event.type === "snapshot" && event.job) {
        setJob(event.job);
        setEvents(event.job.events ?? []);
        applyImageEvents(event.job.events ?? []);
        if (["finished", "failed", "cancelled"].includes(event.job.status)) {
          void performScan(folderPath, true);
        }
        return;
      }
      setEvents((current) => [...current.slice(-199), event]);
      updateImageStatusFromEvent(event);

      if (event.type === "job_finished" || event.type === "job_summary") {
        setJob((current) => (current ? { ...current, status: "finished", summary: event.summary } : current));
        setIsRunning(false);
        void performScan(folderPath, true);
      }
      if (event.type === "job_error") {
        setError(event.error ?? "任务失败");
        setIsRunning(false);
      }
      if (event.type === "process_exit") {
        setIsRunning(false);
        void performScan(folderPath, true);
      }
    };
    source.onerror = () => {
      source.close();
      // Polling continues to refresh this job when a reverse proxy buffers or closes SSE.
    };
  }

  function applyImageEvents(nextEvents: JobEvent[]) {
    setImages((current) => nextEvents.reduce(applyImageEvent, current));
    const latestForSelection = [...nextEvents].reverse().find(
      (event) => event.image_path && imagePathsMatch(selectedImage?.path, event.image_path),
    );
    if (latestForSelection?.image_path && ["image_done", "image_failed", "image_skipped"].includes(latestForSelection.type)) {
      void loadAnnotation(latestForSelection.image_path);
    }
  }

  function updateImageStatusFromEvent(event: JobEvent) {
    if (!event.image_path) return;
    setImages((current) => applyImageEvent(current, event));
    if (imagePathsMatch(selectedImage?.path, event.image_path) && ["image_done", "image_failed", "image_skipped"].includes(event.type)) {
      void loadAnnotation(event.image_path);
    }
  }

  async function cancelJob() {
    if (!job) return;
    await fetch(`/api/jobs/${job.id}/cancel`, { method: "POST" });
    eventSourceRef.current?.close();
    setIsRunning(false);
  }

  function selectImageByIndex(index: number) {
    if (!images.length) return;
    if (isDirty && !window.confirm("当前图片有未保存修改，切换图片会丢失这些修改。是否继续？")) {
      return;
    }
    setSelectedIndex(Math.min(Math.max(index, 0), images.length - 1));
  }

  function selectImage(image: FolderImage) {
    const nextIndex = images.findIndex((item) => item.path === image.path);
    if (nextIndex >= 0) selectImageByIndex(nextIndex);
  }

  function fitImageToViewport(width: number, height: number) {
    const canvas = canvasRef.current;
    if (!canvas || !width || !height) return;
    const fit = Math.min((canvas.clientWidth - 48) / width, (canvas.clientHeight - 48) / height, 1);
    setZoom(Number.isFinite(fit) ? Math.max(fit, 0.05) : 1);
  }

  function ensureAnnotation(): LabelMeAnnotation | null {
    if (!selectedImage || !naturalSize.width || !naturalSize.height) return annotation;
    if (annotation) return annotation;
    return {
      version: "4.0.0-beta.5",
      flags: {},
      checked: false,
      imagePath: selectedImage.name,
      imageData: null,
      imageWidth: naturalSize.width,
      imageHeight: naturalSize.height,
      description: "",
      shapes: [],
    };
  }

  function addShape() {
    const base = ensureAnnotation();
    if (!base || labels.length === 0) {
      setError("请先选择图片并填写类别词");
      return;
    }
    const width = base.imageWidth * 0.18;
    const height = base.imageHeight * 0.14;
    const left = (base.imageWidth - width) / 2;
    const top = (base.imageHeight - height) / 2;
    const nextShape = createShape(labels[0], { left, top, width, height });
    setAnnotation({ ...base, shapes: [...base.shapes, nextShape] });
    setVisibleShapes((current) => ({ ...current, [base.shapes.length]: true }));
    setSelectedShapeIndex(base.shapes.length);
    setIsDirty(true);
  }

  function deleteSelectedShape() {
    if (!annotation || selectedShapeIndex === null) return;
    const nextShapes = annotation.shapes.filter((_, index) => index !== selectedShapeIndex);
    setAnnotation({ ...annotation, shapes: nextShapes });
    setSelectedShapeIndex(null);
    setVisibleShapes(toVisibleShapeMap(nextShapes));
    setIsDirty(true);
  }

  function changeShapeLabel(index: number, label: string) {
    if (!annotation) return;
    setAnnotation({
      ...annotation,
      shapes: annotation.shapes.map((shape, shapeIndex) =>
        shapeIndex === index ? { ...shape, label } : shape,
      ),
    });
    setSelectedShapeIndex(index);
    setIsDirty(true);
  }

  function startShapeDrag(
    event: ReactPointerEvent,
    shapeIndex: number,
    mode: DragState["mode"],
  ) {
    if (!annotation) return;
    event.preventDefault();
    event.stopPropagation();
    setSelectedShapeIndex(shapeIndex);
    setDragState({
      shapeIndex,
      mode,
      startX: event.clientX,
      startY: event.clientY,
      startBox: shapeToBox(annotation.shapes[shapeIndex]),
    });
  }

  function updateDraggedShape(event: PointerEvent) {
    if (!dragState || !annotation || !naturalSize.width || !naturalSize.height) return;
    const scaleX = naturalSize.width / Math.max(renderSize.width, 1);
    const scaleY = naturalSize.height / Math.max(renderSize.height, 1);
    const dx = (event.clientX - dragState.startX) * scaleX;
    const dy = (event.clientY - dragState.startY) * scaleY;
    const nextBox = resizeBox(dragState.startBox, dragState.mode, dx, dy, naturalSize);
    const nextShapes = annotation.shapes.map((shape, index) =>
      index === dragState.shapeIndex ? shapeWithBox(shape, nextBox) : shape,
    );
    setAnnotation({ ...annotation, shapes: nextShapes });
    setIsDirty(true);
  }

  function changeGuideField(
    guideIndex: number,
    field: Exclude<keyof CategoryGuide, "label">,
    value: string,
  ) {
    setGuideDraft((current) =>
      current.map((guide, index) =>
        index === guideIndex ? { ...guide, [field]: textToList(value) } : guide,
      ),
    );
  }

  return (
    <TooltipProvider>
      <GuideReviewModal
        guides={guideDraft}
        mode={guideMode}
        open={isGuideModalOpen}
        isStarting={isRunning || isGeneratingGuides}
        onClose={() => setIsGuideModalOpen(false)}
        onChange={changeGuideField}
        onResearch={researchTemporaryGuides}
        onStart={confirmGuidesAndStart}
      />
      <div className="grid h-full min-w-0 grid-rows-[auto_minmax(0,1fr)_auto] overflow-hidden bg-[#f4f6f3] text-slate-950">
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-300 bg-[#f7f7f2] px-3 py-2 text-sm">
          <div className="flex min-w-0 flex-wrap items-center gap-3">
            <Badge className="rounded-sm bg-[#f28c18] text-white">AI 标注</Badge>
            <span className="shrink-0 font-semibold">当前模式：编辑</span>
            <span className="truncate text-slate-600">
              快捷键：上一张 A、下一张 D、新增矩形 R、删除 Delete
            </span>
          </div>
          <div className="flex shrink-0 items-center gap-2 text-xs text-slate-600">
            {reviewMode ? (
              <Button
                size="sm"
                className="h-7 bg-[#0d9488] text-white hover:bg-[#0f766e]"
                onClick={markCurrentReviewed}
                disabled={!selectedImage || isSaving || annotation?.checked === true}
              >
                <Check className="mr-1 size-3.5" />
                {annotation?.checked ? "已审核" : "标记当前已审核"}
              </Button>
            ) : null}
            <span>{images.length ? selectedIndex + 1 : 0} / {images.length}</span>
            <Separator orientation="vertical" className="h-4 bg-slate-300" />
            <span>缩放 {Math.round(zoom * 100)}%</span>
            <Badge variant={isRunning ? "default" : "secondary"} className="rounded-sm">
              {isRunning ? "正在标注" : "就绪"}
            </Badge>
          </div>
        </header>

        <main className="grid min-h-0 min-w-0 grid-cols-[44px_minmax(0,1fr)] grid-rows-[minmax(0,1fr)_260px] overflow-hidden lg:grid-cols-[48px_minmax(0,1fr)_clamp(260px,22vw,360px)] lg:grid-rows-1">
          <Toolbar
            zoom={zoom}
            onAddShape={addShape}
            onDeleteShape={deleteSelectedShape}
            onPrevious={() => selectImageByIndex(selectedIndex - 1)}
            onNext={() => selectImageByIndex(selectedIndex + 1)}
            onZoomIn={() => setZoom((value) => Math.min(value + 0.1, 3))}
            onZoomOut={() => setZoom((value) => Math.max(value - 0.1, 0.05))}
          />

          <section ref={canvasRef} className="min-h-0 overflow-auto border-r border-slate-300 bg-white">
            <div className="flex h-full min-h-full items-start justify-center p-4">
              {selectedImage ? (
                <div
                  key={selectedImage.path}
                  className="relative max-w-none shadow-sm"
                  style={{
                    width: renderSize.width || undefined,
                    height: renderSize.height || undefined,
                  }}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    key={selectedImage.path}
                    ref={imgRef}
                    alt={selectedImage.name}
                    src={imageUrl(selectedImage.path)}
                    className="block max-w-none select-none"
                    draggable={false}
                    style={{ width: renderSize.width || undefined }}
                    onLoad={(event) => {
                      const width = event.currentTarget.naturalWidth;
                      const height = event.currentTarget.naturalHeight;
                      const size = { width, height };
                      setNaturalSize(size);
                      setAnnotation((current) => normalizeAnnotationImageSize(current, size));
                      fitImageToViewport(width, height);
                    }}
                  />
                  <AnnotationOverlay
                    annotation={annotation}
                    renderSize={renderSize}
                    selectedShapeIndex={selectedShapeIndex}
                    visibleShapes={visibleShapes}
                    onSelectShape={setSelectedShapeIndex}
                    onStartDrag={startShapeDrag}
                  />
                </div>
              ) : (
                <div className="mt-28 flex flex-col items-center gap-3 text-center text-slate-500">
                  <ImageIcon className="h-12 w-12 text-[#f28c18]" />
                  <p>等待加载上一阶段清洗完成的图像帧。</p>
                </div>
              )}
            </div>
          </section>

          <RightPanel
            labels={labels}
            images={filteredImages}
            selectedImagePath={selectedImage?.path}
            annotation={annotation}
            visibleShapes={visibleShapes}
            selectedShapeIndex={selectedShapeIndex}
            searchQuery={searchQuery}
            onSearchQueryChange={setSearchQuery}
            onToggleShape={(index, checked) =>
              setVisibleShapes((current) => ({ ...current, [index]: checked }))
            }
            onSelectShape={setSelectedShapeIndex}
            onSelectImage={selectImage}
            onChangeShapeLabel={changeShapeLabel}
          />
        </main>

        <BottomPanel
          folderPath={folderPath}
          labelsText={labelsText}
          modelProvider={modelProvider}
          modelProviderSummary={modelProviderSummary}
          isRunning={isRunning}
          isGeneratingGuides={isGeneratingGuides}
          isSaving={isSaving}
          isDirty={isDirty}
          progress={progress}
          job={job}
          events={events}
          error={error}
          selectedShapeIndex={selectedShapeIndex}
          onLabelsTextChange={setLabelsText}
          onModelProviderChange={setModelProvider}
          onStart={startJobFromDefault}
          onCancel={cancelJob}
          onAddShape={addShape}
          onDeleteShape={deleteSelectedShape}
          onSave={saveAnnotation}
          integratedMode={integratedMode}
          completionBusy={completionBusy || isCompletingReview}
          onCompleteReview={() => void completeReview()}
        />
      </div>
    </TooltipProvider>
  );
}

function GuideReviewModal({
  guides,
  mode,
  open,
  isStarting,
  onClose,
  onChange,
  onResearch,
  onStart,
}: {
  guides: CategoryGuide[];
  mode: "default" | "temporary";
  open: boolean;
  isStarting: boolean;
  onClose: () => void;
  onChange: (
    guideIndex: number,
    field: Exclude<keyof CategoryGuide, "label">,
    value: string,
  ) => void;
  onResearch: () => void;
  onStart: () => void;
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4 backdrop-blur-sm">
      <div className="flex h-[min(92dvh,900px)] w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-slate-300 bg-[#f8faf5] shadow-2xl">
        <div className="shrink-0 border-b border-slate-300 px-5 py-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold text-slate-950">检查类别指南</h2>
              <p className="mt-1 text-sm text-slate-600">
                {mode === "default"
                  ? "当前展示所有项目共用的全局默认指南。修改并保存后，其他项目也会使用更新后的内容。"
                  : "当前展示按本项目类别词重新调研的结果，仅保存并用于当前项目，不会覆盖全局默认指南。"}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className={mode === "default" ? "border-teal-300 bg-teal-50 text-teal-800" : "border-amber-300 bg-amber-50 text-amber-800"}>
                {mode === "default" ? "全局默认指南" : "项目专用指南"}
              </Badge>
              <Badge className="rounded-sm bg-[#0d9488] text-white">{guides.length} 个类别</Badge>
            </div>
          </div>
        </div>

        <ScrollArea className="min-h-0 flex-1">
          <div className="space-y-4 p-5 pb-8">
            {guides.map((guide, guideIndex) => (
              <Card key={`${guide.label}-${guideIndex}`} className="rounded-lg border-slate-300 bg-white p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm text-slate-500">类别</div>
                    <div className="text-base font-semibold text-slate-950">{guide.label}</div>
                  </div>
                  <span className="text-xs text-slate-500">每行一条规则，空行会自动忽略</span>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  <GuideTextarea
                    label="视觉特征"
                    value={listToText(guide.visual_cues)}
                    onChange={(value) => onChange(guideIndex, "visual_cues", value)}
                    placeholder="例如：花瓣形状、颜色、叶片、整体外观..."
                  />
                  <GuideTextarea
                    label="OCR/文字线索"
                    value={listToText(guide.ocr_cues)}
                    onChange={(value) => onChange(guideIndex, "ocr_cues", value)}
                    placeholder="例如：包装文字、牌匾、标签、警示语..."
                  />
                  <GuideTextarea
                    label="排除规则"
                    value={listToText(guide.negative_cues)}
                    onChange={(value) => onChange(guideIndex, "negative_cues", value)}
                    placeholder="例如：容易误判但不属于该类的对象..."
                  />
                  <GuideTextarea
                    label="歧义说明"
                    value={listToText(guide.ambiguity_notes)}
                    onChange={(value) => onChange(guideIndex, "ambiguity_notes", value)}
                    placeholder="例如：遮挡、相似类别、低置信度处理..."
                  />
                </div>
              </Card>
            ))}
          </div>
        </ScrollArea>

        <div className="shrink-0 flex flex-wrap items-center justify-end gap-2 border-t border-slate-300 bg-white px-5 py-4 shadow-[0_-8px_18px_rgba(15,23,42,0.06)]">
          <Button variant="secondary" onClick={onClose} disabled={isStarting}>
            取消
          </Button>
          <Button variant="outline" onClick={onResearch} disabled={isStarting}>
            <RefreshCw className={`mr-2 h-4 w-4 ${isStarting ? "animate-spin" : ""}`} />
            重新调研
          </Button>
          <Button className="bg-[#0d9488] hover:bg-[#0f766e]" onClick={onStart} disabled={isStarting}>
            <Play className="mr-2 h-4 w-4" />
            {isStarting ? "处理中" : mode === "default" ? "保存全局默认并开始标注" : "保存为项目指南并开始标注"}
          </Button>
        </div>
      </div>
    </div>
  );
}

function GuideTextarea({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string;
  value: string;
  placeholder: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="space-y-1 text-sm">
      <span className="font-medium text-slate-700">{label}</span>
      <textarea
        className="min-h-28 w-full resize-y rounded-md border border-slate-300 bg-white px-3 py-2 text-sm leading-5 text-slate-950 shadow-sm outline-none transition focus:border-[#0d9488] focus:ring-2 focus:ring-[#0d9488]/20"
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function Toolbar({
  zoom,
  onAddShape,
  onDeleteShape,
  onPrevious,
  onNext,
  onZoomIn,
  onZoomOut,
}: {
  zoom: number;
  onAddShape: () => void;
  onDeleteShape: () => void;
  onPrevious: () => void;
  onNext: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}) {
  const tools = [
    { icon: ArrowLeft, label: "上一张", action: onPrevious },
    { icon: ArrowRight, label: "下一张", action: onNext },
    { icon: Square, label: "新增矩形框", action: onAddShape, active: true },
    { icon: MousePointer2, label: "选择 / 拖拽" },
    { icon: Trash2, label: "删除选中框", action: onDeleteShape },
    { icon: ZoomIn, label: "放大", action: onZoomIn },
    { icon: ZoomOut, label: "缩小", action: onZoomOut },
    { icon: Bot, label: "AI 标注" },
  ];
  return (
    <aside className="row-span-2 flex min-h-0 flex-col items-center gap-2 border-r border-slate-300 bg-[#f2f3ed] py-2 lg:row-span-1">
      {tools.map((tool) => (
        <Tooltip key={tool.label}>
          <TooltipTrigger asChild>
            <button
              className={`flex h-8 w-8 items-center justify-center rounded-md border text-white shadow-sm transition hover:brightness-95 ${
                tool.active ? "border-[#0d9488] bg-[#0d9488]" : "border-[#d47b00] bg-[#ff970f]"
              }`}
              onClick={tool.action}
              type="button"
            >
              <tool.icon className="h-4 w-4" />
            </button>
          </TooltipTrigger>
          <TooltipContent side="right">{tool.label}</TooltipContent>
        </Tooltip>
      ))}
      <div className="mt-auto text-xs text-slate-700">{Math.round(zoom * 100)}%</div>
    </aside>
  );
}

function AnnotationOverlay({
  annotation,
  renderSize,
  selectedShapeIndex,
  visibleShapes,
  onSelectShape,
  onStartDrag,
}: {
  annotation: LabelMeAnnotation | null;
  renderSize: ImageSize;
  selectedShapeIndex: number | null;
  visibleShapes: Record<number, boolean>;
  onSelectShape: (index: number) => void;
  onStartDrag: (event: ReactPointerEvent, index: number, mode: DragState["mode"]) => void;
}) {
  if (!annotation) return null;
  const scaleX = renderSize.width / Math.max(annotation.imageWidth, 1);
  const scaleY = renderSize.height / Math.max(annotation.imageHeight, 1);
  return (
    <div className="absolute inset-0">
      {annotation.shapes.map((shape, index) => {
        if (visibleShapes[index] === false) return null;
        const box = shapeToBox(shape);
        const left = box.left * scaleX;
        const top = box.top * scaleY;
        const width = box.width * scaleX;
        const height = box.height * scaleY;
        const selected = index === selectedShapeIndex;
        return (
          <div
            key={`${shape.label}-${index}`}
            role="button"
            tabIndex={0}
            className={`absolute cursor-move border-2 bg-[#8b83f6]/10 ${
              selected ? "border-[#f28c18]" : "border-[#8b83f6]"
            }`}
            style={{ left, top, width, height }}
            onPointerDown={(event) => onStartDrag(event, index, "move")}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") onSelectShape(index);
            }}
            onClick={(event) => {
              event.stopPropagation();
              onSelectShape(index);
            }}
          >
            <span className="absolute -top-5 left-0 whitespace-nowrap bg-[#d8d4ff] px-1 text-[11px] font-semibold text-slate-900">
              {shape.label} {typeof shape.score === "number" ? shape.score.toFixed(2) : ""}
            </span>
            {selected
              ? (["nw", "ne", "sw", "se"] as const).map((mode) => (
                  <button
                    key={mode}
                    className={`absolute h-3 w-3 rounded-full border border-white bg-[#f28c18] ${
                      mode.includes("n") ? "-top-1.5" : "-bottom-1.5"
                    } ${mode.includes("w") ? "-left-1.5" : "-right-1.5"}`}
                    onPointerDown={(event) => onStartDrag(event, index, mode)}
                    type="button"
                    aria-label={`resize ${mode}`}
                  />
                ))
              : null}
          </div>
        );
      })}
    </div>
  );
}

function RightPanel({
  labels,
  images,
  selectedImagePath,
  annotation,
  visibleShapes,
  selectedShapeIndex,
  searchQuery,
  onSearchQueryChange,
  onToggleShape,
  onSelectShape,
  onSelectImage,
  onChangeShapeLabel,
}: {
  labels: string[];
  images: FolderImage[];
  selectedImagePath?: string;
  annotation: LabelMeAnnotation | null;
  visibleShapes: Record<number, boolean>;
  selectedShapeIndex: number | null;
  searchQuery: string;
  onSearchQueryChange: (value: string) => void;
  onToggleShape: (index: number, checked: boolean) => void;
  onSelectShape: (index: number) => void;
  onSelectImage: (image: FolderImage) => void;
  onChangeShapeLabel: (index: number, label: string) => void;
}) {
  return (
    <aside className="col-start-2 grid min-h-0 grid-rows-[88px_minmax(0,1fr)] border-t border-slate-300 bg-[#f9faf7] text-xs lg:col-start-auto lg:border-t-0">
      <Panel title="标签">
        <div className="flex flex-wrap gap-1">
          {labels.length ? (
            labels.map((label) => (
              <Badge key={label} className="rounded-sm bg-[#d9d5ff] text-slate-900">
                {label}
              </Badge>
            ))
          ) : (
            <span className="text-slate-500">请在底部输入类别词</span>
          )}
        </div>
      </Panel>
      <Panel title="对象 / 文件">
        <div className="mb-2 max-h-36 space-y-1 overflow-auto">
          {annotation?.shapes.length ? (
            annotation.shapes.map((shape, index) => (
              <div
                key={`${shape.label}-${index}`}
                role="button"
                tabIndex={0}
                className={`flex items-center gap-2 px-1 py-1 ${
                  selectedShapeIndex === index ? "bg-[#f3c77a]" : "bg-[#c7c4ff]"
                }`}
                onClick={() => onSelectShape(index)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") onSelectShape(index);
                }}
              >
                <input
                  checked={visibleShapes[index] !== false}
                  type="checkbox"
                  onChange={(event) => onToggleShape(index, event.target.checked)}
                />
                {labels.length > 1 ? (
                  <select
                    className="min-w-0 flex-1 rounded border border-slate-300 bg-white px-1 py-0.5"
                    value={shape.label}
                    onChange={(event) => onChangeShapeLabel(index, event.target.value)}
                  >
                    {labels.map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                ) : (
                  <span className="min-w-0 flex-1 truncate">{shape.label}</span>
                )}
                <span className="shrink-0">{typeof shape.score === "number" ? shape.score.toFixed(2) : ""}</span>
              </div>
            ))
          ) : (
            <div className="rounded border border-dashed border-slate-300 p-2 text-slate-500">
              当前图片暂无矩形框，可点击左侧或底部“新增框”。
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 border-y border-slate-200 py-2">
          <Search className="h-4 w-4 text-slate-400" />
          <Input
            className="h-7 border-none bg-transparent px-0 shadow-none focus-visible:ring-0"
            value={searchQuery}
            onChange={(event) => onSearchQueryChange(event.target.value)}
            placeholder="搜索文件名、路径、状态..."
          />
        </div>

        <div className="min-h-32 flex-1 overflow-y-auto overscroll-contain">
          <div className="space-y-1 py-2">
            {images.length ? (
              images.map((image) => (
                <button
                  key={image.path}
                  className={`flex w-full items-center gap-2 truncate px-2 py-1 text-left ${
                    image.path === selectedImagePath ? "bg-[#006de5] text-white" : "hover:bg-slate-100"
                  }`}
                  onClick={() => onSelectImage(image)}
                  title={image.path}
                  type="button"
                  style={{ contentVisibility: "auto", containIntrinsicSize: "28px" }}
                >
                  {image.hasAnnotation ? (
                    <Check className="h-3 w-3 shrink-0" />
                  ) : (
                    <span className="h-3 w-3 shrink-0 rounded-full bg-slate-300" />
                  )}
                  <span className="truncate">{image.name}</span>
                  {image.status ? <span className="ml-auto shrink-0 opacity-70">{image.status}</span> : null}
                </button>
              ))
            ) : (
              <div className="px-2 py-4 text-slate-500">没有匹配图片</div>
            )}
          </div>
        </div>
      </Panel>
    </aside>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="min-h-0 border-b border-slate-300">
      <div className="flex h-7 items-center justify-between border-b border-slate-200 bg-[#f2f2f2] px-2 font-medium">
        <span>{title}</span>
      </div>
      <div className="flex h-[calc(100%-1.75rem)] min-h-0 flex-col overflow-hidden p-2">{children}</div>
    </section>
  );
}

function BottomPanel({
  folderPath,
  labelsText,
  modelProvider,
  modelProviderSummary,
  isRunning,
  isGeneratingGuides,
  isSaving,
  isDirty,
  progress,
  job,
  events,
  error,
  selectedShapeIndex,
  onLabelsTextChange,
  onModelProviderChange,
  onStart,
  onCancel,
  onAddShape,
  onDeleteShape,
  onSave,
  integratedMode,
  completionBusy,
  onCompleteReview,
}: {
  folderPath: string;
  labelsText: string;
  modelProvider: ModelProvider;
  modelProviderSummary: ModelProviderSummary | null;
  isRunning: boolean;
  isGeneratingGuides: boolean;
  isSaving: boolean;
  isDirty: boolean;
  progress: number;
  job: JobState | null;
  events: JobEvent[];
  error: string | null;
  selectedShapeIndex: number | null;
  onLabelsTextChange: (value: string) => void;
  onModelProviderChange: (value: ModelProvider) => void;
  onStart: () => void;
  onCancel: () => void;
  onAddShape: () => void;
  onDeleteShape: () => void;
  onSave: () => void;
  integratedMode: boolean;
  completionBusy: boolean;
  onCompleteReview: () => void;
}) {
  const lastEvent = events.at(-1);
  const summary = job?.summary;
  return (
    <footer className="max-h-[44dvh] min-w-0 overflow-x-hidden overflow-y-auto border-t border-slate-300 bg-[#eef1ea] p-3">
      <div className="grid min-w-0 grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-12">
        <div className="min-w-0 space-y-1 xl:col-span-5">
          <Label>图像文件夹</Label>
          <Input value={folderPath} readOnly className="bg-slate-50 text-slate-600" />
        </div>
        <div className="min-w-0 space-y-1 xl:col-span-3">
          <Label>类别词</Label>
          <Input value={labelsText} onChange={(event) => onLabelsTextChange(event.target.value)} />
        </div>
        <div className="min-w-0 space-y-1 xl:col-span-2">
          <Label>检测框类型</Label>
          <Select value="rectangle" disabled>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="rectangle">矩形框</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="min-w-0 space-y-1 xl:col-span-2">
          <Label>视觉模型</Label>
          <Select value={modelProvider} onValueChange={(value) => onModelProviderChange(value as ModelProvider)}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="dashscope">API</SelectItem>
              <SelectItem value="ollama">本地 Ollama</SelectItem>
            </SelectContent>
          </Select>
          <div className="truncate text-[11px] text-slate-500" title={providerStatusTitle(modelProviderSummary)}>
            {providerStatusLabel(modelProvider, modelProviderSummary)}
          </div>
        </div>
        <div className="flex min-w-0 flex-wrap items-end justify-end gap-2 md:col-span-2 xl:col-span-12">
          <Button variant="secondary" onClick={onAddShape}>
            <Plus className="mr-2 h-4 w-4" />
            新增框
          </Button>
          <Button variant="secondary" onClick={onDeleteShape} disabled={selectedShapeIndex === null}>
            <Trash2 className="mr-2 h-4 w-4" />
            删除
          </Button>
          <Button variant={isDirty ? "default" : "secondary"} onClick={onSave} disabled={!isDirty || isSaving}>
            <Save className="mr-2 h-4 w-4" />
            {isSaving ? "保存中" : "保存"}
          </Button>
          {isRunning ? (
            <Button variant="destructive" onClick={onCancel}>
              <CircleStop className="mr-2 h-4 w-4" />
              停止
            </Button>
          ) : (
            <Button className="bg-[#0d9488] hover:bg-[#0f766e]" onClick={onStart} disabled={isGeneratingGuides}>
              {isGeneratingGuides ? (
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              {isGeneratingGuides ? "生成指南中" : "开始标注"}
            </Button>
          )}
          {integratedMode ? <Button className="bg-emerald-600 hover:bg-emerald-700" onClick={onCompleteReview} disabled={completionBusy || isRunning || isSaving}><Check className="mr-2 h-4 w-4" />{completionBusy ? "提交审核中…" : "完成审核"}</Button> : null}
        </div>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-[1fr_320px]">
        <Card className="rounded-md border-slate-300 bg-white p-3">
          <div className="mb-2 flex items-center justify-between gap-3 text-sm">
            <span className="font-semibold">
              {isGeneratingGuides ? "生成类别指南中" : isRunning ? "正在标注" : isDirty ? "有未保存修改" : "任务状态"}
            </span>
            <span className="truncate text-slate-500">{lastEvent?.image_path ?? lastEvent?.type ?? "等待任务"}</span>
          </div>
          <Progress value={Math.round(progress * 100)} />
          {error ? <p className="mt-2 text-sm text-red-600">{error}</p> : null}
        </Card>
        <Card className="grid grid-cols-3 gap-2 rounded-md border-slate-300 bg-white p-3 text-xs">
          <Metric label="成功标注" value={summary?.annotated ?? 0} />
          <Metric label="空结果" value={summary?.empty ?? 0} />
          <Metric label="失败" value={summary?.failed ?? 0} />
          <Metric label="耗时" value={summary ? `${summary.total_seconds.toFixed(1)}s` : "-"} />
          <Metric
            label="Token"
            value={summary?.token_usage_images ? formatNumber(summary.total_tokens) : "unavailable"}
          />
          <Metric label="缓存命中" value={summary?.total_cached_tokens ?? 0} />
        </Card>
      </div>
    </footer>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div className="text-slate-500">{label}</div>
      <div className="text-base font-semibold text-slate-900">{value}</div>
    </div>
  );
}

function parseLabels(value: string) {
  return value
    .split(/[\s,，、]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function filterImages(images: FolderImage[], queryText: string) {
  const query = queryText.trim().toLowerCase();
  if (!query) return images;
  return images.filter((image) =>
    [image.name, image.path, image.status ?? "", image.hasAnnotation ? "已标注 done" : "未标注 idle"]
      .join(" ")
      .toLowerCase()
      .includes(query),
  );
}

function toVisibleShapeMap(shapes: LabelMeShape[]) {
  return Object.fromEntries(shapes.map((_, index) => [index, true])) as Record<number, boolean>;
}

function normalizeAnnotationImageSize(
  annotation: LabelMeAnnotation | null,
  imageSize: ImageSize,
): LabelMeAnnotation | null {
  if (
    !annotation ||
    !imageSize.width ||
    !imageSize.height ||
    !annotation.imageWidth ||
    !annotation.imageHeight ||
    (annotation.imageWidth === imageSize.width && annotation.imageHeight === imageSize.height)
  ) {
    return annotation;
  }

  const scaleX = imageSize.width / annotation.imageWidth;
  const scaleY = imageSize.height / annotation.imageHeight;
  return {
    ...annotation,
    imageWidth: imageSize.width,
    imageHeight: imageSize.height,
    shapes: annotation.shapes.map((shape) => ({
      ...shape,
      points: shape.points.map((point) => [
        clamp(point[0] * scaleX, 0, imageSize.width),
        clamp(point[1] * scaleY, 0, imageSize.height),
      ]),
    })),
  };
}

function jobProgressFromEvents(events: JobEvent[]) {
  const last = [...events].reverse().find((event) => typeof event.total === "number");
  if (!last?.total) return 0;
  return ((last.processed ?? 0) + (last.skipped ?? 0) + (last.failed ?? 0)) / last.total;
}

function applyImageEvent(images: FolderImage[], event: JobEvent): FolderImage[] {
  if (!event.image_path) return images;
  if (!["image_started", "image_done", "image_failed", "image_skipped"].includes(event.type)) return images;
  return images.map((image) => {
    if (!imagePathsMatch(image.path, event.image_path!)) return image;
    if (event.type === "image_started") return { ...image, status: "processing" };
    return {
      ...image,
      hasAnnotation: event.type !== "image_failed" && (event.type !== "image_skipped" || image.hasAnnotation),
      status: event.type === "image_failed" ? "failed" : event.type === "image_skipped" ? "skipped" : "done",
    };
  });
}

function imagePathsMatch(left: string | undefined, right: string | undefined) {
  if (!left || !right) return false;
  const normalize = (value: string) => value.replaceAll("\\", "/").replace(/\/+$/, "");
  const normalizedLeft = normalize(left);
  const normalizedRight = normalize(right);
  if (normalizedLeft === normalizedRight) return true;
  return normalizedLeft.split("/").at(-1) === normalizedRight.split("/").at(-1);
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("zh-CN").format(value);
}

function providerStatusLabel(provider: ModelProvider, summary: ModelProviderSummary | null) {
  if (!summary) return "config.json";
  if (provider === "ollama") {
    return summary.ollamaConfigured ? summary.ollamaVisionModel ?? "Ollama" : "未配置 Ollama";
  }
  return summary.dashscopeConfigured ? "DashScope API" : "未配置 API key";
}

function providerStatusTitle(summary: ModelProviderSummary | null) {
  if (!summary) return "Provider is read from config.json by default.";
  return [
    `default=${summary.provider}`,
    `dashscope=${summary.dashscopeConfigured ? "configured" : "missing api_key"}`,
    `ollama=${summary.ollamaConfigured ? "configured" : "missing model"}`,
    summary.ollamaBaseUrl ? `ollama_base_url=${summary.ollamaBaseUrl}` : null,
  ]
    .filter(Boolean)
    .join("; ");
}

function normalizeGuideDraft(guides: CategoryGuide[], labels: string[]) {
  const byLabel = new Map(guides.map((guide) => [guide.label, guide]));
  return labels.map((label, index) => {
    const guide = byLabel.get(label) ?? guides[index];
    if (!guide) return guideFromLabel(label);
    return {
      label,
      visual_cues: sanitizeGuideField(guide.visual_cues),
      ocr_cues: sanitizeGuideField(guide.ocr_cues),
      negative_cues: sanitizeGuideField(guide.negative_cues),
      ambiguity_notes: sanitizeGuideField(guide.ambiguity_notes),
    };
  });
}

function guideFromLabel(label: string): CategoryGuide {
  return {
    label,
    visual_cues: [],
    ocr_cues: [],
    negative_cues: [],
    ambiguity_notes: [],
  };
}

function listToText(values: string[]) {
  return values.join("\n");
}

function sanitizeGuideField(value: unknown) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  if (typeof value === "string") {
    return textToList(value);
  }
  return [];
}

function textToList(value: string) {
  return value
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function shapeToBox(shape: LabelMeShape): BBox {
  const points = shape.points.length ? shape.points : [[0, 0], [MIN_BOX_SIZE, MIN_BOX_SIZE]];
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const left = Math.min(...xs);
  const top = Math.min(...ys);
  return {
    left,
    top,
    width: Math.max(...xs) - left,
    height: Math.max(...ys) - top,
  };
}

function shapeWithBox(shape: LabelMeShape, box: BBox): LabelMeShape {
  const right = box.left + box.width;
  const bottom = box.top + box.height;
  return {
    ...shape,
    shape_type: "rectangle",
    points: [
      [box.left, box.top],
      [right, box.top],
      [right, bottom],
      [box.left, bottom],
    ],
  };
}

function createShape(label: string, box: BBox): LabelMeShape {
  return shapeWithBox(
    {
      label,
      score: 1,
      points: [],
      group_id: null,
      description: null,
      difficult: false,
      shape_type: "rectangle",
      flags: {},
      attributes: {},
      kie_linking: [],
    },
    box,
  );
}

function resizeBox(
  startBox: BBox,
  mode: DragState["mode"],
  dx: number,
  dy: number,
  bounds: ImageSize,
): BBox {
  let left = startBox.left;
  let top = startBox.top;
  let right = startBox.left + startBox.width;
  let bottom = startBox.top + startBox.height;

  if (mode === "move") {
    const width = right - left;
    const height = bottom - top;
    left = clamp(left + dx, 0, bounds.width - width);
    top = clamp(top + dy, 0, bounds.height - height);
    return { left, top, width, height };
  }

  if (mode.includes("w")) left += dx;
  if (mode.includes("e")) right += dx;
  if (mode.includes("n")) top += dy;
  if (mode.includes("s")) bottom += dy;

  left = clamp(left, 0, bounds.width - MIN_BOX_SIZE);
  top = clamp(top, 0, bounds.height - MIN_BOX_SIZE);
  right = clamp(right, left + MIN_BOX_SIZE, bounds.width);
  bottom = clamp(bottom, top + MIN_BOX_SIZE, bounds.height);
  return { left, top, width: right - left, height: bottom - top };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function isTypingTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable;
}
