from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
import re
from typing import Any, AsyncIterator
from uuid import uuid4

from .constants import GPU_STAGES, STAGES, VIDEO_SUFFIXES, JobStatus, StageName, StageStatus
from .database import Database, utc_now
from .dataset import IMAGE_EXTENSIONS
from .schemas import CategoryGuideDocument, Job, Project, ProjectCreate, ProjectPatch, StageCompleteRequest, StageImportRequest, StageRunRequest
from .settings import Settings


class WorkflowService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = Database(settings.database_path)
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self.gpu_lock = asyncio.Lock()

    def create_project(self, request: ProjectCreate) -> Project:
        return self.db.create_project(request, self.settings.workspace_root)

    def patch_project(self, project_id: str, request: ProjectPatch) -> Project:
        previous = self.db.get_project(project_id)
        values = request.model_dump(exclude_unset=True)
        if "labels" in values:
            values["labels"] = ProjectCreate(name="validate", labels=values["labels"]).labels
        updated = self.db.update_project(project_id, values)
        if updated.labels != previous.labels:
            self.db.set_stage(project_id, StageName.DEFINITION, StageStatus.COMPLETED,
                              summary={"labels": updated.labels, "task_type": "detection"},
                              bump_revision=True)
            # Category edits affect pre-annotation and every consumer after it,
            # but do not require selecting or cleaning the source videos again.
            self.db.invalidate_downstream(project_id, StageName.CLEANING)
            cleaning = updated.stages[STAGES.index(StageName.CLEANING)]
            if cleaning.status in {StageStatus.COMPLETED, StageStatus.IMPORTED}:
                self.db.unlock_next(project_id, StageName.CLEANING)
            updated = self.db.get_project(project_id)
        return updated

    def delete_project(self, project_id: str) -> dict[str, Any]:
        project = self.db.get_project(project_id)
        active = [
            stage.active_job_id for stage in project.stages
            if stage.active_job_id and self.db.get_job(stage.active_job_id).status in {JobStatus.QUEUED, JobStatus.RUNNING}
        ]
        if active:
            raise ValueError("Stop the running project task before deleting the project")
        workspace = Path(project.workspace_path).resolve()
        workspace_root = self.settings.workspace_root.resolve()
        if workspace.parent != workspace_root or workspace.name != project.id:
            raise ValueError(f"Invalid project workspace: {workspace}")
        preserved = [
            artifact.path for artifact in project.artifacts
            if artifact.kind in {"labeled_dataset", "best_pt", "last_pt", "onnx"}
            and not (Path(artifact.path).resolve() == workspace or workspace in Path(artifact.path).resolve().parents)
        ]
        if workspace.is_dir():
            shutil.rmtree(workspace)
        self.db.delete_project(project.id)
        return {"status": "deleted", "project_id": project.id, "preserved": preserved}

    def get_default_category_guides(self, project_id: str) -> dict[str, Any]:
        project = self.db.get_project(project_id)
        path = self._global_category_guides_path()
        guides, updated_at = self._read_guide_library(path)
        matched = [guides[label] for label in project.labels if label in guides]
        missing = [label for label in project.labels if label not in guides]
        return {
            "exists": not missing,
            "source": "global",
            "labels": project.labels,
            "guides": [guide.model_dump(mode="json") for guide in matched],
            "missing_labels": missing,
            "path": str(path),
            "updated_at": updated_at,
        }

    def save_default_category_guides(
        self, project_id: str, request: CategoryGuideDocument
    ) -> dict[str, Any]:
        project = self.db.get_project(project_id)
        if request.labels != project.labels:
            raise ValueError("Guide labels must exactly match the current project labels")
        if [guide.label for guide in request.guides] != project.labels:
            raise ValueError("A complete guide in project class order is required")
        path = self._global_category_guides_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        existing, _ = self._read_guide_library(path)
        existing.update({guide.label: guide for guide in request.guides})
        payload = {
            "version": 1,
            "scope": "global",
            "labels": list(existing),
            "guides": [guide.model_dump(mode="json") for guide in existing.values()],
            "updated_at": utc_now(),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        return {"exists": True, "source": "global", "path": str(path), **payload}

    def save_project_category_guides(
        self, project_id: str, request: CategoryGuideDocument
    ) -> dict[str, Any]:
        project = self.db.get_project(project_id)
        if request.labels != project.labels or [guide.label for guide in request.guides] != project.labels:
            raise ValueError("A complete guide in current project class order is required")
        path = Path(project.workspace_path) / "annotations" / "project_category_guides.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1, "scope": "project", "project_id": project.id,
            "labels": request.labels,
            "guides": [guide.model_dump(mode="json") for guide in request.guides],
            "updated_at": utc_now(),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        return {"exists": True, "source": "project", "path": str(path), **payload}

    def _global_category_guides_path(self) -> Path:
        return Path(getattr(
            self.settings, "global_category_guides_path", self.settings.root / "default_category_guides.json"
        )).resolve()

    @staticmethod
    def _read_guide_library(path: Path) -> tuple[dict[str, Any], str | None]:
        if not path.is_file():
            return {}, None
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            raw_guides = payload.get("guides", []) if isinstance(payload, dict) else []
            guides = {}
            for value in raw_guides:
                guide = CategoryGuideDocument.model_validate({
                    "labels": [value.get("label")], "guides": [value]
                }).guides[0]
                guides[guide.label] = guide
            return guides, payload.get("updated_at") if isinstance(payload, dict) else None
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            return {}, None

    def validate_path(self, value: str, *, file: bool | None = None) -> Path:
        path = Path(value).expanduser().resolve()
        if not any(path == root or root in path.parents for root in self.settings.allowed_roots):
            raise ValueError(f"Path is outside VTAX_ALLOWED_ROOTS: {path}")
        if not path.exists():
            raise FileNotFoundError(path)
        if file is True and not path.is_file():
            raise ValueError(f"Expected a file: {path}")
        if file is False and not path.is_dir():
            raise ValueError(f"Expected a directory: {path}")
        return path

    def import_stage(self, project_id: str, stage: StageName, request: StageImportRequest) -> Project:
        path = self.validate_path(request.path)
        self._validate_artifact(request.artifact_type, path)
        project = self.db.get_project(project_id)
        expected = {
            StageName.DATA: {"videos", "frames"},
            StageName.CLEANING: {"frames", "cleaned_images"},
            StageName.ANNOTATION: {"cleaned_images", "labelme"},
            StageName.REVIEW: {"labelme"},
            StageName.DATASET: {"labelme", "yolo_dataset"},
            StageName.CONFIGURATION: {"yolo_dataset", "training_config"},
            StageName.TRAINING: {"yolo_dataset", "checkpoint"},
            StageName.EXPORT: {"checkpoint"},
        }.get(stage, set())
        if request.artifact_type not in expected:
            raise ValueError(f"{request.artifact_type} cannot be imported at stage {stage}")
        self.db.add_artifact(project_id, stage, request.artifact_type, path, request.metadata)
        self.db.set_stage(project_id, stage, StageStatus.IMPORTED,
                          summary={"artifact_type": request.artifact_type, "path": str(path), **request.metadata},
                          bump_revision=True)
        self._mark_prior_as_imported(project, stage)
        self.db.invalidate_downstream(project_id, stage)
        self.db.unlock_next(project_id, stage)
        return self.db.get_project(project_id)

    async def run_stage(self, project_id: str, stage: StageName, request: StageRunRequest) -> Job:
        project = self.db.get_project(project_id)
        current = project.stages[STAGES.index(stage)]
        if current.status == StageStatus.LOCKED:
            raise ValueError(f"Stage {stage} is locked; complete or import the required upstream stage first")
        if current.active_job_id:
            active = self.db.get_job(current.active_job_id)
            if active.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                raise ValueError(f"Stage already has an active job: {active.id}")
        command, config = self._build_command(project, stage, request.config)
        log_path = Path(project.workspace_path) / "logs" / f"{stage}-{uuid4().hex[:10]}.log"
        job = self.db.create_job(project_id, stage, command, config, log_path)
        task = asyncio.create_task(self._execute(job.id))
        self.tasks[job.id] = task
        task.add_done_callback(lambda _: self.tasks.pop(job.id, None))
        return job

    async def cancel_job(self, job_id: str) -> Job:
        job = self.db.get_job(job_id)
        process = self.processes.get(job_id)
        if process and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
        now = utc_now()
        self.db.update_job(job_id, status=JobStatus.CANCELLED, error="Cancelled by user", finished_at=now)
        self.db.add_event(job_id, "job_cancelled", {"status": JobStatus.CANCELLED})
        self.db.set_stage(job.project_id, job.stage, StageStatus.CANCELLED, warning="Task cancelled by user")
        return self.db.get_job(job_id)

    async def retry_job(self, job_id: str, *, resume: bool = False) -> Job:
        previous = self.db.get_job(job_id)
        config = dict(previous.config)
        if resume:
            run_dir = Path(str(config.get("project", ""))) / str(config.get("name", ""))
            last = run_dir / "weights" / "last.pt"
            if not last.is_file():
                raise ValueError(f"No resumable last.pt was found for job {job_id}")
            config["resume_from"] = str(last)
        return await self.run_stage(previous.project_id, previous.stage, StageRunRequest(config=config))

    def complete_stage(
        self, project_id: str, stage: StageName, request: StageCompleteRequest
    ) -> Project:
        project = self.db.get_project(project_id)
        current = project.stages[STAGES.index(stage)]
        if current.status == StageStatus.LOCKED:
            raise ValueError(f"Stage {stage} is locked; complete or import the required upstream stage first")
        summary = dict(request.metadata)
        warning = None
        if stage == StageName.REVIEW:
            annotation_dir = self._latest_path(project, ("labelme", "annotation_dir"))
            if annotation_dir is None:
                annotation_dir = Path(project.workspace_path) / "annotations"
            review = self.review_summary(annotation_dir)
            summary.update(review)
            if review["total"] == 0:
                raise ValueError("No LabelMe annotations are available for review")
            if review["reviewed"] < review["total"]:
                if not request.force:
                    raise ValueError("Not every image has been reviewed; use force=true to complete with a warning")
                warning = f"Forced completion with {review['reviewed']}/{review['total']} reviewed images"
            annotation_stage = project.stages[STAGES.index(StageName.ANNOTATION)]
            if annotation_stage.status == StageStatus.REVIEW_REQUIRED:
                self.db.set_stage(project_id, StageName.ANNOTATION, StageStatus.COMPLETED,
                                  summary=annotation_stage.summary)
            labeled = self._materialize_labeled_dataset(project, annotation_dir)
            summary.update(labeled)
            self.db.add_artifact(
                project_id, stage, "labeled_dataset", Path(labeled["labeled_dataset_path"]), labeled
            )
        if stage == StageName.DATA:
            registered = 0
            source_directories: list[Path] = []
            source_values = request.metadata.get("source_directories", [])
            selected_values = request.metadata.get("selected_videos", [])
            if not isinstance(source_values, list) or not isinstance(selected_values, list):
                raise ValueError("source_directories and selected_videos must be arrays")
            for value in source_values:
                source = self.validate_path(str(value), file=False)
                if source not in source_directories:
                    source_directories.append(source)
            selected_videos: list[Path] = []
            for value in selected_values:
                video = self.validate_path(str(value), file=True)
                if video.suffix.lower() not in VIDEO_SUFFIXES:
                    raise ValueError(f"Unsupported video format: {video.name}")
                if source_directories and not any(video == root or root in video.parents for root in source_directories):
                    raise ValueError(f"Selected video is outside the saved source folders: {video}")
                if video not in selected_videos:
                    selected_videos.append(video)
            if selected_videos and not source_directories:
                raise ValueError("Save at least one source folder for the selected videos")
            if source_directories and not selected_videos:
                raise ValueError("Select at least one video for the cleaning stage")
            if selected_videos:
                selection_dir = Path(project.workspace_path) / "inputs" / f"videos-{uuid4().hex[:10]}"
                selection_dir.mkdir(parents=True, exist_ok=False)
                for index, video in enumerate(selected_videos, start=1):
                    (selection_dir / f"{index:04d}-{video.name}").symlink_to(video)
                manifest = {
                    "source_directories": [str(path) for path in source_directories],
                    "selected_videos": [str(path) for path in selected_videos],
                    "selected_count": len(selected_videos),
                }
                (selection_dir / "selection.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self.db.add_artifact(project_id, stage, "videos", selection_dir, manifest)
                summary.update({**manifest, "selection_path": str(selection_dir)})
                registered += 1
            for key, kind in (("new_data", "videos"), ("replay_data", "videos"), ("frames_path", "frames")):
                value = request.metadata.get(key)
                if not value:
                    continue
                path = self.validate_path(str(value), file=False)
                self._validate_artifact(kind, path)
                self.db.add_artifact(project_id, stage, kind, path, {"source": key, "read_only": True})
                registered += 1
            if not registered:
                raise ValueError("Select at least one video directory or an existing frame directory")
        if stage == StageName.CONFIGURATION:
            config = self._normalize_training_config(project, request.metadata)
            self.db.save_training_config(project_id, config)
            summary = config
        self.db.set_stage(project_id, stage, StageStatus.COMPLETED, summary=summary,
                          warning=warning, bump_revision=True)
        self.db.invalidate_downstream(project_id, stage)
        self.db.unlock_next(project_id, stage)
        return self.db.get_project(project_id)

    def review_summary(self, annotation_dir: Path) -> dict[str, Any]:
        total = reviewed = empty = 0
        for path in annotation_dir.rglob("*.json"):
            if path.name in {"summary.json", "run_config.json"}:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or "shapes" not in payload:
                continue
            total += 1
            reviewed += int(bool(payload.get("checked")))
            empty += int(not payload.get("shapes"))
        return {"total": total, "reviewed": reviewed, "unreviewed": total - reviewed,
                "reviewed_percent": round(reviewed * 100 / total, 2) if total else 0,
                "empty": empty}

    def list_labeled_datasets(self) -> list[dict[str, Any]]:
        root = self._labeled_dataset_root()
        result: list[dict[str, Any]] = []
        for directory in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            manifest_path = directory / "dataset_manifest.json"
            manifest: dict[str, Any] = {}
            if manifest_path.is_file():
                try:
                    value = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(value, dict):
                        manifest = value
                except (OSError, json.JSONDecodeError):
                    pass
            images = [
                path for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            annotated = sum(1 for image in images if image.with_suffix(".json").is_file())
            result.append({
                "id": directory.name,
                "name": str(manifest.get("project_name") or directory.name.removesuffix("_labeled")),
                "path": str(directory),
                "project_id": manifest.get("project_id"),
                "labels": manifest.get("labels", []),
                "image_count": len(images),
                "annotated_count": annotated,
                "updated_at": manifest.get("updated_at") or utc_now(),
            })
        return result

    def list_weights(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        weight_root = self._weight_root()
        for path in sorted(weight_root.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
            metadata = self._sidecar_metadata(path)
            result.append({
                "id": f"permanent:{path.name}", "name": path.name, "path": str(path),
                "source": "project", "project_id": metadata.get("project_id"),
                "project_name": metadata.get("project_name"), "kind": metadata.get("kind", "trained"),
                "bytes": path.stat().st_size, "updated_at": metadata.get("updated_at"),
            })
        known = {item["path"] for item in result}
        for size in "nsmlx":
            path = self.settings.model_path(size).resolve()
            if path.is_file() and str(path) not in known:
                result.append({
                    "id": f"base:yolo26{size}", "name": f"YOLO26{size} 基础权重", "path": str(path),
                    "source": "base", "project_id": None, "project_name": "系统模型",
                    "kind": "pretrained", "bytes": path.stat().st_size, "updated_at": None,
                })
        return result

    def finish_project(self, project_id: str) -> dict[str, Any]:
        project = self.db.get_project(project_id)
        export_stage = project.stages[STAGES.index(StageName.EXPORT)]
        if export_stage.status != StageStatus.COMPLETED:
            raise ValueError("Export must complete successfully before finishing the project")
        permanent_onnx = [
            artifact for artifact in project.artifacts
            if artifact.stage == StageName.EXPORT and artifact.kind == "onnx"
            and Path(artifact.path).is_file() and Path(artifact.path).parent == self._onnx_root()
        ]
        if not permanent_onnx:
            raise ValueError("No permanent ONNX artifact is available")
        workspace = Path(project.workspace_path).resolve()
        workspace_root = self.settings.workspace_root.resolve()
        if workspace.parent != workspace_root or not workspace.is_dir():
            raise ValueError(f"Invalid project workspace: {workspace}")
        removed: list[str] = []
        for name in ("inputs", "frames", "cleaned", "dataset", "runs", "exports", "logs"):
            target = workspace / name
            if target.is_dir():
                shutil.rmtree(target)
                removed.append(name)
        annotations = workspace / "annotations"
        if annotations.is_dir():
            for item in annotations.iterdir():
                if item.name in {"default_category_guides.json", "project_category_guides.json"}:
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            removed.append("annotations/intermediate")
        return {
            "status": "completed", "project_id": project.id, "removed": removed,
            "preserved": {
                "labeled_datasets": [artifact.path for artifact in project.artifacts if artifact.kind == "labeled_dataset"],
                "weights": [artifact.path for artifact in project.artifacts if artifact.kind in {"best_pt", "last_pt"}],
                "onnx": [artifact.path for artifact in permanent_onnx],
                "category_guides": str(annotations / "project_category_guides.json"),
            },
        }

    def _materialize_labeled_dataset(self, project: Project, source: Path) -> dict[str, Any]:
        source = source.resolve()
        root = self._labeled_dataset_root()
        safe_name = re.sub(r"[^\w.-]+", "_", project.name, flags=re.UNICODE).strip("._")
        dataset_name = f"{safe_name or project.id}_labeled"
        destination = root / dataset_name
        temporary = root / f".{dataset_name}.{uuid4().hex[:10]}.tmp"
        temporary.mkdir(parents=True, exist_ok=False)
        copied = 0
        try:
            for image in sorted(
                path for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                annotation = image.with_suffix(".json")
                if not annotation.is_file():
                    continue
                relative = image.relative_to(source)
                image_target = temporary / relative
                json_target = image_target.with_suffix(".json")
                image_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, image_target)
                shutil.copy2(annotation, json_target)
                copied += 1
            if copied == 0:
                raise ValueError("No image and LabelMe JSON pairs are available to save")
            provenance = source / "provenance.jsonl"
            if provenance.is_file():
                shutil.copy2(provenance, temporary / provenance.name)
            manifest = {
                "version": 1,
                "project_id": project.id,
                "project_name": project.name,
                "labels": project.labels,
                "image_count": copied,
                "source": str(source),
                "updated_at": utc_now(),
            }
            (temporary / "dataset_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if destination.exists():
                backup = root / f".{dataset_name}.previous.{uuid4().hex[:8]}"
                destination.replace(backup)
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return {
            "labeled_dataset_path": str(destination),
            "labeled_dataset_name": dataset_name,
            "labeled_images": copied,
        }

    def _labeled_dataset_root(self) -> Path:
        root = Path(getattr(self.settings, "labeled_dataset_root", self.settings.root / "LabeledDataset"))
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    def _weight_root(self) -> Path:
        root = Path(getattr(self.settings, "weight_root", self.settings.root / "Weight"))
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    def _onnx_root(self) -> Path:
        root = Path(getattr(self.settings, "onnx_root", self.settings.root / "ONNX"))
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    @staticmethod
    def _sidecar_metadata(path: Path) -> dict[str, Any]:
        sidecar = path.with_suffix(path.suffix + ".json")
        if not sidecar.is_file():
            return {}
        try:
            value = json.loads(sidecar.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _validate_labeled_dataset(self, path: Path) -> Path:
        path = path.resolve()
        root = self._labeled_dataset_root()
        if path.parent != root or not path.is_dir() or path.name.startswith("."):
            raise ValueError(f"Expected a dataset directly under {root}: {path}")
        if not any(
            item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS for item in path.rglob("*")
        ):
            raise ValueError(f"No labeled images found in dataset: {path}")
        return path

    async def events(self, job_id: str, after: int = 0) -> AsyncIterator[str]:
        terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED}
        cursor = after
        while True:
            events = self.db.events(job_id, cursor)
            for event in events:
                cursor = max(cursor, int(event["sequence"]))
                yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            job = self.db.get_job(job_id)
            if job.status in terminal and not events:
                break
            if not events:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.75)

    async def _execute(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        lock = self.gpu_lock if job.stage in GPU_STAGES else _null_lock()
        async with lock:
            self.db.update_job(job_id, status=JobStatus.RUNNING, started_at=utc_now())
            self.db.set_stage(job.project_id, job.stage, StageStatus.RUNNING, active_job_id=job_id)
            self.db.add_event(job_id, "job_started", {"command": job.command})
            log_path = Path(job.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            process: asyncio.subprocess.Process | None = None
            summary: dict[str, Any] = {}
            error: str | None = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *job.command,
                    cwd=self.settings.repo_root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "PYTHONUTF8": "1",
                        "PYTHONIOENCODING": "utf-8",
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": os.pathsep.join(
                            value for value in (str(self.settings.root), str(self.settings.repo_root), os.environ.get("PYTHONPATH", ""))
                            if value
                        ),
                    },
                )
                self.processes[job_id] = process
                self.db.update_job(job_id, pid=process.pid)
                assert process.stdout is not None
                with log_path.open("a", encoding="utf-8") as log:
                    buffer = ""

                    def record_line(line: str) -> None:
                        nonlocal summary, error
                        if not line.strip():
                            return
                        log.write(line + "\n")
                        log.flush()
                        event = self._parse_line(line)
                        self.db.add_event(job_id, event.pop("type"), event)
                        if "summary" in event and isinstance(event["summary"], dict):
                            summary = event["summary"]
                        if event.get("error"):
                            error = str(event["error"])
                        progress = _progress_from_event(event)
                        if progress:
                            self.db.update_job(job_id, progress=progress)

                    while True:
                        raw = await process.stdout.read(4096)
                        if not raw:
                            break
                        buffer += raw.decode("utf-8", errors="replace")
                        parts = re.split(r"[\r\n]+", buffer)
                        buffer = parts.pop()
                        for line in parts:
                            record_line(line)
                    if buffer.strip():
                        record_line(buffer)
                exit_code = await process.wait()
                if exit_code == 0 and not error:
                    self.db.update_job(job_id, status=JobStatus.COMPLETED, exit_code=0,
                                       progress=summary, finished_at=utc_now())
                    final_status = StageStatus.REVIEW_REQUIRED if job.stage == StageName.ANNOTATION else StageStatus.COMPLETED
                    self.db.set_stage(job.project_id, job.stage, final_status, summary=summary,
                                      active_job_id=None, bump_revision=True)
                    self._register_output(job, summary)
                    self.db.invalidate_downstream(job.project_id, job.stage)
                    self.db.unlock_next(job.project_id, job.stage)
                    self.db.add_event(job_id, "process_exit", {"status": "completed", "exit_code": 0})
                else:
                    message = error or f"Process exited with status {exit_code}"
                    self.db.update_job(job_id, status=JobStatus.FAILED, exit_code=exit_code,
                                       error=message, finished_at=utc_now())
                    self.db.set_stage(job.project_id, job.stage, StageStatus.FAILED,
                                      active_job_id=None, warning=message)
                    self.db.add_event(job_id, "process_exit", {"status": "failed", "exit_code": exit_code, "error": message})
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                self.db.update_job(job_id, status=JobStatus.FAILED, error=message, finished_at=utc_now())
                self.db.set_stage(job.project_id, job.stage, StageStatus.FAILED, active_job_id=None, warning=message)
                self.db.add_event(job_id, "job_error", {"error": message})
            finally:
                self.processes.pop(job_id, None)

    def _build_command(
        self, project: Project, stage: StageName, supplied: dict[str, Any]
    ) -> tuple[list[str], dict[str, Any]]:
        workspace = Path(project.workspace_path)
        token = uuid4().hex[:10]
        if stage == StageName.CLEANING:
            input_path = self._path_from_config_or_artifact(project, supplied, "input_path", ("videos", "frames"))
            output = workspace / "cleaned" / token
            frames = workspace / "frames" / token
            config = {
                "input_path": str(input_path), "output_path": str(output), "frames_path": str(frames),
                "interval": float(supplied.get("interval", 0.5)), "jpeg_quality": int(supplied.get("jpeg_quality", 95)),
                "target_count": int(supplied.get("target_count", 1000)), "batch_size": int(supplied.get("batch_size", 16)),
                "device": supplied.get("device", "cuda"), "seed": int(supplied.get("seed", 42)),
                "max_images": supplied.get("max_images"),
            }
            command = [str(self.settings.depth_python), "-m", "frame_cluster_filter.workflow",
                       "--input", str(input_path), "--frames", str(frames),
                       "--model", str(self.settings.dino_model), "--output", str(output),
                       "--interval", str(config["interval"]), "--quality", str(config["jpeg_quality"]),
                       "--target-count", str(config["target_count"]), "--batch-size", str(config["batch_size"]),
                       "--device", str(config["device"]), "--seed", str(config["seed"])]
            if config["max_images"]:
                command.extend(("--max-images", str(config["max_images"])))
            return command, config
        if stage == StageName.ANNOTATION:
            input_path = self._path_from_config_or_artifact(project, supplied, "input_path", ("cleaned_images", "frames"))
            output = workspace / "annotations" / token
            config = {"input_path": str(input_path), "output_path": str(output),
                      "provider": supplied.get("provider", "dashscope"), "labels": project.labels,
                      "overwrite": bool(supplied.get("overwrite", False))}
            command = [str(self.settings.annotation_python), "-m", "data_annotation_agent.vtax.workflow_cli", "annotation",
                       "--input-dir", str(input_path), "--output-dir", str(output), "--labels", *project.labels,
                       "--config", str(self.settings.config_path), "--provider", str(config["provider"])]
            if config["overwrite"]:
                command.append("--overwrite")
            return command, config
        if stage == StageName.DATASET:
            input_path = self._path_from_config_or_artifact(
                project, supplied, "new_data_path", ("labeled_dataset", "labelme", "annotation_dir")
            )
            if supplied.get("new_data_path"):
                input_path = self._validate_labeled_dataset(input_path)
            replay_values = supplied.get("replay_paths", [])
            if not isinstance(replay_values, list):
                raise ValueError("replay_paths must be an array")
            replay_paths = [
                self._validate_labeled_dataset(self.validate_path(str(value), file=False))
                for value in replay_values
            ]
            if input_path in replay_paths:
                raise ValueError("The new dataset cannot also be selected as replay data")
            output = workspace / "dataset" / token
            config = {"new_data_path": str(input_path), "replay_paths": [str(path) for path in replay_paths],
                      "replay_ratio": float(supplied.get("replay_ratio", 0.4)),
                      "output_path": str(output), "labels": project.labels,
                      "seed": int(supplied.get("seed", 42)), "ratios": supplied.get("ratios", [0.8, 0.15, 0.05])}
            ratios = config["ratios"]
            command = [str(self.settings.annotation_python), "-m", "data_annotation_agent.vtax.workflow_cli", "dataset",
                     "--input-dir", str(input_path), "--output-dir", str(output), "--labels", *project.labels,
                     "--seed", str(config["seed"]), "--train-ratio", str(ratios[0]),
                     "--val-ratio", str(ratios[1]), "--test-ratio", str(ratios[2]),
                     "--replay-ratio", str(config["replay_ratio"])]
            for replay_path in replay_paths:
                command.extend(("--replay-dir", str(replay_path)))
            return command, config
        if stage == StageName.TRAINING:
            config = self._normalize_training_config(project, {**(self.db.training_config(project.id) or {}), **supplied})
            dataset_kinds = ("yolo_dataset", "dataset_yaml")
            # The YAML selector was removed from the UI. Always prefer the newest
            # registered dataset so retrying an old failed job cannot revive a stale path.
            data_source = {} if self._latest_path(project, dataset_kinds) is not None else config
            data = self._path_from_config_or_artifact(project, data_source, "data", dataset_kinds)
            model = self.validate_path(
                str(config.get("model") or self.settings.model_path(config["model_size"])), file=True
            )
            if model.suffix.lower() != ".pt":
                raise ValueError("Training weights must be a .pt file")
            run_root = workspace / "runs" / "training"
            name = f"training-{token}"
            artifact_name = f"{self._safe_project_name(project)}_{token}"
            config.update({"data": str(data), "model": str(model), "project": str(run_root), "name": name})
            command = [str(self.settings.yolo_python), "-m", "data_annotation_agent.vtax.workflow_cli", "training",
                       "--model", str(model), "--data", str(data), "--project", str(run_root), "--name", name,
                       "--weight-output-dir", str(self._weight_root()), "--artifact-name", artifact_name,
                       "--project-id", project.id, "--project-name", project.name,
                       "--epochs", str(config["epochs"]), "--imgsz", str(config["imgsz"]),
                       "--batch", str(config["batch"]), "--device", str(config["device"]),
                       "--workers", str(config["workers"]), "--optimizer", str(config["optimizer"]),
                       "--patience", str(config["patience"]), "--seed", str(config["seed"])]
            if supplied.get("resume_from"):
                resume_from = self.validate_path(str(supplied["resume_from"]), file=True)
                config["resume_from"] = str(resume_from)
                command.extend(("--resume-from", str(resume_from)))
            command.append("--amp" if config["amp"] else "--no-amp")
            return command, config
        if stage == StageName.EXPORT:
            weights = self._path_from_config_or_artifact(project, supplied, "weights", ("best_pt", "checkpoint"))
            dataset_kinds = ("dataset_yaml", "yolo_dataset")
            data_source = {} if self._latest_path(project, dataset_kinds) is not None else supplied
            data = self._path_from_config_or_artifact(project, data_source, "data", dataset_kinds)
            precision = str(supplied.get("precision", "fp32")).lower()
            if precision not in {"fp32", "fp16", "int8"}:
                raise ValueError("precision must be fp32, fp16, or int8")
            config = {"weights": str(weights), "data": str(data), "imgsz": int(supplied.get("imgsz", 960)),
                      "device": supplied.get("device", "0"), "precision": precision}
            artifact_name = f"{self._safe_project_name(project)}_{precision}_{token}"
            return ([str(self.settings.yolo_python), "-m", "data_annotation_agent.vtax.workflow_cli", "export",
                     "--weights", str(weights), "--data", str(data), "--imgsz", str(config["imgsz"]),
                     "--device", str(config["device"]), "--precision", precision,
                     "--output-dir", str(self._onnx_root()), "--artifact-name", artifact_name,
                     "--project-id", project.id, "--project-name", project.name], config)
        raise ValueError(f"Stage {stage} is completed synchronously and cannot be run as a job")

    def _normalize_training_config(self, project: Project, supplied: dict[str, Any]) -> dict[str, Any]:
        size = str(supplied.get("model_size", "m"))
        if size not in {"n", "s", "m", "l", "x"}:
            raise ValueError("model_size must be one of n/s/m/l/x")
        config = {
            "model_size": size, "model": supplied.get("model"), "data": supplied.get("data"),
            "epochs": int(supplied.get("epochs", 80)), "imgsz": int(supplied.get("imgsz", 960)),
            "batch": int(supplied.get("batch", 8)), "device": str(supplied.get("device", "0")),
            "workers": int(supplied.get("workers", 8)), "optimizer": str(supplied.get("optimizer", "auto")),
            "patience": int(supplied.get("patience", 25)), "seed": int(supplied.get("seed", 42)),
            "amp": bool(supplied.get("amp", False)),
        }
        if config["epochs"] <= 0 or config["imgsz"] <= 0 or config["batch"] == 0 or config["workers"] < 0:
            raise ValueError("Invalid training parameters")
        return config

    @staticmethod
    def _safe_project_name(project: Project) -> str:
        value = re.sub(r"[^\w.-]+", "_", project.name, flags=re.UNICODE).strip("._")
        return value or project.id

    def _register_output(self, job: Job, summary: dict[str, Any]) -> None:
        config = job.config
        if job.stage == StageName.CLEANING:
            path = Path(config["output_path"])
            selected = summary.get("selected_directory")
            self.db.add_artifact(job.project_id, job.stage, "cleaned_images", Path(selected or path), summary)
        elif job.stage == StageName.ANNOTATION:
            self.db.add_artifact(job.project_id, job.stage, "annotation_dir", Path(config["output_path"]), summary)
        elif job.stage == StageName.DATASET:
            output = Path(config["output_path"])
            self.db.add_artifact(job.project_id, job.stage, "dataset_yaml", output / "dataset.yaml", summary)
        elif job.stage == StageName.TRAINING:
            if summary.get("best_pt"):
                self.db.add_artifact(job.project_id, job.stage, "best_pt", Path(summary["best_pt"]), summary)
            if summary.get("last_pt"):
                self.db.add_artifact(job.project_id, job.stage, "last_pt", Path(summary["last_pt"]), summary)
        elif job.stage == StageName.EXPORT and summary.get("onnx"):
            self.db.add_artifact(job.project_id, job.stage, "onnx", Path(summary["onnx"]), summary)

    def _latest_path(self, project: Project, kinds: tuple[str, ...]) -> Path | None:
        for artifact in reversed(project.artifacts):
            if artifact.kind in kinds:
                return Path(artifact.path)
        return None

    def _path_from_config_or_artifact(
        self, project: Project, config: dict[str, Any], key: str, kinds: tuple[str, ...]
    ) -> Path:
        if config.get(key):
            return self.validate_path(str(config[key]))
        path = self._latest_path(project, kinds)
        if path is None:
            raise ValueError(f"No input is available for {key}; import an upstream artifact first")
        if path.is_dir() and any(kind in {"yolo_dataset", "dataset_yaml"} for kind in kinds):
            yaml_path = path / "dataset.yaml"
            if yaml_path.is_file():
                return yaml_path
        return path

    def _validate_artifact(self, kind: str, path: Path) -> None:
        if kind == "yolo_dataset":
            yaml_path = path if path.is_file() else path / "dataset.yaml"
            if not yaml_path.is_file():
                raise ValueError("YOLO dataset import requires a dataset.yaml file")
        elif kind == "checkpoint" and path.suffix.lower() != ".pt":
            raise ValueError("Checkpoint import requires a .pt file")
        elif kind == "training_config":
            if not path.is_file() or path.suffix.lower() != ".json":
                raise ValueError("Training config import requires a JSON file")
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                raise ValueError("Training config JSON must contain an object")
        elif kind == "labelme":
            root = path.parent if path.is_file() else path
            if not any(item.is_file() and item.suffix.lower() == ".json" for item in root.rglob("*.json")):
                raise ValueError("No LabelMe JSON files found")
        elif kind in {"videos", "frames", "cleaned_images"} and not path.is_dir():
            raise ValueError(f"{kind} import requires a directory")

    def _mark_prior_as_imported(self, project: Project, stage: StageName) -> None:
        current = STAGES.index(stage)
        for item in project.stages[:current]:
            if item.status == StageStatus.LOCKED:
                self.db.set_stage(project.id, item.name, StageStatus.IMPORTED,
                                  summary={"provided_by": stage, "note": "Satisfied by downstream artifact import"})

    @staticmethod
    def _parse_line(line: str) -> dict[str, Any]:
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                payload.setdefault("type", "log")
                return payload
        except json.JSONDecodeError:
            pass
        clean = _ANSI_ESCAPE.sub("", line)
        number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        for segment in reversed([item.strip() for item in clean.replace("\r", "\n").splitlines() if item.strip()]):
            epoch = re.search(
                rf"(?P<epoch>\d+)/(?P<epochs>\d+)\s+(?P<memory>{number})G\s+"
                rf"(?P<box>{number})\s+(?P<cls>{number})\s+(?P<dfl>{number})\s+"
                rf"(?P<instances>\d+)\s+(?P<imgsz>\d+):\s*100%",
                segment,
            )
            if epoch:
                values = epoch.groupdict()
                return {
                    "type": "training_epoch", "epoch": int(values["epoch"]),
                    "epochs": int(values["epochs"]), "gpu_memory_gb": float(values["memory"]),
                    "box_loss": float(values["box"]), "cls_loss": float(values["cls"]),
                    "dfl_loss": float(values["dfl"]), "instances": int(values["instances"]),
                    "imgsz": int(values["imgsz"]),
                }
            metrics = re.match(
                rf"^all\s+(?P<images>\d+)\s+(?P<instances>\d+)\s+"
                rf"(?P<precision>{number})\s+(?P<recall>{number})\s+"
                rf"(?P<map50>{number})\s+(?P<map5095>{number})$",
                segment,
            )
            if metrics:
                values = metrics.groupdict()
                return {
                    "type": "training_metrics", "images": int(values["images"]),
                    "instances": int(values["instances"]), "precision": float(values["precision"]),
                    "recall": float(values["recall"]), "map50": float(values["map50"]),
                    "map50_95": float(values["map5095"]),
                }
        return {"type": "log", "message": clean}


class _null_lock:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: object) -> None:
        return None


def _progress_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    keys = {"current", "total", "percent", "phase", "processed", "failed", "skipped", "epoch", "epochs", "message",
            "box_loss", "cls_loss", "dfl_loss", "precision", "recall", "map50", "map50_95"}
    progress = {key: event[key] for key in keys if key in event}
    return progress or None


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
