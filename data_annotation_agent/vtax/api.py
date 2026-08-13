from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from .constants import VIDEO_SUFFIXES, StageName
from .schemas import CategoryGuideDocument, ProjectCreate, ProjectPatch, StageCompleteRequest, StageImportRequest, StageRunRequest
from .service import WorkflowService
from .settings import settings


app = FastAPI(title="VTAX Workflow API", version="1.0.0")
service = WorkflowService(settings)


def failure(error: Exception) -> HTTPException:
    if isinstance(error, KeyError):
        return HTTPException(status_code=404, detail="Resource not found")
    if isinstance(error, (ValueError, FileNotFoundError)):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=500, detail=str(error))


@app.get("/api/v1/projects")
def list_projects():
    return service.db.list_projects()


@app.post("/api/v1/projects", status_code=201)
def create_project(request: ProjectCreate):
    try:
        return service.create_project(request)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.get("/api/v1/projects/{project_id}")
def get_project(project_id: str):
    try:
        return service.db.get_project(project_id)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.patch("/api/v1/projects/{project_id}")
def patch_project(project_id: str, request: ProjectPatch):
    try:
        return service.patch_project(project_id, request)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.delete("/api/v1/projects/{project_id}")
def delete_project(project_id: str):
    try:
        return service.delete_project(project_id)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.get("/api/v1/projects/{project_id}/category-guides")
def get_default_category_guides(project_id: str):
    try:
        return service.get_default_category_guides(project_id)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.put("/api/v1/projects/{project_id}/category-guides")
def save_default_category_guides(project_id: str, request: CategoryGuideDocument):
    try:
        return service.save_default_category_guides(project_id, request)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.put("/api/v1/projects/{project_id}/project-category-guides")
def save_project_category_guides(project_id: str, request: CategoryGuideDocument):
    try:
        return service.save_project_category_guides(project_id, request)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.post("/api/v1/projects/{project_id}/stages/{stage}/import")
def import_stage(project_id: str, stage: StageName, request: StageImportRequest):
    try:
        return service.import_stage(project_id, stage, request)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.post("/api/v1/projects/{project_id}/stages/{stage}/run", status_code=202)
async def run_stage(project_id: str, stage: StageName, request: StageRunRequest):
    try:
        return await service.run_stage(project_id, stage, request)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.post("/api/v1/projects/{project_id}/stages/{stage}/complete")
def complete_stage(project_id: str, stage: StageName, request: StageCompleteRequest):
    try:
        return service.complete_stage(project_id, stage, request)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return service.db.get_job(job_id)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.get("/api/v1/projects/{project_id}/stages/{stage}/jobs/latest")
def latest_stage_job(project_id: str, stage: StageName):
    try:
        service.db.get_project(project_id)
        job = service.db.latest_job_for_stage(project_id, stage)
        return {"job": job, "events": service.db.events(job.id) if job else []}
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.get("/api/v1/labeled-datasets")
def list_labeled_datasets():
    try:
        return {"datasets": service.list_labeled_datasets()}
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.get("/api/v1/weights")
def list_weights():
    try:
        return {"weights": service.list_weights()}
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.post("/api/v1/projects/{project_id}/finish")
def finish_project(project_id: str):
    try:
        return service.finish_project(project_id)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.get("/api/v1/jobs/{job_id}/events")
def job_events(job_id: str, request: Request):
    try:
        service.db.get_job(job_id)
        after = int(request.headers.get("last-event-id", "0") or "0")
        return StreamingResponse(service.events(job_id, after), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.post("/api/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    try:
        return await service.cancel_job(job_id)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.post("/api/v1/jobs/{job_id}/retry", status_code=202)
async def retry_job(job_id: str):
    try:
        return await service.retry_job(job_id)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.post("/api/v1/jobs/{job_id}/resume", status_code=202)
async def resume_job(job_id: str):
    try:
        return await service.retry_job(job_id, resume=True)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.get("/api/v1/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str):
    try:
        with service.db.connect() as db:
            row = db.execute("SELECT path FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        path = service.validate_path(row["path"], file=True)
        return FileResponse(path, filename=path.name)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.post("/api/v1/files/scan")
def scan_files(payload: dict[str, Any]):
    try:
        path = service.validate_path(str(payload.get("path", "")), file=False)
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        videos = images = bytes_total = 0
        duration_seconds = 0.0
        formats: dict[str, int] = {}
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            suffix = item.suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                videos += 1
                bytes_total += item.stat().st_size
                formats[suffix.lstrip(".")] = formats.get(suffix.lstrip("."), 0) + 1
                try:
                    probe = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(item)],
                        capture_output=True, text=True, timeout=15, check=False,
                    )
                    if probe.returncode == 0:
                        duration_seconds += float(probe.stdout.strip() or 0)
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
            elif suffix in image_suffixes:
                images += 1
                bytes_total += item.stat().st_size
        return {"path": str(path), "videos": videos, "images": images, "bytes": bytes_total,
                "duration_seconds": round(duration_seconds, 3), "formats": formats}
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.post("/api/v1/files/videos")
def list_videos(payload: dict[str, Any]):
    try:
        values = payload.get("paths", [])
        if not isinstance(values, list) or not values:
            raise ValueError("Select at least one video folder")
        if len(values) > 50:
            raise ValueError("At most 50 source folders can be browsed at once")
        videos: dict[str, dict[str, Any]] = {}
        for value in values:
            root = service.validate_path(str(value), file=False)
            for item in root.rglob("*"):
                if not item.is_file() or item.suffix.lower() not in VIDEO_SUFFIXES:
                    continue
                video = service.validate_path(str(item), file=True)
                key = str(video)
                if key in videos:
                    continue
                stat = video.stat()
                videos[key] = {
                    "path": key,
                    "name": video.name,
                    "directory": str(video.parent),
                    "bytes": stat.st_size,
                    "format": video.suffix.lower().lstrip("."),
                }
                if len(videos) >= 1000:
                    break
            if len(videos) >= 1000:
                break
        return {
            "videos": sorted(videos.values(), key=lambda item: (item["directory"], item["name"])),
            "total": len(videos),
            "truncated": len(videos) >= 1000,
        }
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.get("/api/v1/files/video")
def preview_video(path: str):
    try:
        video = service.validate_path(path, file=True)
        if video.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video format: {video.name}")
        media_type = mimetypes.guess_type(video.name)[0] or "application/octet-stream"
        return FileResponse(video, media_type=media_type)
    except Exception as error:  # noqa: BLE001
        raise failure(error) from error


@app.get("/api/v1/system/health")
def system_health():
    gpu = False
    try:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
            nvidia_smi = "/usr/lib/wsl/lib/nvidia-smi"
        gpu = bool(nvidia_smi) and subprocess.run(
            [str(nvidia_smi), "-L"], capture_output=True, timeout=5, check=False
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        pass
    environments = {
        "vtax": settings.annotation_python.exists(),
        "depthv3": settings.depth_python.exists(),
        "yolo": settings.yolo_python.exists(),
    }
    models = {
        "dinov3": settings.dino_model.is_dir(),
        **{f"yolo26{size}": settings.model_path(size).is_file() for size in "nsmlx"},
    }
    providers = {"dashscope": False, "ollama": False}
    if settings.config_path.is_file():
        try:
            config = json.loads(settings.config_path.read_text(encoding="utf-8"))
            providers["dashscope"] = bool(config.get("api_key"))
            providers["ollama"] = bool(config.get("ollama_vision_model"))
        except (OSError, json.JSONDecodeError):
            pass
    disk = shutil.disk_usage(settings.workspace_root)
    ok = gpu and all(environments.values()) and models["dinov3"] and any(models[key] for key in models if key.startswith("yolo"))
    return {"status": "ok" if ok else "degraded", "workspace_root": str(settings.workspace_root),
            "database": settings.database_path.is_file(), "gpu": gpu, "environments": environments,
            "models": models, "disk_free_bytes": disk.free, "providers": providers}


def main() -> None:
    import uvicorn

    uvicorn.run("data_annotation_agent.vtax.api:app", host=os.environ.get("VTAX_API_HOST", "127.0.0.1"),
                port=int(os.environ.get("VTAX_API_PORT", "8100")), reload=False)


if __name__ == "__main__":
    main()
