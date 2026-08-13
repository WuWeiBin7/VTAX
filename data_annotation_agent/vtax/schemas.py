from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from data_annotation_agent.schemas import CategoryGuide

from .constants import JobStatus, StageName, StageStatus


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    task_type: Literal["detection"] = "detection"
    labels: list[str] = Field(min_length=1)
    start_stage: StageName = StageName.DATA

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Project name cannot be empty")
        return value

    @field_validator("labels")
    @classmethod
    def clean_labels(cls, labels: list[str]) -> list[str]:
        cleaned = [label.strip() for label in labels if label.strip()]
        if not cleaned:
            raise ValueError("At least one label is required")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Labels must be unique")
        return cleaned


class ProjectPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    labels: list[str] | None = None


class CategoryGuideDocument(BaseModel):
    """A complete ordered guide set for the current project labels."""

    labels: list[str] = Field(min_length=1)
    guides: list[CategoryGuide] = Field(min_length=1)


class StageImportRequest(BaseModel):
    artifact_type: Literal[
        "videos",
        "frames",
        "cleaned_images",
        "labelme",
        "yolo_dataset",
        "training_config",
        "checkpoint",
    ]
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageRunRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


class StageCompleteRequest(BaseModel):
    force: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    id: str
    project_id: str
    stage: StageName
    kind: str
    path: str
    metadata: dict[str, Any]
    created_at: datetime


class Stage(BaseModel):
    project_id: str
    name: StageName
    position: int
    status: StageStatus
    revision: int
    config: dict[str, Any]
    summary: dict[str, Any]
    active_job_id: str | None
    warning: str | None
    updated_at: datetime


class Project(BaseModel):
    id: str
    name: str
    description: str
    task_type: Literal["detection"] = "detection"
    labels: list[str]
    start_stage: StageName
    workspace_path: str
    created_at: datetime
    updated_at: datetime
    stages: list[Stage] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)


class Job(BaseModel):
    id: str
    project_id: str
    stage: StageName
    status: JobStatus
    command: list[str]
    config: dict[str, Any]
    progress: dict[str, Any]
    error: str | None
    log_path: str
    pid: int | None
    exit_code: int | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class SystemHealth(BaseModel):
    status: Literal["ok", "degraded"]
    workspace_root: str
    database: bool
    gpu: bool
    environments: dict[str, bool]
    models: dict[str, bool]
    disk_free_bytes: int
    providers: dict[str, bool]


def as_path(value: str) -> Path:
    return Path(value).expanduser().resolve()
