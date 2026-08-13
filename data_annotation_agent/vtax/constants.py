from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # Python 3.10 deployment compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value


class StageName(StrEnum):
    DEFINITION = "definition"
    DATA = "data"
    CLEANING = "cleaning"
    ANNOTATION = "annotation"
    REVIEW = "review"
    DATASET = "dataset"
    CONFIGURATION = "configuration"
    TRAINING = "training"
    EXPORT = "export"


STAGES: tuple[StageName, ...] = tuple(StageName)


class StageStatus(StrEnum):
    LOCKED = "locked"
    READY = "ready"
    IMPORTED = "imported"
    QUEUED = "queued"
    RUNNING = "running"
    REVIEW_REQUIRED = "review_required"
    COMPLETED = "completed"
    FAILED = "failed"
    STALE = "stale"
    CANCELLED = "cancelled"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


TERMINAL_STAGE_STATUSES = {
    StageStatus.IMPORTED,
    StageStatus.COMPLETED,
    StageStatus.REVIEW_REQUIRED,
}

GPU_STAGES = {
    StageName.CLEANING,
    StageName.ANNOTATION,
    StageName.TRAINING,
    StageName.EXPORT,
}

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}
