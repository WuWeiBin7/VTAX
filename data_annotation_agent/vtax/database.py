from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
from uuid import uuid4

from .constants import JobStatus, STAGES, StageName, StageStatus
from .schemas import Artifact, Job, Project, ProjectCreate, Stage


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
                    task_type TEXT NOT NULL, labels_json TEXT NOT NULL, start_stage TEXT NOT NULL,
                    workspace_path TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stages (
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    name TEXT NOT NULL, position INTEGER NOT NULL, status TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1, config_json TEXT NOT NULL DEFAULT '{}',
                    summary_json TEXT NOT NULL DEFAULT '{}', active_job_id TEXT,
                    warning TEXT, updated_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, name)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    stage TEXT NOT NULL, status TEXT NOT NULL, command_json TEXT NOT NULL,
                    config_json TEXT NOT NULL, progress_json TEXT NOT NULL DEFAULT '{}', error TEXT,
                    log_path TEXT NOT NULL, pid INTEGER, exit_code INTEGER,
                    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL, type TEXT NOT NULL, payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, UNIQUE(job_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    stage TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_configs (
                    project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
                    config_json TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_job ON job_events(job_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id, created_at DESC);
                """
            )
            interrupted = utc_now()
            db.execute(
                "UPDATE jobs SET status=?, error=?, finished_at=? WHERE status IN (?, ?)",
                (JobStatus.INTERRUPTED, "Worker restarted while this job was active", interrupted,
                 JobStatus.RUNNING, JobStatus.QUEUED),
            )
            db.execute(
                "UPDATE stages SET status=?, active_job_id=NULL, warning=?, updated_at=? "
                "WHERE status IN (?, ?)",
                (StageStatus.FAILED, "Task interrupted by worker restart", interrupted,
                 StageStatus.RUNNING, StageStatus.QUEUED),
            )
            # Category/task definition now lives in the project creation form.
            # Migrate projects created by the previous UI so they enter data
            # selection without requiring the removed definition page.
            db.execute(
                "UPDATE stages SET status=?, updated_at=? WHERE name=? AND status IN (?, ?) "
                "AND project_id IN (SELECT id FROM projects WHERE start_stage=?)",
                (StageStatus.COMPLETED, interrupted, StageName.DEFINITION,
                 StageStatus.READY, StageStatus.LOCKED, StageName.DEFINITION),
            )
            db.execute(
                "UPDATE stages SET status=?, updated_at=? WHERE name=? AND status=? "
                "AND project_id IN (SELECT id FROM projects WHERE start_stage=?)",
                (StageStatus.READY, interrupted, StageName.DATA,
                 StageStatus.LOCKED, StageName.DEFINITION),
            )
            db.execute(
                "UPDATE projects SET start_stage=?, updated_at=? WHERE start_stage=?",
                (StageName.DATA, interrupted, StageName.DEFINITION),
            )
            # Reconcile projects created by builds that left pre-annotation in
            # review_required after the review stage was explicitly completed.
            # This is a state migration only: no downstream stage or job changes.
            db.execute(
                """
                UPDATE stages
                   SET status=?, updated_at=?
                 WHERE name=? AND status=?
                   AND EXISTS (
                       SELECT 1
                         FROM stages AS review_stage
                        WHERE review_stage.project_id=stages.project_id
                          AND review_stage.name=?
                          AND review_stage.status IN (?, ?)
                   )
                """,
                (
                    StageStatus.COMPLETED,
                    interrupted,
                    StageName.ANNOTATION,
                    StageStatus.REVIEW_REQUIRED,
                    StageName.REVIEW,
                    StageStatus.COMPLETED,
                    StageStatus.IMPORTED,
                ),
            )

    def create_project(self, request: ProjectCreate, workspace_root: Path) -> Project:
        project_id = uuid4().hex[:12]
        workspace = (workspace_root / project_id).resolve()
        for name in ("inputs", "frames", "cleaned", "annotations", "dataset", "runs", "exports", "logs"):
            (workspace / name).mkdir(parents=True, exist_ok=True)
        now = utc_now()
        start_stage = StageName.DATA if request.start_stage == StageName.DEFINITION else request.start_stage
        start_index = STAGES.index(start_stage)
        with self.connect() as db:
            db.execute(
                "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (project_id, request.name, request.description, "detection",
                 _dump(request.labels), start_stage, str(workspace), now, now),
            )
            for index, stage in enumerate(STAGES):
                if stage == StageName.DEFINITION:
                    status = StageStatus.COMPLETED
                    summary = {"labels": request.labels, "task_type": request.task_type, "provided_by": "project_creation"}
                elif index < start_index:
                    status = StageStatus.LOCKED
                    summary = {}
                elif index == start_index:
                    status = StageStatus.READY
                    summary = {}
                else:
                    status = StageStatus.LOCKED
                    summary = {}
                db.execute(
                    "INSERT INTO stages(project_id,name,position,status,summary_json,updated_at) VALUES(?,?,?,?,?,?)",
                    (project_id, stage, index, status, _dump(summary), now),
                )
        return self.get_project(project_id)

    def list_projects(self) -> list[Project]:
        with self.connect() as db:
            rows = db.execute("SELECT id FROM projects ORDER BY updated_at DESC").fetchall()
        return [self.get_project(row["id"]) for row in rows]

    def get_project(self, project_id: str) -> Project:
        with self.connect() as db:
            row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
            if row is None:
                raise KeyError(project_id)
            stages = db.execute(
                "SELECT * FROM stages WHERE project_id=? ORDER BY position", (project_id,)
            ).fetchall()
            artifacts = db.execute(
                "SELECT * FROM artifacts WHERE project_id=? ORDER BY created_at", (project_id,)
            ).fetchall()
        return Project(
            id=row["id"], name=row["name"], description=row["description"],
            task_type=row["task_type"], labels=_load(row["labels_json"]),
            start_stage=row["start_stage"], workspace_path=row["workspace_path"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            stages=[_stage(item) for item in stages], artifacts=[_artifact(item) for item in artifacts],
        )

    def delete_project(self, project_id: str) -> None:
        with self.connect() as db:
            cursor = db.execute("DELETE FROM projects WHERE id=?", (project_id,))
            if cursor.rowcount == 0:
                raise KeyError(project_id)

    def update_project(self, project_id: str, values: dict[str, Any]) -> Project:
        allowed = {"name", "description", "labels"}
        values = {key: value for key, value in values.items() if key in allowed and value is not None}
        if not values:
            return self.get_project(project_id)
        if "labels" in values:
            values["labels_json"] = _dump(values.pop("labels"))
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.connect() as db:
            cursor = db.execute(
                f"UPDATE projects SET {assignments} WHERE id=?", (*values.values(), project_id)
            )
            if cursor.rowcount == 0:
                raise KeyError(project_id)
        return self.get_project(project_id)

    def set_stage(
        self, project_id: str, stage: StageName, status: StageStatus, *,
        config: dict[str, Any] | None = None, summary: dict[str, Any] | None = None,
        active_job_id: str | None = None, warning: str | None = None,
        bump_revision: bool = False,
    ) -> None:
        now = utc_now()
        parts = ["status=?", "active_job_id=?", "warning=?", "updated_at=?"]
        params: list[Any] = [status, active_job_id, warning, now]
        if config is not None:
            parts.append("config_json=?")
            params.append(_dump(config))
        if summary is not None:
            parts.append("summary_json=?")
            params.append(_dump(summary))
        if bump_revision:
            parts.append("revision=revision+1")
        params.extend((project_id, stage))
        with self.connect() as db:
            cursor = db.execute(
                f"UPDATE stages SET {', '.join(parts)} WHERE project_id=? AND name=?", params
            )
            if cursor.rowcount == 0:
                raise KeyError((project_id, stage))
            db.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))

    def unlock_next(self, project_id: str, stage: StageName) -> None:
        index = STAGES.index(stage)
        if index + 1 >= len(STAGES):
            return
        next_stage = STAGES[index + 1]
        with self.connect() as db:
            db.execute(
                "UPDATE stages SET status=?, updated_at=? WHERE project_id=? AND name=? AND status IN (?, ?)",
                (StageStatus.READY, utc_now(), project_id, next_stage, StageStatus.LOCKED, StageStatus.STALE),
            )

    def invalidate_downstream(self, project_id: str, stage: StageName) -> None:
        position = STAGES.index(stage)
        with self.connect() as db:
            db.execute(
                "UPDATE stages SET status=?, active_job_id=NULL, warning=?, updated_at=? "
                "WHERE project_id=? AND position>? AND status!=?",
                (StageStatus.STALE, "Upstream stage has a newer revision", utc_now(),
                 project_id, position, StageStatus.LOCKED),
            )

    def create_job(
        self, project_id: str, stage: StageName, command: list[str], config: dict[str, Any], log_path: Path
    ) -> Job:
        job_id = uuid4().hex
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO jobs(id,project_id,stage,status,command_json,config_json,log_path,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (job_id, project_id, stage, JobStatus.QUEUED, _dump(command), _dump(config), str(log_path), now),
            )
        self.set_stage(project_id, stage, StageStatus.QUEUED, config=config, active_job_id=job_id)
        return self.get_job(job_id)

    def update_job(self, job_id: str, **values: Any) -> Job:
        if "progress" in values:
            values["progress_json"] = _dump(values.pop("progress"))
        allowed = {"status", "progress_json", "error", "pid", "exit_code", "started_at", "finished_at"}
        values = {key: value for key, value in values.items() if key in allowed}
        with self.connect() as db:
            assignments = ", ".join(f"{key}=?" for key in values)
            cursor = db.execute(f"UPDATE jobs SET {assignments} WHERE id=?", (*values.values(), job_id))
            if cursor.rowcount == 0:
                raise KeyError(job_id)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Job:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return _job(row)

    def latest_job_for_stage(self, project_id: str, stage: StageName) -> Job | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE project_id=? AND stage=? ORDER BY created_at DESC LIMIT 1",
                (project_id, stage),
            ).fetchone()
        return _job(row) if row else None

    def add_event(self, job_id: str, event_type: str, payload: dict[str, Any]) -> int:
        with self.connect() as db:
            sequence = db.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 FROM job_events WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            db.execute(
                "INSERT INTO job_events(job_id,sequence,type,payload_json,created_at) VALUES(?,?,?,?,?)",
                (job_id, sequence, event_type, _dump(payload), utc_now()),
            )
        return int(sequence)

    def events(self, job_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM job_events WHERE job_id=? AND sequence>? ORDER BY sequence LIMIT 1000",
                (job_id, after),
            ).fetchall()
        return [{"sequence": row["sequence"], "type": row["type"],
                 "created_at": row["created_at"], **_load(row["payload_json"])} for row in rows]

    def add_artifact(
        self, project_id: str, stage: StageName, kind: str, path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        artifact_id = uuid4().hex
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)",
                (artifact_id, project_id, stage, kind, str(path), _dump(metadata or {}), now),
            )
        return Artifact(id=artifact_id, project_id=project_id, stage=stage, kind=kind,
                        path=str(path), metadata=metadata or {}, created_at=now)

    def save_training_config(self, project_id: str, config: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO training_configs VALUES(?,?,?) ON CONFLICT(project_id) DO UPDATE SET "
                "config_json=excluded.config_json,updated_at=excluded.updated_at",
                (project_id, _dump(config), utc_now()),
            )

    def training_config(self, project_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT config_json FROM training_configs WHERE project_id=?", (project_id,)).fetchone()
        return _load(row[0]) if row else None


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _load(value: str) -> Any:
    return json.loads(value)


def _stage(row: sqlite3.Row) -> Stage:
    return Stage(project_id=row["project_id"], name=row["name"], position=row["position"],
                 status=row["status"], revision=row["revision"], config=_load(row["config_json"]),
                 summary=_load(row["summary_json"]), active_job_id=row["active_job_id"],
                 warning=row["warning"], updated_at=row["updated_at"])


def _artifact(row: sqlite3.Row) -> Artifact:
    return Artifact(id=row["id"], project_id=row["project_id"], stage=row["stage"],
                    kind=row["kind"], path=row["path"], metadata=_load(row["metadata_json"]),
                    created_at=row["created_at"])


def _job(row: sqlite3.Row) -> Job:
    return Job(id=row["id"], project_id=row["project_id"], stage=row["stage"], status=row["status"],
               command=_load(row["command_json"]), config=_load(row["config_json"]),
               progress=_load(row["progress_json"]), error=row["error"], log_path=row["log_path"],
               pid=row["pid"], exit_code=row["exit_code"], created_at=row["created_at"],
               started_at=row["started_at"], finished_at=row["finished_at"])
