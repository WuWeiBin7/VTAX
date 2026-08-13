from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from data_annotation_agent.schemas import CategoryGuide
from data_annotation_agent.vtax.constants import JobStatus, StageName, StageStatus
from data_annotation_agent.vtax.database import Database, utc_now
from data_annotation_agent.vtax.dataset import prepare_yolo_dataset
from data_annotation_agent.vtax.schemas import (
    CategoryGuideDocument,
    ProjectCreate,
    ProjectPatch,
    StageCompleteRequest,
    StageImportRequest,
)
from data_annotation_agent.vtax.service import WorkflowService
from data_annotation_agent.vtax.workflow_cli import prepare_annotation_workspace


@pytest.fixture
def workflow(tmp_path: Path) -> WorkflowService:
    root = tmp_path / "vtax"
    settings = SimpleNamespace(
        root=root,
        workspace_root=root / "workspaces",
        labeled_dataset_root=root / "LabeledDataset",
        weight_root=root / "Weight",
        onnx_root=root / "ONNX",
        global_category_guides_path=root / "default_category_guides.json",
        database_path=root / "var" / "vtax.sqlite3",
        repo_root=Path(__file__).parents[1],
        depth_python=Path("python"),
        yolo_python=Path("python"),
        annotation_python=Path("python"),
        dino_model=root / "models" / "dinov3",
        yolo_model_root=root / "models" / "yolo26",
        config_path=root / "config.json",
        allowed_roots=(tmp_path.resolve(),),
        model_path=lambda size: root / "models" / "yolo26" / f"yolo26{size}.pt",
    )
    return WorkflowService(settings)


def project(workflow: WorkflowService, start: StageName = StageName.DEFINITION):
    return workflow.create_project(ProjectCreate(name="010 safety", labels=["helmet", "person"], start_stage=start))


def test_project_creation_embeds_definition_and_opens_data_stage(workflow: WorkflowService) -> None:
    created = project(workflow)

    assert created.start_stage == StageName.DATA
    assert created.stages[0].status == StageStatus.COMPLETED
    assert created.stages[0].summary == {
        "labels": ["helmet", "person"],
        "task_type": "detection",
        "provided_by": "project_creation",
    }
    assert created.stages[1].status == StageStatus.READY


def test_project_default_category_guides_are_complete_and_persistent(workflow: WorkflowService) -> None:
    created = project(workflow)
    missing = workflow.get_default_category_guides(created.id)
    assert missing["exists"] is False

    saved = workflow.save_default_category_guides(
        created.id,
        CategoryGuideDocument(
            labels=created.labels,
            guides=[
                CategoryGuide(label="helmet", visual_cues=["hard shell"]),
                CategoryGuide(label="person", visual_cues=["human body"]),
            ],
        ),
    )

    assert saved["exists"] is True
    loaded = workflow.get_default_category_guides(created.id)
    assert loaded["labels"] == ["helmet", "person"]
    assert loaded["guides"][0]["visual_cues"] == ["hard shell"]
    assert Path(loaded["path"]).is_file()


def test_project_default_category_guides_reject_incomplete_order(workflow: WorkflowService) -> None:
    created = project(workflow)
    with pytest.raises(ValueError, match="complete guide"):
        workflow.save_default_category_guides(
            created.id,
            CategoryGuideDocument(
                labels=created.labels,
                guides=[
                    CategoryGuide(label="person"),
                    CategoryGuide(label="helmet"),
                ],
            ),
        )


def test_global_category_guides_are_shared_across_projects(workflow: WorkflowService) -> None:
    first = project(workflow)
    workflow.save_default_category_guides(
        first.id,
        CategoryGuideDocument(
            labels=first.labels,
            guides=[
                CategoryGuide(label="helmet", visual_cues=["hard shell"]),
                CategoryGuide(label="person", visual_cues=["human body"]),
            ],
        ),
    )
    second = workflow.create_project(ProjectCreate(
        name="shared guide", labels=["person", "helmet"], start_stage=StageName.DATA
    ))

    loaded = workflow.get_default_category_guides(second.id)

    assert loaded["exists"] is True
    assert loaded["source"] == "global"
    assert [guide["label"] for guide in loaded["guides"]] == ["person", "helmet"]
    assert Path(loaded["path"]) == workflow.settings.global_category_guides_path.resolve()


def test_project_category_guides_do_not_overwrite_global_library(workflow: WorkflowService) -> None:
    created = project(workflow)
    workflow.save_default_category_guides(
        created.id,
        CategoryGuideDocument(
            labels=created.labels,
            guides=[CategoryGuide(label="helmet"), CategoryGuide(label="person")],
        ),
    )
    saved = workflow.save_project_category_guides(
        created.id,
        CategoryGuideDocument(
            labels=created.labels,
            guides=[
                CategoryGuide(label="helmet", visual_cues=["project-only helmet"]),
                CategoryGuide(label="person", visual_cues=["project-only person"]),
            ],
        ),
    )

    assert saved["source"] == "project"
    assert Path(saved["path"]).name == "project_category_guides.json"
    assert Path(saved["path"]).is_file()
    global_guides = workflow.get_default_category_guides(created.id)
    assert global_guides["guides"][0]["visual_cues"] == []


def test_global_category_guides_report_missing_project_labels(workflow: WorkflowService) -> None:
    created = project(workflow)
    workflow.settings.global_category_guides_path.write_text(json.dumps({
        "version": 1,
        "labels": ["helmet"],
        "guides": [{"label": "helmet", "visual_cues": ["hard shell"]}],
    }), encoding="utf-8")

    loaded = workflow.get_default_category_guides(created.id)

    assert loaded["exists"] is False
    assert loaded["missing_labels"] == ["person"]
    assert [guide["label"] for guide in loaded["guides"]] == ["helmet"]


def test_arbitrary_stage_import_marks_prior_stages_imported(workflow: WorkflowService, tmp_path: Path) -> None:
    created = project(workflow, StageName.CONFIGURATION)
    dataset = tmp_path / "external-yolo"
    dataset.mkdir()
    (dataset / "dataset.yaml").write_text("path: .\ntrain: images/train\n", encoding="utf-8")

    updated = workflow.import_stage(
        created.id,
        StageName.CONFIGURATION,
        StageImportRequest(artifact_type="yolo_dataset", path=str(dataset)),
    )

    assert updated.stages[6].status == StageStatus.IMPORTED
    assert updated.stages[0].status == StageStatus.COMPLETED
    assert all(item.status == StageStatus.IMPORTED for item in updated.stages[1:6])
    assert updated.stages[7].status == StageStatus.READY
    assert updated.artifacts[-1].path == str(dataset.resolve())


def test_locked_stage_cannot_run_or_complete(workflow: WorkflowService) -> None:
    created = project(workflow)
    with pytest.raises(ValueError, match="locked"):
        workflow.complete_stage(created.id, StageName.TRAINING, StageCompleteRequest())


def test_data_video_selection_is_materialized_for_cleaning(workflow: WorkflowService, tmp_path: Path) -> None:
    source = tmp_path / "videos"
    source.mkdir()
    selected = source / "selected.mp4"
    skipped = source / "skipped.mp4"
    selected.write_bytes(b"selected-video")
    skipped.write_bytes(b"skipped-video")
    created = project(workflow)

    updated = workflow.complete_stage(
        created.id,
        StageName.DATA,
        StageCompleteRequest(metadata={
            "source_directories": [str(source)],
            "selected_videos": [str(selected)],
        }),
    )

    data_stage = updated.stages[1]
    assert data_stage.status == StageStatus.COMPLETED
    assert data_stage.summary["selected_videos"] == [str(selected.resolve())]
    assert data_stage.summary["selected_count"] == 1
    selection_dir = Path(data_stage.summary["selection_path"])
    links = [path for path in selection_dir.iterdir() if path.suffix == ".mp4"]
    assert len(links) == 1
    assert links[0].resolve() == selected.resolve()
    assert not any(path.resolve() == skipped.resolve() for path in links)
    command, config = workflow._build_command(updated, StageName.CLEANING, {})
    assert config["input_path"] == str(selection_dir)
    assert command[command.index("--input") + 1] == str(selection_dir)


def test_upstream_label_change_invalidates_downstream(workflow: WorkflowService, tmp_path: Path) -> None:
    created = project(workflow, StageName.DATASET)
    labelme = tmp_path / "labelme"
    labelme.mkdir()
    (labelme / "frame.json").write_text(json.dumps({"imageWidth": 10, "imageHeight": 10, "shapes": []}), encoding="utf-8")
    workflow.import_stage(created.id, StageName.DATASET, StageImportRequest(artifact_type="labelme", path=str(labelme)))

    updated = workflow.patch_project(created.id, ProjectPatch(labels=["helmet", "person", "vest"]))

    assert updated.labels[-1] == "vest"
    assert updated.stages[0].status == StageStatus.COMPLETED
    assert updated.stages[1].status == StageStatus.IMPORTED
    assert updated.stages[2].status == StageStatus.IMPORTED
    assert updated.stages[3].status == StageStatus.READY
    assert all(item.status == StageStatus.STALE for item in updated.stages[4:7])
    assert all(item.status == StageStatus.LOCKED for item in updated.stages[7:])


def test_path_whitelist_rejects_external_path(workflow: WorkflowService, tmp_path: Path) -> None:
    external = tmp_path.parent / "outside-vtax.txt"
    external.write_text("x", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="VTAX_ALLOWED_ROOTS"):
            workflow.validate_path(str(external))
    finally:
        external.unlink(missing_ok=True)


def test_review_force_persists_coverage_warning(workflow: WorkflowService, tmp_path: Path) -> None:
    created = project(workflow, StageName.REVIEW)
    labelme = tmp_path / "review-labelme"
    labelme.mkdir()
    (labelme / "a.jpg").write_bytes(b"image-a")
    (labelme / "b.jpg").write_bytes(b"image-b")
    (labelme / "a.json").write_text(json.dumps({"imageWidth": 20, "imageHeight": 10, "shapes": [], "checked": True}), encoding="utf-8")
    (labelme / "b.json").write_text(json.dumps({"imageWidth": 20, "imageHeight": 10, "shapes": []}), encoding="utf-8")
    workflow.import_stage(created.id, StageName.REVIEW, StageImportRequest(artifact_type="labelme", path=str(labelme)))

    with pytest.raises(ValueError, match="Not every image"):
        workflow.complete_stage(created.id, StageName.REVIEW, StageCompleteRequest())
    updated = workflow.complete_stage(created.id, StageName.REVIEW, StageCompleteRequest(force=True))
    review = updated.stages[4]
    assert review.status == StageStatus.COMPLETED
    assert review.summary["reviewed_percent"] == 50.0
    assert "1/2" in (review.warning or "")
    labeled_path = Path(review.summary["labeled_dataset_path"])
    assert labeled_path.name == "010_safety_labeled"
    assert (labeled_path / "a.jpg").is_file()
    assert (labeled_path / "dataset_manifest.json").is_file()
    assert updated.artifacts[-1].kind == "labeled_dataset"


def test_combined_annotation_review_unlocks_dataset(workflow: WorkflowService, tmp_path: Path) -> None:
    created = project(workflow, StageName.ANNOTATION)
    annotation_dir = tmp_path / "combined-annotation"
    annotation_dir.mkdir()
    (annotation_dir / "frame.jpg").write_bytes(b"image")
    (annotation_dir / "frame.json").write_text(
        json.dumps({"imageWidth": 20, "imageHeight": 10, "shapes": [], "checked": False}),
        encoding="utf-8",
    )
    workflow.patch_project(created.id, ProjectPatch(labels=["helmet", "vest"]))
    workflow.complete_stage(
        created.id,
        StageName.ANNOTATION,
        StageCompleteRequest(metadata={"annotation_dir": str(annotation_dir), "labels": ["helmet", "vest"]}),
    )
    workflow.import_stage(
        created.id,
        StageName.REVIEW,
        StageImportRequest(artifact_type="labelme", path=str(annotation_dir)),
    )
    updated = workflow.complete_stage(
        created.id,
        StageName.REVIEW,
        StageCompleteRequest(force=True),
    )

    assert updated.labels == ["helmet", "vest"]
    assert updated.stages[3].status == StageStatus.COMPLETED
    assert updated.stages[4].status == StageStatus.COMPLETED
    assert updated.stages[5].status == StageStatus.READY


def test_multiclass_labelme_conversion_and_group_split(tmp_path: Path) -> None:
    source = tmp_path / "annotations"
    source.mkdir()
    provenance = []
    for index, (video, label) in enumerate((("v1", "helmet"), ("v1", "person"), ("v2", None), ("v3", "helmet"))):
        image = source / f"frame_{index:04d}.jpg"
        image.write_bytes(b"jpeg-placeholder")
        shapes = [] if label is None else [{"label": label, "shape_type": "rectangle", "points": [[10, 20], [50, 80]]}]
        image.with_suffix(".json").write_text(json.dumps({"imageWidth": 100, "imageHeight": 100, "shapes": shapes}), encoding="utf-8")
        provenance.append(json.dumps({"target": image.name, "source_video": video}))
    (source / "provenance.jsonl").write_text("\n".join(provenance), encoding="utf-8")

    summary = prepare_yolo_dataset(source, tmp_path / "dataset", ["helmet", "person"], seed=42)

    assert summary["source_images"] == 4
    assert summary["class_counts"] == {"helmet": 2, "person": 1}
    manifest = (tmp_path / "dataset" / "manifest.csv").read_text(encoding="utf-8-sig")
    v1_splits = {line.split(",")[5] for line in manifest.splitlines()[1:] if ":v1," in line}
    assert len(v1_splits) == 1
    assert sum(item["negative_images"] for item in summary["splits"].values()) == 1


def test_single_video_dataset_uses_contiguous_non_empty_fallback(tmp_path: Path) -> None:
    source = tmp_path / "one-video"
    source.mkdir()
    provenance = []
    for index in range(20):
        image = source / f"video_01_{index:06d}_{index / 2:010.3f}s.jpg"
        image.write_bytes(b"image")
        image.with_suffix(".json").write_text(
            json.dumps({"imageWidth": 100, "imageHeight": 100, "shapes": []}),
            encoding="utf-8",
        )
        provenance.append(json.dumps({"target": image.name, "source_video": "video_01"}))
    (source / "provenance.jsonl").write_text("\n".join(provenance), encoding="utf-8")

    summary = prepare_yolo_dataset(source, tmp_path / "dataset", ["hand"], seed=42)

    assert summary["source_groups"] == 1
    assert summary["split_strategy"] == "contiguous_fallback"
    assert [summary["splits"][split]["images"] for split in ("train", "val", "test")] == [16, 3, 1]
    manifest_rows = list(csv.DictReader((tmp_path / "dataset" / "manifest.csv").open(encoding="utf-8-sig")))
    assert [row["split"] for row in manifest_rows] == ["train"] * 16 + ["val"] * 3 + ["test"]


def test_dataset_rejects_too_few_images_for_requested_splits(tmp_path: Path) -> None:
    source = tmp_path / "tiny"
    source.mkdir()
    for index in range(2):
        image = source / f"frame_{index:04d}.jpg"
        image.write_bytes(b"image")
        image.with_suffix(".json").write_text(
            json.dumps({"imageWidth": 10, "imageHeight": 10, "shapes": []}), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="At least 3 images"):
        prepare_yolo_dataset(source, tmp_path / "dataset", ["hand"])


def test_dataset_replay_samples_forty_percent_of_new_data(tmp_path: Path) -> None:
    new = tmp_path / "new_labeled"
    replay_a = tmp_path / "replay_a_labeled"
    replay_b = tmp_path / "replay_b_labeled"
    for root, count in ((new, 10), (replay_a, 12), (replay_b, 8)):
        root.mkdir()
        for index in range(count):
            image = root / f"frame_{index:04d}.jpg"
            image.write_bytes(b"image")
            image.with_suffix(".json").write_text(
                json.dumps({"imageWidth": 100, "imageHeight": 100, "shapes": []}),
                encoding="utf-8",
            )

    summary = prepare_yolo_dataset(
        new,
        tmp_path / "mixed-yolo",
        ["helmet"],
        replay_sources=[replay_a, replay_b],
        replay_ratio=0.4,
        seed=7,
    )

    assert summary["new_images"] == 10
    assert summary["replay_candidates"] == 20
    assert summary["replay_images"] == 4
    assert summary["source_images"] == 14
    manifest = (tmp_path / "mixed-yolo" / "manifest.csv").read_text(encoding="utf-8-sig")
    assert manifest.count(",replay,") == 4


def test_annotation_workspace_preserves_video_group_from_flat_frame_names(tmp_path: Path) -> None:
    source = tmp_path / "selected"
    output = tmp_path / "managed"
    source.mkdir()
    (source / "video-A_000004_000002.000s.jpg").write_bytes(b"image")
    (source / "video-B_000008_000004.000s.jpg").write_bytes(b"image")

    prepare_annotation_workspace(source, output)

    rows = [json.loads(line) for line in (output / "provenance.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["source_video"] for row in rows} == {"video-A", "video-B"}


def test_database_restart_marks_active_job_interrupted(workflow: WorkflowService) -> None:
    created = project(workflow, StageName.TRAINING)
    job = workflow.db.create_job(created.id, StageName.TRAINING, ["stub"], {}, Path(created.workspace_path) / "logs" / "job.log")
    workflow.db.update_job(job.id, status=JobStatus.RUNNING, started_at=utc_now())
    workflow.db.set_stage(created.id, StageName.TRAINING, StageStatus.RUNNING, active_job_id=job.id)

    restarted = Database(workflow.settings.database_path)

    assert restarted.get_job(job.id).status == JobStatus.INTERRUPTED
    stage = restarted.get_project(created.id).stages[7]
    assert stage.status == StageStatus.FAILED
    assert stage.active_job_id is None


def test_training_command_can_disable_amp(workflow: WorkflowService, tmp_path: Path) -> None:
    created = project(workflow, StageName.TRAINING)
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("path: .\n", encoding="utf-8")
    model = tmp_path / "yolo26n.pt"
    model.write_bytes(b"weights")

    command, config = workflow._build_command(created, StageName.TRAINING, {
        "data": str(dataset), "model": str(model), "model_size": "n", "amp": False,
        "epochs": 2, "imgsz": 640, "batch": 4,
    })

    assert config["amp"] is False
    assert "--no-amp" in command
    assert "--weight-output-dir" in command
    assert str(workflow.settings.weight_root.resolve()) in command


def test_training_command_can_explicitly_enable_amp(workflow: WorkflowService, tmp_path: Path) -> None:
    created = project(workflow, StageName.TRAINING)
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("path: .\n", encoding="utf-8")
    model = tmp_path / "yolo26n.pt"
    model.write_bytes(b"weights")

    command, config = workflow._build_command(created, StageName.TRAINING, {
        "data": str(dataset), "model": str(model), "model_size": "n", "amp": True,
    })

    assert config["amp"] is True
    assert "--amp" in command
    assert "--no-amp" not in command


def test_training_retry_prefers_latest_dataset_artifact(workflow: WorkflowService, tmp_path: Path) -> None:
    created = project(workflow, StageName.TRAINING)
    old_dataset = tmp_path / "old-dataset.yaml"
    old_dataset.write_text("path: old\n", encoding="utf-8")
    latest_dataset = tmp_path / "latest-dataset.yaml"
    latest_dataset.write_text("path: latest\n", encoding="utf-8")
    model = tmp_path / "yolo26n.pt"
    model.write_bytes(b"weights")
    workflow.db.add_artifact(created.id, StageName.DATASET, "dataset_yaml", old_dataset)
    workflow.db.add_artifact(created.id, StageName.DATASET, "dataset_yaml", latest_dataset)
    created = workflow.db.get_project(created.id)

    command, config = workflow._build_command(created, StageName.TRAINING, {
        "data": str(old_dataset), "model": str(model), "model_size": "n",
    })

    assert config["data"] == str(latest_dataset)
    assert command[command.index("--data") + 1] == str(latest_dataset)


def test_weight_registry_lists_permanent_and_base_weights(workflow: WorkflowService) -> None:
    workflow.settings.weight_root.mkdir(parents=True, exist_ok=True)
    trained = workflow.settings.weight_root / "helmet_best.pt"
    trained.write_bytes(b"trained")
    trained.with_suffix(".pt.json").write_text(json.dumps({
        "project_id": "project-1", "project_name": "helmet project", "kind": "best",
    }), encoding="utf-8")
    base = workflow.settings.model_path("m")
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_bytes(b"base")

    weights = workflow.list_weights()

    assert weights[0]["path"] == str(trained.resolve())
    assert weights[0]["project_name"] == "helmet project"
    assert any(item["id"] == "base:yolo26m" and item["path"] == str(base.resolve()) for item in weights)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
def test_export_command_uses_permanent_onnx_directory(
    workflow: WorkflowService, tmp_path: Path, precision: str
) -> None:
    created = project(workflow, StageName.EXPORT)
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("path: .\n", encoding="utf-8")
    best = tmp_path / "best.pt"
    best.write_bytes(b"weights")

    command, config = workflow._build_command(created, StageName.EXPORT, {
        "weights": str(best), "data": str(dataset), "precision": precision,
    })

    assert config["precision"] == precision
    assert command[command.index("--precision") + 1] == precision
    assert command[command.index("--output-dir") + 1] == str(workflow.settings.onnx_root.resolve())


def test_finish_project_removes_only_intermediate_workspace_data(workflow: WorkflowService) -> None:
    created = project(workflow, StageName.EXPORT)
    workspace = Path(created.workspace_path)
    original = workflow.settings.root / "original-videos" / "source.mp4"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"video")
    for name in ("inputs", "frames", "cleaned", "dataset", "runs", "exports", "logs"):
        directory = workspace / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "temporary.bin").write_bytes(b"temporary")
    annotations = workspace / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    guide = annotations / "default_category_guides.json"
    guide.write_text("{}", encoding="utf-8")
    (annotations / "frame.json").write_text("{}", encoding="utf-8")
    permanent_onnx = workflow.settings.onnx_root / "project_fp32.onnx"
    permanent_onnx.parent.mkdir(parents=True, exist_ok=True)
    permanent_onnx.write_bytes(b"onnx")
    workflow.db.add_artifact(created.id, StageName.EXPORT, "onnx", permanent_onnx)
    workflow.db.set_stage(created.id, StageName.EXPORT, StageStatus.COMPLETED)

    result = workflow.finish_project(created.id)

    assert result["status"] == "completed"
    assert permanent_onnx.is_file()
    assert original.is_file()
    assert guide.is_file()
    assert not (annotations / "frame.json").exists()
    assert all(not (workspace / name).exists() for name in ("inputs", "frames", "cleaned", "dataset", "runs", "exports", "logs"))


def test_delete_project_removes_workspace_but_preserves_permanent_artifacts(workflow: WorkflowService) -> None:
    created = project(workflow)
    workspace = Path(created.workspace_path)
    permanent_dataset = workflow.settings.labeled_dataset_root / "shared_labeled"
    permanent_weight = workflow.settings.weight_root / "shared_best.pt"
    permanent_onnx = workflow.settings.onnx_root / "shared_fp32.onnx"
    permanent_dataset.mkdir(parents=True)
    permanent_weight.parent.mkdir(parents=True, exist_ok=True)
    permanent_onnx.parent.mkdir(parents=True, exist_ok=True)
    permanent_weight.write_bytes(b"weight")
    permanent_onnx.write_bytes(b"onnx")
    workflow.db.add_artifact(created.id, StageName.REVIEW, "labeled_dataset", permanent_dataset)
    workflow.db.add_artifact(created.id, StageName.TRAINING, "best_pt", permanent_weight)
    workflow.db.add_artifact(created.id, StageName.EXPORT, "onnx", permanent_onnx)

    result = workflow.delete_project(created.id)

    assert result["status"] == "deleted"
    assert not workspace.exists()
    assert permanent_dataset.is_dir()
    assert permanent_weight.is_file()
    assert permanent_onnx.is_file()
    with pytest.raises(KeyError):
        workflow.db.get_project(created.id)


def test_delete_project_rejects_running_job(workflow: WorkflowService) -> None:
    created = project(workflow, StageName.CLEANING)
    job = workflow.db.create_job(
        created.id, StageName.CLEANING, ["stub"], {}, Path(created.workspace_path) / "logs" / "active.log"
    )
    workflow.db.update_job(job.id, status=JobStatus.RUNNING, started_at=utc_now())

    with pytest.raises(ValueError, match="Stop the running"):
        workflow.delete_project(created.id)

    assert Path(created.workspace_path).is_dir()


def test_ultralytics_log_parser_extracts_epoch_and_metrics() -> None:
    epoch = WorkflowService._parse_line(
        "\r\x1b[K  2/80  2.52G  1.342  5.474  0.01918  23  640: 100% 2/2"
    )
    metrics = WorkflowService._parse_line(" all  3  8  0.00333  0.5  0.0369  0.0167")
    assert epoch == {
        "type": "training_epoch", "epoch": 2, "epochs": 80, "gpu_memory_gb": 2.52,
        "box_loss": 1.342, "cls_loss": 5.474, "dfl_loss": 0.01918,
        "instances": 23, "imgsz": 640,
    }
    assert metrics["type"] == "training_metrics"
    assert metrics["precision"] == pytest.approx(0.00333)
    assert metrics["map50_95"] == pytest.approx(0.0167)


def test_api_create_locked_guard_and_sse(workflow: WorkflowService, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_annotation_agent.vtax import api

    monkeypatch.setattr(api, "service", workflow)
    client = TestClient(api.app)
    response = client.post("/api/v1/projects", json={"name": "API", "labels": ["helmet"]})
    assert response.status_code == 201
    project_id = response.json()["id"]
    missing_guides = client.get(f"/api/v1/projects/{project_id}/category-guides")
    assert missing_guides.status_code == 200
    assert missing_guides.json()["exists"] is False
    saved_guides = client.put(
        f"/api/v1/projects/{project_id}/category-guides",
        json={
            "labels": ["helmet"],
            "guides": [{"label": "helmet", "visual_cues": ["hard shell"]}],
        },
    )
    assert saved_guides.status_code == 200
    assert client.get(f"/api/v1/projects/{project_id}/category-guides").json()["guides"][0]["visual_cues"] == ["hard shell"]
    locked = client.post(f"/api/v1/projects/{project_id}/stages/training/run", json={"config": {}})
    assert locked.status_code == 400

    created_job = workflow.db.create_job(project_id, StageName.DEFINITION, ["stub"], {}, Path(response.json()["workspace_path"]) / "logs" / "stub.log")
    workflow.db.add_event(created_job.id, "progress", {"current": 1, "total": 1})
    workflow.db.update_job(created_job.id, status=JobStatus.COMPLETED, finished_at=utc_now())
    stream = client.get(f"/api/v1/jobs/{created_job.id}/events")
    assert stream.status_code == 200
    assert '"type": "progress"' in stream.text


def test_api_delete_project(workflow: WorkflowService, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_annotation_agent.vtax import api

    monkeypatch.setattr(api, "service", workflow)
    client = TestClient(api.app)
    response = client.post("/api/v1/projects", json={"name": "delete me", "labels": ["hand"]})

    deleted = client.delete(f"/api/v1/projects/{response.json()['id']}")

    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
