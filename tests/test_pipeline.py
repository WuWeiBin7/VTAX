from pathlib import Path
import json

from PIL import Image

from data_annotation_agent.images import PreparedImage
from data_annotation_agent.pipeline import annotate_directory
from data_annotation_agent.schemas import (
    CategoryGuide,
    DetectionResult,
    DetectionShape,
    TokenUsage,
)


class FakeClient:
    def __init__(self) -> None:
        self.last_coordinate_system: str | None = None

    def research_categories(self, labels: list[str]) -> list[CategoryGuide]:
        return [CategoryGuide(label=label, visual_cues=["cue"]) for label in labels]

    def detect_objects(
        self,
        *,
        image_path: Path,
        prepared_image: PreparedImage,
        labels: list[str],
        guides: list[CategoryGuide],
        coordinate_system: str,
    ) -> DetectionResult:
        self.last_coordinate_system = coordinate_system
        return DetectionResult(
            shapes=[
                DetectionShape(
                    label=labels[0],
                    score=0.75,
                    bbox=[0, 0, 500, 500],
                )
            ],
            token_usage=TokenUsage(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                cached_tokens=60,
                cache_creation_input_tokens=40,
            ),
        )


def test_annotate_directory_writes_labelme_json_and_stats(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "annotations"
    input_dir.mkdir()
    Image.new("RGB", (100, 80), color="white").save(input_dir / "one.jpg")

    stats = annotate_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        labels=["poppy"],
        client=FakeClient(),
    )

    output_file = output_dir / "one.json"
    assert stats.total == 1
    assert stats.processed == 1
    assert stats.annotated == 1
    assert stats.total_seconds > 0
    assert stats.image_seconds > 0
    assert stats.total_tokens == 120
    assert stats.total_input_tokens == 100
    assert stats.total_output_tokens == 20
    assert stats.total_cached_tokens == 60
    assert stats.total_cache_creation_input_tokens == 40
    assert len(stats.image_records) == 1
    assert stats.image_records[0].shapes == 1
    assert stats.image_records[0].input_image is not None
    assert stats.image_records[0].input_image["coordinate_system"] == "pixel"
    assert stats.image_records[0].input_image["canvas_width"] == 1024
    assert output_file.exists()
    payload = output_file.read_text(encoding="utf-8")
    assert '"imageWidth": 100' in payload
    assert '"shape_type": "rectangle"' in payload
    assert '"label": "poppy"' in payload
    log_payload = json.loads((output_dir / "annotation_run_log.json").read_text(encoding="utf-8"))
    assert log_payload["summary"]["processed"] == 1
    assert log_payload["summary"]["image_records"][0]["image_path"].endswith("one.jpg")
    assert log_payload["summary"]["image_records"][0]["seconds"] > 0
    assert log_payload["summary"]["image_records"][0]["token_usage"]["total_tokens"] == 120
    assert log_payload["summary"]["image_records"][0]["input_image"]["canvas_width"] == 1024
    assert log_payload["summary"]["image_records"][0]["input_image"]["coordinate_system"] == "pixel"


def test_annotate_directory_emits_progress_events(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "annotations"
    events: list[dict[str, object]] = []
    input_dir.mkdir()
    Image.new("RGB", (100, 80), color="white").save(input_dir / "one.jpg")

    stats = annotate_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        labels=["poppy"],
        client=FakeClient(),
        progress_callback=events.append,
    )

    event_types = [event["type"] for event in events]
    assert event_types[0] == "job_started"
    assert "research_done" in event_types
    assert "image_started" in event_types
    assert "image_done" in event_types
    assert event_types[-1] == "job_finished"
    summary = events[-1]["summary"]
    assert isinstance(summary, dict)
    assert summary["processed"] == stats.processed
    assert summary["annotated"] == stats.annotated
    image_done_event = next(event for event in events if event["type"] == "image_done")
    assert image_done_event["input_image"]["canvas_width"] == 1024


def test_annotate_directory_passes_coordinate_system_to_client(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "annotations"
    input_dir.mkdir()
    Image.new("RGB", (40, 30), color="white").save(input_dir / "one.jpg")
    client = FakeClient()

    annotate_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        labels=["poppy"],
        client=client,
        coordinate_system="pixel",
    )

    assert client.last_coordinate_system == "pixel"


def test_annotate_directory_skips_existing_json(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "annotations"
    input_dir.mkdir()
    output_dir.mkdir()
    Image.new("RGB", (10, 10), color="white").save(input_dir / "one.jpg")
    (output_dir / "one.json").write_text("{}", encoding="utf-8")

    stats = annotate_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        labels=["poppy"],
        client=FakeClient(),
    )

    assert stats.skipped == 1
    assert stats.processed == 0
    log_payload = json.loads((output_dir / "annotation_run_log.json").read_text(encoding="utf-8"))
    assert log_payload["summary"]["skipped"] == 1
    assert log_payload["summary"]["image_records"][0]["status"] == "skipped"
    assert log_payload["summary"]["image_records"][0]["seconds"] == 0.0
