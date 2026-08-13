from pathlib import Path

from PIL import Image

from data_annotation_agent.web_job import build_parser, main


def test_web_job_parser_accepts_inputs() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--input-dir",
            "C:/tmp/images",
            "--labels",
            "poppy",
            "--output-dir",
            "C:/tmp/output",
            "--provider",
            "ollama",
        ]
    )

    assert args.input_dir.name == "images"
    assert args.labels == ["poppy"]
    assert args.coordinate_system == "pixel"
    assert args.provider == "ollama"


def test_web_job_main_emits_jsonl_for_mock_client(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "annotations"
    input_dir.mkdir()
    Image.new("RGB", (64, 64), color="white").save(input_dir / "one.jpg")

    class FakeClient:
        def research_categories(self, labels: list[str]):
            from data_annotation_agent.schemas import CategoryGuide

            return [CategoryGuide(label=label, visual_cues=["cue"]) for label in labels]

        def detect_objects(self, **kwargs):
            from data_annotation_agent.schemas import DetectionResult, DetectionShape

            return DetectionResult(
                shapes=[
                    DetectionShape(label="poppy", score=0.8, bbox=[0, 0, 10, 10]),
                ]
            )

    events: list[str] = []

    def fake_emit(event):
        import json

        events.append(json.dumps(event, ensure_ascii=False))

    monkeypatch.setattr("data_annotation_agent.web_job._emit", fake_emit)
    monkeypatch.setattr(
        "data_annotation_agent.web_job.build_annotation_client",
        lambda config: FakeClient(),
    )
    monkeypatch.setattr(
        "data_annotation_agent.web_job.ModelConfig.from_sources",
        lambda **kwargs: object(),
    )

    code = main(
        [
            "--input-dir",
            str(input_dir),
            "--labels",
            "poppy",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    assert any('"type": "job_finished"' in line for line in events)
