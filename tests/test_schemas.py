from data_annotation_agent.schemas import (
    DetectionResult,
    DetectionShape,
    build_annotation,
    normalize_detections,
    validate_labels,
)


def test_validate_labels_deduplicates_and_strips() -> None:
    assert validate_labels([" person ", "weapon", "person", ""]) == ["person", "weapon"]


def test_normalize_detections_clips_and_filters() -> None:
    result = DetectionResult(
        shapes=[
            DetectionShape(label="person", score=0.8, bbox=[-10, 5, 100, 120]),
            DetectionShape(label="other", score=0.9, bbox=[0, 0, 10, 10]),
            DetectionShape(label="person", score=0.5, bbox=[20, 20, 10, 10]),
        ]
    )

    shapes = normalize_detections(
        result,
        labels=["person"],
        source_width=100,
        source_height=100,
        target_width=200,
        target_height=300,
    )

    assert len(shapes) == 1
    assert shapes[0].label == "person"
    assert shapes[0].points == [[0.0, 15.0], [200.0, 15.0], [200.0, 300.0], [0.0, 300.0]]


def test_normalize_detections_maps_qwen_normalized_coordinates() -> None:
    result = DetectionResult(
        shapes=[
            DetectionShape(label="poppy", score=0.9, bbox=[250, 100, 750, 600]),
        ]
    )

    shapes = normalize_detections(
        result,
        labels=["poppy"],
        source_width=1000,
        source_height=1000,
        target_width=2000,
        target_height=1000,
    )

    assert shapes[0].points == [
        [500.0, 100.0],
        [1500.0, 100.0],
        [1500.0, 600.0],
        [500.0, 600.0],
    ]


def test_normalize_detections_maps_letterboxed_normalized_coordinates() -> None:
    result = DetectionResult(
        shapes=[
            DetectionShape(label="poppy", score=0.9, bbox=[250, 375, 750, 625]),
        ]
    )

    shapes = normalize_detections(
        result,
        labels=["poppy"],
        source_width=1000,
        source_height=1000,
        target_width=200,
        target_height=100,
        source_offset_x=0,
        source_offset_y=250,
        source_content_width=1000,
        source_content_height=500,
    )

    assert shapes[0].points == [
        [50.0, 25.0],
        [150.0, 25.0],
        [150.0, 75.0],
        [50.0, 75.0],
    ]


def test_normalize_detections_maps_tall_letterboxed_normalized_coordinates() -> None:
    result = DetectionResult(
        shapes=[
            DetectionShape(label="poppy", score=0.9, bbox=[375, 250, 625, 750]),
        ]
    )

    shapes = normalize_detections(
        result,
        labels=["poppy"],
        source_width=1000,
        source_height=1000,
        target_width=100,
        target_height=200,
        source_offset_x=250,
        source_offset_y=0,
        source_content_width=500,
        source_content_height=1000,
    )

    assert shapes[0].points == [
        [25.0, 50.0],
        [75.0, 50.0],
        [75.0, 150.0],
        [25.0, 150.0],
    ]


def test_normalize_detections_maps_letterboxed_pixel_coordinates() -> None:
    result = DetectionResult(
        shapes=[
            DetectionShape(label="poppy", score=0.9, bbox=[256, 384, 768, 640]),
        ]
    )

    shapes = normalize_detections(
        result,
        labels=["poppy"],
        source_width=1024,
        source_height=1024,
        target_width=200,
        target_height=100,
        source_offset_x=0,
        source_offset_y=256,
        source_content_width=1024,
        source_content_height=512,
    )

    assert shapes[0].points == [
        [50.0, 25.0],
        [150.0, 25.0],
        [150.0, 75.0],
        [50.0, 75.0],
    ]


def test_normalize_detections_clips_boxes_in_padding_region() -> None:
    result = DetectionResult(
        shapes=[
            DetectionShape(label="poppy", score=0.9, bbox=[0, 0, 1024, 1024]),
        ]
    )

    shapes = normalize_detections(
        result,
        labels=["poppy"],
        source_width=1024,
        source_height=1024,
        target_width=200,
        target_height=100,
        source_offset_x=0,
        source_offset_y=256,
        source_content_width=1024,
        source_content_height=512,
    )

    assert shapes[0].points == [
        [0.0, 0.0],
        [200.0, 0.0],
        [200.0, 100.0],
        [0.0, 100.0],
    ]


def test_build_annotation_matches_labelme_shape() -> None:
    annotation = build_annotation(
        image_path="image.jpg",
        image_width=640,
        image_height=480,
        shapes=[],
    )

    payload = annotation.model_dump(mode="json")
    assert payload["version"] == "4.0.0-beta.5"
    assert payload["imagePath"] == "image.jpg"
    assert payload["imageData"] is None
    assert payload["imageWidth"] == 640
    assert payload["imageHeight"] == 480
    assert payload["shapes"] == []
