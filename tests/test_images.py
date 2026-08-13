from pathlib import Path

from PIL import Image

from data_annotation_agent.images import discover_images, prepare_image_for_model


def test_discover_images_recursive(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    image_path = nested / "sample.JPG"
    image_path.write_bytes(b"fake")
    (tmp_path / "sample.txt").write_text("not image", encoding="utf-8")

    assert discover_images(tmp_path) == [image_path]
    assert discover_images(tmp_path, recursive=False) == []


def test_prepare_image_for_model_resizes_large_image(tmp_path: Path) -> None:
    image_path = tmp_path / "large.png"
    Image.new("RGB", (200, 100), color="white").save(image_path)

    prepared = prepare_image_for_model(image_path, max_edge=100, max_bytes=100_000)

    assert prepared.original_width == 200
    assert prepared.original_height == 100
    assert prepared.width == 100
    assert prepared.height == 100
    assert prepared.canvas_width == 100
    assert prepared.canvas_height == 100
    assert prepared.scaled_width == 100
    assert prepared.scaled_height == 50
    assert prepared.pad_left == 0
    assert prepared.pad_top == 25
    assert prepared.scale == 0.5
    assert prepared.data_url.startswith("data:image/jpeg;base64,")
