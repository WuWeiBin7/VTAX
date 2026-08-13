"""Image discovery and model payload preparation."""

from __future__ import annotations

from dataclasses import dataclass
import base64
from io import BytesIO
import mimetypes
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PreparedImage:
    """Image payload sent to the vision model."""

    data_url: str
    width: int
    height: int
    original_width: int
    original_height: int
    canvas_width: int
    canvas_height: int
    scaled_width: int
    scaled_height: int
    scale: float
    pad_left: int
    pad_top: int


def discover_images(input_dir: Path, *, recursive: bool = True) -> list[Path]:
    """Find supported image files in a directory."""

    if not input_dir.exists() or not input_dir.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise ValueError(msg)

    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_image_size(image_path: Path) -> tuple[int, int]:
    """Return image dimensions as `(width, height)`."""

    with Image.open(image_path) as image:
        return ImageOps.exif_transpose(image).size


def prepare_image_for_model(
    image_path: Path,
    *,
    max_edge: int = 1024,
    max_bytes: int = 7_000_000,
) -> PreparedImage:
    """Create a square letterboxed canvas and encode it as a Base64 data URL."""

    with Image.open(image_path) as opened_image:
        image = ImageOps.exif_transpose(opened_image)
        original_width, original_height = image.size
        canvas_edge = max(int(max_edge), 1)
        min_canvas_edge = min(canvas_edge, 512)
        rgb_image = image.convert("RGB")

        prepared = _render_letterboxed_canvas(
            rgb_image,
            original_width=original_width,
            original_height=original_height,
            canvas_edge=canvas_edge,
        )
        quality = 90
        output = _encode_jpeg(prepared["canvas"], quality=quality)
        while len(output) > max_bytes and quality > 45:
            quality -= 10
            output = _encode_jpeg(prepared["canvas"], quality=quality)

        while len(output) > max_bytes and canvas_edge > min_canvas_edge:
            next_edge = max(int(canvas_edge * 0.85), min_canvas_edge)
            if next_edge == canvas_edge:
                break
            canvas_edge = next_edge
            prepared = _render_letterboxed_canvas(
                rgb_image,
                original_width=original_width,
                original_height=original_height,
                canvas_edge=canvas_edge,
            )
            output = _encode_jpeg(prepared["canvas"], quality=quality)

        encoded = base64.b64encode(output).decode("ascii")
        return PreparedImage(
            data_url=f"data:image/jpeg;base64,{encoded}",
            width=prepared["canvas_width"],
            height=prepared["canvas_height"],
            original_width=original_width,
            original_height=original_height,
            canvas_width=prepared["canvas_width"],
            canvas_height=prepared["canvas_height"],
            scaled_width=prepared["scaled_width"],
            scaled_height=prepared["scaled_height"],
            scale=prepared["scale"],
            pad_left=prepared["pad_left"],
            pad_top=prepared["pad_top"],
        )


def _render_letterboxed_canvas(
    image: Image.Image,
    *,
    original_width: int,
    original_height: int,
    canvas_edge: int,
) -> dict[str, object]:
    """Fit an image into a square canvas and keep reversible transform metadata."""

    scale = canvas_edge / max(original_width, original_height)
    scaled_width = max(int(round(original_width * scale)), 1)
    scaled_height = max(int(round(original_height * scale)), 1)
    resized = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (canvas_edge, canvas_edge), color="white")
    pad_left = (canvas_edge - scaled_width) // 2
    pad_top = (canvas_edge - scaled_height) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return {
        "canvas": canvas,
        "canvas_width": canvas_edge,
        "canvas_height": canvas_edge,
        "scaled_width": scaled_width,
        "scaled_height": scaled_height,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
    }


def _encode_jpeg(image: Image.Image, *, quality: int) -> bytes:
    """Encode an image as JPEG bytes."""

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()
