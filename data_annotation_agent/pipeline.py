"""End-to-end directory annotation pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Protocol

from data_annotation_agent.cache import load_guides_from_cache, save_guides_to_cache
from data_annotation_agent.images import PreparedImage, discover_images, prepare_image_for_model
from data_annotation_agent.schemas import (
    CategoryGuide,
    DetectionResult,
    TokenUsage,
    build_annotation,
    normalize_detections,
    validate_labels,
)

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[dict[str, object]], None]


class AnnotationClient(Protocol):
    """Client interface used by the pipeline."""

    def research_categories(self, labels: list[str]) -> list[CategoryGuide]:
        """Return category guides for labels."""

    def detect_objects(
        self,
        *,
        image_path: Path,
        prepared_image: PreparedImage,
        labels: list[str],
        guides: list[CategoryGuide],
        coordinate_system: str,
    ) -> DetectionResult:
        """Detect objects in one image."""


@dataclass
class ImageProcessRecord:
    """Runtime details for one image."""

    image_path: str
    output_path: str
    status: str
    seconds: float
    shapes: int = 0
    token_usage: TokenUsage | None = None
    error: str | None = None
    input_image: dict[str, object] | None = None


@dataclass
class AnnotationStats:
    """Summary of one annotation run."""

    total: int = 0
    processed: int = 0
    annotated: int = 0
    skipped: int = 0
    empty: int = 0
    failed: int = 0
    total_seconds: float = 0.0
    image_seconds: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cached_tokens: int = 0
    total_cache_creation_input_tokens: int = 0
    token_usage_images: int = 0
    label_counts: Counter[str] = field(default_factory=Counter)
    image_records: list[ImageProcessRecord] = field(default_factory=list)
    log_path: str | None = None


def annotate_directory(
    *,
    input_dir: Path,
    output_dir: Path,
    labels: list[str],
    client: AnnotationClient,
    recursive: bool = True,
    overwrite: bool = False,
    max_edge: int = 1024,
    max_bytes: int = 7_000_000,
    retries: int = 3,
    coordinate_system: str = "pixel",
    category_guides: list[CategoryGuide] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnnotationStats:
    """Annotate every supported image in a directory."""

    normalized_labels = validate_labels(labels)
    image_paths = discover_images(input_dir, recursive=recursive)
    if not image_paths:
        msg = f"No supported images found in {input_dir}"
        raise ValueError(msg)

    stats = AnnotationStats(total=len(image_paths))
    output_dir.mkdir(parents=True, exist_ok=True)
    stats.log_path = str(output_dir / "annotation_run_log.json")
    run_started_at = time.perf_counter()
    _emit_progress(
        progress_callback,
        {
            "type": "job_started",
            "total": len(image_paths),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "labels": normalized_labels,
        },
    )
    guides = get_category_guides(
        output_dir=output_dir,
        labels=normalized_labels,
        client=client,
        category_guides=category_guides,
    )
    _emit_progress(
        progress_callback,
        {
            "type": "research_done",
            "labels": normalized_labels,
            "guides": [guide.model_dump(mode="json") for guide in guides],
        },
    )

    for image_path in image_paths:
        output_path = _output_path_for(image_path, input_dir=input_dir, output_dir=output_dir)
        if output_path.exists() and not overwrite:
            stats.skipped += 1
            stats.image_records.append(
                ImageProcessRecord(
                    image_path=str(image_path),
                    output_path=str(output_path),
                    status="skipped",
                    seconds=0.0,
                    shapes=0,
                    token_usage=None,
                    error=None,
                    input_image=None,
                )
            )
            LOGGER.info("Skipping existing annotation: %s", output_path)
            _emit_progress(
                progress_callback,
                {
                    "type": "image_skipped",
                    "image_path": str(image_path),
                    "output_path": str(output_path),
                    "processed": stats.processed,
                    "skipped": stats.skipped,
                    "total": stats.total,
                },
            )
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_started_at = time.perf_counter()
        detection_result: DetectionResult | None = None
        status = "processed"
        error_message: str | None = None
        _emit_progress(
            progress_callback,
            {
                "type": "image_started",
                "image_path": str(image_path),
                "output_path": str(output_path),
                "processed": stats.processed,
                "total": stats.total,
            },
        )
        try:
            prepared = prepare_image_for_model(
                image_path,
                max_edge=max_edge,
                max_bytes=max_bytes,
            )
            detection_result = _detect_with_retries(
                client,
                image_path=image_path,
                prepared_image=prepared,
                labels=normalized_labels,
                guides=guides,
                coordinate_system=coordinate_system,
                retries=retries,
            )
            shapes = normalize_detections(
                detection_result,
                labels=normalized_labels,
                source_width=_coordinate_source_width(prepared, coordinate_system),
                source_height=_coordinate_source_height(prepared, coordinate_system),
                target_width=prepared.original_width,
                target_height=prepared.original_height,
                source_offset_x=_coordinate_source_offset_x(prepared, coordinate_system),
                source_offset_y=_coordinate_source_offset_y(prepared, coordinate_system),
                source_content_width=_coordinate_source_content_width(prepared, coordinate_system),
                source_content_height=_coordinate_source_content_height(prepared, coordinate_system),
            )
            stats.processed += 1
        except Exception as exc:  # noqa: BLE001 - keep batch processing alive.
            LOGGER.exception("Failed to annotate %s: %s", image_path, exc)
            prepared = _empty_prepared_image(image_path)
            shapes = []
            stats.failed += 1
            status = "failed"
            error_message = str(exc)

        seconds = time.perf_counter() - image_started_at
        stats.image_seconds += seconds
        if shapes:
            stats.annotated += 1
        elif status == "processed":
            stats.empty += 1
        for shape in shapes:
            stats.label_counts[shape.label] += 1

        token_usage = detection_result.token_usage if detection_result is not None else None
        _add_token_usage(stats, token_usage)

        annotation = build_annotation(
            image_path=image_path.name,
            image_width=prepared.original_width,
            image_height=prepared.original_height,
            shapes=shapes,
        )
        output_path.write_text(
            json.dumps(annotation.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        stats.image_records.append(
            ImageProcessRecord(
                image_path=str(image_path),
                output_path=str(output_path),
                status=status,
                seconds=seconds,
                shapes=len(shapes),
                token_usage=token_usage,
                error=error_message,
                input_image=_prepared_image_metadata(prepared, coordinate_system),
            )
        )
        _emit_progress(
            progress_callback,
            {
                "type": "image_failed" if status == "failed" else "image_done",
                "image_path": str(image_path),
                "output_path": str(output_path),
                "status": status,
                "seconds": seconds,
                "shapes": len(shapes),
                "token_usage": token_usage.model_dump(mode="json") if token_usage else None,
                "error": error_message,
                "input_image": _prepared_image_metadata(prepared, coordinate_system),
                "processed": stats.processed,
                "annotated": stats.annotated,
                "empty": stats.empty,
                "failed": stats.failed,
                "skipped": stats.skipped,
                "total": stats.total,
            },
        )
        LOGGER.info(
            "Image processed: image=%s status=%s time=%.2fs shapes=%s tokens=%s output=%s input=%s",
            image_path.name,
            status,
            seconds,
            len(shapes),
            _format_token_usage(token_usage),
            output_path,
            _prepared_image_metadata(prepared, coordinate_system),
        )

    stats.total_seconds = time.perf_counter() - run_started_at
    _write_run_log(output_dir=output_dir, stats=stats)
    _emit_progress(progress_callback, {"type": "job_finished", "summary": stats_to_dict(stats)})
    return stats


def get_category_guides(
    *,
    output_dir: Path,
    labels: list[str],
    client: AnnotationClient,
    category_guides: list[CategoryGuide] | None = None,
    force_research: bool = False,
) -> list[CategoryGuide]:
    """Load, research, or normalize user-confirmed category guides."""

    cache_path = output_dir / ".annotation_cache" / "category_guides.json"
    if category_guides is not None:
        guides = _normalize_guide_list(category_guides, labels)
        save_guides_to_cache(cache_path, labels, guides)
        LOGGER.info("Saved user-confirmed category guides to cache: %s", cache_path)
        return guides

    cached = None if force_research else load_guides_from_cache(cache_path, labels)
    if cached is not None:
        LOGGER.info("Loaded category guides from cache: %s", cache_path)
        return cached

    guides = client.research_categories(labels)
    save_guides_to_cache(cache_path, labels, guides)
    LOGGER.info("Saved category guides to cache: %s", cache_path)
    return guides


def _normalize_guide_list(guides: list[CategoryGuide], labels: list[str]) -> list[CategoryGuide]:
    """Keep externally provided guides aligned with requested labels."""

    by_label = {guide.label: guide for guide in guides}
    normalized: list[CategoryGuide] = []
    for index, label in enumerate(labels):
        guide = by_label.get(label)
        if guide is None and index < len(guides):
            guide = guides[index].model_copy(update={"label": label})
        normalized.append(guide or CategoryGuide(label=label))
    return normalized


def stats_to_dict(stats: AnnotationStats) -> dict[str, object]:
    """Convert run statistics into a JSON-serializable dictionary."""

    return {
        "total": stats.total,
        "processed": stats.processed,
        "annotated": stats.annotated,
        "skipped": stats.skipped,
        "empty": stats.empty,
        "failed": stats.failed,
        "total_seconds": stats.total_seconds,
        "image_seconds": stats.image_seconds,
        "total_input_tokens": stats.total_input_tokens,
        "total_output_tokens": stats.total_output_tokens,
        "total_tokens": stats.total_tokens,
        "total_cached_tokens": stats.total_cached_tokens,
        "total_cache_creation_input_tokens": stats.total_cache_creation_input_tokens,
        "token_usage_images": stats.token_usage_images,
        "label_counts": dict(stats.label_counts),
        "log_path": stats.log_path,
        "image_records": [
            {
                "image_path": record.image_path,
                "output_path": record.output_path,
                "status": record.status,
                "seconds": record.seconds,
                "shapes": record.shapes,
                "token_usage": record.token_usage.model_dump(mode="json")
                if record.token_usage
                else None,
                "error": record.error,
                "input_image": record.input_image,
            }
            for record in stats.image_records
        ],
    }


def _detect_with_retries(
    client: AnnotationClient,
    *,
    image_path: Path,
    prepared_image: PreparedImage,
    labels: list[str],
    guides: list[CategoryGuide],
    coordinate_system: str,
    retries: int,
) -> DetectionResult:
    """Call the vision model with bounded retries."""

    attempts = max(retries, 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return client.detect_objects(
                image_path=image_path,
                prepared_image=prepared_image,
                labels=labels,
                guides=guides,
                coordinate_system=coordinate_system,
            )
        except Exception as exc:  # noqa: BLE001 - retry transient model failures.
            last_error = exc
            LOGGER.warning(
                "Detection attempt %s/%s failed for %s: %s",
                attempt,
                attempts,
                image_path,
                exc,
            )
    assert last_error is not None
    raise last_error


def _output_path_for(image_path: Path, *, input_dir: Path, output_dir: Path) -> Path:
    """Map an image path to its annotation JSON path."""

    relative = image_path.relative_to(input_dir)
    return output_dir / relative.with_suffix(".json")


def _empty_prepared_image(image_path: Path) -> PreparedImage:
    """Recover original dimensions for an empty failure annotation."""

    from data_annotation_agent.images import get_image_size

    width, height = get_image_size(image_path)
    return PreparedImage(
        data_url="",
        width=width,
        height=height,
        original_width=width,
        original_height=height,
        canvas_width=width,
        canvas_height=height,
        scaled_width=width,
        scaled_height=height,
        scale=1.0,
        pad_left=0,
        pad_top=0,
    )


def _coordinate_source_width(prepared: PreparedImage, coordinate_system: str) -> int:
    """Return the coordinate width used by model bboxes."""

    if coordinate_system == "pixel":
        return prepared.canvas_width
    if coordinate_system == "normalized-1000":
        return 1000
    msg = f"Unsupported coordinate system: {coordinate_system}"
    raise ValueError(msg)


def _coordinate_source_height(prepared: PreparedImage, coordinate_system: str) -> int:
    """Return the coordinate height used by model bboxes."""

    if coordinate_system == "pixel":
        return prepared.canvas_height
    if coordinate_system == "normalized-1000":
        return 1000
    msg = f"Unsupported coordinate system: {coordinate_system}"
    raise ValueError(msg)


def _coordinate_source_offset_x(prepared: PreparedImage, coordinate_system: str) -> float:
    """Return the X offset of the true content inside the model input."""

    if coordinate_system == "pixel":
        return float(prepared.pad_left)
    if coordinate_system == "normalized-1000":
        return prepared.pad_left / prepared.canvas_width * 1000.0
    msg = f"Unsupported coordinate system: {coordinate_system}"
    raise ValueError(msg)


def _coordinate_source_offset_y(prepared: PreparedImage, coordinate_system: str) -> float:
    """Return the Y offset of the true content inside the model input."""

    if coordinate_system == "pixel":
        return float(prepared.pad_top)
    if coordinate_system == "normalized-1000":
        return prepared.pad_top / prepared.canvas_height * 1000.0
    msg = f"Unsupported coordinate system: {coordinate_system}"
    raise ValueError(msg)


def _coordinate_source_content_width(prepared: PreparedImage, coordinate_system: str) -> float | None:
    """Return the width of the image content region inside the model input."""

    if coordinate_system == "pixel":
        return float(prepared.scaled_width)
    if coordinate_system == "normalized-1000":
        return prepared.scaled_width / prepared.canvas_width * 1000.0
    msg = f"Unsupported coordinate system: {coordinate_system}"
    raise ValueError(msg)


def _coordinate_source_content_height(prepared: PreparedImage, coordinate_system: str) -> float | None:
    """Return the height of the image content region inside the model input."""

    if coordinate_system == "pixel":
        return float(prepared.scaled_height)
    if coordinate_system == "normalized-1000":
        return prepared.scaled_height / prepared.canvas_height * 1000.0
    msg = f"Unsupported coordinate system: {coordinate_system}"
    raise ValueError(msg)


def _prepared_image_metadata(prepared: PreparedImage, coordinate_system: str) -> dict[str, object]:
    """Serialize the transform metadata used to prepare one image."""

    return {
        "original_width": prepared.original_width,
        "original_height": prepared.original_height,
        "canvas_width": prepared.canvas_width,
        "canvas_height": prepared.canvas_height,
        "scaled_width": prepared.scaled_width,
        "scaled_height": prepared.scaled_height,
        "scale": prepared.scale,
        "pad_left": prepared.pad_left,
        "pad_top": prepared.pad_top,
        "coordinate_system": coordinate_system,
    }


def _add_token_usage(stats: AnnotationStats, token_usage: TokenUsage | None) -> None:
    """Accumulate token usage when provider metadata is available."""

    if token_usage is None:
        return

    has_any_usage = False
    if token_usage.input_tokens is not None:
        stats.total_input_tokens += token_usage.input_tokens
        has_any_usage = True
    if token_usage.output_tokens is not None:
        stats.total_output_tokens += token_usage.output_tokens
        has_any_usage = True
    if token_usage.total_tokens is not None:
        stats.total_tokens += token_usage.total_tokens
        has_any_usage = True
    elif token_usage.input_tokens is not None or token_usage.output_tokens is not None:
        stats.total_tokens += (token_usage.input_tokens or 0) + (token_usage.output_tokens or 0)
        has_any_usage = True
    if token_usage.cached_tokens is not None:
        stats.total_cached_tokens += token_usage.cached_tokens
        has_any_usage = True
    if token_usage.cache_creation_input_tokens is not None:
        stats.total_cache_creation_input_tokens += token_usage.cache_creation_input_tokens
        has_any_usage = True

    if has_any_usage:
        stats.token_usage_images += 1


def _format_token_usage(token_usage: TokenUsage | None) -> str:
    """Format per-image token usage for logs."""

    if token_usage is None:
        return "unavailable"
    parts: list[str] = []
    if token_usage.input_tokens is not None:
        parts.append(f"input={token_usage.input_tokens}")
    if token_usage.output_tokens is not None:
        parts.append(f"output={token_usage.output_tokens}")
    if token_usage.total_tokens is not None:
        parts.append(f"total={token_usage.total_tokens}")
    if token_usage.cached_tokens is not None:
        parts.append(f"cached={token_usage.cached_tokens}")
    if token_usage.cache_creation_input_tokens is not None:
        parts.append(f"cache_created={token_usage.cache_creation_input_tokens}")
    return ", ".join(parts) if parts else "unavailable"


def _emit_progress(callback: ProgressCallback | None, event: dict[str, object]) -> None:
    """Emit progress events without letting UI reporting break annotation."""

    if callback is None:
        return
    try:
        callback(event)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Progress callback failed: %s", exc)


def _write_run_log(*, output_dir: Path, stats: AnnotationStats) -> None:
    """Persist per-image and summary statistics to the output directory."""

    log_path = output_dir / "annotation_run_log.json"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": stats_to_dict(stats),
    }
    log_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
