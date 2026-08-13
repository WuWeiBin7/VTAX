"""Pydantic schemas and LabelMe conversion helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CategoryGuide(BaseModel):
    """Visual and OCR guidance for a single requested class."""

    label: str
    visual_cues: list[str] = Field(default_factory=list)
    ocr_cues: list[str] = Field(default_factory=list)
    negative_cues: list[str] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)


class CategoryGuides(BaseModel):
    """Research output for all requested classes."""

    guides: list[CategoryGuide] = Field(default_factory=list)


class DetectionShape(BaseModel):
    """Compact model output for one rectangular detection."""

    label: str
    score: float = Field(ge=0.0, le=1.0)
    bbox: list[float]

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        """Ensure bbox has the expected `[x1, y1, x2, y2]` shape."""

        if len(value) != 4:
            msg = "bbox must contain exactly four numbers"
            raise ValueError(msg)
        return [float(item) for item in value]


class TokenUsage(BaseModel):
    """Token usage metadata returned by the model provider when available."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


class DetectionResult(BaseModel):
    """Compact model output for all detections in one image."""

    shapes: list[DetectionShape] = Field(default_factory=list)
    token_usage: TokenUsage | None = None


class LabelMeShape(BaseModel):
    """LabelMe-compatible rectangle shape."""

    label: str
    score: float
    points: list[list[float]]
    group_id: None = None
    description: None = None
    difficult: bool = False
    shape_type: str = "rectangle"
    flags: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)
    kie_linking: list[Any] = Field(default_factory=list)


class LabelMeAnnotation(BaseModel):
    """LabelMe-compatible annotation file."""

    version: str = "4.0.0-beta.5"
    flags: dict[str, Any] = Field(default_factory=dict)
    checked: bool = False
    shapes: list[LabelMeShape] = Field(default_factory=list)
    imagePath: str
    imageData: None = None
    imageHeight: int
    imageWidth: int
    description: str = ""


def validate_labels(labels: Iterable[str]) -> list[str]:
    """Normalize user labels while preserving order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for label in labels:
        clean_label = label.strip()
        if not clean_label or clean_label in seen:
            continue
        normalized.append(clean_label)
        seen.add(clean_label)

    if not normalized:
        msg = "At least one non-empty label is required."
        raise ValueError(msg)
    return normalized


def normalize_guides(guides: CategoryGuides, labels: list[str]) -> list[CategoryGuide]:
    """Keep guides aligned with the requested labels."""

    by_label = {guide.label: guide for guide in guides.guides}
    normalized: list[CategoryGuide] = []
    for index, label in enumerate(labels):
        guide = by_label.get(label)
        if guide is None and index < len(guides.guides):
            # Models may rewrite labels slightly; preserve the user's label but keep the generated cues.
            guide = guides.guides[index].model_copy(update={"label": label})
        normalized.append(guide or CategoryGuide(label=label))
    return normalized


def bbox_to_points(bbox: list[float]) -> list[list[float]]:
    """Convert `[x1, y1, x2, y2]` into LabelMe rectangle points."""

    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def normalize_detections(
    result: DetectionResult,
    *,
    labels: list[str],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    source_offset_x: float = 0.0,
    source_offset_y: float = 0.0,
    source_content_width: float | None = None,
    source_content_height: float | None = None,
) -> list[LabelMeShape]:
    """Validate, scale, clip, and convert detections into LabelMe shapes."""

    allowed_labels = set(labels)
    content_width = float(source_content_width if source_content_width is not None else source_width)
    content_height = float(
        source_content_height if source_content_height is not None else source_height
    )
    x_scale = target_width / content_width
    y_scale = target_height / content_height
    shapes: list[LabelMeShape] = []

    for detection in result.shapes:
        if detection.label not in allowed_labels:
            continue

        x1, y1, x2, y2 = detection.bbox
        clamped_to_canvas = [
            min(max(x1, 0.0), float(source_width)),
            min(max(y1, 0.0), float(source_height)),
            min(max(x2, 0.0), float(source_width)),
            min(max(y2, 0.0), float(source_height)),
        ]
        content_coords = [
            clamped_to_canvas[0] - source_offset_x,
            clamped_to_canvas[1] - source_offset_y,
            clamped_to_canvas[2] - source_offset_x,
            clamped_to_canvas[3] - source_offset_y,
        ]
        clipped_to_content = [
            min(max(content_coords[0], 0.0), content_width),
            min(max(content_coords[1], 0.0), content_height),
            min(max(content_coords[2], 0.0), content_width),
            min(max(content_coords[3], 0.0), content_height),
        ]
        scaled = [
            min(max(clipped_to_content[0] * x_scale, 0.0), float(target_width)),
            min(max(clipped_to_content[1] * y_scale, 0.0), float(target_height)),
            min(max(clipped_to_content[2] * x_scale, 0.0), float(target_width)),
            min(max(clipped_to_content[3] * y_scale, 0.0), float(target_height)),
        ]
        left, top, right, bottom = scaled
        if right <= left or bottom <= top:
            continue

        shapes.append(
            LabelMeShape(
                label=detection.label,
                score=float(detection.score),
                points=bbox_to_points([left, top, right, bottom]),
            )
        )

    return shapes


def build_annotation(
    *,
    image_path: str,
    image_width: int,
    image_height: int,
    shapes: list[LabelMeShape],
) -> LabelMeAnnotation:
    """Create a LabelMe-compatible annotation."""

    return LabelMeAnnotation(
        imagePath=image_path,
        imageWidth=image_width,
        imageHeight=image_height,
        shapes=shapes,
    )
