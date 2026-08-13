"""Category guide cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data_annotation_agent.schemas import CategoryGuide


def labels_cache_key(labels: list[str]) -> str:
    """Create a stable cache key for a label list."""

    payload = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_guides_from_cache(cache_path: Path, labels: list[str]) -> list[CategoryGuide] | None:
    """Load cached category guides for a label list."""

    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        entry = payload.get("entries", {}).get(labels_cache_key(labels))
        if not entry or entry.get("labels") != labels:
            return None
        guides = [CategoryGuide.model_validate(item) for item in entry.get("guides", [])]
        if any(_guide_contains_mojibake(guide) for guide in guides):
            return None
        return guides
    except (OSError, ValueError, TypeError):
        return None


def save_guides_to_cache(cache_path: Path, labels: list[str], guides: list[CategoryGuide]) -> None:
    """Persist category guides for reuse."""

    payload: dict[str, object] = {"entries": {}}
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = {"entries": {}}

    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        payload["entries"] = entries

    entries[labels_cache_key(labels)] = {
        "labels": labels,
        "guides": [guide.model_dump(mode="json") for guide in guides],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _guide_contains_mojibake(guide: CategoryGuide) -> bool:
    """Detect common replacement/mojibake characters in cached guides."""

    values = [
        guide.label,
        *guide.visual_cues,
        *guide.ocr_cues,
        *guide.negative_cues,
        *guide.ambiguity_notes,
    ]
    markers = ("�", "锛", "绫", "鏍", "妯", "涓", "鐢", "瑙", "鍥")
    return any(any(marker in value for marker in markers) for value in values)
