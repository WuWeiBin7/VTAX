"""JSONL runner used by the local web frontend."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

from data_annotation_agent.config import DEFAULT_CONFIG_PATH, ModelConfig
from data_annotation_agent.langchain_client import build_annotation_client
from data_annotation_agent.pipeline import annotate_directory, get_category_guides, stats_to_dict
from data_annotation_agent.schemas import CategoryGuide, validate_labels


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the web job runner."""

    parser = argparse.ArgumentParser(prog="data-annotation-agent-web-job")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--labels", required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--base-url")
    parser.add_argument("--research-model")
    parser.add_argument("--vision-model")
    parser.add_argument("--provider", choices=["dashscope", "ollama"])
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--max-bytes", type=int, default=7_000_000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--research-only", action="store_true")
    parser.add_argument("--guides-json", type=Path)
    parser.add_argument(
        "--coordinate-system",
        choices=["normalized-1000", "pixel"],
        default="pixel",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run an annotation job and stream JSONL progress events to stdout."""

    _configure_stdio()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.input_dir

    try:
        guides = _load_guides(args.guides_json)
        config = ModelConfig.from_sources(
            config_path=args.config,
            base_url=args.base_url,
            research_model=args.research_model,
            vision_model=args.vision_model,
            provider=args.provider,
            require_research_credentials=args.research_only or guides is None,
        )
        client = build_annotation_client(config)
        if args.research_only:
            labels = validate_labels(args.labels)
            output_dir.mkdir(parents=True, exist_ok=True)
            researched_guides = get_category_guides(
                output_dir=output_dir,
                labels=labels,
                client=client,
                category_guides=guides,
                force_research=guides is None,
            )
            _emit(
                {
                    "type": "research_done",
                    "labels": labels,
                    "guides": [guide.model_dump(mode="json") for guide in researched_guides],
                }
            )
            return 0

        stats = annotate_directory(
            input_dir=args.input_dir,
            output_dir=output_dir,
            labels=args.labels,
            client=client,
            recursive=not args.no_recursive,
            overwrite=args.overwrite,
            max_edge=args.max_edge,
            max_bytes=args.max_bytes,
            retries=args.retries,
            coordinate_system=args.coordinate_system,
            category_guides=guides,
            progress_callback=_emit,
        )
        _emit({"type": "job_summary", "summary": stats_to_dict(stats)})
        return 0
    except Exception as exc:  # noqa: BLE001
        _emit({"type": "job_error", "error": str(exc)})
        return 1


def _emit(event: dict[str, Any]) -> None:
    """Write one JSON event per line for the Next.js job manager."""

    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _configure_stdio() -> None:
    """Force UTF-8 JSONL output so the web frontend can decode Chinese text correctly."""

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _load_guides(path: Path | None) -> list[CategoryGuide] | None:
    """Load user-confirmed category guides from a JSON file."""

    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("guides", [])
    return [CategoryGuide.model_validate(item) for item in payload]


if __name__ == "__main__":
    raise SystemExit(main())
