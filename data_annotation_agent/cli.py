"""Command line interface for the data annotation agent."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_annotation_agent.config import (
    DEFAULT_CONFIG_PATH,
    ModelConfig,
)
from data_annotation_agent.langchain_client import build_annotation_client
from data_annotation_agent.pipeline import annotate_directory


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="data-annotation-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser("annotate", help="Annotate images in a folder.")
    annotate.add_argument("--input-dir", required=True, type=Path)
    annotate.add_argument("--labels", required=True, nargs="+")
    annotate.add_argument("--output-dir", type=Path)
    annotate.add_argument("--overwrite", action="store_true")
    annotate.add_argument("--no-recursive", action="store_true")
    annotate.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    annotate.add_argument("--base-url")
    annotate.add_argument("--research-model")
    annotate.add_argument("--vision-model")
    annotate.add_argument("--provider", choices=["dashscope", "ollama"])
    annotate.add_argument("--max-edge", type=int, default=1024)
    annotate.add_argument("--max-bytes", type=int, default=7_000_000)
    annotate.add_argument("--retries", type=int, default=3)
    annotate.add_argument(
        "--coordinate-system",
        choices=["normalized-1000", "pixel"],
        default="pixel",
        help="Coordinate system returned by the vision model.",
    )
    annotate.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    if args.command == "annotate":
        output_dir = args.output_dir or args.input_dir
        config = ModelConfig.from_sources(
            config_path=args.config,
            base_url=args.base_url,
            research_model=args.research_model,
            vision_model=args.vision_model,
            provider=args.provider,
        )
        client = build_annotation_client(config)
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
        )
        logging.info(
            "Finished: total=%s processed=%s annotated=%s empty=%s skipped=%s failed=%s "
            "total_time=%.2fs image_time=%.2fs tokens=%s label_counts=%s",
            stats.total,
            stats.processed,
            stats.annotated,
            stats.empty,
            stats.skipped,
            stats.failed,
            stats.total_seconds,
            stats.image_seconds,
            (
                f"total={stats.total_tokens}, input={stats.total_input_tokens}, "
                f"output={stats.total_output_tokens}, cached={stats.total_cached_tokens}, "
                f"cache_created={stats.total_cache_creation_input_tokens}, "
                f"images_with_usage={stats.token_usage_images}"
                if stats.token_usage_images
                else "unavailable"
            ),
            dict(stats.label_counts),
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
