from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

from .dataset import IMAGE_EXTENSIONS, prepare_yolo_dataset


def emit(event_type: str, **payload: Any) -> None:
    print(json.dumps({"type": event_type, **payload}, ensure_ascii=False), flush=True)


def prepare_annotation_workspace(source: Path, output: Path) -> dict[str, Any]:
    images = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images found under {source}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "provenance.jsonl"
    rows: list[dict[str, str]] = []
    for index, image in enumerate(images, 1):
        relative = image.relative_to(source)
        target_name = relative.name if len(relative.parts) == 1 else f"{relative.stem}_{index:06d}{relative.suffix.lower()}"
        target = output / target_name
        if not target.exists():
            try:
                os.link(image, target)
            except OSError:
                shutil.copy2(image, target)
        inferred_group = relative.parts[0] if len(relative.parts) > 1 else re.sub(
            r"_\d{6}_\d{6}\.\d{3}s$", "", relative.stem
        )
        rows.append({"target": target.name, "source": str(image), "source_video": inferred_group or relative.stem})
        if index == 1 or index % 50 == 0 or index == len(images):
            emit("progress", current=index, total=len(images), message="Preparing annotation workspace")
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return {"images": len(images), "output_dir": str(output), "manifest": str(manifest)}


def run_annotation(args: argparse.Namespace) -> int:
    from data_annotation_agent.web_job import main as annotation_main

    summary = prepare_annotation_workspace(args.input_dir.resolve(), args.output_dir.resolve())
    emit("workspace_ready", summary=summary)
    command = [
        "--input-dir", str(args.output_dir.resolve()),
        "--output-dir", str(args.output_dir.resolve()),
        "--labels", *args.labels,
        "--coordinate-system", "normalized-1000",
        "--config", str(args.config.resolve()),
    ]
    if args.provider:
        command.extend(("--provider", args.provider))
    if args.overwrite:
        command.append("--overwrite")
    if args.guides_json:
        command.extend(("--guides-json", str(args.guides_json.resolve())))
    return annotation_main(command)


def run_dataset(args: argparse.Namespace) -> int:
    summary = prepare_yolo_dataset(
        args.input_dir, args.output_dir, args.labels,
        replay_sources=args.replay_dir,
        replay_ratio=args.replay_ratio,
        seed=args.seed, ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
    )
    emit("job_summary", summary=summary)
    return 0


def run_training(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    source = args.resume_from or args.model
    emit("training_started", model=str(source), data=str(args.data), epochs=args.epochs, resume=bool(args.resume_from))
    model = YOLO(str(source))

    def report_epoch(trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total = int(getattr(trainer, "epochs", args.epochs) or args.epochs)
        metrics = getattr(trainer, "metrics", {}) or {}
        losses = getattr(trainer, "tloss", None)
        payload: dict[str, Any] = {
            "current": epoch, "total": total, "percent": round(epoch * 100 / max(total, 1), 2),
            "message": f"Epoch {epoch}/{total}",
        }
        if losses is not None:
            values = losses.detach().cpu().tolist() if hasattr(losses, "detach") else list(losses)
            for key, value in zip(("box_loss", "cls_loss", "dfl_loss"), values, strict=False):
                payload[key] = float(value)
        for source_key, target_key in (
            ("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall"),
            ("metrics/mAP50(B)", "map50"), ("metrics/mAP50-95(B)", "map50_95"),
        ):
            if source_key in metrics:
                payload[target_key] = float(metrics[source_key])
        emit("training_progress", **payload)

    model.add_callback("on_train_epoch_end", report_epoch)
    options: dict[str, Any] = dict(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, project=str(args.project), name=args.name,
        exist_ok=False, pretrained=True, optimizer=args.optimizer, patience=args.patience,
        cos_lr=True, close_mosaic=10, seed=args.seed, deterministic=True,
        plots=True, val=True, save=True, amp=args.amp,
    )
    if args.resume_from:
        options["resume"] = True
    results = model.train(**options)
    save_dir = Path(results.save_dir).resolve()
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce best.pt: {best}")
    args.weight_output_dir.mkdir(parents=True, exist_ok=True)
    permanent_best = args.weight_output_dir / f"{args.artifact_name}_best.pt"
    permanent_last = args.weight_output_dir / f"{args.artifact_name}_last.pt"
    shutil.copy2(best, permanent_best)
    if last.is_file():
        shutil.copy2(last, permanent_last)
    metadata = {
        "project_id": args.project_id, "project_name": args.project_name,
        "source_weight": str(args.model.resolve()), "updated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
    }
    permanent_best.with_suffix(".pt.json").write_text(
        json.dumps({**metadata, "kind": "best"}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if permanent_last.is_file():
        permanent_last.with_suffix(".pt.json").write_text(
            json.dumps({**metadata, "kind": "last"}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    emit("weights_saved", best_pt=str(permanent_best), last_pt=str(permanent_last))
    emit("job_summary", summary={"run_dir": str(save_dir), "working_best_pt": str(best),
                                  "best_pt": str(permanent_best),
                                  "last_pt": str(permanent_last) if permanent_last.is_file() else None,
                                  "pretrained_weight": str(args.model.resolve())})
    return 0


def run_export(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.data), split="test", imgsz=args.imgsz, device=args.device, plots=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"{args.artifact_name}.onnx"
    if args.precision == "fp32":
        exported = _export_fp32(model, args)
        shutil.copy2(exported, target)
    elif args.precision == "fp16":
        exported = _export_fp16(model, args)
        shutil.copy2(exported, target)
    else:
        exported = _export_fp32(model, args)
        _export_int8(Path(exported), target)
    metric_values = getattr(metrics, "results_dict", {})
    target.with_suffix(".onnx.json").write_text(json.dumps({
        "project_id": args.project_id, "project_name": args.project_name,
        "weights": str(args.weights.resolve()), "precision": args.precision,
        "imgsz": args.imgsz, "updated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    emit("job_summary", summary={
        "weights": str(args.weights.resolve()), "onnx": str(target.resolve()), "precision": args.precision,
        "metrics": {str(key): float(value) for key, value in metric_values.items() if isinstance(value, (int, float))},
        "save_dir": str(Path(metrics.save_dir).resolve()),
    })
    return 0


def _export_fp32(model: Any, args: argparse.Namespace) -> Path:
    return Path(model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True,
                             half=False, device=args.device)).resolve()


def _export_fp16(model: Any, args: argparse.Namespace) -> Path:
    return Path(model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True,
                             half=True, device=args.device)).resolve()


def _export_int8(source: Path, target: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="vtax-workflow")
    commands = root.add_subparsers(dest="command", required=True)
    annotation = commands.add_parser("annotation")
    annotation.add_argument("--input-dir", type=Path, required=True)
    annotation.add_argument("--output-dir", type=Path, required=True)
    annotation.add_argument("--labels", nargs="+", required=True)
    annotation.add_argument("--config", type=Path, required=True)
    annotation.add_argument("--provider", choices=("dashscope", "ollama"))
    annotation.add_argument("--guides-json", type=Path)
    annotation.add_argument("--overwrite", action="store_true")

    dataset = commands.add_parser("dataset")
    dataset.add_argument("--input-dir", type=Path, required=True)
    dataset.add_argument("--output-dir", type=Path, required=True)
    dataset.add_argument("--labels", nargs="+", required=True)
    dataset.add_argument("--replay-dir", type=Path, action="append", default=[])
    dataset.add_argument("--replay-ratio", type=float, default=0.4)
    dataset.add_argument("--seed", type=int, default=42)
    dataset.add_argument("--train-ratio", type=float, default=0.8)
    dataset.add_argument("--val-ratio", type=float, default=0.15)
    dataset.add_argument("--test-ratio", type=float, default=0.05)

    training = commands.add_parser("training")
    training.add_argument("--model", type=Path, required=True)
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--project", type=Path, required=True)
    training.add_argument("--name", required=True)
    training.add_argument("--weight-output-dir", type=Path, required=True)
    training.add_argument("--artifact-name", required=True)
    training.add_argument("--project-id", required=True)
    training.add_argument("--project-name", required=True)
    training.add_argument("--epochs", type=int, default=80)
    training.add_argument("--imgsz", type=int, default=960)
    training.add_argument("--batch", type=int, default=8)
    training.add_argument("--device", default="0")
    training.add_argument("--workers", type=int, default=8)
    training.add_argument("--optimizer", default="auto")
    training.add_argument("--patience", type=int, default=25)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--resume-from", type=Path)
    amp = training.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    training.set_defaults(amp=False)

    export = commands.add_parser("export")
    export.add_argument("--weights", type=Path, required=True)
    export.add_argument("--data", type=Path, required=True)
    export.add_argument("--imgsz", type=int, default=960)
    export.add_argument("--device", default="0")
    export.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32")
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--artifact-name", required=True)
    export.add_argument("--project-id", required=True)
    export.add_argument("--project-name", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "annotation":
            return run_annotation(args)
        if args.command == "dataset":
            return run_dataset(args)
        if args.command == "training":
            return run_training(args)
        if args.command == "export":
            return run_export(args)
        raise AssertionError(args.command)
    except Exception as exc:  # noqa: BLE001
        emit("job_error", error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
