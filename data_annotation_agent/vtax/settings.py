from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        default_root = Path(os.environ.get("VTAX_ROOT", repo_root.parent))
        self.root = default_root.resolve()
        self.workspace_root = Path(os.environ.get("VTAX_WORKSPACE_ROOT", self.root / "workspaces")).resolve()
        self.labeled_dataset_root = Path(
            os.environ.get("VTAX_LABELED_DATASET_ROOT", self.root / "LabeledDataset")
        ).resolve()
        self.weight_root = Path(os.environ.get("VTAX_WEIGHT_ROOT", self.root / "Weight")).resolve()
        self.onnx_root = Path(os.environ.get("VTAX_ONNX_ROOT", self.root / "ONNX")).resolve()
        self.global_category_guides_path = Path(
            os.environ.get("VTAX_GLOBAL_CATEGORY_GUIDES", self.root / "default_category_guides.json")
        ).resolve()
        self.database_path = Path(os.environ.get("VTAX_DATABASE", self.root / "var" / "vtax.sqlite3")).resolve()
        self.repo_root = Path(os.environ.get("VTAX_REPO_ROOT", repo_root)).resolve()
        self.depth_python = Path(os.environ.get("VTAX_DEPTH_PYTHON", "/root/miniconda3/envs/depthv3/bin/python"))
        self.yolo_python = Path(os.environ.get("VTAX_YOLO_PYTHON", "/root/miniconda3/envs/yolo/bin/python"))
        self.annotation_python = Path(os.environ.get("VTAX_PYTHON", os.sys.executable))
        self.dino_model = Path(os.environ.get("VTAX_DINO_MODEL", self.root / "models" / "dinov3" / "vit_l16"))
        self.yolo_model_root = Path(os.environ.get("VTAX_YOLO_MODELS", self.root / "models" / "yolo26"))
        self.config_path = Path(os.environ.get("VTAX_MODEL_CONFIG", self.repo_root / "config.json"))
        raw_roots = os.environ.get("VTAX_ALLOWED_ROOTS", f"{self.root}{os.pathsep}/mnt/d/wwb/VTAX")
        self.allowed_roots = tuple(Path(item).expanduser().resolve() for item in raw_roots.split(os.pathsep) if item.strip())
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.labeled_dataset_root.mkdir(parents=True, exist_ok=True)
        self.weight_root.mkdir(parents=True, exist_ok=True)
        self.onnx_root.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def model_path(self, size: str) -> Path:
        if size not in {"n", "s", "m", "l", "x"}:
            raise ValueError(f"Unsupported YOLO26 model size: {size}")
        return self.yolo_model_root / f"yolo26{size}.pt"


settings = Settings()
