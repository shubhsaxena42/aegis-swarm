"""
Aegis Swarm — DVC Data Version Manager
═══════════════════════════════════════

Programmatic DVC operations for dataset and model versioning.

Integrates with MLFlow: when a new dataset version is created with DVC,
the commit hash is logged to the active MLFlow run for full provenance.

Tracked assets:
  - datasets/visdrone/        → VisDrone2019 training data
  - datasets/custom/          → Custom labeled drone imagery
  - models/weights/           → Trained model weights (.pt, .onnx)
  - models/exports/           → Exported inference models

Usage:
    from infrastructure.dvc_versioning.data_manager import AegisDataManager

    dm = AegisDataManager()
    dm.track_dataset("datasets/visdrone", "Initial VisDrone2019-DET import")
    dm.track_model_weights("runs/detect/train/weights/best.pt", "YOLOv26s finetune v1")
    dm.pull_dataset("datasets/visdrone")  # Restore specific version
"""

import os
import json
import subprocess
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aegis.dvc")

# ─── Project Root ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AegisDataManager:
    """
    DVC-based data versioning manager for Aegis Swarm.

    Provides typed methods for tracking datasets, model weights,
    and maintaining dataset-model provenance chains.
    """

    def __init__(self, project_root: Optional[str] = None):
        self._root = Path(project_root) if project_root else PROJECT_ROOT
        self._dvc_dir = self._root / ".dvc"

        if not self._dvc_dir.exists():
            logger.warning(
                "DVC not initialized. Run 'dvc init' in the project root."
            )

    # ═══════════════════════════════════════════════════════════
    #                 DATASET OPERATIONS
    # ═══════════════════════════════════════════════════════════

    def track_dataset(
        self,
        dataset_path: str,
        message: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Track a dataset directory with DVC.

        This will:
          1. Run `dvc add` on the dataset path
          2. Create/update the .dvc tracking file
          3. Compute dataset statistics
          4. Return metadata for MLFlow logging

        Args:
            dataset_path: Relative path to dataset directory
            message:      Description of this dataset version
            tags:         Optional tags for this version

        Returns:
            Metadata dict with hash, stats, and version info
        """
        full_path = self._root / dataset_path

        if not full_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {full_path}")

        # Compute dataset statistics before tracking
        stats = self._compute_dataset_stats(full_path)

        # Run DVC add
        result = self._run_dvc(["add", str(dataset_path)])

        # Get the DVC file hash
        dvc_file = full_path.with_suffix(full_path.suffix + ".dvc")
        if not dvc_file.exists():
            dvc_file = Path(str(full_path) + ".dvc")

        dvc_hash = self._get_dvc_hash(dvc_file) if dvc_file.exists() else "unknown"

        metadata = {
            "path": dataset_path,
            "dvc_hash": dvc_hash,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "stats": stats,
            "tags": tags or {},
        }

        # Save metadata alongside .dvc file
        meta_path = full_path.parent / f".{full_path.name}.meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Tracked dataset: {dataset_path} "
            f"({stats.get('total_files', '?')} files, hash={dvc_hash[:8]}...)"
        )

        return metadata

    def track_model_weights(
        self,
        weights_path: str,
        message: str = "",
        model_name: str = "yolov26s",
    ) -> Dict[str, Any]:
        """
        Track model weight files with DVC.

        Args:
            weights_path: Path to the weights file (e.g., best.pt)
            message:      Description of this model version
            model_name:   Model architecture name

        Returns:
            Metadata dict
        """
        full_path = self._root / weights_path

        if not full_path.exists():
            raise FileNotFoundError(f"Weights file not found: {full_path}")

        # File hash
        file_hash = self._compute_file_hash(full_path)
        file_size_mb = full_path.stat().st_size / (1024 * 1024)

        # Run DVC add
        self._run_dvc(["add", str(weights_path)])

        metadata = {
            "path": weights_path,
            "model_name": model_name,
            "file_hash": file_hash,
            "file_size_mb": round(file_size_mb, 2),
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"Tracked model weights: {weights_path} "
            f"({file_size_mb:.1f} MB, hash={file_hash[:8]}...)"
        )

        return metadata

    def pull_dataset(self, dataset_path: str, remote: str = "minio"):
        """Pull a dataset from DVC remote storage."""
        logger.info(f"Pulling dataset: {dataset_path} from {remote}")
        self._run_dvc(["pull", str(dataset_path), "-r", remote])

    def push_dataset(self, dataset_path: str, remote: str = "minio"):
        """Push a tracked dataset to DVC remote storage."""
        logger.info(f"Pushing dataset: {dataset_path} to {remote}")
        self._run_dvc(["push", str(dataset_path), "-r", remote])

    def get_dataset_version(self, dataset_path: str) -> Optional[str]:
        """Get the current DVC hash for a tracked dataset."""
        dvc_file = self._root / (str(dataset_path) + ".dvc")
        if dvc_file.exists():
            return self._get_dvc_hash(dvc_file)
        return None

    # ═══════════════════════════════════════════════════════════
    #                  DVC PIPELINES
    # ═══════════════════════════════════════════════════════════

    def create_training_pipeline(self) -> Dict[str, Any]:
        """
        Generate a DVC pipeline definition (dvc.yaml) for the
        full training workflow.

        Stages:
          1. prepare  — Convert VisDrone annotations to YOLO format
          2. train    — Train YOLOv26s model
          3. evaluate — Run evaluation on test set
          4. export   — Export to ONNX/TorchScript
        """
        pipeline = {
            "stages": {
                "prepare": {
                    "cmd": "python visdrone_to_yolo_converter.py",
                    "deps": [
                        "visdrone_to_yolo_converter.py",
                        "datasets/visdrone/raw",
                    ],
                    "outs": [
                        "datasets/visdrone/yolo/train",
                        "datasets/visdrone/yolo/val",
                    ],
                    "params": ["params.yaml:prepare"],
                },
                "train": {
                    "cmd": "python kaggle_train_yolo_native.py",
                    "deps": [
                        "kaggle_train_yolo_native.py",
                        "datasets/visdrone/yolo/train",
                        "datasets/visdrone/yolo/val",
                    ],
                    "outs": [
                        {"runs/detect/train/weights/best.pt": {"cache": False}},
                    ],
                    "params": ["params.yaml:train"],
                    "metrics": [
                        {"runs/detect/train/results.csv": {"cache": False}},
                    ],
                    "plots": [
                        {"runs/detect/train/results.png": {"cache": False}},
                    ],
                },
                "evaluate": {
                    "cmd": "python -c \"from ultralytics import YOLO; YOLO('runs/detect/train/weights/best.pt').val()\"",
                    "deps": [
                        "runs/detect/train/weights/best.pt",
                        "datasets/visdrone/yolo/val",
                    ],
                    "metrics": [
                        {"evaluation/metrics.json": {"cache": False}},
                    ],
                },
                "benchmark": {
                    "cmd": "python realtime_drone_inference.py --benchmark-only",
                    "deps": [
                        "realtime_drone_inference.py",
                        "runs/detect/train/weights/best.pt",
                    ],
                    "metrics": [
                        {"evaluation/speed_benchmarks.json": {"cache": False}},
                    ],
                },
            }
        }

        # Write dvc.yaml
        import yaml
        pipeline_path = self._root / "dvc.yaml"
        with open(pipeline_path, "w") as f:
            yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created DVC pipeline: {pipeline_path}")
        return pipeline

    def create_params_file(self) -> Dict[str, Any]:
        """Generate default params.yaml for DVC pipeline."""
        params = {
            "prepare": {
                "source_dir": "datasets/visdrone/raw",
                "output_dir": "datasets/visdrone/yolo",
                "train_split": 0.8,
                "val_split": 0.15,
                "test_split": 0.05,
                "ignore_classes": ["others", "occluded"],
            },
            "train": {
                "model": "yolov26s.pt",
                "imgsz": 640,
                "batch": 16,
                "epochs": 100,
                "optimizer": "AdamW",
                "lr0": 0.001,
                "lrf": 0.01,
                "weight_decay": 0.0005,
                "warmup_epochs": 3,
                "mosaic": 1.0,
                "mixup": 0.1,
                "copy_paste": 0.1,
                "degrees": 10.0,
                "translate": 0.2,
                "scale": 0.5,
                "flipud": 0.5,
                "fliplr": 0.5,
                "device": "0",
                "workers": 8,
                "patience": 20,
            },
        }

        import yaml
        params_path = self._root / "params.yaml"
        with open(params_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created params file: {params_path}")
        return params

    # ═══════════════════════════════════════════════════════════
    #                  INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════

    def _run_dvc(self, args: List[str]) -> str:
        """Run a DVC command and return stdout."""
        try:
            result = subprocess.run(
                ["dvc"] + args,
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.error(f"DVC command failed: dvc {' '.join(args)}\n{result.stderr}")
            return result.stdout
        except FileNotFoundError:
            logger.error("DVC not installed. Install with: pip install dvc[s3]")
            return ""
        except subprocess.TimeoutExpired:
            logger.error(f"DVC command timed out: dvc {' '.join(args)}")
            return ""

    @staticmethod
    def _get_dvc_hash(dvc_file: Path) -> str:
        """Extract the MD5 hash from a .dvc tracking file."""
        try:
            import yaml
            with open(dvc_file) as f:
                dvc_data = yaml.safe_load(f)
            outs = dvc_data.get("outs", [])
            if outs:
                return outs[0].get("md5", "unknown")
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute MD5 hash of a file."""
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def _compute_dataset_stats(dataset_path: Path) -> Dict[str, Any]:
        """Compute statistics for a dataset directory."""
        stats = {
            "total_files": 0,
            "total_size_mb": 0,
            "file_types": {},
            "subdirectories": [],
        }

        for item in dataset_path.rglob("*"):
            if item.is_file():
                stats["total_files"] += 1
                stats["total_size_mb"] += item.stat().st_size / (1024 * 1024)

                ext = item.suffix.lower()
                stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1

            elif item.is_dir() and item != dataset_path:
                rel = str(item.relative_to(dataset_path))
                if rel not in stats["subdirectories"]:
                    stats["subdirectories"].append(rel)

        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats
