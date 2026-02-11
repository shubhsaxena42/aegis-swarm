"""
Aegis Swarm — MLFlow Experiment Tracker
═══════════════════════════════════════

Integrated MLFlow wrapper for tracking YOLO drone detection experiments.

Tracks:
  - Training hyperparameters (imgsz, batch, epochs, augmentation, etc.)
  - Training metrics (mAP, precision, recall, loss curves)
  - Speed mode benchmarks (ULTRA_FAST, FAST, BALANCED, ACCURATE)
  - Model artifacts (weights, ONNX exports, TorchScript)
  - Dataset metadata (VisDrone version, split ratios, class distribution)
  - Hardware context (GPU, VRAM, inference device)

Usage:
    from infrastructure.mlflow_tracking.experiment_tracker import AegisExperimentTracker

    tracker = AegisExperimentTracker()

    # Start a training run
    with tracker.start_training_run("yolov26s-visdrone-finetune") as run:
        tracker.log_training_config(model="yolov26s", imgsz=640, epochs=100)
        tracker.log_epoch_metrics(epoch=1, train_loss=0.5, val_map50=0.42)
        tracker.log_model_artifact("runs/detect/train/weights/best.pt")
        tracker.log_speed_benchmark(mode="FAST", fps=25.3, latency_ms=39.5)

    # Register production model
    tracker.register_model("yolov26s-visdrone-finetune", stage="Production")
"""

import os
import sys
import json
import time
import platform
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# ─── Configuration ────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_S3_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
EXPERIMENT_PREFIX = "aegis-swarm"

# MinIO credentials for artifact storage
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", MLFLOW_S3_ENDPOINT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aegis.mlflow")


class AegisExperimentTracker:
    """
    MLFlow experiment tracker specialized for Aegis Swarm drone detection models.
    """

    def __init__(self, tracking_uri: str = MLFLOW_TRACKING_URI):
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._active_run = None
        logger.info(f"MLFlow tracker connected to {tracking_uri}")

    # ═══════════════════════════════════════════════════════════
    #                  EXPERIMENT MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def get_or_create_experiment(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Get or create an MLFlow experiment."""
        full_name = f"{EXPERIMENT_PREFIX}/{name}"
        experiment = self._client.get_experiment_by_name(full_name)

        if experiment is None:
            experiment_id = self._client.create_experiment(
                full_name,
                tags={
                    "project": "aegis-swarm",
                    "team": "drone-ml",
                    **(tags or {}),
                },
            )
            logger.info(f"Created experiment: {full_name} (id={experiment_id})")
        else:
            experiment_id = experiment.experiment_id

        return experiment_id

    # ═══════════════════════════════════════════════════════════
    #                  RUN LIFECYCLE
    # ═══════════════════════════════════════════════════════════

    @contextmanager
    def start_training_run(
        self,
        run_name: str,
        experiment_name: str = "yolo-training",
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Context manager for a training run.

        Usage:
            with tracker.start_training_run("yolov26s-finetune-v3") as run:
                tracker.log_training_config(...)
                tracker.log_epoch_metrics(...)
        """
        experiment_id = self.get_or_create_experiment(experiment_name)

        run_tags = {
            "mlflow.runName": run_name,
            "project": "aegis-swarm",
            "run_type": "training",
            "host": platform.node(),
            "python_version": platform.python_version(),
            **(tags or {}),
        }

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=run_tags,
        ) as run:
            self._active_run = run
            logger.info(f"Started MLFlow run: {run_name} (id={run.info.run_id})")

            # Log system info
            self._log_system_info()

            try:
                yield run
            except Exception as e:
                mlflow.set_tag("run_status", "failed")
                mlflow.set_tag("error", str(e))
                logger.error(f"Run failed: {e}")
                raise
            finally:
                self._active_run = None

    @contextmanager
    def start_inference_benchmark(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Context manager for inference benchmark runs."""
        experiment_id = self.get_or_create_experiment("inference-benchmarks")

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags={"run_type": "benchmark", **(tags or {})},
        ) as run:
            self._active_run = run
            self._log_system_info()
            yield run
            self._active_run = None

    # ═══════════════════════════════════════════════════════════
    #                   LOGGING METHODS
    # ═══════════════════════════════════════════════════════════

    def log_training_config(
        self,
        model: str = "yolov26s",
        imgsz: int = 640,
        batch_size: int = 16,
        epochs: int = 100,
        optimizer: str = "AdamW",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        momentum: float = 0.937,
        warmup_epochs: int = 3,
        augmentation: Optional[Dict[str, Any]] = None,
        dataset: str = "VisDrone2019-DET",
        dataset_version: str = "1.0",
        num_classes: int = 10,
        freeze_layers: int = 0,
        use_sahi: bool = False,
        sahi_slices: str = "2x2",
        speed_mode: str = "BALANCED",
        **extra_params,
    ):
        """Log training hyperparameters."""
        params = {
            "model": model,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "warmup_epochs": warmup_epochs,
            "dataset": dataset,
            "dataset_version": dataset_version,
            "num_classes": num_classes,
            "freeze_layers": freeze_layers,
            "use_sahi": use_sahi,
            "sahi_slices": sahi_slices,
            "speed_mode": speed_mode,
            **extra_params,
        }

        if augmentation:
            for k, v in augmentation.items():
                params[f"aug_{k}"] = v

        mlflow.log_params(params)
        logger.info(f"Logged training config: {model} @ {imgsz}px, {epochs} epochs")

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_map50: Optional[float] = None,
        val_map50_95: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        learning_rate: Optional[float] = None,
        **extra_metrics,
    ):
        """Log per-epoch training metrics."""
        metrics = {
            k: v
            for k, v in {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/mAP50": val_map50,
                "val/mAP50-95": val_map50_95,
                "val/precision": precision,
                "val/recall": recall,
                "train/lr": learning_rate,
                **{f"custom/{k}": v for k, v in extra_metrics.items()},
            }.items()
            if v is not None
        }

        mlflow.log_metrics(metrics, step=epoch)

    def log_final_metrics(
        self,
        best_map50: float,
        best_map50_95: float,
        best_epoch: int,
        total_training_time_s: float,
        final_model_size_mb: Optional[float] = None,
    ):
        """Log final training metrics."""
        mlflow.log_metrics({
            "best/mAP50": best_map50,
            "best/mAP50-95": best_map50_95,
            "best/epoch": best_epoch,
            "training_time_seconds": total_training_time_s,
            "training_time_hours": total_training_time_s / 3600,
        })
        if final_model_size_mb:
            mlflow.log_metric("model_size_mb", final_model_size_mb)

    def log_speed_benchmark(
        self,
        mode: str,
        fps: float,
        latency_ms: float,
        imgsz: int = 640,
        use_sahi: bool = False,
        half_precision: bool = True,
        device: str = "cuda",
    ):
        """Log inference speed benchmark results."""
        mlflow.log_metrics({
            f"speed/{mode}/fps": fps,
            f"speed/{mode}/latency_ms": latency_ms,
            f"speed/{mode}/imgsz": imgsz,
        })
        mlflow.log_params({
            f"speed/{mode}/use_sahi": use_sahi,
            f"speed/{mode}/half_precision": half_precision,
            f"speed/{mode}/device": device,
        })

    def log_per_class_metrics(
        self,
        class_metrics: Dict[str, Dict[str, float]],
    ):
        """
        Log per-class detection metrics.

        Args:
            class_metrics: {"car": {"ap50": 0.87, "ap50_95": 0.62, "count": 5000}, ...}
        """
        for class_name, metrics in class_metrics.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(
                    f"class/{class_name}/{metric_name}", value
                )

    # ═══════════════════════════════════════════════════════════
    #                  ARTIFACT LOGGING
    # ═══════════════════════════════════════════════════════════

    def log_model_artifact(
        self,
        model_path: str,
        artifact_subdir: str = "model",
    ):
        """Log a model weights file as an artifact."""
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_subdir)
            logger.info(f"Logged model artifact: {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")

    def log_training_curves(self, results_csv_path: str):
        """Log YOLO training results CSV."""
        if os.path.exists(results_csv_path):
            mlflow.log_artifact(results_csv_path, "training_curves")

    def log_confusion_matrix(self, confusion_matrix_path: str):
        """Log confusion matrix plot."""
        if os.path.exists(confusion_matrix_path):
            mlflow.log_artifact(confusion_matrix_path, "evaluation")

    def log_dataset_info(
        self,
        dataset_name: str,
        train_images: int,
        val_images: int,
        test_images: int,
        class_distribution: Dict[str, int],
        dvc_version: Optional[str] = None,
    ):
        """Log dataset metadata."""
        mlflow.log_params({
            "dataset/name": dataset_name,
            "dataset/train_images": train_images,
            "dataset/val_images": val_images,
            "dataset/test_images": test_images,
            "dataset/total_images": train_images + val_images + test_images,
            "dataset/num_classes": len(class_distribution),
        })
        if dvc_version:
            mlflow.log_param("dataset/dvc_version", dvc_version)

        # Log class distribution as artifact
        mlflow.log_dict(
            {"class_distribution": class_distribution},
            "dataset/class_distribution.json",
        )

    # ═══════════════════════════════════════════════════════════
    #                  MODEL REGISTRY
    # ═══════════════════════════════════════════════════════════

    def register_model(
        self,
        run_id: str,
        model_name: str = "aegis-swarm-yolo",
        artifact_path: str = "model",
        stage: str = "Staging",
        description: str = "",
    ):
        """
        Register a trained model in the MLFlow Model Registry.

        Args:
            run_id:        Run ID containing the model artifact
            model_name:    Registry model name
            artifact_path: Artifact subdirectory containing the model
            stage:         Stage to transition to (Staging, Production, Archived)
            description:   Model version description
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"

        result = mlflow.register_model(model_uri, model_name)
        logger.info(
            f"Registered model: {model_name} v{result.version} (run: {run_id})"
        )

        # Transition to target stage
        self._client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )
        logger.info(f"Transitioned {model_name} v{result.version} → {stage}")

        if description:
            self._client.update_model_version(
                name=model_name,
                version=result.version,
                description=description,
            )

        return result

    def get_production_model_uri(
        self, model_name: str = "aegis-swarm-yolo"
    ) -> Optional[str]:
        """Get the URI of the current production model."""
        try:
            versions = self._client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                return f"models:/{model_name}/Production"
        except Exception:
            pass
        return None

    # ═══════════════════════════════════════════════════════════
    #                  INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════

    def _log_system_info(self):
        """Log system and hardware information."""
        mlflow.log_params({
            "system/os": platform.system(),
            "system/python": platform.python_version(),
            "system/hostname": platform.node(),
            "system/architecture": platform.machine(),
        })

        # GPU info (if available)
        try:
            import torch
            if torch.cuda.is_available():
                mlflow.log_params({
                    "gpu/name": torch.cuda.get_device_name(0),
                    "gpu/vram_gb": round(
                        torch.cuda.get_device_properties(0).total_mem / 1e9, 1
                    ),
                    "gpu/cuda_version": torch.version.cuda,
                    "gpu/count": torch.cuda.device_count(),
                })
        except ImportError:
            pass

    def compare_runs(
        self,
        experiment_name: str = "yolo-training",
        metric: str = "best/mAP50",
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Compare top N runs by a metric.

        Returns a list of dicts with run_id, metrics, and params.
        """
        experiment_id = self.get_or_create_experiment(experiment_name)
        runs = self._client.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.`{metric}` DESC"],
            max_results=top_n,
        )

        results = []
        for run in runs:
            results.append({
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "status": run.info.status,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
            })

        return results
