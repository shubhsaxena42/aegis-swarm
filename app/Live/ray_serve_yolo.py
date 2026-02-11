"""
Ray Serve — YOLOv26 Inference Endpoint with Adaptive Autoscaling
════════════════════════════════════════════════════════════════

Deploys YOLO + SAHI + BoT-SORT as a Ray Serve HTTP endpoint.
Receives frames from the Colosseum bridge (or any HTTP client)
and returns detection + tracking results as JSON.

Features:
  - Adaptive autoscaling (1-5 replicas based on request load)
  - GPU-accelerated inference with FP16
  - SAHI sliced inference for small-object detection
  - BoT-SORT multi-object tracking per drone stream
  - Health check endpoint

ARCHITECTURE:
  ┌──────────────┐    HTTP POST     ┌───────────────────────┐
  │  Colosseum   │ ──────────────▶  │  Ray Serve            │
  │  Bridge      │                  │  ┌─────────────────┐  │
  │  (5 drones)  │  ◀──────────── ─ │  │ YOLOv26 + SAHI  │  │
  └──────────────┘    JSON result   │  │ + BoT-SORT      │  │
                                    │  └─────────────────┘  │
                                    │  Replicas: 1→5 (auto) │
                                    └───────────────────────┘

INSTALL:
    pip install "ray[serve]" ultralytics boxmot numpy opencv-python

USAGE:
    # Start Ray Serve
    serve run ray_serve_yolo:deployment

    # Or programmatically:
    python ray_serve_yolo.py
"""

import os
import io
import time
import json
import base64
import logging
import numpy as np
from typing import Dict, List, Optional, Any

import ray
from ray import serve

import torch
import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aegis.ray_serve")

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov26s.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
DEFAULT_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))

# SAHI-lite config
SAHI_SLICES = (2, 2)    # 2x2 grid for speed
SAHI_OVERLAP = 0.1


# ============================================================
# SAHI-LITE (Inlined for zero-dependency deployment)
# ============================================================

def sahi_lite_inference(model, frame, slices=(2, 2), overlap=0.1, conf=0.25, imgsz=640):
    """Lightweight SAHI: slice → infer → NMS merge."""
    h, w = frame.shape[:2]
    slice_h, slice_w = h // slices[0], w // slices[1]
    overlap_h, overlap_w = int(slice_h * overlap), int(slice_w * overlap)

    all_boxes, all_scores, all_classes = [], [], []

    for row in range(slices[0]):
        for col in range(slices[1]):
            y1 = max(0, row * slice_h - overlap_h)
            y2 = min(h, (row + 1) * slice_h + overlap_h)
            x1 = max(0, col * slice_w - overlap_w)
            x2 = min(w, (col + 1) * slice_w + overlap_w)

            results = model.predict(frame[y1:y2, x1:x2], imgsz=imgsz, conf=conf, verbose=False)[0]

            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                all_boxes.append(boxes)
                all_scores.append(results.boxes.conf.cpu().numpy())
                all_classes.append(results.boxes.cls.cpu().numpy())

    if not all_boxes:
        return np.empty((0, 6))

    boxes = np.vstack(all_boxes)
    scores = np.hstack(all_scores)
    classes = np.hstack(all_classes)

    dets = np.hstack([boxes, scores.reshape(-1, 1), classes.reshape(-1, 1)])
    keep = _nms(boxes, scores)
    return dets[keep]


def _nms(boxes, scores, iou_thresh=0.5):
    """Simple NMS."""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return np.array(keep)


# ============================================================
# RAY SERVE DEPLOYMENT
# ============================================================

@serve.deployment(
    name="yolo_detector",

    # ── Adaptive Autoscaling ──────────────────────────────────
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 5,
        "initial_replicas": 1,
        "target_ongoing_requests": 3,          # Scale up when >3 concurrent requests
        "upscale_delay_s": 10,                 # Wait 10s before adding replica
        "downscale_delay_s": 60,               # Wait 60s before removing replica
        "smoothing_factor": 0.5,
        "upscaling_factor": 1.0,               # Add 1 replica at a time
        "downscaling_factor": 1.0,
        "metrics_interval_s": 5,
    },

    # ── Resources per replica ─────────────────────────────────
    ray_actor_options={
        "num_gpus": 1,
        "num_cpus": 2,
    },

    # ── Health check ──────────────────────────────────────────
    health_check_period_s=30,
    health_check_timeout_s=10,
)
class YOLODetector:
    """
    GPU-accelerated YOLO + SAHI + BoT-SORT inference server.

    Each replica loads its own model and maintains per-drone trackers.
    Autoscaling adds/removes replicas based on incoming request volume.
    """

    def __init__(self):
        logger.info("Initializing YOLODetector replica...")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"  Device: {self.device}")

        # Load YOLO model
        self.model = YOLO(MODEL_PATH)
        self.model.to(self.device)

        # FP16 for speed
        if self.device == "cuda:0":
            self.model.model.half()
            logger.info("  FP16 enabled")

        # Per-drone BoT-SORT trackers (keyed by drone_id)
        self._trackers: Dict[str, Any] = {}
        self._class_names = self.model.names

        # Metrics
        self._request_count = 0
        self._total_latency = 0.0

        logger.info(f"  ✅ Model loaded: {MODEL_PATH}")

    def _get_tracker(self, drone_id: str):
        """Get or create a BoT-SORT tracker for a specific drone."""
        if drone_id not in self._trackers:
            from boxmot import BotSort
            self._trackers[drone_id] = BotSort(
                reid_weights=None,
                device=self.device,
                half=(self.device == "cuda:0"),
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=30,
                match_thresh=0.8,
                frame_rate=10,
            )
        return self._trackers[drone_id]

    async def __call__(self, request) -> Dict[str, Any]:
        """
        Handle an inference request.

        Expected JSON payload:
            {
                "image": "<base64-encoded JPEG>",
                "drone_id": "Drone1",
                "timestamp": 1234567890.0
            }

        Returns:
            {
                "drone_id": "Drone1",
                "model": "yolov26s",
                "speed_mode": "SAHI_LITE",
                "latency_s": 0.045,
                "fps": 22.2,
                "detections": [
                    {"track_id": 1, "class": "car", "confidence": 0.92, "bbox": [x1,y1,x2,y2]},
                    ...
                ]
            }
        """
        t0 = time.time()

        # Parse request
        body = await request.json()
        drone_id = body.get("drone_id", "unknown")
        img_b64 = body.get("image", "")

        # Decode image
        img_bytes = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Failed to decode image", "drone_id": drone_id}

        # Run SAHI-lite inference
        detections = sahi_lite_inference(
            self.model, frame,
            slices=SAHI_SLICES,
            overlap=SAHI_OVERLAP,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=DEFAULT_IMGSZ,
        )

        # Run BoT-SORT tracking
        tracker = self._get_tracker(drone_id)
        tracks = tracker.update(detections, frame)

        # Format results
        det_list = []
        for t in tracks:
            cls_id = int(t[6])
            det_list.append({
                "track_id": int(t[4]),
                "class": self._class_names.get(cls_id, str(cls_id)),
                "confidence": round(float(t[5]), 3),
                "bbox": [round(float(x), 1) for x in t[0:4]],
            })

        latency = time.time() - t0
        self._request_count += 1
        self._total_latency += latency

        return {
            "drone_id": drone_id,
            "model": MODEL_PATH,
            "speed_mode": "SAHI_LITE",
            "latency_s": round(latency, 4),
            "fps": round(1.0 / latency if latency > 0 else 0, 1),
            "detections": det_list,
            "tracks_active": len(tracks),
        }

    async def check_health(self):
        """Health check for Ray Serve."""
        avg_latency = (self._total_latency / self._request_count) if self._request_count > 0 else 0
        return {
            "status": "healthy",
            "device": self.device,
            "model": MODEL_PATH,
            "requests_served": self._request_count,
            "avg_latency_ms": round(avg_latency * 1000, 1),
        }


# ── Bind the deployment ──────────────────────────────────────
deployment = YOLODetector.bind()


# ============================================================
# PROGRAMMATIC STARTUP (alternative to `serve run`)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Aegis Swarm — Ray Serve YOLO Deployment")
    print("=" * 60)
    print(f"  Model:        {MODEL_PATH}")
    print(f"  Autoscaling:  1 → 5 replicas")
    print(f"  Endpoint:     http://localhost:8000/detect")
    print("=" * 60)

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Deploy
    # Deploy with HTTP options to match K8s Service (port 8080)
    serve.start(http_options={"host": "0.0.0.0", "port": 8080})
    serve.run(deployment, route_prefix="/detect", name="yolo_detector")

    print("\n✅ Ray Serve is running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down Ray Serve...")
        serve.shutdown()
