"""
YOLO26s + BoT-SORT Real-Time Drone Feed Processing
══════════════════════════════════════════════════

This script is optimized for real-time inference at 10+ FPS.
Runs LOCALLY (not on Kaggle) with RTSP/webcam/video streams.
Integrates with Redpanda to publish inference telemetry.

SPEED MODES:
- ULTRA_FAST: Native YOLO, 640px, no SAHI → 30-50 FPS
- FAST: Native YOLO, 1280px, no SAHI → 15-25 FPS  
- BALANCED: SAHI-lite (2x2 slices) → 8-12 FPS
- ACCURATE: Full SAHI (3x3 slices) → 3-5 FPS

USAGE:
    python realtime_drone_inference.py

REQUIREMENTS:
    pip install ultralytics boxmot opencv-python numpy confluent-kafka
"""

import cv2
import time
import sys
import os
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from boxmot import BotSort

# Add project root to path for infrastructure imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from infrastructure.redpanda.producer import AegisTelemetryProducer
    REDPANDA_AVAILABLE = True
except ImportError:
    print("⚠️ Redpanda infrastructure not found. Running in standalone mode.")
    REDPANDA_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================

# Input source (uncomment one)
VIDEO_SOURCE = "rtsp://your-drone-ip:8554/stream"  # RTSP stream
# VIDEO_SOURCE = 0  # Webcam
# VIDEO_SOURCE = "drone_feed.mp4"  # Video file

# Model path
MODEL_PATH = "yolov26s.pt"

# Speed mode: "ULTRA_FAST", "FAST", "BALANCED", "ACCURATE"
SPEED_MODE = "FAST"

# Display settings
SHOW_FPS = True
SHOW_TRACKS = True
WINDOW_NAME = "YOLO26s + BoT-SORT | Press 'Q' to quit"

# Output recording (set to None to disable)
OUTPUT_PATH = "output_realtime.mp4"  # or None

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25

# Drone ID for telemetry
DRONE_ID = "drone-001"

# ============================================================
# SPEED MODE CONFIGURATIONS
# ============================================================

SPEED_CONFIGS = {
    "ULTRA_FAST": {
        "imgsz": 640,
        "use_sahi": False,
        "half": True,  # FP16 inference
        "target_fps": 30,
        "description": "Max speed, may miss small objects"
    },
    "FAST": {
        "imgsz": 1280,
        "use_sahi": False,
        "half": True,
        "target_fps": 20,
        "description": "Good balance for 10fps streams"
    },
    "BALANCED": {
        "imgsz": 640,
        "use_sahi": True,
        "sahi_slices": (2, 2),  # 2x2 grid = 4 slices
        "sahi_overlap": 0.1,
        "half": True,
        "target_fps": 10,
        "description": "SAHI-lite for small objects"
    },
    "ACCURATE": {
        "imgsz": 640,
        "use_sahi": True,
        "sahi_slices": (3, 3),  # 3x3 grid = 9 slices
        "sahi_overlap": 0.2,
        "half": False,
        "target_fps": 5,
        "description": "Full accuracy, not real-time"
    }
}

# ============================================================
# SAHI-LITE IMPLEMENTATION (Optimized for speed)
# ============================================================

def sahi_lite_inference(model, frame, slices=(2, 2), overlap=0.1, conf=0.25, imgsz=640):
    """
    Lightweight SAHI implementation for real-time.
    Only slices the image, no full-frame pass.
    """
    h, w = frame.shape[:2]
    slice_h = h // slices[0]
    slice_w = w // slices[1]
    
    overlap_h = int(slice_h * overlap)
    overlap_w = int(slice_w * overlap)
    
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for row in range(slices[0]):
        for col in range(slices[1]):
            # Calculate slice coordinates with overlap
            y1 = max(0, row * slice_h - overlap_h)
            y2 = min(h, (row + 1) * slice_h + overlap_h)
            x1 = max(0, col * slice_w - overlap_w)
            x2 = min(w, (col + 1) * slice_w + overlap_w)
            
            # Extract slice
            slice_img = frame[y1:y2, x1:x2]
            
            # Run inference on slice
            results = model.predict(
                slice_img, 
                imgsz=imgsz, 
                conf=conf, 
                verbose=False
            )[0]
            
            # Offset boxes back to original coordinates
            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                boxes[:, [0, 2]] += x1  # Offset x
                boxes[:, [1, 3]] += y1  # Offset y
                
                all_boxes.append(boxes)
                all_scores.append(results.boxes.conf.cpu().numpy())
                all_classes.append(results.boxes.cls.cpu().numpy())
    
    if len(all_boxes) == 0:
        return np.empty((0, 6))
    
    # Concatenate all detections
    boxes = np.vstack(all_boxes)
    scores = np.hstack(all_scores)
    classes = np.hstack(all_classes)
    
    # Apply NMS to remove duplicates from overlapping slices
    dets = np.hstack([boxes, scores.reshape(-1, 1), classes.reshape(-1, 1)])
    keep = nms(boxes, scores, iou_threshold=0.5)
    
    return dets[keep]


def nms(boxes, scores, iou_threshold=0.5):
    """Simple NMS implementation."""
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
        
        # Calculate IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)


# ============================================================
# NATIVE YOLO INFERENCE
# ============================================================

def native_inference(model, frame, imgsz=640, conf=0.25):
    """Standard YOLO inference without slicing."""
    results = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    
    if len(results.boxes) == 0:
        return np.empty((0, 6))
    
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy().reshape(-1, 1)
    classes = results.boxes.cls.cpu().numpy().reshape(-1, 1)
    
    return np.hstack([boxes, scores, classes])


# ============================================================
# FPS CALCULATOR
# ============================================================

class FPSCounter:
    def __init__(self, window_size=30):
        self.times = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def update(self):
        current_time = time.time()
        self.times.append(current_time - self.last_time)
        self.last_time = current_time
    
    def get_fps(self):
        if len(self.times) == 0:
            return 0
        return 1.0 / (sum(self.times) / len(self.times))


# ============================================================
# VISUALIZATION
# ============================================================

def draw_tracks(frame, tracks, fps=0, mode=""):
    """Draw bounding boxes and track IDs on frame."""
    
    # Define colors for different track IDs
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)
    ]
    
    for track in tracks:
        x1, y1, x2, y2, track_id, conf, cls, _ = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        
        color = colors[track_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"#{track_id} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw FPS and mode
    if fps > 0:
        cv2.putText(frame, f"FPS: {fps:.1f} | Mode: {mode} | Tracks: {len(tracks)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame


# ============================================================
# MAIN REAL-TIME LOOP
# ============================================================

def main():
    print("=" * 60)
    print("🚀 YOLO26s + BoT-SORT Real-Time Drone Inference")
    print("=" * 60)
    
    # Initialize Redpanda producer
    producer = None
    if REDPANDA_AVAILABLE:
        try:
            producer = AegisTelemetryProducer()
            print("✅ Connected to Redpanda (AegisTelemetryProducer)")
        except Exception as e:
            print(f"⚠️ Failed to connect to Redpanda: {e}")

    # Load configuration
    config = SPEED_CONFIGS[SPEED_MODE]
    print(f"\n📌 Speed Mode: {SPEED_MODE}")
    print(f"   {config['description']}")
    print(f"   Target FPS: {config['target_fps']}")
    
    # Initialize device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n📍 Device: {device}")
    if device == "cuda:0":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load YOLO model
    print(f"\n🔧 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.to(device)
    
    # Enable FP16 if specified
    if config.get("half", False) and device == "cuda:0":
        model.model.half()
        print("   ✅ FP16 (half precision) enabled")
    
    # Initialize BoT-SORT tracker
    print("🔧 Initializing BoT-SORT tracker...")
    tracker = BotSort(
        reid_weights=None,
        device=device,
        half=config.get("half", False),
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=10  # Match drone feed rate
    )
    
    # Open video source
    print(f"\n📹 Opening source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print("❌ Failed to open video source!")
        # Fallback to test file if RTSP fails
        if VIDEO_SOURCE.startswith("rtsp"):
            print("   → RTSP failed. Trying 'drone_feed.mp4'...")
            cap = cv2.VideoCapture("drone_feed.mp4")
            if not cap.isOpened():
                 return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 10
    print(f"   Resolution: {width}x{height}")
    print(f"   Input FPS: {input_fps}")
    
    # Initialize video writer if output specified
    writer = None
    if OUTPUT_PATH:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, input_fps, (width, height))
        print(f"   📼 Recording to: {OUTPUT_PATH}")
    
    # Initialize FPS counter
    fps_counter = FPSCounter()
    
    print("\n" + "=" * 60)
    print("▶️  Starting real-time processing... Press 'Q' to quit")
    print("=" * 60 + "\n")
    
    frame_count = 0
    inference_start_time = time.time()
    
    try:
        while True:
            iter_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("⚠️ End of stream or read error")
                break
            
            # Run detection
            if config["use_sahi"]:
                detections = sahi_lite_inference(
                    model, frame,
                    slices=config["sahi_slices"],
                    overlap=config["sahi_overlap"],
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=config["imgsz"]
                )
            else:
                detections = native_inference(
                    model, frame,
                    imgsz=config["imgsz"],
                    conf=CONFIDENCE_THRESHOLD
                )
            
            # Run tracking
            tracks = tracker.update(detections, frame)
            
            # Calculate latency
            latency = time.time() - iter_start
            fps_counter.update()
            current_fps = fps_counter.get_fps()
            
            # Publish to Redpanda
            if producer and frame_count % 3 == 0:  # Don't flood: every 3rd frame (~3-4 fps)
                # Format detections for JSON
                det_list = []
                for t in tracks:
                    # track: [x1, y1, x2, y2, id, conf, cls, ind]
                    det_list.append({
                        "track_id": int(t[4]),
                        "class": model.names[int(t[6])] if hasattr(model, 'names') else str(int(t[6])),
                        "confidence": float(t[5]),
                        "bbox": [float(x) for x in t[0:4]]
                    })
                
                producer.send_inference_result(
                    drone_id=DRONE_ID,
                    model_name=f"yolov26s-{SPEED_MODE}",
                    speed_mode=SPEED_MODE,
                    latency_s=latency,
                    fps=current_fps,
                    detections=det_list
                )

            # Draw results
            annotated_frame = draw_tracks(frame, tracks, current_fps, SPEED_MODE)
            
            # Display
            cv2.imshow(WINDOW_NAME, annotated_frame)
            
            # Write to output
            if writer:
                writer.write(annotated_frame)
            
            frame_count += 1
            
            # Status update every 100 frames
            if frame_count % 100 == 0:
                print(f"   Processed {frame_count} frames | FPS: {current_fps:.1f} | Tracks: {len(tracks)}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n👋 Quit requested")
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        if producer:
            producer.flush()
        
        print(f"\n✅ Total frames processed: {frame_count}")
        if OUTPUT_PATH:
            print(f"📼 Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
