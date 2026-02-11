"""
Autonomous Drone Car-Following System
PID Controller + Bounding-Box Tracking Consumer

Delegates ALL inference (YOLO + SAHI + BoT-SORT) to realtime_drone_inference.py
and focuses solely on the control loop that converts bounding-box tracks
into smooth drone velocity commands via PID.

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────┐
│  realtime_drone_inference.py  (YOLO26s + SAHI-lite + BoT-SORT)     │
│         ↓  tracks (N, 8)                                           │
├─────────────────────────────────────────────────────────────────────┤
│  THIS FILE — drone_follow_car.py                                    │
│  VisualFollowController → PID → DroneCommand → DroneInterface       │
├─────────────────────────────────────────────────────────────────────┤
│  langgraph_mission_controller.py  (Coordinator/Tactical/Analyst)    │
│         ↕  SharedState                                              │
└─────────────────────────────────────────────────────────────────────┘

REQUIREMENTS:
    pip install ultralytics boxmot opencv-python numpy
"""

import cv2
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class DroneConfig:
    """Drone following configuration."""
    
    # Target selection
    target_class: str = "car"  # VisDrone class to follow
    target_track_id: Optional[int] = None  # Specific track ID (None = auto-select largest)
    
    # Following behavior
    follow_distance_m: float = 10.0  # Desired distance in meters
    max_speed_ms: float = 15.0  # Max drone speed m/s
    
    # Frame zones (normalized 0-1)
    center_zone: float = 0.15  # Within 15% of center = no movement
    slow_zone: float = 0.35  # 15-35% = slow movement
    # Beyond 35% = fast movement
    
    # PID gains (tune for your drone)
    pid_yaw: Tuple[float, float, float] = (0.5, 0.01, 0.1)  # Kp, Ki, Kd
    pid_pitch: Tuple[float, float, float] = (0.4, 0.01, 0.08)
    pid_altitude: Tuple[float, float, float] = (0.3, 0.005, 0.05)
    
    # Safety
    min_altitude_m: float = 5.0
    max_altitude_m: float = 50.0
    connection_timeout_s: float = 5.0

# ============================================================
# DRONE STATE
# ============================================================

class FollowState(Enum):
    """Drone following states."""
    IDLE = "idle"
    SEARCHING = "searching"  # Looking for target
    ACQUIRING = "acquiring"  # Found target, confirming ID
    FOLLOWING = "following"  # Actively following
    LOST = "lost"  # Target lost, recovering
    RETURNING = "returning"  # Returning to home
    EMERGENCY = "emergency"  # Emergency stop

@dataclass
class TargetInfo:
    """Information about the tracked target."""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # cx, cy
    area: int  # bounding box area (proxy for distance)
    confidence: float
    frames_tracked: int = 0
    last_seen: float = 0.0

@dataclass 
class DroneCommand:
    """Command to send to drone."""
    yaw_rate: float = 0.0  # -1 to 1 (left/right rotation)
    pitch: float = 0.0  # -1 to 1 (forward/backward)
    roll: float = 0.0  # -1 to 1 (left/right strafe)
    throttle: float = 0.0  # -1 to 1 (up/down)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "yaw_rate": self.yaw_rate,
            "pitch": self.pitch,
            "roll": self.roll,
            "throttle": self.throttle
        }

# ============================================================
# PID CONTROLLER
# ============================================================

class PIDController:
    """PID controller for smooth drone control."""
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_min: float = -1.0, output_max: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
    
    def compute(self, error: float) -> float:
        """Compute PID output."""
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt <= 0:
            dt = 0.01
        
        # Proportional
        p_term = self.kp * error
        
        # Integral (with anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)
        i_term = self.ki * self.integral
        
        # Derivative
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Update state
        self.prev_error = error
        self.prev_time = current_time
        
        # Compute output
        output = p_term + i_term + d_term
        return np.clip(output, self.output_min, self.output_max)
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

# ============================================================
# VISUAL FOLLOWING CONTROLLER
# ============================================================

class VisualFollowController:
    """
    Core visual following logic.
    Takes tracked detections and outputs drone commands.
    """
    
    def __init__(self, config: DroneConfig, frame_size: Tuple[int, int]):
        self.config = config
        self.frame_w, self.frame_h = frame_size
        self.frame_center = (frame_size[0] // 2, frame_size[1] // 2)
        
        # PID controllers
        self.pid_yaw = PIDController(*config.pid_yaw)
        self.pid_pitch = PIDController(*config.pid_pitch)
        self.pid_alt = PIDController(*config.pid_altitude)
        
        # State
        self.state = FollowState.SEARCHING
        self.target: Optional[TargetInfo] = None
        self.target_history: deque = deque(maxlen=30)  # Last 30 frames
        
        # Reference area (proxy for distance)
        self.reference_area: Optional[float] = None
    
    def update(self, tracks: np.ndarray, class_names: Dict[int, str]) -> Tuple[DroneCommand, FollowState]:
        """
        Process tracks and generate drone command.
        
        Args:
            tracks: BoT-SORT output (N, 8): [x1, y1, x2, y2, track_id, conf, cls, idx]
            class_names: Mapping of class ID to name
        
        Returns:
            DroneCommand and current FollowState
        """
        # Find target class objects
        target_tracks = self._filter_target_class(tracks, class_names)
        
        if len(target_tracks) == 0:
            return self._handle_target_lost()
        
        # Select/update target
        target_info = self._select_target(target_tracks, class_names)
        
        if target_info is None:
            return self._handle_target_lost()
        
        self.target = target_info
        self.target_history.append(target_info)
        self.state = FollowState.FOLLOWING
        
        # Calculate errors (normalized -1 to 1)
        error_x = (target_info.center[0] - self.frame_center[0]) / (self.frame_w / 2)
        error_y = (target_info.center[1] - self.frame_center[1]) / (self.frame_h / 2)
        
        # Distance error (using bbox area as proxy)
        if self.reference_area is None:
            self.reference_area = target_info.area
        error_distance = (self.reference_area - target_info.area) / self.reference_area
        
        # Apply dead zone
        error_x = self._apply_dead_zone(error_x)
        error_y = self._apply_dead_zone(error_y)
        
        # Compute control outputs
        yaw_rate = self.pid_yaw.compute(error_x)  # Left/right to center target
        pitch = self.pid_pitch.compute(-error_distance)  # Forward/backward for distance
        throttle = self.pid_alt.compute(-error_y)  # Up/down to center vertically
        
        command = DroneCommand(
            yaw_rate=yaw_rate,
            pitch=pitch,
            roll=0.0,  # No strafing in simple follow
            throttle=throttle
        )
        
        return command, self.state
    
    def _filter_target_class(self, tracks: np.ndarray, 
                             class_names: Dict[int, str]) -> np.ndarray:
        """Filter tracks to only target class."""
        if len(tracks) == 0:
            return tracks
        
        mask = []
        for track in tracks:
            cls_id = int(track[6])
            cls_name = class_names.get(cls_id, "unknown")
            mask.append(cls_name == self.config.target_class)
        
        return tracks[np.array(mask)]
    
    def _select_target(self, tracks: np.ndarray, 
                       class_names: Dict[int, str]) -> Optional[TargetInfo]:
        """Select which track to follow."""
        
        # If we have a locked target, try to find it
        if self.target and self.config.target_track_id:
            for track in tracks:
                if int(track[4]) == self.config.target_track_id:
                    return self._track_to_info(track, class_names)
        
        # Otherwise, select largest (closest) target
        if len(tracks) == 0:
            return None
        
        # Calculate areas
        areas = (tracks[:, 2] - tracks[:, 0]) * (tracks[:, 3] - tracks[:, 1])
        largest_idx = np.argmax(areas)
        
        return self._track_to_info(tracks[largest_idx], class_names)
    
    def _track_to_info(self, track: np.ndarray, 
                       class_names: Dict[int, str]) -> TargetInfo:
        """Convert track array to TargetInfo."""
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        conf = float(track[5])
        cls_id = int(track[6])
        
        return TargetInfo(
            track_id=track_id,
            class_name=class_names.get(cls_id, "unknown"),
            bbox=(x1, y1, x2, y2),
            center=((x1 + x2) // 2, (y1 + y2) // 2),
            area=(x2 - x1) * (y2 - y1),
            confidence=conf,
            last_seen=time.time()
        )
    
    def _apply_dead_zone(self, error: float) -> float:
        """Apply dead zone to reduce jitter."""
        if abs(error) < self.config.center_zone:
            return 0.0
        elif abs(error) < self.config.slow_zone:
            # Scale down in slow zone
            return error * 0.5
        return error
    
    def _handle_target_lost(self) -> Tuple[DroneCommand, FollowState]:
        """Handle when target is lost."""
        if self.target and (time.time() - self.target.last_seen) < 2.0:
            # Recently lost - hover in place
            self.state = FollowState.LOST
            return DroneCommand(), self.state
        
        # Lost for too long - search
        self.state = FollowState.SEARCHING
        self.target = None
        self.reference_area = None
        
        # Reset PIDs
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        self.pid_alt.reset()
        
        # Slow rotation to search
        return DroneCommand(yaw_rate=0.2), self.state
    
    def lock_target(self, track_id: int):
        """Lock onto a specific track ID."""
        self.config.target_track_id = track_id
        print(f"🎯 Locked onto track ID: {track_id}")
    
    def release_target(self):
        """Release target lock, revert to auto-selection."""
        self.config.target_track_id = None
        self.target = None
        self.reference_area = None
        print("🔓 Target lock released")

# ============================================================
# DRONE SDK INTERFACES (Abstract - implement for your drone)
# ============================================================

class DroneInterface:
    """Abstract interface for drone control. Implement for your specific drone."""
    
    def connect(self) -> bool:
        """Connect to drone."""
        raise NotImplementedError
    
    def disconnect(self):
        """Disconnect from drone."""
        raise NotImplementedError
    
    def takeoff(self) -> bool:
        """Takeoff."""
        raise NotImplementedError
    
    def land(self) -> bool:
        """Land."""
        raise NotImplementedError
    
    def send_command(self, cmd: DroneCommand):
        """Send velocity command to drone."""
        raise NotImplementedError
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get current drone telemetry."""
        raise NotImplementedError
    
    def emergency_stop(self):
        """Emergency stop."""
        raise NotImplementedError


class SimulatedDrone(DroneInterface):
    """Simulated drone for testing."""
    
    def __init__(self):
        self.connected = False
        self.flying = False
        self.position = {"x": 0, "y": 0, "z": 0}
        self.velocity = {"vx": 0, "vy": 0, "vz": 0}
    
    def connect(self) -> bool:
        print("🔌 [SIM] Connecting to simulated drone...")
        time.sleep(0.5)
        self.connected = True
        print("✅ [SIM] Connected!")
        return True
    
    def disconnect(self):
        print("🔌 [SIM] Disconnecting...")
        self.connected = False
    
    def takeoff(self) -> bool:
        print("🛫 [SIM] Taking off...")
        time.sleep(1)
        self.flying = True
        self.position["z"] = 10
        print("✅ [SIM] Airborne at 10m")
        return True
    
    def land(self) -> bool:
        print("🛬 [SIM] Landing...")
        time.sleep(1)
        self.flying = False
        self.position["z"] = 0
        print("✅ [SIM] Landed")
        return True
    
    def send_command(self, cmd: DroneCommand):
        # Simulate movement
        self.velocity["vx"] = cmd.pitch * 5
        self.velocity["vy"] = cmd.roll * 5
        self.velocity["vz"] = cmd.throttle * 2
        # Print for debugging
        # print(f"[SIM] Cmd: yaw={cmd.yaw_rate:.2f} pitch={cmd.pitch:.2f} throttle={cmd.throttle:.2f}")
    
    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "connected": self.connected,
            "flying": self.flying,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "battery": 85
        }
    
    def emergency_stop(self):
        print("🚨 [SIM] EMERGENCY STOP!")
        self.velocity = {"vx": 0, "vy": 0, "vz": 0}


class DJITelloDrone(DroneInterface):
    """DJI Tello drone interface (requires djitellopy)."""
    
    def __init__(self):
        try:
            from djitellopy import Tello
            self.tello = Tello()
        except ImportError:
            raise ImportError("Install djitellopy: pip install djitellopy")
    
    def connect(self) -> bool:
        self.tello.connect()
        print(f"✅ Tello connected. Battery: {self.tello.get_battery()}%")
        return True
    
    def disconnect(self):
        self.tello.end()
    
    def takeoff(self) -> bool:
        self.tello.takeoff()
        return True
    
    def land(self) -> bool:
        self.tello.land()
        return True
    
    def send_command(self, cmd: DroneCommand):
        # Tello uses -100 to 100 range
        self.tello.send_rc_control(
            left_right_velocity=int(cmd.roll * 100),
            forward_backward_velocity=int(cmd.pitch * 100),
            up_down_velocity=int(cmd.throttle * 100),
            yaw_velocity=int(cmd.yaw_rate * 100)
        )
    
    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "battery": self.tello.get_battery(),
            "height": self.tello.get_height(),
            "temperature": self.tello.get_temperature()
        }
    
    def emergency_stop(self):
        self.tello.emergency()


class MAVLinkDrone(DroneInterface):
    """PX4/ArduPilot drone via MAVLink (requires pymavlink or mavsdk)."""
    
    def __init__(self, connection_string: str = "udp:127.0.0.1:14540"):
        try:
            from pymavlink import mavutil
            self.connection_string = connection_string
            self.mav = None
        except ImportError:
            raise ImportError("Install pymavlink: pip install pymavlink")
    
    def connect(self) -> bool:
        from pymavlink import mavutil
        self.mav = mavutil.mavlink_connection(self.connection_string)
        self.mav.wait_heartbeat()
        print(f"✅ MAVLink connected to {self.connection_string}")
        return True
    
    def disconnect(self):
        if self.mav:
            self.mav.close()
    
    def takeoff(self) -> bool:
        # Arm and takeoff via MAVLink
        self.mav.arducopter_arm()
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            22,  # MAV_CMD_NAV_TAKEOFF
            0, 0, 0, 0, 0, 0, 0, 10  # Takeoff to 10m
        )
        return True
    
    def land(self) -> bool:
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            21,  # MAV_CMD_NAV_LAND
            0, 0, 0, 0, 0, 0, 0, 0
        )
        return True
    
    def send_command(self, cmd: DroneCommand):
        # Send velocity command via MANUAL_CONTROL
        self.mav.mav.manual_control_send(
            self.mav.target_system,
            int(cmd.pitch * 1000),  # x: pitch
            int(cmd.roll * 1000),   # y: roll
            int((cmd.throttle + 1) * 500),  # z: throttle (0-1000)
            int(cmd.yaw_rate * 1000),  # r: yaw
            0  # buttons
        )
    
    def get_telemetry(self) -> Dict[str, Any]:
        msg = self.mav.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg:
            return {
                "lat": msg.lat / 1e7,
                "lon": msg.lon / 1e7,
                "alt": msg.alt / 1000,
                "relative_alt": msg.relative_alt / 1000
            }
        return {}
    
    def emergency_stop(self):
        self.mav.arducopter_disarm()

# ============================================================
# VISUALIZATION
# ============================================================

def draw_follow_overlay(frame: np.ndarray, 
                        controller: VisualFollowController,
                        command: DroneCommand,
                        tracks: np.ndarray) -> np.ndarray:
    """Draw following visualization on frame."""
    
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Draw center crosshair
    cv2.line(overlay, (w//2 - 30, h//2), (w//2 + 30, h//2), (0, 255, 255), 2)
    cv2.line(overlay, (w//2, h//2 - 30), (w//2, h//2 + 30), (0, 255, 255), 2)
    
    # Draw dead zone
    dz = int(controller.config.center_zone * w / 2)
    cv2.rectangle(overlay, 
                  (w//2 - dz, h//2 - dz), 
                  (w//2 + dz, h//2 + dz), 
                  (0, 255, 0), 1)
    
    # Draw all tracks
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, track in enumerate(tracks):
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        
        is_target = (controller.target and 
                     controller.target.track_id == track_id)
        
        color = (0, 255, 255) if is_target else colors[track_id % len(colors)]
        thickness = 3 if is_target else 1
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(overlay, f"#{track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if is_target:
            # Draw line from target to center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.arrowedLine(overlay, (cx, cy), (w//2, h//2), 
                           (0, 255, 255), 2, tipLength=0.05)
    
    # Draw state and command info
    state_colors = {
        FollowState.FOLLOWING: (0, 255, 0),
        FollowState.SEARCHING: (0, 255, 255),
        FollowState.LOST: (0, 165, 255),
        FollowState.EMERGENCY: (0, 0, 255)
    }
    state_color = state_colors.get(controller.state, (255, 255, 255))
    
    # Info panel
    cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (250, 120), state_color, 2)
    
    cv2.putText(overlay, f"State: {controller.state.value}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
    
    if controller.target:
        cv2.putText(overlay, f"Target: #{controller.target.track_id}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(overlay, f"Yaw: {command.yaw_rate:+.2f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"Pitch: {command.pitch:+.2f}", (120, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"Throttle: {command.throttle:+.2f}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return overlay

# ============================================================
# MAIN FOLLOWING LOOP
# (Inference delegated to realtime_drone_inference.py)
# ============================================================

def run_follow_loop(
    video_source: str,
    model_path: str,
    drone: DroneInterface,
    config: DroneConfig = DroneConfig(),
    speed_mode: str = "FAST",
    visualize: bool = True
):
    """
    Main car-following loop.

    Inference (YOLO + SAHI + BoT-SORT) is handled by functions imported
    from realtime_drone_inference.py — this file only does PID control.

    Args:
        video_source: Video source (RTSP URL, webcam ID, or file path)
        model_path:   Path to YOLO model weights
        drone:        DroneInterface implementation
        config:       Following configuration
        speed_mode:   One of ULTRA_FAST, FAST, BALANCED, ACCURATE
        visualize:    Whether to show visualization
    """
    import torch
    from ultralytics import YOLO
    from boxmot import BotSort

    # Import inference from the single source of truth
    from realtime_drone_inference import (
        sahi_lite_inference,
        native_inference,
        SPEED_CONFIGS,
        CONFIDENCE_THRESHOLD,
    )

    print("=" * 60)
    print("🚁 Autonomous Drone Car-Following System")
    print(f"   Inference engine: realtime_drone_inference.py ({speed_mode})")
    print("=" * 60)

    # Resolve speed config
    cfg = SPEED_CONFIGS.get(speed_mode, SPEED_CONFIGS["FAST"])
    use_sahi = cfg["use_sahi"]
    imgsz    = cfg["imgsz"]
    use_half  = cfg.get("half", False)

    # Load YOLO model (shared with inference module's logic)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"📍 Device: {device}")
    print(f"🔧 Loading model: {model_path}")
    model = YOLO(model_path)
    model.to(device)
    if use_half and device != "cpu":
        model.model.half()
    class_names = model.names

    # Initialize BoT-SORT tracker (stateful, must be per-session)
    tracker = BotSort(
        reid_weights=None,
        device=device,
        half=(device != "cpu"),
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=10,
    )
    print("✅ BoT-SORT tracker ready")

    # Open video
    print(f"📹 Opening: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        # Fallback to local video file
        fallback = "drone_feed.mp4"
        print(f"⚠️  Primary source failed, trying {fallback}...")
        cap = cv2.VideoCapture(fallback)
        if not cap.isOpened():
            print("❌ No video source available!")
            return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Resolution: {w}x{h}")

    # Initialize follow controller
    controller = VisualFollowController(config, (w, h))

    # Connect drone
    if not drone.connect():
        print("❌ Failed to connect to drone!")
        return

    print("\n" + "=" * 60)
    print("Controls:")
    print("  T - Takeoff      L - Land")
    print("  1-9 - Lock to track ID")
    print("  R - Release lock  E - Emergency stop")
    print("  Q - Quit")
    print("=" * 60 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── Inference (delegated to realtime_drone_inference) ──
            if use_sahi:
                slices = cfg.get("sahi_slices", (2, 2))
                dets = sahi_lite_inference(
                    model, frame,
                    slices=slices,
                    overlap=0.1,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=imgsz,
                )
            else:
                dets = native_inference(
                    model, frame,
                    imgsz=imgsz,
                    conf=CONFIDENCE_THRESHOLD,
                )

            # ── Tracking ──────────────────────────────────────────
            tracks = tracker.update(dets, frame)

            # ── PID Follow Control ────────────────────────────────
            command, state = controller.update(tracks, class_names)

            # ── Send to drone (only if airborne) ──────────────────
            telemetry = drone.get_telemetry()
            if telemetry.get("flying", False):
                drone.send_command(command)

            # ── Visualization ─────────────────────────────────────
            if visualize:
                vis_frame = draw_follow_overlay(frame, controller, command, tracks)
                cv2.imshow("Drone Follow", vis_frame)

            # ── Keyboard ──────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                drone.takeoff()
            elif key == ord('l'):
                drone.land()
            elif key == ord('e'):
                drone.emergency_stop()
                controller.state = FollowState.EMERGENCY
            elif key == ord('r'):
                controller.release_target()
            elif ord('1') <= key <= ord('9'):
                controller.lock_target(key - ord('1') + 1)

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")

    finally:
        print("🛬 Landing and disconnecting...")
        drone.land()
        drone.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Shutdown complete")


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    run_follow_loop(
        video_source="drone_feed.mp4",   # or 0 for webcam, or RTSP URL
        model_path="yolov26s.pt",
        drone=SimulatedDrone(),
        config=DroneConfig(target_class="car"),
        speed_mode="FAST",               # uses realtime_drone_inference speed configs
        visualize=True,
    )
