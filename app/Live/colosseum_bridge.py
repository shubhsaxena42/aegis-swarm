"""
Colosseum ↔ Aegis Swarm Bridge (5-Drone Swarm)
═══════════════════════════════════════════════

Connects Unreal Engine 5 + Colosseum simulation to the full Aegis pipeline:
  UE5 Drones → [Colosseum API] → This Bridge → Redpanda → Flink/Prometheus
                                             → Ray Serve → YOLO Inference
                                             → LangGraph → Mission Control

Supports N drones (default 5) running concurrently.

PREREQUISITES:
    1. Colosseum plugin installed in your UE5 project
    2. settings.json configured for multi-vehicle (see generate_settings())
    3. Redpanda running (docker-compose up -d redpanda)

INSTALL:
    pip install airsim confluent-kafka requests numpy opencv-python Pillow

USAGE:
    python colosseum_bridge.py
"""

import os
import sys
import cv2
import json
import time
import base64
import logging
import threading
import numpy as np
from io import BytesIO
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# Add project root for infrastructure imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    import airsim
    COLOSSEUM_AVAILABLE = True
except ImportError:
    COLOSSEUM_AVAILABLE = False
    print("⚠️  'airsim' package not installed. Install via: pip install airsim")

try:
    from infrastructure.redpanda.producer import AegisTelemetryProducer
    REDPANDA_AVAILABLE = True
except ImportError:
    REDPANDA_AVAILABLE = False
    print("⚠️  Redpanda producer not found. Telemetry will not be streamed.")

try:
    import requests
    RAY_SERVE_AVAILABLE = True
except ImportError:
    RAY_SERVE_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aegis.colosseum")

# ============================================================
# CONFIGURATION
# ============================================================

NUM_DRONES = 5
SWARM_ID = "swarm-alpha"

# Colosseum / AirSim
COLOSSEUM_IP = os.getenv("COLOSSEUM_IP", "127.0.0.1")

# Ray Serve inference endpoint
RAY_SERVE_URL = os.getenv("RAY_SERVE_URL", "http://localhost:8000/detect")

# Telemetry rate (Hz) and camera capture rate (Hz)
TELEMETRY_HZ = 10       # 10 telemetry packets per second per drone
CAMERA_HZ = 3           # 3 camera captures per second per drone (sent to YOLO)

# Drone vehicle names (must match settings.json)
DRONE_NAMES = [f"Drone{i+1}" for i in range(NUM_DRONES)]

# ============================================================
# SETTINGS.JSON GENERATOR
# ============================================================

def generate_settings(num_drones: int = 5, output_path: str = None) -> dict:
    """
    Generate Colosseum settings.json for a multi-drone swarm.

    Place the output at:
        Windows: %USERPROFILE%/Documents/AirSim/settings.json
        Linux:   ~/Documents/AirSim/settings.json

    Args:
        num_drones: Number of drones in the swarm.
        output_path: Optional path to write the file to.

    Returns:
        The settings dictionary.
    """
    vehicles = {}
    spacing = 3.0  # meters between drones at spawn

    for i in range(num_drones):
        name = f"Drone{i+1}"
        vehicles[name] = {
            "VehicleType": "SimpleFlight",
            "DefaultVehicleState": "Armed",
            "AutoCreate": True,
            "EnableCollisionPassthrough": False,
            "EnableCollisions": True,
            "X": float(i * spacing),
            "Y": 0.0,
            "Z": 0.0,
            "Yaw": 0.0,
            "Cameras": {
                "front_center": {
                    "CaptureSettings": [
                        {
                            "ImageType": 0,        # Scene
                            "Width": 1280,
                            "Height": 720,
                            "FOV_Degrees": 90,
                            "TargetGamma": 1.0,
                        }
                    ],
                    "X": 0.25,
                    "Y": 0.0,
                    "Z": -0.1,
                    "Pitch": -15.0,
                    "Roll": 0.0,
                    "Yaw": 0.0,
                }
            },
            "Sensors": {
                "Imu": {"SensorType": 2, "Enabled": True},
                "Gps": {"SensorType": 3, "Enabled": True},
                "Barometer": {"SensorType": 1, "Enabled": True},
            },
        }

    settings = {
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "ClockType": "SteppableClock",
        "ViewMode": "NoDisplay",
        "Vehicles": vehicles,
        "SubWindows": [
            {"WindowID": 0, "CameraName": "front_center", "ImageType": 0, "VehicleName": "Drone1"}
        ],
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(settings, f, indent=2)
        logger.info(f"Settings written to {output_path}")

    return settings


# ============================================================
# DRONE STATE TRACKER
# ============================================================

@dataclass
class DroneSnapshot:
    """Point-in-time state of a single drone."""
    drone_id: str
    timestamp: float = 0.0
    # Kinematics
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    velocity: Dict[str, float] = field(default_factory=lambda: {"vx": 0, "vy": 0, "vz": 0})
    orientation: Dict[str, float] = field(default_factory=lambda: {"pitch": 0, "roll": 0, "yaw": 0})
    # Health
    battery_percent: float = 100.0
    # GPS
    latitude: float = 37.7749      # SF default
    longitude: float = -122.4194
    altitude_m: float = 0.0
    # Status
    landed: bool = True
    collided: bool = False


# ============================================================
# COLOSSEUM CLIENT WRAPPER
# ============================================================

class ColosseumSwarmClient:
    """
    Manages connections to N drones in a Colosseum (AirSim-compatible) simulation.
    """

    def __init__(self, ip: str = COLOSSEUM_IP, drone_names: List[str] = None):
        self.ip = ip
        self.drone_names = drone_names or DRONE_NAMES
        self.client: Optional[airsim.MultirotorClient] = None
        self.snapshots: Dict[str, DroneSnapshot] = {}
        self._connected = False

    # ── Connection ────────────────────────────────────────────
    def connect(self) -> bool:
        """Connect to the Colosseum simulation."""
        if not COLOSSEUM_AVAILABLE:
            logger.error("airsim package not installed")
            return False

        try:
            self.client = airsim.MultirotorClient(ip=self.ip)
            self.client.confirmConnection()
            logger.info(f"Connected to Colosseum at {self.ip}")

            # Enable API control for all drones
            for name in self.drone_names:
                self.client.enableApiControl(True, vehicle_name=name)
                self.client.armDisarm(True, vehicle_name=name)
                self.snapshots[name] = DroneSnapshot(drone_id=name)
                logger.info(f"  ✅ {name} armed and API-controlled")

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    # ── Takeoff All ───────────────────────────────────────────
    def takeoff_all(self, altitude: float = 10.0):
        """Command all drones to take off to a specified altitude."""
        futures = []
        for name in self.drone_names:
            f = self.client.takeoffAsync(vehicle_name=name)
            futures.append((name, f))

        for name, f in futures:
            f.join()
            logger.info(f"  🛫 {name} airborne")

        # Move to altitude
        for name in self.drone_names:
            self.client.moveToZAsync(-altitude, 3, vehicle_name=name)

    # ── Telemetry ─────────────────────────────────────────────
    def get_telemetry(self, drone_name: str) -> DroneSnapshot:
        """Read current state from a single drone."""
        state = self.client.getMultirotorState(vehicle_name=drone_name)
        kin = state.kinematics_estimated

        pos = kin.position
        vel = kin.linear_velocity
        ori = airsim.to_eularian_angles(kin.orientation)

        gps = self.client.getGpsData(vehicle_name=drone_name)

        speed = float((vel.x_val**2 + vel.y_val**2 + vel.z_val**2) ** 0.5)
        # Simulated battery drain based on speed and time
        prev = self.snapshots.get(drone_name)
        dt = time.time() - (prev.timestamp if prev else time.time())
        drain_rate = 0.01 + speed * 0.005   # %/sec base + speed penalty
        battery = max(0.0, (prev.battery_percent if prev else 100.0) - drain_rate * dt)

        snap = DroneSnapshot(
            drone_id=drone_name,
            timestamp=time.time(),
            position={"x": pos.x_val, "y": pos.y_val, "z": -pos.z_val},  # NED → ENU
            velocity={"vx": vel.x_val, "vy": vel.y_val, "vz": vel.z_val},
            orientation={"pitch": ori[0], "roll": ori[1], "yaw": ori[2]},
            battery_percent=round(battery, 2),
            latitude=gps.gnss.geo_point.latitude,
            longitude=gps.gnss.geo_point.longitude,
            altitude_m=gps.gnss.geo_point.altitude,
            landed=state.landed_state == airsim.LandedState.Landed,
            collided=state.collision.has_collided,
        )

        self.snapshots[drone_name] = snap
        return snap

    # ── Camera Capture ────────────────────────────────────────
    def capture_frame(self, drone_name: str, camera: str = "front_center") -> Optional[np.ndarray]:
        """Capture a BGR frame from a drone's camera."""
        responses = self.client.simGetImages(
            [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)],
            vehicle_name=drone_name,
        )

        if not responses or responses[0].width == 0:
            return None

        img_raw = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img_raw.reshape(responses[0].height, responses[0].width, 3)
        return img  # BGR

    # ── Movement Commands ─────────────────────────────────────
    def move_drone(self, drone_name: str, vx: float, vy: float, vz: float, yaw_rate: float, duration: float = 0.5):
        """Send velocity command to a drone."""
        self.client.moveByVelocityAsync(
            vx, vy, vz, duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
            vehicle_name=drone_name,
        )

    def land_all(self):
        """Land all drones."""
        for name in self.drone_names:
            self.client.landAsync(vehicle_name=name)
        logger.info("All drones landing...")


# ============================================================
# RAY SERVE INFERENCE CLIENT
# ============================================================

def send_frame_to_ray_serve(frame: np.ndarray, drone_id: str, url: str = RAY_SERVE_URL) -> Optional[Dict]:
    """
    Send a frame to the Ray Serve YOLO endpoint for inference.

    Args:
        frame: BGR numpy array
        drone_id: Drone identifier for tracking

    Returns:
        Inference result dict or None
    """
    if not RAY_SERVE_AVAILABLE:
        return None

    try:
        # Encode frame as JPEG for efficient transfer
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        payload = {
            "image": img_b64,
            "drone_id": drone_id,
            "timestamp": time.time(),
        }

        resp = requests.post(url, json=payload, timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
        else:
            logger.warning(f"Ray Serve returned {resp.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        logger.warning(f"Inference request failed: {e}")
        return None


# ============================================================
# MAIN BRIDGE LOOP
# ============================================================

class AegisSwarmBridge:
    """
    Orchestrates the full data pipeline:
    Colosseum → Telemetry (Redpanda) + Frames (Ray Serve) → Results (Redpanda)
    """

    def __init__(self, num_drones: int = NUM_DRONES):
        self.num_drones = num_drones
        self.drone_names = [f"Drone{i+1}" for i in range(num_drones)]
        self.swarm_client: Optional[ColosseumSwarmClient] = None
        self.producer: Optional[AegisTelemetryProducer] = None
        self._running = False
        self._telemetry_interval = 1.0 / TELEMETRY_HZ
        self._camera_interval = 1.0 / CAMERA_HZ
        self._executor = ThreadPoolExecutor(max_workers=num_drones * 2)

    def start(self):
        """Initialize and start the bridge."""
        logger.info("=" * 60)
        logger.info("🚀 Aegis Swarm Bridge — Colosseum ↔ Pipeline")
        logger.info("=" * 60)

        # 1. Connect to Colosseum
        self.swarm_client = ColosseumSwarmClient(drone_names=self.drone_names)
        if not self.swarm_client.connect():
            logger.error("Cannot start without Colosseum connection.")
            return

        # 2. Connect to Redpanda
        if REDPANDA_AVAILABLE:
            try:
                self.producer = AegisTelemetryProducer()
                logger.info("✅ Redpanda producer connected")
            except Exception as e:
                logger.warning(f"Redpanda unavailable: {e}")

        # 3. Takeoff all drones
        logger.info(f"🛫 Taking off {self.num_drones} drones...")
        self.swarm_client.takeoff_all(altitude=15.0)
        time.sleep(3)  # Let them stabilize

        # 4. Start data loops
        self._running = True

        # Telemetry thread (all drones)
        telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        telemetry_thread.start()

        # Camera/inference threads (one per drone)
        camera_threads = []
        for name in self.drone_names:
            t = threading.Thread(target=self._camera_loop, args=(name,), daemon=True)
            t.start()
            camera_threads.append(t)

        logger.info(f"▶️  Bridge running — {self.num_drones} drones streaming")
        logger.info("   Press Ctrl+C to stop\n")

        # Main loop (keep alive + status)
        try:
            frame_count = 0
            while self._running:
                time.sleep(5)
                frame_count += 5
                self._print_status()
        except KeyboardInterrupt:
            logger.info("\n⏹️  Shutting down...")
        finally:
            self.stop()

    def stop(self):
        """Graceful shutdown."""
        self._running = False
        if self.swarm_client:
            self.swarm_client.land_all()
        if self.producer:
            self.producer.flush()
        self._executor.shutdown(wait=False)
        logger.info("✅ Bridge stopped")

    # ── Telemetry Loop ────────────────────────────────────────
    def _telemetry_loop(self):
        """Stream telemetry from all drones to Redpanda at TELEMETRY_HZ."""
        while self._running:
            loop_start = time.time()

            for name in self.drone_names:
                try:
                    snap = self.swarm_client.get_telemetry(name)

                    if self.producer:
                        speed = (snap.velocity["vx"]**2 + snap.velocity["vy"]**2 + snap.velocity["vz"]**2) ** 0.5
                        self.producer.send_telemetry(
                            drone_id=snap.drone_id,
                            swarm_id=SWARM_ID,
                            battery_percent=snap.battery_percent,
                            altitude_m=snap.altitude_m,
                            velocity_ms=speed,
                            latitude=snap.latitude,
                            longitude=snap.longitude,
                            connected=not snap.collided,
                        )

                    # Battery alerts
                    if snap.battery_percent < 20 and self.producer:
                        self.producer.send_battery_event(
                            drone_id=snap.drone_id,
                            event_type="critical_battery" if snap.battery_percent < 10 else "low_battery",
                            battery_percent=snap.battery_percent,
                            details=f"Battery at {snap.battery_percent}%",
                        )

                except Exception as e:
                    logger.error(f"Telemetry error for {name}: {e}")

            # Sleep to maintain rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, self._telemetry_interval - elapsed)
            time.sleep(sleep_time)

    # ── Camera / Inference Loop ───────────────────────────────
    def _camera_loop(self, drone_name: str):
        """Capture frames and send to Ray Serve for inference."""
        while self._running:
            loop_start = time.time()

            try:
                frame = self.swarm_client.capture_frame(drone_name)

                if frame is not None:
                    # Send to Ray Serve for YOLO inference
                    result = send_frame_to_ray_serve(frame, drone_name)

                    if result and self.producer:
                        self.producer.send_inference_result(
                            drone_id=drone_name,
                            model_name=result.get("model", "yolov26s"),
                            speed_mode=result.get("speed_mode", "FAST"),
                            latency_s=result.get("latency_s", 0),
                            fps=result.get("fps", 0),
                            detections=result.get("detections", []),
                        )

            except Exception as e:
                logger.error(f"Camera error for {drone_name}: {e}")

            elapsed = time.time() - loop_start
            sleep_time = max(0, self._camera_interval - elapsed)
            time.sleep(sleep_time)

    # ── Status Report ─────────────────────────────────────────
    def _print_status(self):
        """Print a compact swarm status line."""
        statuses = []
        for name in self.drone_names:
            snap = self.swarm_client.snapshots.get(name)
            if snap:
                statuses.append(f"{name}: bat={snap.battery_percent:.0f}% alt={snap.altitude_m:.1f}m")
        logger.info(" | ".join(statuses))

        if self.producer:
            stats = self.producer.stats
            logger.info(f"  Redpanda: sent={stats['messages_sent']} errs={stats['delivery_errors']}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aegis Swarm — Colosseum Bridge")
    parser.add_argument("--drones", type=int, default=5, help="Number of drones")
    parser.add_argument("--generate-settings", action="store_true",
                        help="Generate settings.json and exit")
    parser.add_argument("--settings-output", type=str,
                        default=os.path.expanduser("~/Documents/AirSim/settings.json"),
                        help="Path for generated settings.json")
    args = parser.parse_args()

    if args.generate_settings:
        settings = generate_settings(args.drones, args.settings_output)
        print(json.dumps(settings, indent=2))
        print(f"\n✅ Settings written to {args.settings_output}")
        print("   Restart Colosseum/UE5 to apply.")
    else:
        bridge = AegisSwarmBridge(num_drones=args.drones)
        bridge.start()
