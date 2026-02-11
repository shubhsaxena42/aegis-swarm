"""
Aegis Swarm — Drone Fleet Prometheus Metrics Exporter
═══════════════════════════════════════════════════════

Custom Prometheus exporter that:
  1. Consumes drone telemetry from Redpanda (Kafka)
  2. Maintains per-drone gauge metrics
  3. Exposes /metrics endpoint for Prometheus scraping

Metrics Exported:
  - aegis_drone_battery_percent          — Battery level per drone
  - aegis_drone_altitude_meters          — Current altitude
  - aegis_drone_velocity_ms              — Ground speed
  - aegis_drone_connected                — Connection status (1/0)
  - aegis_drone_motor_rpm                — Per-motor RPM
  - aegis_drone_latitude / longitude     — GPS position
  - aegis_drone_temperature_celsius      — Onboard temperature
  - aegis_drone_signal_strength_dbm      — Radio signal strength
  - aegis_inference_latency_seconds      — YOLO inference time
  - aegis_inference_fps                  — Inference throughput
  - aegis_inference_detections_total     — Cumulative detections
  - aegis_swarm_active_drones            — Active drones in fleet
  - aegis_swarm_health_score             — Overall swarm health [0-1]
  - aegis_telemetry_messages_total       — Messages consumed
"""

import os
import json
import time
import logging
import threading
from typing import Dict, Any

from prometheus_client import (
    start_http_server,
    Gauge,
    Counter,
    Histogram,
    Info,
    REGISTRY,
    CollectorRegistry,
)
from confluent_kafka import Consumer, KafkaError

# ─── Configuration ────────────────────────────────────────────
REDPANDA_BROKER = os.getenv("REDPANDA_BROKER", "localhost:19092")
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "9100"))
TELEMETRY_TOPIC = os.getenv("TELEMETRY_TOPIC", "drone.telemetry")
INFERENCE_TOPIC = os.getenv("INFERENCE_TOPIC", "drone.inference.results")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "prometheus-exporter")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("aegis.exporter")


# ═══════════════════════════════════════════════════════════════
#                    METRIC DEFINITIONS
# ═══════════════════════════════════════════════════════════════

# ── Per-Drone Gauges ──────────────────────────────────────────
DRONE_BATTERY = Gauge(
    "aegis_drone_battery_percent",
    "Drone battery level percentage",
    ["drone_id", "swarm_id"],
)
DRONE_ALTITUDE = Gauge(
    "aegis_drone_altitude_meters",
    "Drone altitude in meters above ground level",
    ["drone_id", "swarm_id"],
)
DRONE_VELOCITY = Gauge(
    "aegis_drone_velocity_ms",
    "Drone ground speed in m/s",
    ["drone_id", "swarm_id"],
)
DRONE_CONNECTED = Gauge(
    "aegis_drone_connected",
    "Drone connection status (1=connected, 0=disconnected)",
    ["drone_id", "swarm_id"],
)
DRONE_MOTOR_RPM = Gauge(
    "aegis_drone_motor_rpm",
    "Individual motor RPM",
    ["drone_id", "swarm_id", "motor_id"],
)
DRONE_LATITUDE = Gauge(
    "aegis_drone_latitude",
    "Drone GPS latitude",
    ["drone_id", "swarm_id"],
)
DRONE_LONGITUDE = Gauge(
    "aegis_drone_longitude",
    "Drone GPS longitude",
    ["drone_id", "swarm_id"],
)
DRONE_TEMPERATURE = Gauge(
    "aegis_drone_temperature_celsius",
    "Drone onboard temperature",
    ["drone_id", "swarm_id"],
)
DRONE_SIGNAL = Gauge(
    "aegis_drone_signal_strength_dbm",
    "Radio signal strength in dBm",
    ["drone_id", "swarm_id"],
)
DRONE_VOLTAGE = Gauge(
    "aegis_drone_voltage",
    "Battery voltage",
    ["drone_id", "swarm_id"],
)

# ── Inference Metrics ─────────────────────────────────────────
INFERENCE_LATENCY = Histogram(
    "aegis_inference_latency_seconds",
    "YOLO inference latency distribution",
    ["drone_id", "model_name", "speed_mode"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
)
INFERENCE_FPS = Gauge(
    "aegis_inference_fps",
    "Current inference FPS",
    ["drone_id", "model_name"],
)
INFERENCE_DETECTIONS = Counter(
    "aegis_inference_detections_total",
    "Total objects detected",
    ["drone_id", "class_name"],
)

# ── Swarm-Level Metrics ──────────────────────────────────────
SWARM_ACTIVE = Gauge(
    "aegis_swarm_active_drones",
    "Number of currently active drones",
    ["swarm_id"],
)
SWARM_HEALTH = Gauge(
    "aegis_swarm_health_score",
    "Overall swarm health score (0.0 to 1.0)",
    ["swarm_id"],
)

# ── Exporter Internal Metrics ────────────────────────────────
TELEMETRY_MESSAGES = Counter(
    "aegis_telemetry_messages_total",
    "Total telemetry messages consumed from Redpanda",
    ["topic"],
)
TELEMETRY_ERRORS = Counter(
    "aegis_telemetry_errors_total",
    "Total telemetry processing errors",
    ["error_type"],
)


# ═══════════════════════════════════════════════════════════════
#                  TELEMETRY CONSUMER
# ═══════════════════════════════════════════════════════════════

class DroneMetricsExporter:
    """Consumes drone telemetry from Redpanda and updates Prometheus metrics."""

    def __init__(self):
        self._drone_states: Dict[str, Dict[str, Any]] = {}
        self._last_seen: Dict[str, float] = {}
        self._running = False
        self._lock = threading.Lock()

        self._consumer_config = {
            "bootstrap.servers": REDPANDA_BROKER,
            "group.id": CONSUMER_GROUP,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
            "session.timeout.ms": 30000,
            "max.poll.interval.ms": 300000,
        }

    def start(self):
        """Start the exporter — HTTP server + Kafka consumer threads."""
        logger.info(f"Starting Prometheus exporter on :{EXPORTER_PORT}")
        start_http_server(EXPORTER_PORT)

        self._running = True

        # Telemetry consumer thread
        telemetry_thread = threading.Thread(
            target=self._consume_loop,
            args=(TELEMETRY_TOPIC, self._process_telemetry),
            daemon=True,
            name="telemetry-consumer",
        )
        telemetry_thread.start()

        # Inference consumer thread
        inference_thread = threading.Thread(
            target=self._consume_loop,
            args=(INFERENCE_TOPIC, self._process_inference),
            daemon=True,
            name="inference-consumer",
        )
        inference_thread.start()

        # Swarm health calculator thread
        health_thread = threading.Thread(
            target=self._swarm_health_loop,
            daemon=True,
            name="health-calculator",
        )
        health_thread.start()

        logger.info("All consumer threads started. Exporter ready.")

        # Keep main thread alive
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down exporter...")
            self._running = False

    def _consume_loop(self, topic: str, handler):
        """Generic Kafka consumer loop."""
        consumer = Consumer(self._consumer_config)
        consumer.subscribe([topic])
        logger.info(f"Subscribed to topic: {topic}")

        while self._running:
            try:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    TELEMETRY_ERRORS.labels(error_type="kafka_error").inc()
                    logger.error(f"Kafka error: {msg.error()}")
                    continue

                value = json.loads(msg.value().decode("utf-8"))
                handler(value)
                TELEMETRY_MESSAGES.labels(topic=topic).inc()

            except json.JSONDecodeError:
                TELEMETRY_ERRORS.labels(error_type="json_decode").inc()
            except Exception as e:
                TELEMETRY_ERRORS.labels(error_type="processing").inc()
                logger.exception(f"Error processing message from {topic}: {e}")
                time.sleep(1)

        consumer.close()

    def _process_telemetry(self, data: Dict[str, Any]):
        """
        Process a drone telemetry message.

        Expected schema:
        {
            "drone_id": "drone-001",
            "swarm_id": "swarm-alpha",
            "timestamp": 1707660000.0,
            "battery_percent": 87.5,
            "battery_voltage": 22.1,
            "altitude_m": 45.2,
            "velocity_ms": 12.3,
            "position": {"lat": 37.7749, "lon": -122.4194},
            "motor_rpm": [8500, 8520, 8490, 8510],
            "temperature_c": 42.0,
            "signal_dbm": -65,
            "connected": true
        }
        """
        drone_id = data.get("drone_id", "unknown")
        swarm_id = data.get("swarm_id", "default")
        labels = {"drone_id": drone_id, "swarm_id": swarm_id}

        with self._lock:
            self._drone_states[drone_id] = data
            self._last_seen[drone_id] = time.time()

        # Update gauges
        if "battery_percent" in data:
            DRONE_BATTERY.labels(**labels).set(data["battery_percent"])

        if "battery_voltage" in data:
            DRONE_VOLTAGE.labels(**labels).set(data["battery_voltage"])

        if "altitude_m" in data:
            DRONE_ALTITUDE.labels(**labels).set(data["altitude_m"])

        if "velocity_ms" in data:
            DRONE_VELOCITY.labels(**labels).set(data["velocity_ms"])

        if "connected" in data:
            DRONE_CONNECTED.labels(**labels).set(1 if data["connected"] else 0)

        if "temperature_c" in data:
            DRONE_TEMPERATURE.labels(**labels).set(data["temperature_c"])

        if "signal_dbm" in data:
            DRONE_SIGNAL.labels(**labels).set(data["signal_dbm"])

        if "position" in data:
            pos = data["position"]
            DRONE_LATITUDE.labels(**labels).set(pos.get("lat", 0))
            DRONE_LONGITUDE.labels(**labels).set(pos.get("lon", 0))

        if "motor_rpm" in data:
            for i, rpm in enumerate(data["motor_rpm"]):
                DRONE_MOTOR_RPM.labels(
                    drone_id=drone_id, swarm_id=swarm_id, motor_id=f"motor_{i}"
                ).set(rpm)

    def _process_inference(self, data: Dict[str, Any]):
        """
        Process an inference result message.

        Expected schema:
        {
            "drone_id": "drone-001",
            "model_name": "yolov26s",
            "speed_mode": "FAST",
            "latency_s": 0.045,
            "fps": 22.1,
            "detections": [
                {"class": "car", "confidence": 0.92},
                {"class": "person", "confidence": 0.85}
            ]
        }
        """
        drone_id = data.get("drone_id", "unknown")
        model = data.get("model_name", "unknown")
        mode = data.get("speed_mode", "unknown")

        if "latency_s" in data:
            INFERENCE_LATENCY.labels(
                drone_id=drone_id, model_name=model, speed_mode=mode
            ).observe(data["latency_s"])

        if "fps" in data:
            INFERENCE_FPS.labels(drone_id=drone_id, model_name=model).set(data["fps"])

        for det in data.get("detections", []):
            INFERENCE_DETECTIONS.labels(
                drone_id=drone_id, class_name=det.get("class", "unknown")
            ).inc()

    def _swarm_health_loop(self):
        """Periodically calculate swarm-level health metrics."""
        while self._running:
            time.sleep(5)

            with self._lock:
                now = time.time()
                active_drones = {}

                for drone_id, last_seen in self._last_seen.items():
                    if now - last_seen < 30:  # Active if seen in last 30s
                        state = self._drone_states.get(drone_id, {})
                        swarm_id = state.get("swarm_id", "default")

                        if swarm_id not in active_drones:
                            active_drones[swarm_id] = []
                        active_drones[swarm_id].append(state)
                    else:
                        # Mark as disconnected
                        state = self._drone_states.get(drone_id, {})
                        swarm_id = state.get("swarm_id", "default")
                        DRONE_CONNECTED.labels(
                            drone_id=drone_id, swarm_id=swarm_id
                        ).set(0)

                # Update swarm metrics
                for swarm_id, drones in active_drones.items():
                    SWARM_ACTIVE.labels(swarm_id=swarm_id).set(len(drones))

                    # Health score: weighted average of battery, signal, temperature
                    if drones:
                        health_scores = []
                        for d in drones:
                            battery_health = min(d.get("battery_percent", 0) / 100, 1.0)
                            signal_health = min(
                                (d.get("signal_dbm", -100) + 100) / 60, 1.0
                            )
                            temp_health = 1.0 - max(
                                (d.get("temperature_c", 25) - 60) / 40, 0
                            )
                            score = (
                                0.4 * battery_health
                                + 0.3 * signal_health
                                + 0.3 * max(temp_health, 0)
                            )
                            health_scores.append(score)

                        avg_health = sum(health_scores) / len(health_scores)
                        SWARM_HEALTH.labels(swarm_id=swarm_id).set(round(avg_health, 3))


# ═══════════════════════════════════════════════════════════════
#                         MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    exporter = DroneMetricsExporter()
    exporter.start()
