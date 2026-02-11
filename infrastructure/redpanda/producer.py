"""
Aegis Swarm — Redpanda Telemetry Producer
══════════════════════════════════════════

Publishes drone telemetry to Redpanda topics.
Use this module in your drone simulation or real drone bridge to feed
data into the streaming pipeline (Flink, Prometheus exporter, etc.)

Usage:
    from infrastructure.redpanda.producer import AegisTelemetryProducer

    producer = AegisTelemetryProducer()
    producer.send_telemetry(drone_entity, swarm_id="swarm-alpha")
    producer.send_inference_result(drone_id, model_name, latency, detections)
"""

import os
import json
import time
import logging
import atexit
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

# ─── Configuration ────────────────────────────────────────────
REDPANDA_BROKER = os.getenv("REDPANDA_BROKER", "localhost:19092")

TOPICS = {
    "telemetry": "drone.telemetry",
    "health_alerts": "drone.health.alerts",
    "inference": "drone.inference.results",
    "commands": "drone.mission.commands",
    "battery_events": "drone.battery.events",
    "aggregated": "drone.metrics.aggregated",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aegis.redpanda.producer")


class AegisTelemetryProducer:
    """
    Publishes drone telemetry and inference results to Redpanda.

    Thread-safe. Handles delivery callbacks and buffering internally.
    """

    def __init__(
        self,
        broker: str = REDPANDA_BROKER,
        batch_size: int = 100,
        linger_ms: int = 10,
    ):
        self._producer = Producer({
            "bootstrap.servers": broker,
            "client.id": "aegis-swarm-producer",
            "acks": "all",
            "retries": 3,
            "retry.backoff.ms": 100,
            "batch.size": batch_size * 1024,       # bytes
            "linger.ms": linger_ms,
            "compression.type": "lz4",
            "queue.buffering.max.messages": 100000,
            "queue.buffering.max.kbytes": 1048576,  # 1 GB
        })
        self._delivery_errors = 0
        self._messages_sent = 0

        atexit.register(self.flush)
        logger.info(f"Telemetry producer connected to {broker}")

    # ── Delivery Callback ─────────────────────────────────────
    def _on_delivery(self, err, msg):
        if err:
            self._delivery_errors += 1
            logger.error(f"Delivery failed for {msg.topic()}: {err}")
        else:
            self._messages_sent += 1

    # ── Core Send Method ──────────────────────────────────────
    def _send(self, topic: str, key: str, value: Dict[str, Any]):
        """Send a message to a Redpanda topic."""
        try:
            self._producer.produce(
                topic=topic,
                key=key.encode("utf-8"),
                value=json.dumps(value).encode("utf-8"),
                callback=self._on_delivery,
            )
            # Trigger delivery of buffered messages
            self._producer.poll(0)
        except BufferError:
            logger.warning("Producer queue full — flushing...")
            self._producer.flush(timeout=5)
            self._producer.produce(
                topic=topic,
                key=key.encode("utf-8"),
                value=json.dumps(value).encode("utf-8"),
                callback=self._on_delivery,
            )

    # ═══════════════════════════════════════════════════════════
    #                   PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def send_telemetry(
        self,
        drone_id: str,
        swarm_id: str = "default",
        battery_percent: float = 100.0,
        battery_voltage: float = 22.2,
        altitude_m: float = 0.0,
        velocity_ms: float = 0.0,
        latitude: float = 0.0,
        longitude: float = 0.0,
        motor_rpm: Optional[List[float]] = None,
        temperature_c: float = 25.0,
        signal_dbm: float = -50.0,
        connected: bool = True,
    ):
        """
        Publish a drone telemetry record.

        Args:
            drone_id:         Unique drone identifier (e.g., "drone-001")
            swarm_id:         Swarm group identifier
            battery_percent:  Battery level (0-100)
            battery_voltage:  Current battery voltage
            altitude_m:       Altitude in meters AGL
            velocity_ms:      Ground speed in m/s
            latitude/longitude: GPS coordinates
            motor_rpm:        List of 4 motor RPMs
            temperature_c:    Onboard temperature
            signal_dbm:       Radio signal strength
            connected:        Connection status
        """
        payload = {
            "drone_id": drone_id,
            "swarm_id": swarm_id,
            "timestamp": time.time(),
            "battery_percent": battery_percent,
            "battery_voltage": battery_voltage,
            "altitude_m": altitude_m,
            "velocity_ms": velocity_ms,
            "position": {"lat": latitude, "lon": longitude},
            "motor_rpm": motor_rpm or [0, 0, 0, 0],
            "temperature_c": temperature_c,
            "signal_dbm": signal_dbm,
            "connected": connected,
        }
        self._send(TOPICS["telemetry"], drone_id, payload)

    def send_telemetry_from_entity(self, drone_entity, swarm_id: str = "default"):
        """
        Publish telemetry directly from a DroneEntity object.

        Compatible with: Python/Before/core/drone_entity.py::DroneEntity

        Args:
            drone_entity:  DroneEntity instance from the simulation
            swarm_id:       Swarm group identifier
        """
        pos = drone_entity.position
        vel = drone_entity.velocity
        speed = float((vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2) ** 0.5)

        self.send_telemetry(
            drone_id=f"drone-{drone_entity.id:03d}",
            swarm_id=swarm_id,
            battery_percent=drone_entity.battery_level,
            battery_voltage=drone_entity.battery_voltage_current,
            altitude_m=max(pos[2], 0),
            velocity_ms=speed,
            latitude=pos[0] * 0.000009,    # Sim meters → approx lat offset
            longitude=pos[1] * 0.000011,   # Sim meters → approx lon offset
            motor_rpm=list(drone_entity.motor_speeds),
            temperature_c=25.0 + speed * 0.5,  # Estimate: ambient + velocity heating
            signal_dbm=-40 - max(pos[2], 0) * 0.3,  # Signal degrades with altitude
            connected=drone_entity.battery_level > 0,
        )

    def send_inference_result(
        self,
        drone_id: str,
        model_name: str = "yolov26s",
        speed_mode: str = "FAST",
        latency_s: float = 0.0,
        fps: float = 0.0,
        detections: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Publish YOLO inference results.

        Args:
            drone_id:     Drone identifier
            model_name:   Model name (e.g., "yolov26s")
            speed_mode:   Speed mode used (ULTRA_FAST, FAST, BALANCED, ACCURATE)
            latency_s:    Inference latency in seconds
            fps:          Current FPS
            detections:   List of {"class": "car", "confidence": 0.92} dicts
        """
        payload = {
            "drone_id": drone_id,
            "timestamp": time.time(),
            "model_name": model_name,
            "speed_mode": speed_mode,
            "latency_s": latency_s,
            "fps": fps,
            "detections": detections or [],
        }
        self._send(TOPICS["inference"], drone_id, payload)

    def send_mission_command(
        self,
        drone_id: str,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
        issued_by: str = "langgraph",
    ):
        """
        Publish a mission command.

        Args:
            drone_id:    Target drone
            command:     Command name (follow, patrol, rtl, land, etc.)
            parameters:  Command-specific parameters
            issued_by:   Source of command (langgraph, operator, auto)
        """
        payload = {
            "drone_id": drone_id,
            "timestamp": time.time(),
            "command": command,
            "parameters": parameters or {},
            "issued_by": issued_by,
        }
        self._send(TOPICS["commands"], drone_id, payload)

    def send_battery_event(
        self,
        drone_id: str,
        event_type: str,
        battery_percent: float,
        details: str = "",
    ):
        """
        Publish a battery lifecycle event.

        Args:
            event_type:  low_battery, critical_battery, charging, full_charge
        """
        payload = {
            "drone_id": drone_id,
            "timestamp": time.time(),
            "event_type": event_type,
            "battery_percent": battery_percent,
            "details": details,
        }
        self._send(TOPICS["battery_events"], drone_id, payload)

    # ── Lifecycle ─────────────────────────────────────────────

    def flush(self, timeout: float = 10.0):
        """Flush all pending messages. Call before shutdown."""
        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning(f"{remaining} messages still in queue after flush")
        logger.info(
            f"Producer stats: sent={self._messages_sent}, errors={self._delivery_errors}"
        )

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "messages_sent": self._messages_sent,
            "delivery_errors": self._delivery_errors,
        }
