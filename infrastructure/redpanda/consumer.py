"""
Aegis Swarm — Redpanda Consumer Utilities
══════════════════════════════════════════

Reusable consumer helpers for reading from Aegis Swarm topics.

Usage:
    from infrastructure.redpanda.consumer import AegisTelemetryConsumer

    consumer = AegisTelemetryConsumer()
    for telemetry in consumer.consume_telemetry(max_messages=100):
        print(telemetry["drone_id"], telemetry["battery_percent"])
"""

import os
import json
import logging
from typing import Dict, Any, Generator, List, Optional

from confluent_kafka import Consumer, KafkaError, TopicPartition

REDPANDA_BROKER = os.getenv("REDPANDA_BROKER", "localhost:19092")
logger = logging.getLogger("aegis.redpanda.consumer")


class AegisTelemetryConsumer:
    """
    Consumer for Aegis Swarm Redpanda topics.
    Supports both streaming (infinite loop) and batch consumption.
    """

    def __init__(
        self,
        broker: str = REDPANDA_BROKER,
        group_id: str = "aegis-consumer",
        auto_offset_reset: str = "latest",
    ):
        self._config = {
            "bootstrap.servers": broker,
            "group.id": group_id,
            "auto.offset.reset": auto_offset_reset,
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
            "session.timeout.ms": 30000,
        }

    def consume_telemetry(
        self,
        max_messages: Optional[int] = None,
        timeout_s: float = 1.0,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Consume drone telemetry records.

        Args:
            max_messages: Max messages to consume (None = infinite)
            timeout_s:    Poll timeout in seconds
        """
        yield from self._consume("drone.telemetry", max_messages, timeout_s)

    def consume_health_alerts(
        self,
        max_messages: Optional[int] = None,
        timeout_s: float = 1.0,
    ) -> Generator[Dict[str, Any], None, None]:
        """Consume health alert records from Flink pipeline."""
        yield from self._consume("drone.health.alerts", max_messages, timeout_s)

    def consume_inference_results(
        self,
        max_messages: Optional[int] = None,
        timeout_s: float = 1.0,
    ) -> Generator[Dict[str, Any], None, None]:
        """Consume YOLO inference results."""
        yield from self._consume("drone.inference.results", max_messages, timeout_s)

    def consume_aggregated_metrics(
        self,
        max_messages: Optional[int] = None,
        timeout_s: float = 1.0,
    ) -> Generator[Dict[str, Any], None, None]:
        """Consume aggregated health metrics from Flink."""
        yield from self._consume("drone.metrics.aggregated", max_messages, timeout_s)

    def _consume(
        self,
        topic: str,
        max_messages: Optional[int],
        timeout_s: float,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generic consumer loop."""
        consumer = Consumer(self._config)
        consumer.subscribe([topic])
        count = 0

        try:
            while max_messages is None or count < max_messages:
                msg = consumer.poll(timeout=timeout_s)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(f"Consumer error: {msg.error()}")
                    continue

                try:
                    data = json.loads(msg.value().decode("utf-8"))
                    data["_meta"] = {
                        "topic": msg.topic(),
                        "partition": msg.partition(),
                        "offset": msg.offset(),
                        "timestamp": msg.timestamp()[1],
                    }
                    yield data
                    count += 1
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode message from {topic}")
        finally:
            consumer.close()
