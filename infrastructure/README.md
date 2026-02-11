# 🏗️ Aegis Swarm — Infrastructure Stack

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────────────────────────┐
                    │                     AEGIS SWARM INFRASTRUCTURE                      │
                    ├─────────────────────────────────────────────────────────────────────┤
                    │                                                                     │
  DRONE FLEET       │   ┌─────────────┐     ┌──────────────┐     ┌───────────────┐       │
  ┌──────┐          │   │  REDPANDA   │     │ APACHE FLINK │     │  PROMETHEUS   │       │
  │ 🛸 1 │──────────┼──►│  (Kafka)    │────►│  Stream Proc │────►│  + Grafana    │       │
  │ 🛸 2 │──────────┼──►│             │     │              │     │               │       │
  │ 🛸 N │──────────┼──►│  Topics:    │     │  Jobs:       │     │  Metrics:     │       │
  └──────┘          │   │  telemetry  │     │  health_     │     │  battery      │       │
                    │   │  inference  │     │  analytics   │     │  altitude     │       │
  YOLO INFERENCE    │   │  commands   │     │              │     │  motor_rpm    │       │
  ┌──────┐          │   │  battery    │     │  Detects:    │     │  inference    │       │
  │ 🧠   │──────────┼──►│  alerts     │     │  anomalies   │     │  swarm_health │       │
  └──────┘          │   │  aggregated │     │  drain rate  │     │               │       │
                    │   └─────────────┘     │  motor imbal │     │  Alerts:      │       │
                    │                       │  GPS drift   │     │  battery_crit │       │
                    │   ┌─────────────┐     │  overheating │     │  conn_lost    │       │
                    │   │   MLFLOW    │     └──────────────┘     │  motor_anom   │       │
                    │   │  Experiment │                           └───────────────┘       │
                    │   │  Tracking   │     ┌──────────────┐                              │
                    │   │  + Registry │     │    MinIO      │     ┌───────────────┐       │
                    │   │             │────►│  (S3 Storage) │◄────│     DVC       │       │
                    │   └─────────────┘     │              │     │  Data Version │       │
                    │                       │  Buckets:     │     │  Control      │       │
                    │   ┌─────────────┐     │  mlflow-art.  │     └───────────────┘       │
                    │   │ POSTGRESQL  │     │  dvc-storage  │                              │
                    │   │ (MLFlow DB) │     │  drone-data   │                              │
                    │   └─────────────┘     └──────────────┘                              │
                    └─────────────────────────────────────────────────────────────────────┘
```

## 🧩 Components

| Component | Technology | Purpose | Port |
|-----------|-----------|---------|------|
| **Stream Broker** | Redpanda | Kafka-compatible event streaming for drone telemetry | `19092` (Kafka), `8080` (Console) |
| **Stream Processing** | Apache Flink | Real-time health scoring, anomaly detection | `8081` (UI) |
| **System Metrics** | Prometheus | Time-series metrics collection & alerting | `9090` |
| **Dashboards** | Grafana | Visualization of drone fleet and infrastructure metrics | `3000` |
| **Experiment Tracking** | MLFlow | YOLO training experiments, model registry | `5000` |
| **Data Versioning** | DVC | Dataset and model weight version control | — |
| **Object Storage** | MinIO | S3-compatible storage for artifacts and DVC data | `9000` (API), `9001` (Console) |
| **Metadata DB** | PostgreSQL | MLFlow backend store | `5432` |

## 📁 File Structure

```
infrastructure/
├── docker-compose.yml                      # All services orchestration
├── .env.example                            # Environment variable template
├── requirements.txt                        # Python dependencies
│
├── redpanda/
│   ├── __init__.py
│   ├── producer.py                         # Telemetry publisher (DroneEntity integration)
│   └── consumer.py                         # Topic consumer utilities
│
├── flink/
│   └── jobs/
│       └── drone_health_analytics.py       # PyFlink health scoring + anomaly detection
│
├── prometheus/
│   ├── prometheus.yml                      # Scrape configuration
│   ├── alerts.yml                          # Alert rules (battery, motor, swarm, infra)
│   ├── grafana-datasources.yml             # Grafana auto-provisioning
│   ├── drone_metrics_exporter.py           # Custom Prometheus exporter
│   └── Dockerfile.exporter                 # Exporter container image
│
├── mlflow_tracking/
│   ├── __init__.py
│   └── experiment_tracker.py               # MLFlow wrapper for YOLO experiments
│
└── dvc_versioning/
    ├── __init__.py
    └── data_manager.py                     # DVC operations + pipeline generation
```

## 🚀 Quick Start

### 1. Start All Services

```bash
cd infrastructure
cp .env.example .env         # Configure credentials
docker compose up -d         # Start everything
```

### 2. Verify Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Redpanda Console | http://localhost:8080 | — |
| Flink Dashboard | http://localhost:8081 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / aegis-swarm-2024 |
| MLFlow | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |

### 3. Install Python Dependencies

```bash
pip install -r infrastructure/requirements.txt
```

## 📖 Usage Examples

### Publish Drone Telemetry to Redpanda

```python
from infrastructure.redpanda import AegisTelemetryProducer
from Python.Before.core.drone_entity import DroneEntity

producer = AegisTelemetryProducer()
drone = DroneEntity(drone_id=1, initial_pos=[0, 0, 50])

# Direct telemetry
producer.send_telemetry(
    drone_id="drone-001",
    battery_percent=85.0,
    altitude_m=50.0,
    motor_rpm=[8500, 8520, 8490, 8510],
)

# From simulation entity (auto-extracts all fields)
producer.send_telemetry_from_entity(drone, swarm_id="swarm-alpha")

# Inference results
producer.send_inference_result(
    drone_id="drone-001",
    model_name="yolov26s",
    speed_mode="FAST",
    latency_s=0.042,
    fps=23.8,
    detections=[
        {"class": "car", "confidence": 0.92},
        {"class": "person", "confidence": 0.87},
    ],
)
```

### Track YOLO Training with MLFlow

```python
from infrastructure.mlflow_tracking import AegisExperimentTracker

tracker = AegisExperimentTracker()

with tracker.start_training_run("yolov26s-visdrone-v3") as run:
    # Log hyperparameters
    tracker.log_training_config(
        model="yolov26s", imgsz=640, epochs=100,
        batch_size=16, optimizer="AdamW",
    )

    # Log per-epoch metrics
    for epoch in range(100):
        tracker.log_epoch_metrics(
            epoch=epoch, train_loss=0.5, val_map50=0.65,
        )

    # Log final model
    tracker.log_model_artifact("runs/detect/train/weights/best.pt")
    tracker.log_final_metrics(
        best_map50=0.72, best_map50_95=0.48,
        best_epoch=87, total_training_time_s=14400,
    )

# Register for production
tracker.register_model(run.info.run_id, stage="Production")
```

### Version Datasets with DVC

```python
from infrastructure.dvc_versioning import AegisDataManager

dm = AegisDataManager()

# Track a dataset
meta = dm.track_dataset("datasets/visdrone", "VisDrone2019-DET v1.0")

# Push to remote storage (MinIO)
dm.push_dataset("datasets/visdrone")

# Pull on another machine
dm.pull_dataset("datasets/visdrone")

# Generate DVC pipeline (prepare → train → evaluate → benchmark)
dm.create_training_pipeline()
dm.create_params_file()
```

## 🔔 Prometheus Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| `DroneBatteryCritical` | Battery < 15% for 30s | 🔴 Critical |
| `DroneBatteryLow` | Battery < 30% for 1m | 🟡 Warning |
| `DroneConnectionLost` | No telemetry for 15s | 🔴 Critical |
| `DroneAltitudeExceeded` | Altitude > 120m for 10s | 🟡 Warning |
| `SwarmDroneCountLow` | Active drones < 3 for 1m | 🟡 Warning |
| `InferenceLatencyHigh` | Latency > 500ms for 2m | 🟡 Warning |
| `DroneMotorAnomaly` | Motor RPM deviation > 30% for 30s | 🟡 Warning |
| `RedpandaConsumerLag` | Lag > 10K messages for 5m | 🟡 Warning |
| `FlinkJobFailed` | Job restarts > 2 in 1m | 🔴 Critical |

## 🏗️ Flink Health Analytics Pipeline

The Flink job (`drone_health_analytics.py`) implements:

1. **Per-drone health scoring** — weighted combination of:
   - Battery health (40%): level + voltage sag
   - Motor health (25%): RPM symmetry across 4 motors
   - Signal health (20%): dBm normalized
   - Thermal health (15%): safe operating range

2. **Anomaly detection** with configurable thresholds:
   - Rapid battery drain (> 0.1%/s)
   - Motor imbalance (RPM CV > 15%)
   - GPS drift (position jump inconsistent with velocity)
   - Overheating (> 70°C)

3. **Alert cooldowns** — prevents flooding (30s for warnings, 10s for critical)
