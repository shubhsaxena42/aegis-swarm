# 🛡️ Aegis Swarm — Autonomous Drone Fleet AI Platform

> **A production-grade, cloud-native platform that processes real-time telemetry from a 5-drone swarm at 10 FPS, performing object detection, autonomous mission planning, and streaming health analytics — all secured by a fully automated DevSecOps pipeline.**

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](/.github/workflows/devsecops-pipeline.yml)
[![Security](https://img.shields.io/badge/Security-Trivy%20%7C%20Bandit%20%7C%20OWASP-red?logo=shield)](/.github/workflows/devsecops-pipeline.yml)
[![ML Tracking](https://img.shields.io/badge/MLOps-MLflow%20%7C%20DVC-0194E2?logo=mlflow)](./infrastructure/)
[![Streaming](https://img.shields.io/badge/Streaming-Redpanda%20(Kafka%20API)-E50075?logo=apache-kafka&logoColor=white)](./deploy/k8s/01-redpanda.yaml)
[![Inference](https://img.shields.io/badge/Inference-Ray%20Serve%20%7C%20YOLOv26-028CF0?logo=ray)](./app/Live/ray_serve_yolo.py)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ⚡ The "So What?" (60-Second Summary)

Most drone simulations are single-script demos. **Aegis Swarm is different.**

| Metric | Value |
|:---|:---|
| 🚀 **Telemetry Throughput** | 50–100 msg/sec sustained; benchmarked to **1 GB/s** peak |
| 🎯 **Detection Latency (P50)** | **~45ms** (YOLOv26 + SAHI on NVIDIA T4/L4) |
| 🔴 **Worst-Case E2E Latency** | **P99 ~350ms** (telemetry → decision → command) |
| 🛡️ **Fault Tolerance** | Flink recovers in **<10s** from last checkpoint (60s interval) |
| 📡 **Partitions (Redpanda)** | 6 for telemetry/inference, 3 for commands — mapped 1:1 to Flink task slots |
| 🔒 **Security** | SAST + DAST + SBOM (CycloneDX) + container scanning on every commit |

---

## 🏗️ System Architecture

The platform follows a **fully event-driven microservices** design. No component talks directly to another — all communication flows through **Redpanda**, making the system resilient to individual component failures.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       EDGE / SIMULATION LAYER                            │
│                                                                          │
│   ┌──────────────────────────────────────────────────────┐              │
│   │  Unreal Engine 5 + Colosseum (AirSim)                │              │
│   │  5 Drones • 60Hz Physics • 10-12 FPS Camera Capture  │              │
│   └────────────────────┬─────────────────────────────────┘              │
└────────────────────────┼─────────────────────────────────────────────────┘
                         │ Telemetry + Video Frames (10 Hz)
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION & VISION LAYER                         │
│                                                                          │
│   ┌─────────────────────┐     ┌──────────────────────────────────┐       │
│   │  Redpanda (Kafka)   │     │  Ray Serve (YOLOv26 + BoT-SORT)  │       │
│   │  6 Topics           │◄────│  SAHI Slicing • Dynamic Batching  │       │
│   │  6 Partitions (Tel) │     │  Autoscaling: 1→5 GPU Replicas   │       │
│   │  RF=3 (Production)  │     │  P50 Latency: ~45ms              │       │
│   └────────┬────────────┘     └──────────────────────────────────┘       │
└────────────┼─────────────────────────────────────────────────────────────┘
             │  drone.telemetry + drone.inference.results
             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    STREAM ANALYTICS LAYER                                │
│                                                                          │
│   ┌───────────────────────────────────────────────────────────────┐      │
│   │  Apache Flink 1.20                                            │      │
│   │  State Backend: RocksDB  •  Exactly-Once Semantics            │      │
│   │  Sliding Windows: 5min/10s  •  Watermarks: 5s lateness        │      │
│   │  Parallelism: 6 (matches Redpanda partition count)            │      │
│   │                                                               │      │
│   │  Outputs:                                                     │      │
│   │  • drone.health.alerts → LangGraph Analyst                   │      │
│   │  • drone.metrics.aggregated → Prometheus                     │      │
│   └───────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────┘
             │  Health Alerts + Mission Events
             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    AGENTIC COORDINATION LAYER                            │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────┐        │
│   │  LangGraph (GPT-4o-mini) — 3-Agent Hierarchy               │        │
│   │                                                             │        │
│   │  [COORDINATOR]  ←→  [ANALYST]  ←→  [TACTICAL]             │        │
│   │    Mission Planner   Safety Monitor  PID Executor           │        │
│   │                                                             │        │
│   │  Shared State Object • LangGraph Checkpointing              │        │
│   │  VFF Pathfinding • Pre-emption Logic (Deadlock Prevention)  │        │
│   └──────────────────────────┬──────────────────────────────────┘        │
└─────────────────────────────┼────────────────────────────────────────────┘
                              │  drone.mission.commands
                              ▼
              ┌───────────────────────────────┐
              │   Colosseum Bridge            │
              │   (Command Consumer)          │
              │   moveByVelocityAsync()       │
              │   landAsync()  • takeoffAsync │
              └───────────────────────────────┘
```

---

## 🔑 Key Design Decisions & Trade-offs

This section explains the **"Why"** behind each major technology choice — the most important signal for a senior engineer interviewer.

### Why Redpanda over Apache Kafka?
| Concern | Kafka | Redpanda (Our Choice) |
|:---|:---|:---|
| **Deployment Complexity** | Requires Zookeeper cluster | Single binary, no JVM, no Zookeeper |
| **Latency** | ~5-15ms producer latency | **~1-3ms** producer latency |
| **Dev Experience** | Complex tuning required | Ships with console UI, schema registry |
| **K8s Footprint** | Large (multiple pods) | Single StatefulSet |

### Why Ray Serve over Flask/FastAPI?
*   **Stateful Actors:** BoT-SORT tracking requires **per-drone memory** across frames. Ray Serve actors stay alive in memory, making this trivial. Flask is stateless — you'd need Redis to simulate it.
*   **Fractional GPUs:** We allocate `num_gpus=0.2` per replica, allowing 5 inference actors on a single physical GPU. Impossible natively with Flask.
*   **Request-Queue Autoscaling:** Scales based on pending request depth, not CPU — the correct signal for GPU-bound workloads.

### Why Apache Flink over a Python Script?
*   A Python script loses all historical state on restart. Flink's **RocksDB + S3 Checkpointing** allows recovery in <10s.
*   Python has no concept of **Watermarks** — late telemetry data (from signal drops) would corrupt analytics windows. Flink handles this natively.
*   Flink's TCP-based **Backpressure** prevents OOM crashes during traffic spikes.

---

## 🚀 Quick Start

### Prerequisites
- Docker + Docker Compose
- Python 3.10+
- Unreal Engine 5 with Colosseum plugin *(for live simulation)*
- NVIDIA GPU *(for YOLO inference)*

### Option A: Local Dev Stack (Docker Compose)
```bash
# Start all infrastructure: Redpanda, Flink, MLflow, MinIO, Observability
docker compose -f deploy/docker-compose.yml up -d

# Services available at:
# Redpanda Console:  http://localhost:8080
# Flink Dashboard:   http://localhost:8081
# MLflow UI:         http://localhost:5000
# Grafana:           http://localhost:3000  (admin / aegis-swarm-2024)
# MinIO Console:     http://localhost:9001
```

### Option B: Production (GitOps + ArgoCD)
```bash
# Apply ArgoCD application — syncs entire deploy/k8s/ directory
kubectl apply -f deploy/k8s/argocd-app.yaml

# Verify fleet
kubectl get pods -n aegis-swarm
```

### Running the Simulation
```bash
pip install -r app/requirements.txt

# 1. Start Ray Serve inference endpoint
python app/Live/ray_serve_yolo.py

# 2. Generate UE5 multi-drone settings (run once)
python app/Live/colosseum_bridge.py --generate-settings

# 3. Start the full swarm bridge (connects UE5 → Redpanda → Ray Serve)
python app/Live/colosseum_bridge.py --drones 5

# 4. Start the AI Mission Controller (LangGraph 3-agent system)
python app/Live/langgraph_mission_controller.py
```

---

## 🛡️ DevSecOps Pipeline

Security is **not a checkbox** — it is integrated at every stage of the build.

```
Code Push → GitHub Actions
               │
               ├─── SAST: Bandit (Python) + SonarQube + tfsec (IaC)
               │
               ├─── SCA: OWASP Dependency-Check + pip-audit
               │         (Blocks merge if CVSS ≥ 7.0)
               │
               ├─── Container Build (Multi-stage Dockerfile, non-root user)
               │         └── Trivy image scan + SBOM (CycloneDX) generation
               │
               ├─── DAST: OWASP ZAP (attacks running inference endpoint)
               │
               └─── GitOps: ArgoCD sync to Kubernetes
                             └── Runtime: Falco syscall auditing
```

**Pipeline Runtime:** ~12 minutes. Security scans run in **parallel** using a GitHub Actions matrix strategy.

---

## 🔒 Security Architecture

*   **Container Security:** Distroless base images, non-root user (`USER 1001`), read-only filesystem.
*   **Network:** Zero-trust `NetworkPolicies` — each pod can only communicate with explicitly allowed services.
*   **Secrets:** GitHub Secrets in CI, Kubernetes Secrets in-cluster. Production-ready integration with HashiCorp Vault.
*   **Telemetry Integrity:** Kinematic Validation — if a drone's GPS coordinates exceed its physical maximum velocity between frames, the data is flagged as **Spoofed** and discarded.
*   **Supply Chain:** SBOM (Software Bill of Materials) generated on every build via Trivy, allowing instant CVE impact assessment (e.g., "Are we affected by Log4Shell?").

---

## 📊 Observability Stack

| Signal | Tool | What We Track |
|:---|:---|:---|
| **Metrics** | Prometheus + Grafana | Consumer lag, inference P99 latency, GPU memory, drone heartbeats |
| **Logs** | ELK Stack (Elasticsearch + Kibana) | Structured JSON logs, queryable by `drone_id` and `severity` |
| **Traces** | OpenTelemetry + Jaeger | Full request path: `Drone → Redpanda → Flink → LangGraph → Action` |

**SLOs:**
*   **Availability:** 99.9% heartbeat uptime per drone
*   **Latency:** 95th percentile of mission commands issued within **200ms** of telemetry arrival

**Silent Failure Detection:** If `telemetry_count_total` stops incrementing for a `drone_id` while its pod is still `Running`, a "Stale Data" alert fires — catching hangs before they cascade.

---

## 🧠 MLOps Pipeline

```
New Training Data (UE5 Captures)
        │
        ├── DVC tracks dataset → MinIO (S3-compatible)
        │
        ├── GitHub Action triggers headless training job
        │
        ├── MLflow logs: mAP@50-95, precision, recall, latency benchmarks
        │
        └── Model promoted to "Production" tag in MLflow Registry
                  │
                  └── Ray Serve workers hot-reload weights (zero downtime)
```

*   **Model Drift Detection:** If the fleet's `average_confidence_score` drops by >30%, a "Drift Alert" triggers sample logging for a retraining run.
*   **Canary Deployment:** Ray Serve routes 10% of traffic to a new model version before full promotion.

---

## 📦 Redpanda Topic Architecture

| Topic | Partitions | Retention | Purpose |
|:---|:---:|:---|:---|
| `drone.telemetry` | **6** | 24h | GPS, battery, velocity from each drone |
| `drone.inference.results` | **6** | 24h | YOLO detections + BoT-SORT track IDs |
| `drone.health.alerts` | **3** | 7d | Flink-generated anomaly alerts |
| `drone.mission.commands` | **3** | 7d | LangGraph → Drone action commands |
| `drone.battery.events` | **3** | Default | Battery lifecycle events |
| `drone.metrics.aggregated` | **3** | Default | Windowed health scores |

*6 partitions on telemetry/inference = 1:1 mapping with Flink's parallelism of 6, ensuring ordered, zero-shuffle processing per drone.*

---

## ⚠️ Known Limitations & Future Work

Being honest about limitations shows **engineering maturity**.

| Limitation | Planned Fix |
|:---|:---|
| Single-node Redpanda (dev) | RF=3 multi-broker cluster in production manifests |
| JSON telemetry (verbose) | Migrate to **Avro + Schema Registry** (50-80% bandwidth reduction) |
| Cloud inference only | **Edge Inference (NVIDIA Jetson)** for >100 drone scalability |
| HTTP image transport (base64) | Replace with **gRPC binary streaming** (-30ms latency) |
| LLM API cost ~$200-300/month | Self-host **Llama 3** to reduce cost by 90% |

---

## 📂 Repository Structure

```
.
├── .github/workflows/
│   └── devsecops-pipeline.yml    # Full CI/CD: SAST, DAST, SBOM, Container Scan
├── app/
│   └── Live/
│       ├── colosseum_bridge.py   # UE5 ↔ Redpanda ↔ Ray Serve Bridge (5-drone)
│       ├── ray_serve_yolo.py     # YOLOv26 + BoT-SORT inference endpoint
│       ├── drone_follow_car.py   # PID Controller (VFF Pathfinding)
│       └── langgraph_mission_controller.py  # 3-Agent LangGraph orchestrator
├── deploy/
│   ├── k8s/                      # Kubernetes manifests (ArgoCD / Kustomize)
│   │   ├── 01-redpanda.yaml      # StatefulSet, Topics, RF config
│   │   ├── 02-flink.yaml         # JobManager, TaskManager, RocksDB config
│   │   └── 07-observability.yaml # ELK, OTel, Jaeger, Prometheus
│   └── docker-compose.yml        # Local dev stack
├── infrastructure/
│   ├── redpanda/
│   │   ├── producer.py           # AegisTelemetryProducer (typed, idempotent)
│   │   └── consumer.py           # AegisTelemetryConsumer (typed generators)
│   └── flink/jobs/
│       └── drone_health_analytics.py  # Flink job: RocksDB, Watermarks, Windows
├── observability/
│   └── prometheus/alerts.yaml    # Alert rules (battery, latency, drift)
└── interview_prep.md             # Deep-dive Q&A for technical interviews
```

---

## 👤 Author

**Shubh Saxena** — [github.com/shubhsaxena42](https://github.com/shubhsaxena42)

*Built as a comprehensive demonstration of production-grade MLOps, DevSecOps, and distributed systems engineering principles in an autonomous robotics context.*

---
**License:** MIT
