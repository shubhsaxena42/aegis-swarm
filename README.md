# 🛡️ Aegis Swarm — Autonomous Drone Fleet AI & DevSecOps Platform

**Aegis Swarm** is a production-grade **DevSecOps** and **MLOps** platform designed for real-time autonomous drone fleet management. It integrates high-performance stream processing via Apache Flink, computer vision inference via Ray Serve, and comprehensive observability into a secure Kubernetes-native architecture.

---

## 🏗️ System Architecture

The system follows an event-driven microservices architecture optimized for low-latency telemetry and high-throughput inference:

```mermaid
graph TD
    subgraph "Edge / Input"
        Drone[Drone Telemetry] -->|MQTT/HTTP| Ingress[NGINX Ingress]
        Camera[Video Feed] -->|RTSP/HTTP| Ingress
    end

    subgraph "Core Processing (Kubernetes)"
        Ingress --> Ray[Ray Serve (YOLOv26)]
        Ingress --> Redpanda[Redpanda (Kafka API)]
        
        Ray -->|Inference Results| Redpanda
        Redpanda -->|Stream| Flink[Apache Flink 1.20]
        Flink -->|Stateful Analytics| Postgres[(PostgreSQL 16)]
    end

    subgraph "MLOps Platform"
        MLFlow[MLFlow v2.19] --> MinIO[(MinIO S3)]
        MLFlow --> Postgres
        Ray -- Load Model --> MLFlow
    end

    subgraph "Observability Stack"
        Prometheus[Prometheus] -->|Scrape| Flink & Redpanda & Ray
        Prometheus --> Grafana[Grafana 11.4]
        
        Filebeat[Filebeat] --> Logstash[Logstash] --> Elastic[(Elasticsearch)]
        Kibana[Kibana] --> Elastic
        
        OTel[OTel Collector] --> Jaeger[Jaeger Traces]
    end

```

---

## 📂 Project Structure

```bash
.
├── .github/workflows/   # DevSecOps Pipeline (SAST, DAST, SBOM, Falco)
├── app/                 # Python Application Source Code
│   ├── Live/            # Core Drone Logic & YOLOv26 Inference
│   └── requirements.txt # Dependencies (Ray, Ultralytics, Flink, Kafka)
├── deploy/              # Deployment Configuration
│   ├── k8s/             # Kubernetes Manifests (Kustomize)
│   │   ├── 00-namespace.yaml
│   │   ├── 01-redpanda.yaml
│   │   └── ...
│   └── docker-compose.yml # Local Dev Stack (Redpanda, MinIO, MLFlow)
├── infrastructure/      # DVC Versioning & Redpanda Producer/Consumer Logic
├── observability/       # Monitoring Configs (Prometheus Alerts & Grafana)
└── README.md            

```

---

## 🚀 Deployment Guide

### Option A: GitOps (ArgoCD) — Recommended

Aegis Swarm is designed for **GitOps**. All cluster state is defined in `deploy/k8s/`.

1. **Apply the Application**:
```bash
kubectl apply -f deploy/k8s/argocd-app.yaml

```

2. **Sync**: ArgoCD will synchronize the cluster state with the repository.

### Option B: Local Development (Docker Compose)

Ideal for testing the streaming and ML infrastructure locally.

```bash
# Start Redpanda, Flink, MLFlow, MinIO, and Observability stack
docker-compose -f deploy/docker-compose.yml up -d

```

**Auto-Configured Topics:**

* `drone.telemetry` (6 partitions, 24h retention)
* `drone.health.alerts` (3 partitions, 7d retention)
* `drone.inference.results` (6 partitions)
* `drone.mission.commands`
* `drone.battery.events`

---

## 📦 Infrastructure Stack Details

| Service | Version | Role | Key Configuration |
| --- | --- | --- | --- |
| **Redpanda** | `v24.3.1` | Message Broker | Kafka-compatible; Ports `19092` (API), `9644` (Metrics). |
| **Apache Flink** | `1.20` | Stream Processor | RocksDB state backend; EXACTLY_ONCE checkpointing. |
| **MLFlow** | `v2.19.0` | Model Registry | Postgres backend; S3-compatible artifact storage. |
| **MinIO** | `2024-11-07` | S3 Storage | Stores models, DVC data, and Flink checkpoints. |
| **Grafana** | `11.4.0` | Dashboards | Default Creds: `admin` / `aegis-swarm-2024`. |

---

## 🧠 Application Modules

### 1. Computer Vision & Inference

* **Engine**: Ray Serve `v2.9.0+` for auto-scaling inference actors.
* **Model**: YOLOv26 via `ultralytics`.
* **Tracking**: `boxmot` (BoT-SORT) for persistent drone/object ID tracking.
* **Optimization**: SAHI (Slicing Aided Hyper Inference) for small object detection.

### 2. Data & Experiment Management

* **Data Versioning**: DVC `v3.56+` with S3/MinIO remote backend.
* **Tracking**: MLFlow integration for logging drone mission parameters and model performance.

### 3. Real-time Analytics

* **Logic**: Flink jobs consume `drone.telemetry` to calculate health scores and detect anomalies in real-time.

---

## 🛡️ DevSecOps Pipeline

Security is automated via GitHub Actions:

1. **SCA**: `Trivy` & `OWASP Dependency-Check` for CVE scanning.
2. **SAST**: `Bandit` (Python) and `tfsec` (Infrastructure).
3. **Container**: Multi-stage `Dockerfile` scanned for vulnerabilities with SBOM generation.
4. **Runtime**: `Falco` integration for Kubernetes syscall auditing.

---

## 🛠️ Operations

**Check Pod Status:**

```bash
kubectl get pods -n aegis-swarm

```

**Local Service Access:**

* **MLFlow UI**: `http://localhost:5000`
* **Redpanda Console**: `http://localhost:8080`
* **Flink Dashboard**: `http://localhost:8081`
* **MinIO Console**: `http://localhost:9001`
* **Grafana**: `http://localhost:3000`

---

**License**: MIT
**Author**: [shubhsaxena42](https://github.com/shubhsaxena42)
