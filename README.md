# 🛡️ Aegis Swarm — Autonomous Drone Fleet AI & DevSecOps Platform

![Build Status](https://img.shields.io/github/actions/workflow/status/shubhsaxena42/aegis-swarm/devsecops-pipeline.yml?label=DevSecOps&style=for-the-badge)
![Kubernetes](https://img.shields.io/badge/kubernetes-v1.28+-326ce5.svg?style=for-the-badge&logo=kubernetes)
![Ray Serve](https://img.shields.io/badge/Ray_Serve-Scaling-028CF0?style=for-the-badge)
![Redpanda](https://img.shields.io/badge/Redpanda-Streaming-F05F40?style=for-the-badge)
![Argocd](https://img.shields.io/badge/ArgoCD-GitOps-orange?style=for-the-badge)

**Aegis Swarm** is a production-grade **DevSecOps** and **MLOps** platform designed for real-time autonomous drone fleet management. It integrates high-performance stream processing, computer vision inference, and comprehensive observability into a secure Kubernetes-native architecture.

---

## 🏗️ System Architecture

The system follows an event-driven microservices architecture:

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
        Redpanda -->|Stream| Flink[Apache Flink]
        Flink -->|Stateful Analytics| Postgres[(PostgreSQL)]
    end

    subgraph "MLOps Platform"
        MLFlow[MLFlow Registry] --> MinIO[(MinIO S3)]
        MLFlow --> Postgres
        Ray -- Load Model --> MLFlow
    end

    subgraph "Observability Stack"
        Prometheus[Prometheus] -->|Scrape| Flink & Redpanda & Ray
        Prometheus --> Grafana[Grafana Dashboards]
        
        Filebeat[Filebeat] --> Logstash[Logstash] --> Elastic[(Elasticsearch)]
        Kibana[Kibana] --> Elastic
        
        OTel[OTel Collector] --> Jaeger[Jaeger Traces]
    end
```

---

## 📂 Project Structure

The codebase is organized into modular directories for application logic, deployment, and configuration.

```bash
.
├── .github/workflows/   # DevSecOps Pipeline (SAST, DAST, SBOM, Falco)
├── app/                 # Python Application Source Code
│   ├── Live/            # Core Drone Logic & Inference Scripts
│   └── requirements.txt # Application Dependencies
├── deploy/              # Deployment Configuration
│   ├── k8s/             # Kubernetes Manifests (Kustomize)
│   │   ├── 00-namespace.yaml
│   │   ├── 01-redpanda.yaml
│   │   ├── ...
│   │   └── argocd-app.yaml
│   └── docker-compose.yml # Local Development Stack
├── observability/       # Monitoring Configs
│   ├── prometheus/      # Alert rules & scrape configs
│   └── grafana/         # Dashboard datasources
├── Dockerfile           # Multi-stage production build
└── README.md            # You are here
```

---

## 🚀 Deployment Guide

### Prerequisites
*   **Kubernetes Cluster** (v1.28+) or Minikube
*   **Kubectl** & **Kustomize**
*   **Docker**
*   **Python 3.11** (for local dev)

### Option A: GitOps (ArgoCD) — Recommended
Aegis Swarm is designed for **GitOps**. All cluster state is defined in `deploy/k8s/`.

1.  **Install ArgoCD**:
    ```bash
    kubectl create namespace argocd
    kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
    ```
2.  **Apply the Application**:
    ```bash
    kubectl apply -f deploy/k8s/argocd-app.yaml
    ```
3.  **Sync**: ArgoCD will automatically synchronize the cluster state with the `deploy/k8s` directory in the repository.

### Option B: Manual Kubernetes Deployment
To deploy directly without ArgoCD:

```bash
# 1. (Optional) Create local persistent volumes for Minikube/Docker Desktop
kubectl apply -f deploy/k8s/00-persistent-volumes.yaml

# 2. Deploy the entire stack via Kustomize
kubectl apply -k deploy/k8s/
```

### Option C: Local Development (Docker Compose)
For strictly local testing without Kubernetes:

```bash
docker-compose -f deploy/docker-compose.yml up -d
```

### 📦 Kubernetes Resources Default Config

#### Manifest Files
| File | Contents |
|------|----------|
| `00-namespace.yaml` | Namespace, ConfigMap, Secrets |
| `00-persistent-volumes.yaml` | **Local HostPath PVs** (for testing/bare-metal) |
| `01-redpanda.yaml` | Redpanda StatefulSet, Console, Topic Init Job |
| `02-flink.yaml` | Flink JobManager + TaskManager, ConfigMap |
| `03-monitoring.yaml` | Prometheus, Grafana, Drone Metrics Exporter |
| `04-mlplatform.yaml` | PostgreSQL, MinIO, MLFlow, Bucket Init Job |
| `05-application.yaml` | Aegis Swarm App, HPA, PDB, ServiceAccount |
| `06-networking.yaml` | Ingress, NetworkPolicies (zero-trust), Rate Limiting |
| `07-observability.yaml` | ELK Stack (Logs) + OTel/Jaeger (Traces) |
| `argocd-app.yaml` | ArgoCD Application Definition |

#### Stateful Services (PVC Configuration)
- **Redpanda**: 10Gi (Telemetry Stream)
- **PostgreSQL**: 10Gi (MLFlow Metadata)
- **Elasticsearch**: 30Gi (Logs)
- **MinIO**: 50Gi (Model Registry & Artifacts)
- **Prometheus**: 20Gi (Metrics Retention)

#### Auto-Scaling & Availability
- **HPA**: App scales 2–8 pods based on CPU (70%) or Ram (80%).
- **PDB**: Ensures min 1 pod available during node upgrades.
- **Ray Serve**: Auto-scales inference actors (1-5 replicas) based on queue depth.

#### Security Implementation
- **Zero-Trust Network**: Default deny ingress/egress; explicit allow lists.
- **Least-Privilege**: Containers run as non-root user `1000`.
- **Secret Management**: All creds mapped via K8s Secrets (base64).
- **Ingress Security**: TLS termination + Rate Limiting (1000 req/m).

---

## 🔍 Observability & Monitoring

The platform implements the **"Three Pillars of Observability"**:

| Pillar | Tool Stack | Access URL | Description |
|:---|:---|:---|:---|
| **Logs** | **ELK Stack** (Filebeat, Logstash, ES, Kibana) | `https://aegis.local/kibana` | Centralized logging for all pods. |
| **Metrics** | **Prometheus** & **Grafana** | `https://aegis.local/grafana` | Real-time dashboards for CPU, memory, drone battery, & throughput. |
| **Traces** | **OpenTelemetry** & **Jaeger** | `https://aegis.local/jaeger` | Distributed tracing for latency analysis of Ray Serve requests. |

**Credentials:**
*   Grafana: `admin` / `aegis-swarm-2024` (or see Secrets)

---

## 🛡️ DevSecOps Pipeline

Security is baked into every commit via GitHub Actions (`.github/workflows/devsecops-pipeline.yml`).

### Stages
1.  **Secret Scanning**: `Gitleaks` detects hardcoded credentials.
2.  **SAST (Static Analysis)**: `Bandit` (Python) & `tfsec` (Terraform) check for code vulnerabilities.
3.  **SCA (Supply Chain)**: `OWASP Dependency-Check` & `Trivy` scan libraries for CVEs (CVSS ≥ 7).
4.  **Container Security**: `Trivy` scans the Docker image for vulnerabilities & generates **SBOM** (CycloneDX).
5.  **DAST (Dynamic Analysis)**: `OWASP ZAP` attacks the running application to find runtime flaws.
6.  **Runtime Security**: `Falco` audits container syscalls (e.g., shell spawning, sensitive file access).

---

## 🧠 Application Modules

### 1. Ray Serve Inference (`app/Live/ray_serve_yolo.py`)
*   **Model**: YOLOv26 (Custom)
*   **Optimizations**: SAHI (Slicing Aided Hyper Inference) for small object detection.
*   **Tracking**: BoT-SORT for persistent ID tracking.
*   **Architecture**: Auto-scaling actors (1-5 replicas) based on request load.

### 2. Stream Processing (`deploy/k8s/02-flink.yaml`)
*   **Framework**: Apache Flink 1.20
*   **Function**: Consumes `drone.telemetry` from Redpanda, calculates Health Scores, and emits alerts to `drone.alerts`.

---

## 🛠️ Operations & Troubleshooting

**Check Pod Status:**
```bash
kubectl get pods -n aegis-swarm
```

**View Application Logs:**
```bash
kubectl logs -l app=aegis-swarm -n aegis-swarm --tail=100 -f
```

**Port Forward Services (Local Access):**
```bash
# Ray Serve / App
kubectl port-forward svc/aegis-swarm-app 8080:8080 -n aegis-swarm

# Redpanda Console
kubectl port-forward svc/redpanda-console 8080:8080 -n aegis-swarm

# Flink Dashboard
kubectl port-forward svc/flink-jobmanager 8081:8081 -n aegis-swarm
```

---

## 📄 License & Attribution
**Aegis Swarm Project**
Developed by [shubhsaxena42](https://github.com/shubhsaxena42)
License: MIT
