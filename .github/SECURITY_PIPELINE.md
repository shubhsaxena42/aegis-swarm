# 🛡️ Aegis Swarm — DevSecOps CI/CD Pipeline

## Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     Aegis Swarm DevSecOps Pipeline                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   SECRET      │  │   PYTHON     │  │    OWASP     │  │  SONARQUBE   │  │
│  │  SCANNING     │  │    SAST      │  │ DEPENDENCY   │  │  ANALYSIS    │  │
│  │  (Gitleaks)   │  │ (Bandit +    │  │   CHECK      │  │ (Code Qual.) │  │
│  │              │  │  Safety)     │  │  (CVE Scan)  │  │              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │                 │          │
│         │    ┌────────────▼─────────────────▼─────────────────▼──┐       │
│         │    │                                                    │       │
│         │    │        ┌──────────────────────────┐               │       │
│         │    │        │   🐳 DOCKER BUILD        │               │       │
│         │    │        │   + TRIVY SCAN            │               │       │
│         │    │        │  (FS + Image + IaC)       │               │       │
│         │    │        └────────────┬─────────────┘               │       │
│         │    │                     │                              │       │
│         │    │        ┌────────────▼─────────────┐               │       │
│         │    │        │   🏗️ IAC SECURITY        │               │       │
│         │    │        │  (tfsec + Checkov)        │               │       │
│         │    │        └────────────┬─────────────┘               │       │
│         │    │                     │                              │       │
│         │    │        ┌────────────▼─────────────┐               │       │
│         │    │        │   🕷️  OWASP ZAP          │               │       │
│         │    │        │  (DAST - Baseline/Full)   │               │       │
│         │    │        └────────────┬─────────────┘               │       │
│         │    └─────────────────────┼──────────────────────────────┘       │
│         │                         │                                       │
│         └─────────────┬───────────┘                                       │
│                       │                                                    │
│              ┌────────▼──────────┐                                        │
│              │  📊 SECURITY      │                                        │
│              │     REPORT        │                                        │
│              │ (Consolidated)    │                                        │
│              └───────────────────┘                                        │
└────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 Security Tools Integrated

| Tool | Type | Purpose | Stage |
|------|------|---------|-------|
| **Gitleaks** | Secret Detection | Scans git history for leaked credentials, API keys, tokens | 1 |
| **Bandit** | SAST | Python-specific security linter — detects common vulnerabilities | 2 |
| **Safety** | SCA | Checks Python dependencies against known CVE databases | 2 |
| **OWASP Dependency-Check** | SCA | Comprehensive CVE scanning for all project dependencies | 3 |
| **SonarQube** | SAST | Code quality, bugs, code smells, security hotspots, test coverage | 4 |
| **Trivy** | Container/FS/IaC | Multi-purpose scanner — filesystem, container images, and IaC configs | 5 & 6 |
| **tfsec** | IaC | Terraform-specific security scanner by Aqua Security | 6 |
| **Checkov** | IaC | Policy-as-code scanner for Terraform, CloudFormation, K8s | 6 |
| **OWASP ZAP** | DAST | Dynamic application security testing — baseline & full active scans | 7 |

## 🚀 Quick Start

### 1. Required GitHub Secrets

Navigate to **Settings → Secrets and variables → Actions** and add:

| Secret | Required | Description |
|--------|----------|-------------|
| `SONAR_TOKEN` | ✅ | SonarQube authentication token |
| `SONAR_HOST_URL` | ✅ | SonarQube instance URL (e.g., `https://sonarcloud.io`) |

> **Note:** `GITHUB_TOKEN` is automatically provided by GitHub Actions.

### 2. SonarQube Setup

**Option A: SonarCloud (Recommended for open-source)**
1. Go to [sonarcloud.io](https://sonarcloud.io) and sign in with GitHub
2. Import the `aegis-swarm` repository
3. Copy the project token → add as `SONAR_TOKEN` secret
4. Set `SONAR_HOST_URL` to `https://sonarcloud.io`

**Option B: Self-hosted SonarQube**
1. Deploy SonarQube (Docker: `docker run -d -p 9000:9000 sonarqube:latest`)
2. Create a project and generate a token
3. Add both secrets to your repository

### 3. Trigger the Pipeline

The pipeline runs automatically on:
- **Push** to `main`, `develop`, or `release/**` branches
- **Pull requests** targeting `main` or `develop`
- **Manual trigger** via workflow dispatch (with optional ZAP and full Trivy toggles)

### 4. Manual Trigger with Options

```bash
# Using GitHub CLI
gh workflow run "🛡️ Aegis Swarm — DevSecOps Pipeline" \
  --field run_zap_scan=true \
  --field run_full_trivy=true
```

## 📂 Files Created

```
aegis-swarm/
├── .github/
│   ├── workflows/
│   │   └── devsecops-pipeline.yml    # Main CI/CD pipeline
│   ├── owasp-suppressions.xml        # OWASP DC false-positive suppressions
│   └── zap-rules.tsv                 # ZAP scan rule configuration
├── .trivyignore                       # Trivy CVE suppressions
├── sonar-project.properties           # SonarQube project config
└── Dockerfile                         # Multi-stage Docker build
```

## 🔒 Security Best Practices Implemented

1. **Principle of Least Privilege** — Minimal `permissions` block in workflow
2. **Concurrency Control** — Prevents redundant parallel runs on the same branch
3. **SARIF Integration** — All results upload to GitHub Security tab for unified view
4. **Multi-layer Scanning** — SAST + SCA + Container + IaC + DAST coverage
5. **Non-root Docker** — Container runs as unprivileged `aegis` user
6. **Multi-stage Build** — Minimal production image without build tools
7. **Gating** — Pipeline fails on leaked secrets; configurable CVE thresholds
8. **Artifact Retention** — Reports stored for 30–90 days for audit trail

## 📊 Viewing Results

### GitHub Security Tab
All SARIF reports automatically appear under **Security → Code scanning alerts**.

### Workflow Summary
Each run generates a rich markdown summary accessible from the **Actions** tab.

### Downloadable Artifacts
Detailed JSON/HTML/SARIF reports are available as downloadable artifacts on each workflow run.

## ⚙️ Customization

### Adjusting Trivy Severity
Edit the `TRIVY_SEVERITY` env in the workflow:
```yaml
TRIVY_SEVERITY: "CRITICAL,HIGH"  # Change to "CRITICAL" for stricter gating
```

### Adding OWASP DC Suppressions
Edit `.github/owasp-suppressions.xml`:
```xml
<suppress>
  <notes>False positive — we don't use feature X.</notes>
  <cve>CVE-2024-XXXXX</cve>
</suppress>
```

### Configuring ZAP Rules
Edit `.github/zap-rules.tsv` — set rules to `FAIL`, `WARN`, or `IGNORE`.

### Adjusting OWASP DC CVE Threshold
In the workflow, change the `--failOnCVSS` value (default: `7`):
```yaml
args: >-
  --failOnCVSS 9    # Only fail on Critical CVEs
```
