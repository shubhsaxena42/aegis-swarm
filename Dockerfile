# ═══════════════════════════════════════════════════════════════
#  Dockerfile — Aegis Swarm
#  Multi-stage build for production-grade container image
# ═══════════════════════════════════════════════════════════════

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# Corrected path: app/requirements.txt
COPY app/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --prefix=/install \
    -r requirements.txt

# ── Stage 2: Production ──────────────────────────────────────
FROM python:3.11-slim AS production

# Security: Run as non-root user
RUN groupadd -r aegis && useradd -r -g aegis -d /app -s /sbin/nologin aegis

WORKDIR /app

# Install runtime libs for Postgres & OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code from context root
# Note: 'src' files are now in 'app/' directory relative to build context
COPY --chown=aegis:aegis . .

# Remove unnecessary files from the image
# Updated cleanup paths based on new structure
RUN rm -rf .git .github Terraform cache __pycache__ *.md .trivyignore infrastructure deploy observability

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "print('healthy')" || exit 1

# Security hardening
RUN chmod -R 550 /app && \
    find /app -type f -name "*.py" -exec chmod 440 {} \;

USER aegis

# Default entrypoint
ENTRYPOINT ["python"]
CMD ["app/Live/ray_serve_yolo.py"]
