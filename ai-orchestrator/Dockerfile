# IntelliFlow AI Orchestrator - Multi-stage Docker Build

# Build stage
FROM python:3.11-slim AS builder

LABEL maintainer="IntelliFlow Team <platform@intelliflow.com>"
LABEL description="AI Orchestrator for IntelliFlow AI Platform"

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        pkg-config \
        libpq-dev \
        && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY src/main/python/requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/main/python/ ./

# Production stage
FROM python:3.11-slim AS production

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        netcat-traditional \
        git \
        libpq5 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r intelliflow && \
    useradd -r -g intelliflow -u 1000 -m intelliflow

# Set working directory
WORKDIR /app

# Create required directories
RUN mkdir -p /app/logs /app/models /app/experiments /app/config && \
    chown -R intelliflow:intelliflow /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder --chown=intelliflow:intelliflow /app ./

# Create health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8080/health || exit 1' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh && \
    chown intelliflow:intelliflow /app/healthcheck.sh

# Switch to non-root user
USER intelliflow

# Expose ports
EXPOSE 8080 8081

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    WORKERS=4 \
    HOST=0.0.0.0 \
    PORT=8080 \
    LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting AI Orchestrator..."\n\
\n\
# Start model management service in background\n\
python model_management_service.py &\n\
MODEL_SERVICE_PID=$!\n\
\n\
# Start experiment monitoring service in background\n\
python experiment_monitoring.py &\n\
MONITOR_SERVICE_PID=$!\n\
\n\
# Start training pipeline service\n\
python automated_training_pipeline.py &\n\
TRAINING_SERVICE_PID=$!\n\
\n\
# Wait for any service to exit\n\
wait $MODEL_SERVICE_PID $MONITOR_SERVICE_PID $TRAINING_SERVICE_PID' > /app/start.sh && \
    chmod +x /app/start.sh

# Start all services
ENTRYPOINT ["/app/start.sh"]

# Metadata
LABEL org.opencontainers.image.title="IntelliFlow AI Orchestrator"
LABEL org.opencontainers.image.description="AI/ML model management and orchestration service"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="IntelliFlow"
LABEL org.opencontainers.image.source="https://github.com/intelliflow/ai-platform"