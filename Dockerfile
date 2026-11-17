# FleetGuard: one image for the API, the dashboard, MLflow and the training pipeline.
# The service to run is chosen by the command (see docker-compose.yml).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    FLEETGUARD_HOME=/app

# libgomp: OpenMP runtime needed by LightGBM and XGBoost; curl: container health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes don't reinstall them (layer caching).
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY configs ./configs
COPY dashboard ./dashboard
COPY data/raw ./data/raw

# Run as an unprivileged user; state folders are mounted as volumes.
RUN useradd --create-home --uid 1000 fleetguard \
    && mkdir -p models outputs mlruns data/incoming \
    && chown -R fleetguard:fleetguard /app
USER fleetguard

EXPOSE 8000 8501 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fs http://127.0.0.1:8000/health || exit 1

CMD ["fleetguard", "serve", "--host", "0.0.0.0", "--port", "8000"]
