# P2: recipe only, not built. P3 should record an immutable base-image digest.
FROM python:3.11.14-slim-bookworm AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
COPY requirements-container.txt /tmp/requirements-container.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements-container.txt \
    && /opt/venv/bin/pip check

FROM python:3.11.14-slim-bookworm
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PORT=8080 TORCH_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    HF_HOME=/tmp/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    SAM_CHECKPOINT_PATH=/models/sam_vit_b_01ec64.pth \
    BIREFNET_MODEL_PATH=/models/birefnet CORS_ORIGINS=""
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app \
    && mkdir /app /models && chown app:app /app /models
COPY --from=dependencies /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app main.py alpha_preservation.py runtime_config.py server_safety.py inference_runtime.py run_server.py ./
USER app
EXPOSE 8080
CMD ["python", "run_server.py"]
# Weights are intentionally not downloaded/copied here. Supply a read-only /models
# mount for P3; missing SAM checkpoint intentionally fails startup.
