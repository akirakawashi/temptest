#
# Речевой монитор — образ приложения.
# Собирается под linux/arm64 (Mac на Apple Silicon) и linux/amd64.
# Базовый образ можно заменить на зеркало: PYTHON_IMAGE в .env.

ARG PYTHON_IMAGE=python:3.13-slim

# ---------------------------------------------------------------- модели
FROM ${PYTHON_IMAGE} AS models
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl bzip2 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ARG MODELS_BASE_URL=https://github.com/k2-fsa/sherpa-onnx/releases/download
COPY scripts/download_models.sh /tmp/download_models.sh
RUN MODELS_BASE_URL="${MODELS_BASE_URL}" sh /tmp/download_models.sh /models

# ----------------------------------------------------------- зависимости
FROM ${PYTHON_IMAGE} AS builder
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# PyTorch нужен только для модели эмоций. На arm64 пакет из PyPI и так без CUDA;
# на amd64 берём облегчённую сборку «только CPU», иначе образ вырастет на несколько гигабайт.
ARG TARGETARCH
ARG TORCH_VERSION=2.10.0
RUN if [ "${TARGETARCH}" = "amd64" ]; then \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu \
            "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}"; \
    else \
        pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}"; \
    fi

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Официальный пакет GigaAM, зафиксированный на проверенном коммите (август 2026).
ARG GIGAAM_REF=7447938d791c4f3e643386ee22c33777004293a5
RUN pip install "gigaam @ git+https://github.com/salute-developers/GigaAM.git@${GIGAAM_REF}"

# ------------------------------------------------------------ приложение
FROM ${PYTHON_IMAGE} AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_DIR=/models \
    EMO_CACHE=/cache/gigaam \
    HOME=/tmp

RUN useradd --system --uid 10001 --no-create-home app \
 && mkdir -p /cache/gigaam \
 && chown -R app /cache

COPY --from=builder /opt/venv /opt/venv
COPY --from=models /models /models
WORKDIR /srv
COPY app /srv/app

USER app
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import json,sys,urllib.request; sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8080/api/status', timeout=4))['ok'] else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
