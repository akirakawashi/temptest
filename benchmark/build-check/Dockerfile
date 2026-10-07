# Контроль: начало рабочего Dockerfile владельца, до копирования кода worker.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
