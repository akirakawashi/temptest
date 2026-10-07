"""Последовательный клиент рабочего Whisper: без повторов и управления сервером."""
from __future__ import annotations

import ipaddress
import json
import logging
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx

log = logging.getLogger("whisper-api")


def validate_url(value: str) -> str:
    url = urlsplit(value.rstrip("/"))
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ValueError("Whisper: нужен HTTP(S)-адрес без учётных данных")
    if url.query or url.fragment or url.path.rstrip("/") != "/v1":
        raise ValueError("Whisper: адрес должен оканчиваться на /v1, без query и fragment")
    try:
        address = ipaddress.ip_address(url.hostname)
        local = (address.is_private or address.is_loopback) and not (address.is_unspecified or address.is_multicast)
    except ValueError:
        local = url.hostname == "localhost"
    if not local:
        raise ValueError("Разрешён только локальный IP Whisper; облачные адреса и DNS-имена запрещены")
    return value.rstrip("/")


def speech_seconds(segments: list, duration: float) -> float:
    intervals = []
    for segment in segments:
        if not isinstance(segment, dict) or not str(segment.get("text", "")).strip():
            continue
        try:
            start, end = max(0.0, float(segment["start"])), min(duration, float(segment["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            intervals.append((start, end))
    total, last = 0.0, 0.0
    for start, end in sorted(intervals):
        total += max(0.0, end - max(start, last))
        last = max(last, end)
    return total


class WhisperAPI:
    def __init__(self, base_url: str, key: str, model: str, timeout: float, transport=None):
        self.base_url = validate_url(base_url)
        if not key:
            raise ValueError("Задайте WHISPER_API_KEY в .env на сервере; ключ не пишется в отчёт")
        self.key, self.model = key, model
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(timeout, connect=5.0, pool=5.0),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            transport=transport or httpx.AsyncHTTPTransport(retries=0),
            trust_env=False, follow_redirects=False,
        )

    async def close(self):
        await self.client.aclose()

    async def check(self) -> dict:
        health = await self.health()
        models = await self.client.get(self.base_url + "/models", timeout=10)
        models.raise_for_status()
        payload = models.json()
        if self.model not in [item.get("id") for item in payload.get("data", [])]:
            raise RuntimeError(f"Рабочий Whisper не объявляет модель {self.model}; менять/загружать её стенд не будет")
        log.info("Whisper API: доступность и авторизация проверены, модель %s", self.model)
        return {"health": health, "models": payload, "checked_at": time.time()}

    async def health(self) -> dict:
        origin = self.base_url.rsplit("/v1", 1)[0]
        response = await self.client.get(origin + "/health", timeout=10)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise RuntimeError("Whisper не подтвердил готовность; аудио не отправляется")
        return payload

    async def run(self, path: Path, duration: float) -> dict:
        parameters = {"model": self.model, "language": "ru", "response_format": "verbose_json", "vad_filter": "true"}
        log.info("Whisper API: один POST, файл %s, байт %d, параметры %s; повторов нет", path.name, path.stat().st_size, parameters)
        started = time.perf_counter()
        result = {"status": "error", "text": "", "segments": [], "error": None,
                  "mode": "existing_server_api", "endpoint": self.base_url + "/audio/transcriptions",
                  "request_parameters": parameters, "attempts": 1,
                  "timer_boundary": "HTTP-запрос: передача WAV, очередь, обработка, получение ответа",
                  "server_may_still_be_processing": False}
        try:
            with path.open("rb") as audio:
                async with self.client.stream("POST", result["endpoint"], data=parameters,
                                              files={"file": (path.name, audio, "audio/wav")}) as response:
                    result["http_status"] = response.status_code
                    result["response_headers"] = {k: v for k, v in response.headers.items()
                                                  if k in {"content-type", "content-length", "x-request-id"}}
                    body = bytearray()
                    async for block in response.aiter_bytes():
                        body.extend(block)
                        if len(body) > 16 * 1024 ** 2:
                            raise ValueError("Ответ Whisper превышает 16 МиБ; запрос не повторяется")
            elapsed = time.perf_counter() - started
            result["response_bytes"] = len(body)
            # Даже ошибочный ответ не должен сохранить секрет, если сервер его отразил.
            text = bytes(body).decode("utf-8", "replace").replace(self.key, "[КЛЮЧ СКРЫТ]")
            if response.status_code != 200:
                result["response_error"] = text
                raise RuntimeError(f"HTTP {response.status_code}; последующие запросы и GigaAM остановлены")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                result["response_error"] = text
                raise
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                result["response_error"] = text
                raise ValueError("Whisper вернул некорректный verbose_json")
            segments = payload.get("segments", [])
            if not isinstance(segments, list):
                raise ValueError("Whisper: segments должен быть списком")
            result.update(status="ok", text=payload["text"], segments=segments, response=payload,
                          elapsed_seconds=elapsed, speech_seconds=speech_seconds(segments, duration),
                          server_processing_seconds=None)
        except Exception as exc:
            result["elapsed_seconds"] = time.perf_counter() - started
            result["error"] = f"{type(exc).__name__}: {exc}".replace(self.key, "[КЛЮЧ СКРЫТ]")
            result["server_may_still_be_processing"] = isinstance(exc, httpx.TransportError)
            log.error("Whisper API: %s. Автоматического повтора нет; после обрыва сервер может продолжать обработку", result["error"])
        return result
