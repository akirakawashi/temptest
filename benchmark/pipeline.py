"""Файловые адаптеры. Модели загружаются только явным вызовом load()."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import importlib.metadata
import gc
import logging
import threading
import time
from typing import Callable

import numpy as np

from app.config import SAMPLE_RATE, Settings
from benchmark.gpu import CudaEngines, configure_whisper_vad
from app.llm import LLM
from app.session import Session
from benchmark.llm_settings import llm_runtime

log = logging.getLogger("сравнение")

STAGES = {
    "vad": "Выделение речи",
    "asr": "Распознавание речи",
    "speaker": "Разделение собеседников",
    "emotion": "Определение эмоций",
}


class StageTimer:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.values = {stage: {"seconds": 0.0, "calls": 0, "errors": 0} for stage in STAGES}
            self.details = []
            self.origin = None

    def measure(self, stage: str, call: Callable, *, audio_seconds: float | None = None):
        started = time.perf_counter()
        failed = False
        detail = {"stage": stage, "started_clock": started, "audio_seconds": audio_seconds,
                  "thread": threading.current_thread().name}
        try:
            result = call()
            if stage == "asr":
                detail.update(text=result[0], tokens=len(result[1]))
            elif stage == "emotion":
                detail["probabilities"] = result
            return result
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = time.perf_counter() - started
            with self.lock:
                entry = self.values[stage]
                entry["seconds"] += elapsed
                entry["calls"] += 1
                entry["errors"] += int(failed)
                if stage != "vad":
                    self.details.append({**detail, "seconds": elapsed, "error": failed})

    def snapshot(self) -> dict:
        with self.lock:
            return {stage: dict(values) for stage, values in self.values.items()}

    def trace(self) -> list[dict]:
        with self.lock:
            ordered = sorted(self.details, key=lambda item: item["started_clock"])
            base = self.origin if self.origin is not None else ordered[0]["started_clock"] if ordered else 0
            return [{**{k: v for k, v in item.items() if k != "started_clock"},
                     "offset_seconds": item["started_clock"] - base} for item in ordered]


class TimedVad:
    def __init__(self, vad, timer: StageTimer):
        self.vad, self.timer = vad, timer

    def __getattr__(self, name):
        return getattr(self.vad, name)

    def accept(self, samples):
        return self.timer.measure("vad", lambda: self.vad.accept(samples))

    def flush(self):
        return self.timer.measure("vad", self.vad.flush)


class TimedEngines(CudaEngines):
    def __init__(self, cfg: Settings):
        self.timer = StageTimer()
        super().__init__(cfg)

    def new_vad(self):
        return TimedVad(super().new_vad(), self.timer)

    def transcribe(self, samples):
        return self.timer.measure("asr", lambda: super(TimedEngines, self).transcribe(samples), audio_seconds=len(samples) / SAMPLE_RATE)

    def embed(self, samples):
        return self.timer.measure("speaker", lambda: super(TimedEngines, self).embed(samples), audio_seconds=len(samples) / SAMPLE_RATE)

    def emotions(self, samples):
        return self.timer.measure("emotion", lambda: super(TimedEngines, self).emotions(samples), audio_seconds=len(samples) / SAMPLE_RATE)


class WhisperPipeline:
    def __init__(self, model: str, cache: str, threads: int, beam_size: int = 5):
        self.model_name, self.cache = model, cache
        self.threads, self.beam_size = threads, beam_size
        self.model = None
        self.vad_providers = []

    def load(self) -> None:
        log.info("Whisper: импорт faster-whisper")
        from faster_whisper import WhisperModel

        log.info("Whisper VAD: начало загрузки Silero ONNX на CUDA")
        self.vad_providers = configure_whisper_vad()
        log.info("Whisper VAD: загрузка завершена, провайдеры %s", self.vad_providers)
        log.info("Whisper ASR: начало загрузки %s на CUDA/float16", self.model_name)
        self.model = WhisperModel(
            self.model_name, device="cuda", device_index=0, compute_type="float16", cpu_threads=self.threads,
            num_workers=1, download_root=self.cache,
        )
        log.info("Whisper ASR: загрузка завершена")

    def run(self, samples: np.ndarray) -> dict:
        """Таймер охватывает и transcribe(), и ленивую генерацию ВСЕХ сегментов."""
        segments, error = [], None
        started = time.perf_counter()
        try:
            iterator, info = self.model.transcribe(
                samples, language="ru", task="transcribe", beam_size=self.beam_size,
                temperature=0.0, vad_filter=True, word_timestamps=True,
            )
            for segment in iterator:
                segments.append(asdict(segment))
            text = " ".join(segment["text"].strip() for segment in segments).strip()
        except Exception as exc:
            error = f"Ошибка распознавания Whisper ({type(exc).__name__}): {exc}"
            text = " ".join(segment["text"].strip() for segment in segments).strip()
        elapsed = time.perf_counter() - started
        return {
            "status": "error" if error else "ok", "elapsed_seconds": elapsed,
            "text": text, "segments": segments, "error": error,
            "language": "ru", "model": self.model_name, "device": "cuda:0", "compute_type": "float16",
            "vad_providers": self.vad_providers,
        }

    def shutdown(self) -> None:
        if self.model is not None:
            self.model.model.unload_model()
            self.model = None
        gc.collect()


class GigaPipeline:
    def __init__(self, cfg: Settings, threads: int, timeout: float, seed: int = 42):
        self.cfg, self.threads, self.timeout, self.seed = cfg, threads, timeout, seed
        self.llm_runtime = llm_runtime(threads, cfg, seed)
        self.engines = TimedEngines(cfg)
        self.llm = LLM(cfg, self.engines)
        self.events: list[dict] = []

    async def load(self) -> None:
        if not self.cfg.emo_enabled or not self.cfg.llm_enabled:
            raise RuntimeError("Для полного прогона GigaAM обязательны эмоции и LLM")
        await asyncio.to_thread(self.engines.load_core)
        if not self.engines.core_ready:
            raise RuntimeError(f"Не загрузились основные модели GigaAM: {self.engines.status()}")
        await asyncio.to_thread(self.engines.load_emotions)
        if not self.engines.emotions_ready:
            raise RuntimeError(f"Не загрузилась модель эмоций: {self.engines.status()['emo']['detail']}")
        # prepare() может ждать сервер бесконечно; у стенда ожидание ограничено.
        log.info("GigaAM LLM: начало подготовки Ollama, модель %s", self.cfg.llm_model)
        log.info("GigaAM LLM: контекст %d, батч %d, запрошен KV-кеш %s и полная загрузка на GPU",
                 self.llm_runtime["options"]["num_ctx"], self.llm_runtime["options"]["num_batch"],
                 self.llm_runtime["kv_cache_type_requested"])
        if self.cfg.llm_autopull:
            await asyncio.wait_for(self.llm.prepare(), timeout=self.timeout)
        elif not await asyncio.wait_for(self.llm._prepare_once(), timeout=self.timeout):
            raise RuntimeError("LLM не подготовлена: запустите этап скачивания весов; внутри закрытого конвейера скачиваний нет")
        log.info("GigaAM LLM: подготовка завершена")

    async def run(self, samples: np.ndarray) -> dict:
        self.engines.timer.reset()
        self.events = []
        analysis = None
        llm_seconds = 0.0
        session = None
        status, error = "ok", None
        started = time.perf_counter()
        self.engines.timer.origin = started

        async def emit(event: dict) -> None:
            # Уровень сигнала и промежуточные счётчики интерфейса не нужны в отчёте.
            if event.get("type") in {"utterance", "emotion", "relabel", "error", "llm"}:
                self.events.append({"elapsed_seconds": time.perf_counter() - started, **event})

        try:
            async with asyncio.timeout(self.timeout):
                session = Session(self.engines, self.cfg, emit)
                session.start()
                # Вход тот же, что у микрофона. Скорость подачи не привязана к часам.
                pcm = (np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2")
                for offset in range(0, pcm.size, 1600):
                    await session.feed(pcm[offset:offset + 1600].tobytes())
                    await asyncio.sleep(0)  # даём выполняться исходным рабочим задачам
                await session.flush()
                await session.wait_idle()
                failures = [e["message"] for e in self.events if e["type"] == "error"]
                if failures:
                    raise RuntimeError("; ".join(failures))
                missing = [u["id"] for u in session.utterances
                           if u["end"] - u["start"] >= 0.5 and u.get("emotion") is None]
                if missing:
                    raise RuntimeError(f"Не получены обязательные эмоции для реплик: {missing}")
                if not session.utterances:
                    status = "no_speech"
                else:
                    llm_started = time.perf_counter()
                    try:
                        await self.llm.analyze(
                            session.transcript(), emit,
                            options=self.llm_runtime["options"],
                        )
                    finally:
                        llm_seconds = time.perf_counter() - llm_started
                    analysis = next((e for e in reversed(self.events) if e["type"] == "llm"), None)
                    if not analysis or analysis.get("state") != "done":
                        raise RuntimeError((analysis or {}).get("message", "LLM не завершила разбор"))
                    if not analysis.get("completed") or not isinstance(analysis.get("result"), dict):
                        raise RuntimeError("LLM не вернула завершённый JSON-объект; исходный ответ сохранён")
        except TimeoutError:
            status, error = "timeout", f"Полный цикл GigaAM превысил {self.timeout:g} с"
        except asyncio.CancelledError:
            if session:
                await session.close()
            raise
        except Exception as exc:
            status, error = "error", f"Ошибка полного цикла GigaAM ({type(exc).__name__}): {exc}"
        elapsed = time.perf_counter() - started
        # Снимки результата и запись файлов находятся за границей замера.
        stages = self.engines.timer.snapshot()
        utterances = list(session.utterances) if session else []
        transcript = session.transcript() if session else ""
        if session:
            await session.close()
        return {
            "status": status, "elapsed_seconds": elapsed, "asr_seconds": stages["asr"]["seconds"],
            "stages": stages, "stage_calls": self.engines.timer.trace(), "llm_seconds": llm_seconds, "text": transcript,
            "utterances": utterances, "analysis": analysis, "events": list(self.events), "error": error,
            "emotions_skipped_short": sum(u.get("emotion") is None for u in utterances),
        }

    async def gpu_model_info(self) -> list[dict]:
        """Проверяем фактическую загрузку LLM после прогрева, а не только настройки."""
        import httpx

        async with httpx.AsyncClient(base_url=self.cfg.ollama_url, timeout=30.0) as client:
            response = await client.get("/api/ps")
            response.raise_for_status()
            models = response.json().get("models", [])
        matching = [m for m in models if m.get("name") == self.cfg.llm_model or m.get("model") == self.cfg.llm_model]
        if not matching:
            raise RuntimeError("После прогрева LLM отсутствует в списке загруженных моделей Ollama")
        if any(not isinstance(m.get("size_vram"), (int, float)) or not isinstance(m.get("size"), (int, float))
               or m["size_vram"] <= 0 or m["size"] <= 0 or m["size_vram"] != m["size"] for m in matching):
            raise RuntimeError("Ollama не загрузила LLM целиком на GPU. Освободите видеопамять; CPU/частичная загрузка не допускаются")
        return matching

    async def unload_llm(self) -> None:
        """Освободить VRAM до переключения на Whisper; вне замеров."""
        import httpx

        async with asyncio.timeout(60), httpx.AsyncClient(base_url=self.cfg.ollama_url, timeout=30) as client:
            response = await client.post("/api/generate", json={"model": self.cfg.llm_model, "keep_alive": 0})
            response.raise_for_status()
            while True:
                response = await client.get("/api/ps")
                response.raise_for_status()
                if not response.json().get("models"):
                    return
                await asyncio.sleep(0.2)

    def shutdown(self) -> None:
        # Отмена asyncio не останавливает нативный inference: ждём завершения потоков.
        for pool in (self.engines.asr_pool, self.engines.emo_pool, self.engines.live_pool):
            pool.shutdown(wait=True, cancel_futures=True)
        for name in ("_asr", "_spk", "_emo", "_vad_sess"):
            setattr(self.engines, name, None)
        gc.collect()


def versions() -> dict:
    result = {}
    for name in ("faster-whisper", "ctranslate2", "sherpa-onnx", "onnxruntime", "onnxruntime-gpu", "numpy", "torch", "gigaam"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "не установлен"
    return result
