"""Flow записи: поиск речи → распознавание → паузы; в полном — ещё голоса и эмоции. Модели подключает движок."""
from __future__ import annotations

import asyncio
import gc
import threading
import time
from typing import Callable

import numpy as np

from app.config import SAMPLE_RATE, Settings
from app.session import Session, tokens_to_words, words_text

STAGES = {"vad": "Выделение речи", "asr": "Распознавание речи"}
FULL_STAGES = {**STAGES, "speaker": "Разделение собеседников", "emotion": "Определение эмоций"}


class StageTimer:
    def __init__(self, stages=STAGES):
        self.stages = tuple(stages)
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.values = {stage: {"seconds": 0.0, "calls": 0, "errors": 0} for stage in self.stages}
            self.details = []
            self.segments = []
            self.origin = None

    def measure(self, stage: str, call: Callable, *, audio_seconds: float | None = None):
        started = time.perf_counter()
        failed = False
        detail = {"stage": stage, "started_clock": started, "audio_seconds": audio_seconds,
                  "thread": threading.current_thread().name}
        try:
            result = call()
            if stage == "asr":
                detail.update(text=result[0], tokens=list(result[1]), token_timestamps=list(result[2]))
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
        return self._remember(self.timer.measure("vad", lambda: self.vad.accept(samples),
                                                audio_seconds=len(samples) / SAMPLE_RATE))

    def flush(self):
        return self._remember(self.timer.measure("vad", self.vad.flush))

    def _remember(self, segments):
        for seg in segments:
            self.timer.segments.append({"start": seg.start / SAMPLE_RATE, "end": seg.end / SAMPLE_RATE,
                "voice_start": (seg.start + seg.lead) / SAMPLE_RATE,
                "voice_end": (seg.end - seg.trail) / SAMPLE_RATE,
                "lead_seconds": seg.lead / SAMPLE_RATE, "trail_seconds": seg.trail / SAMPLE_RATE,
                "probabilities": [round(float(p), 4) for p in seg.probs]})
        return segments


def wall_time_breakdown(calls: list[dict], elapsed: float, stages=STAGES) -> dict:
    """Разбивка 100% времени: пересечения вынесены отдельно, двойного счёта нет."""
    boundaries = [(0.0, None, 0), (elapsed, None, 0)]
    buckets = {key: 0.0 for key in (*stages, "parallel", "other")}
    for call in calls:
        a = max(0.0, min(elapsed, call["offset_seconds"]))
        b = max(a, min(elapsed, a + call["seconds"]))
        boundaries.extend([(a, call["stage"], 1), (b, call["stage"], -1)])
    active = {}
    previous = 0.0
    for at, stage, change in sorted(boundaries, key=lambda entry: entry[0]):
        stages = [key for key, count in active.items() if count]
        key = stages[0] if len(stages) == 1 else "parallel" if stages else "other"
        buckets[key] += at - previous
        if stage is not None:
            active[stage] = active.get(stage, 0) + change
        previous = at
    return {"exclusive_wall_seconds": buckets,
            "exclusive_wall_percent": {key: seconds / elapsed * 100 if elapsed else 0 for key, seconds in buckets.items()},
            "method": "Интервалы perf_counter; пересечение этапов — parallel, работа между вызовами — other; сумма 100%"}


def vad_pauses(segments: list[dict], audio_seconds: float) -> list[dict]:
    """Промежутки вне VAD-речи, включая начало и конец; это оценка детектора."""
    pauses, previous = [], 0.0
    for seg in segments:
        start = min(audio_seconds, max(0.0, seg["voice_start"]))
        end = min(audio_seconds, max(start, seg["voice_end"]))
        if start > previous:
            pauses.append({"start": previous, "end": start, "seconds": start - previous})
        previous = max(previous, end)
    if previous < audio_seconds:
        pauses.append({"start": previous, "end": audio_seconds, "seconds": audio_seconds - previous})
    return pauses


class FirstLineSession(Session):
    """Тот же поток VAD/очередь ASR, без вызовов голоса и эмоций."""

    def _live_guess(self) -> None:
        pass

    def _process(self, seg, clusterer, split_turns, prev_end, prev_speaker) -> dict:
        started = time.perf_counter()
        text, tokens, stamps = self.eng.transcribe(seg.samples)
        if not any(ch.isalnum() for ch in text):
            return {"turns": [], "merges": []}
        duration = len(seg.samples) / SAMPLE_RATE
        words = tokens_to_words(tokens, stamps, duration)
        voice_start = (seg.start + seg.lead) / SAMPLE_RATE
        voice_end = (seg.end - seg.trail) / SAMPLE_RATE
        return {"merges": [], "turns": [{"speaker": None, "emotion": None, "interrupted": None,
            "start": seg.start / SAMPLE_RATE, "end": seg.end / SAMPLE_RATE,
            "voice_start": voice_start, "voice_end": voice_end,
            "gap": None if prev_end is None else max(0.0, voice_start - prev_end),
            "text": words_text(words) if words else text.strip(), "words": len(words),
            "word_timestamps": [{"text": w["w"], "start": seg.start / SAMPLE_RATE + w["s"],
                                 "end": seg.start / SAMPLE_RATE + w["e"]} for w in words],
            "asr_ms": (time.perf_counter() - started) * 1000, "diar_ms": 0, "_audio": seg.samples}]}


class FullSession(Session):
    """Исходная сессия монитора целиком: текст, смена говорящего, голоса, эмоции."""

    def _live_guess(self) -> None:
        # Подсказка «кто говорит сейчас» нужна живому микрофону; на реплики она не влияет.
        pass


class FirstLinePipeline:
    """Подаёт запись движку как поток и собирает реплики, времена этапов и паузы."""

    session_type = FirstLineSession
    event_types = frozenset({"utterance", "error"})

    def __init__(self, cfg: Settings, engines, timeout: float):
        self.cfg, self.engines, self.timeout = cfg, engines, timeout
        self.events: list[dict] = []

    def verify(self, session: Session) -> None:
        """Проверка готового разговора до конца замера; урезанному flow проверять нечего."""

    def details(self, session: Session) -> dict:
        return {}

    async def run(self, samples: np.ndarray) -> dict:
        timer = self.engines.timer
        timer.reset()
        self.events = []
        session = None
        status, error = "ok", None
        started = time.perf_counter()
        timer.origin = started

        async def emit(event: dict) -> None:
            # Уровень сигнала и промежуточные счётчики интерфейса не нужны в отчёте.
            if event.get("type") in self.event_types:
                self.events.append({"elapsed_seconds": time.perf_counter() - started, **event})

        try:
            async with asyncio.timeout(self.timeout):
                session = self.session_type(self.engines, self.cfg, emit)
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
                self.verify(session)
                if not session.utterances:
                    status = "no_speech"
        except TimeoutError:
            status, error = "timeout", f"Обработка записи превысила {self.timeout:g} с"
        except asyncio.CancelledError:
            if session:
                await session.close()
            raise
        except Exception as exc:
            status, error = "error", f"Ошибка обработки записи ({type(exc).__name__}): {exc}"
        elapsed = time.perf_counter() - started
        # Снимки результата и запись файлов находятся за границей замера.
        stages = timer.snapshot()
        utterances = list(session.utterances) if session else []
        details = self.details(session) if session else {}
        counters = session.counters() if session else {}
        calls = timer.trace()
        speech_segments = list(timer.segments)
        pauses = vad_pauses(speech_segments, len(samples) / SAMPLE_RATE)
        if session:
            await session.close()
        return {
            "status": status, "elapsed_seconds": elapsed, "asr_seconds": stages["asr"]["seconds"],
            "enabled_stages": list(timer.stages), "stages": stages, "stage_calls": calls,
            "text": "\n".join(u["text"] for u in utterances),
            "utterances": utterances, "events": list(self.events), "error": error,
            "counters": counters, "speech_segments": speech_segments, "pauses": pauses,
            "pause_seconds": sum(p["seconds"] for p in pauses),
            "timing": wall_time_breakdown(calls, elapsed, timer.stages), **details,
        }

    def shutdown(self) -> None:
        # Отмена asyncio не останавливает нативный inference: ждём завершения потоков.
        for pool in (self.engines.asr_pool, self.engines.emo_pool, self.engines.live_pool):
            pool.shutdown(wait=True, cancel_futures=True)
        for name in ("_asr", "_spk", "_emo", "_vad_sess"):
            setattr(self.engines, name, None)
        gc.collect()


class FullPipeline(FirstLinePipeline):
    """Тот же flow с голосами собеседников и эмоциями; языковой модели нет."""

    session_type = FullSession
    event_types = frozenset({"utterance", "emotion", "relabel", "error"})

    def verify(self, session: Session) -> None:
        missing = [u["id"] for u in session.utterances if u["end"] - u["start"] >= 0.5 and u.get("emotion") is None]
        if missing:
            raise RuntimeError(f"Не получены эмоции для реплик: {missing}")

    def details(self, session: Session) -> dict:
        utterances = session.utterances
        return {"dialog": session.transcript(), "speakers": len({u["speaker"] for u in utterances}),
                "emotions_skipped_short": sum(u.get("emotion") is None for u in utterances)}
