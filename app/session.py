"""Одна живая сессия: поток звука с микрофона -> реплики с собеседниками и эмоциями.

Всё состояние (текст, векторы голосов, счётчики) хранится только в памяти и
пропадает при закрытии соединения — на диск ничего не записывается.
"""
from __future__ import annotations

import asyncio
import bisect
import logging
import re
import time
from collections import deque
from typing import Awaitable, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import SAMPLE_RATE, Settings
from .diarizer import OnlineSpeakerClusterer, find_turns
from .engines import Engines
from .vad import FRAME_SEC, SpeechSegment, find_pauses

log = logging.getLogger("session")

Send = Callable[[dict], Awaitable[None]]

WIN_SEC = 1.5          # окно для поиска смены говорящего внутри фрагмента
HOP_SEC = 0.5
SPLIT_MIN_SEC = 3.2    # фрагменты короче не делим
INTERRUPT_GAP = 0.30   # пауза короче — считаем, что собеседника перебили


def tokens_to_words(tokens: List[str], stamps: List[float], duration: float) -> List[dict]:
    """Собрать слова из токенов модели. Токен, начинающийся с пробела, открывает слово."""
    words: List[dict] = []
    for tok, ts in zip(tokens, stamps):
        if not tok:
            continue
        starts = tok[0] in (" ", "▁")
        if starts or not words:
            words.append({"w": tok.lstrip(" ▁"), "s": float(ts), "e": float(ts), "raw": [tok]})
        else:
            words[-1]["w"] += tok
            words[-1]["raw"].append(tok)
        words[-1]["e"] = float(ts)
    for i, w in enumerate(words):
        nxt = words[i + 1]["s"] if i + 1 < len(words) else duration
        # слово длится до начала следующего, но не дольше ~0.6 с после последнего токена
        w["e"] = round(min(max(w["e"] + 0.08, min(nxt, w["e"] + 0.6)), duration), 3)
        w["s"] = round(w["s"], 3)
    return words


def words_text(words: List[dict]) -> str:
    text = "".join("".join(w["raw"]) for w in words).replace("▁", " ").strip()
    text = text.lstrip("—–- ")            # тире диалога в начале реплики не нужно
    text = re.sub(r"\s+[—–-]+$", "", text)  # и в конце: оно относится уже к следующей реплике
    return text[:1].upper() + text[1:] if text else text


def refine_cuts(
    rough: List[float], words: List[dict], probs: np.ndarray, dur: float, pause_threshold: float
) -> List[Tuple[float, float]]:
    """Уточнить точки смены говорящего.

    rough — примерные границы по голосу (секунды). Для каждой ищем настоящую паузу
    рядом (по вероятностям детектора речи): тогда левая реплика кончается в начале
    паузы, правая начинается в её конце. Если паузы нет (говорят без остановки или
    одновременно) — режем между словами, и пауза считается нулевой.
    """
    if not rough or len(words) < 2:
        return []
    pauses = find_pauses(probs, pause_threshold)
    out: List[Tuple[float, float]] = []
    last = 0.3
    for c in sorted(rough):
        best: Optional[Tuple[float, float]] = None
        best_cost = 1e9
        for a, b in pauses:
            mid = (a + b) / 2
            cost = abs(mid - c) - 1.0 * (b - a)
            if abs(mid - c) <= 1.2 and a > last and b < dur - 0.3 and cost < best_cost:
                best, best_cost = (a, b), cost
        if best is None:
            for i in range(len(words) - 1):
                mid = (words[i]["e"] + words[i + 1]["s"]) / 2
                cost = abs(mid - c)
                if cost <= 1.0 and mid > last and mid < dur - 0.3 and cost < best_cost:
                    best, best_cost = (mid, mid), cost
        if best is None or (out and best == out[-1]):
            continue
        out.append((round(best[0], 3), round(best[1], 3)))
        last = best[1] + 0.3
    return out


class Session:
    def __init__(self, engines: Engines, cfg: Settings, send: Send):
        self.eng = engines
        self.cfg = cfg
        self.send = send
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker: Optional[asyncio.Task] = None
        self._pending_models: set[asyncio.Future] = set()
        self._pending_events: set[asyncio.Task] = set()
        self.speaker_threshold = cfg.speaker_threshold
        self.split_turns = cfg.split_turns
        self.names: Dict[int, str] = {}
        self._reset_state()

    # ----------------------------------------------------------------- state
    def _reset_state(self) -> None:
        self.vad = self.eng.new_vad() if self.eng.components["vad"].state == "ready" else None
        self.clusterer = OnlineSpeakerClusterer(self.speaker_threshold, self.cfg.max_speakers)
        self.utterances: List[dict] = []
        self.next_id = 1
        self.samples_in = 0
        self.speech_samples = 0
        self.chunks_in = 0
        self.bytes_in = 0
        self.empty_segments = 0
        self.busy = False
        self.speaking = False
        self.speech_started_at = 0.0
        self.live_busy = False
        self.last_live = 0.0
        self.last_level_sent = 0.0
        self.peak = 0.0
        self.clipped = 0
        self.clock: Deque[Tuple[int, float]] = deque(maxlen=4000)   # (отсчёт, время прихода)
        self.started_wall = time.time()
        self.epoch = getattr(self, "epoch", 0) + 1                   # растёт при сбросе

    async def reset(self) -> None:
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        self.names.clear()
        self._reset_state()
        await self.send({"type": "reset"})

    def start(self) -> None:
        if self.worker is None:
            self.worker = asyncio.create_task(self._consume())

    async def wait_idle(self) -> None:
        """Дождаться всех реплик, фоновых моделей и доставки их событий.

        Для файлового входа: остановка подачи звука сама по себе не означает,
        что распознавание и эмоции уже закончились. Новых порций во время
        ожидания подавать нельзя.
        """
        await self.queue.join()
        while self._pending_models or self._pending_events:
            pending = tuple(self._pending_models | self._pending_events)
            await asyncio.gather(*pending, return_exceptions=True)
            # На уже завершённых задачах gather может вернуть сразу, прежде чем
            # их done-callback уберёт задачу из набора. Даём callbacks выполниться.
            await asyncio.sleep(0)

    def _emit_later(self, payload: dict) -> None:
        task = asyncio.create_task(self.send(payload))
        self._pending_events.add(task)
        task.add_done_callback(self._pending_events.discard)

    async def close(self) -> None:
        if self.worker:
            self.worker.cancel()
            try:
                await self.worker
            except (asyncio.CancelledError, Exception):
                pass
        self.utterances.clear()
        self.clusterer = OnlineSpeakerClusterer()

    # ------------------------------------------------------------------ audio
    async def feed(self, pcm: bytes) -> None:
        """Принять очередную порцию звука: 16 кГц, моно, 16 бит."""
        if self.vad is None:
            if not self.eng.core_ready:
                return
            self.vad = self.eng.new_vad()
        if len(pcm) % 2:
            pcm = pcm[:-1]                      # обрывок отсчёта: 16-битный звук всегда чётной длины
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if x.size == 0:
            return
        now = time.time()
        self.samples_in += x.size
        self.chunks_in += 1
        self.bytes_in += len(pcm)
        self.clock.append((self.samples_in, now))
        peak = float(np.max(np.abs(x)))
        self.peak = max(self.peak, peak)
        if peak >= 0.985:
            self.clipped += 1

        for seg in self.vad.accept(x):
            await self._enqueue(seg)

        speaking = self.vad.speaking
        if speaking != self.speaking:
            self.speaking = speaking
            if speaking:
                self.speech_started_at = self.samples_in / SAMPLE_RATE
                self.last_live = now
            await self.send({"type": "live", "speaking": speaking, "t": round(self.samples_in / SAMPLE_RATE, 2)})
        elif speaking and now - self.last_live >= 1.0 and not self.live_busy:
            self.last_live = now
            self._live_guess()

        if now - self.last_level_sent >= 0.1:
            self.last_level_sent = now
            rms = float(np.sqrt(np.mean(x * x)) + 1e-9)
            await self.send(
                {
                    "type": "level",
                    "t": round(self.samples_in / SAMPLE_RATE, 2),
                    "db": round(20 * np.log10(rms), 1),
                    "peak_db": round(20 * np.log10(peak + 1e-9), 1),
                    "speaking": speaking,
                    "prob": round(self.vad.last_prob, 2),
                }
            )

    async def flush(self) -> None:
        """Микрофон остановлен: дожать незавершённый фрагмент речи."""
        if self.vad is None:
            return
        for seg in self.vad.flush():
            await self._enqueue(seg)
        if self.speaking:
            self.speaking = False
            await self.send({"type": "live", "speaking": False, "t": round(self.samples_in / SAMPLE_RATE, 2)})

    async def _enqueue(self, seg: SpeechSegment) -> None:
        self.speech_samples += seg.samples.size
        await self.queue.put((self.epoch, seg, self._wall_at(seg.end - seg.trail), time.time()))

    def _wall_at(self, sample: int) -> float:
        """Момент, когда отсчёт с данным номером пришёл на сервер."""
        if not self.clock:
            return time.time()
        idx = bisect.bisect_left(self.clock, (sample, 0.0))
        idx = min(idx, len(self.clock) - 1)
        return self.clock[idx][1]

    def _live_guess(self) -> None:
        """Пока человек ещё говорит — прикинуть по последним секундам, чей это голос."""
        samples = self.vad.current(2.5)
        if samples.size < int(1.2 * SAMPLE_RATE) or not self.clusterer.speakers:
            return
        tail = samples
        epoch, clusterer = self.epoch, self.clusterer
        self.live_busy = True

        def work() -> Optional[int]:
            emb = self.eng.embed(tail)
            if emb is None:
                return None
            sid, _ = clusterer.score(emb, len(tail) / SAMPLE_RATE)
            return sid

        fut = self.loop.run_in_executor(self.eng.live_pool, work)
        self._pending_models.add(fut)

        def done(f: "asyncio.Future") -> None:
            self._pending_models.discard(f)
            self.live_busy = False
            if f.cancelled() or f.exception() or epoch != self.epoch or not self.speaking:
                return
            self._emit_later({"type": "live", "speaking": True, "speaker": f.result(),
                              "t": round(self.samples_in / SAMPLE_RATE, 2)})

        fut.add_done_callback(done)

    # --------------------------------------------------------------- pipeline
    async def _consume(self) -> None:
        while True:
            epoch, seg, end_wall, queued_at = await self.queue.get()
            if epoch != self.epoch:
                self.queue.task_done()
                continue
            self.busy = True
            try:
                picked_at = time.time()
                prev_end, prev_speaker = self._last_turn()
                result = await self.loop.run_in_executor(
                    self.eng.asr_pool, self._process, seg, self.clusterer, self.split_turns, prev_end, prev_speaker
                )
                if epoch != self.epoch:
                    continue
                done_at = time.time()
                for merged_from, merged_to in result["merges"]:
                    for u in self.utterances:
                        if u["speaker"] == merged_from:
                            u["speaker"] = merged_to
                    self.names.pop(merged_from, None)      # номер освободился, имя с ним не переходит
                    await self.send({"type": "relabel", "from": merged_from, "to": merged_to})
                if not result["turns"]:
                    self.empty_segments += 1
                for turn in result["turns"]:
                    audio = turn.pop("_audio")
                    turn["id"] = self.next_id
                    self.next_id += 1
                    turn["latency_ms"] = int((done_at - end_wall) * 1000)
                    turn["queue_ms"] = int((picked_at - queued_at) * 1000)
                    self.utterances.append(turn)
                    await self.send({"type": "utterance", **turn})
                    if self.eng.emotions_ready and turn["end"] - turn["start"] >= 0.5:
                        self._schedule_emotion(turn["id"], audio, epoch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("segment processing failed")
                await self.send({"type": "error", "message": f"Ошибка обработки фрагмента: {exc}"})
            finally:
                self.busy = False
                self.queue.task_done()

    def _process(
        self,
        seg: SpeechSegment,
        clusterer: OnlineSpeakerClusterer,
        split_turns: bool,
        prev_end: Optional[float],
        prev_speaker: Optional[int],
    ) -> dict:
        """Тяжёлая часть (выполняется в потоке моделей): текст, слова, собеседники.

        Всё состояние сессии передаётся аргументами: если разговор сбросили, пока фрагмент
        обрабатывался, он не должен попасть в голоса нового разговора.
        """
        t0 = time.time()
        samples = seg.samples
        dur = samples.size / SAMPLE_RATE
        t_seg = seg.start / SAMPLE_RATE
        lead, trail = seg.lead / SAMPLE_RATE, seg.trail / SAMPLE_RATE

        text, tokens, stamps = self.eng.transcribe(samples)
        asr_ms = (time.time() - t0) * 1000
        if not any(ch.isalnum() for ch in text):
            return {"turns": [], "merges": []}        # шум: модель вернула пусто или одни знаки
        words = tokens_to_words(tokens, stamps, dur)

        # --- смена говорящего внутри фрагмента
        t1 = time.time()
        cuts: List[Tuple[float, float]] = []          # (конец левой реплики, начало правой)
        if split_turns and dur >= SPLIT_MIN_SEC:
            embs, centers = [], []
            pos = 0.0
            while pos + WIN_SEC <= dur + 1e-6:
                e = self.eng.embed(samples[int(pos * SAMPLE_RATE): int((pos + WIN_SEC) * SAMPLE_RATE)])
                if e is not None:
                    embs.append(e)
                    centers.append(pos + WIN_SEC / 2)
                pos += HOP_SEC
            rough = find_turns(embs, centers, dur, clusterer.threshold_for(WIN_SEC))
            cuts = refine_cuts([b for _, b in rough[:-1]], words, seg.probs, dur, self.cfg.vad_threshold - 0.15)

        # --- части фрагмента и их слова
        edges = [0.0] + [c for cut in cuts for c in cut] + [dur]      # a0, b0, a1, b1, ...
        pieces: List[dict] = []
        for k in range(0, len(edges), 2):
            a, b = edges[k], edges[k + 1]
            mid_lo = (edges[k - 1] + a) / 2 if k > 0 else -1.0
            mid_hi = (b + edges[k + 2]) / 2 if k + 2 < len(edges) else dur + 1.0
            part_words = [w for w in words if mid_lo <= (w["s"] + w["e"]) / 2 < mid_hi]
            if part_words:                                           # часть без слов — не реплика
                pieces.append({"a": a, "b": b, "words": part_words})
        if pieces:
            pieces[0]["a"], pieces[-1]["b"] = 0.0, dur               # крайние тянутся до границ фрагмента

        # --- кто говорит в каждой части
        merges: List[Tuple[int, int]] = []
        parts: List[dict] = []
        for k, pc in enumerate(pieces):
            a, b = pc["a"], pc["b"]
            voiced = (b - a) - (lead if k == 0 else 0.0) - (trail if k == len(pieces) - 1 else 0.0)
            chunk = samples[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)]
            res = clusterer.assign(self.eng.embed(chunk), max(0.05, voiced))
            for src, dst in res.merged:
                merges.append((src, dst))
                for p in parts:
                    if p["speaker"] == src:
                        p["speaker"] = dst
                if prev_speaker == src:
                    prev_speaker = dst
            if parts and parts[-1]["speaker"] == res.speaker:
                # соседние части оказались одним голосом — склеиваем обратно
                parts[-1]["b"] = b
                parts[-1]["words"].extend(pc["words"])
                parts[-1]["sim"] = max(parts[-1]["sim"], res.similarity)
                parts[-1]["confident"] = parts[-1]["confident"] or res.confident
                continue
            parts.append({"a": a, "b": b, "speaker": res.speaker, "sim": res.similarity,
                          "new": res.is_new, "confident": res.confident, "words": pc["words"]})
        diar_ms = (time.time() - t1) * 1000

        turns: List[dict] = []
        for k, p in enumerate(parts):
            abs_start = round(t_seg + p["a"], 2)
            abs_end = round(t_seg + p["b"], 2)
            # пауза считается по самой речи, без «полей» тишины вокруг фрагмента
            voice_start = abs_start + (lead if k == 0 else 0.0)
            voice_end = abs_end - (trail if k == len(parts) - 1 else 0.0)
            gap = None if prev_end is None else round(max(0.0, voice_start - prev_end), 2)
            interrupted = bool(
                prev_speaker is not None and prev_speaker != p["speaker"] and gap is not None and gap < INTERRUPT_GAP
            )
            chunk = samples[int(p["a"] * SAMPLE_RATE): int(p["b"] * SAMPLE_RATE)]
            rms = float(np.sqrt(np.mean(chunk * chunk)) + 1e-9) if chunk.size else 1e-9
            share = (p["b"] - p["a"]) / dur
            turns.append(
                {
                    "speaker": p["speaker"],
                    "new_speaker": p["new"],
                    "confident": p["confident"],
                    "similarity": round(p["sim"], 3),
                    "start": abs_start,
                    "end": abs_end,
                    "voice_end": round(voice_end, 2),
                    "text": words_text(p["words"]),
                    "words": len(p["words"]),
                    "gap": gap,
                    "interrupted": interrupted,
                    "db": round(float(20 * np.log10(rms)), 1),
                    "asr_ms": int(asr_ms * share),
                    "diar_ms": int(diar_ms * share),
                    "rtf": round((asr_ms + diar_ms) / 1000 / dur, 3),
                    "emotion": None,
                    "_audio": chunk,
                }
            )
            prev_end, prev_speaker = voice_end, p["speaker"]
        return {"turns": turns, "merges": merges}

    def _last_turn(self) -> Tuple[Optional[float], Optional[int]]:
        if not self.utterances:
            return None, None
        u = self.utterances[-1]
        return u.get("voice_end", u["end"]), u["speaker"]

    # --------------------------------------------------------------- emotions
    def _schedule_emotion(self, uid: int, audio: np.ndarray, epoch: int) -> None:
        def work() -> Tuple[Dict[str, float], float]:
            t = time.time()
            probs = self.eng.emotions(audio)
            return probs, (time.time() - t) * 1000

        fut = self.loop.run_in_executor(self.eng.emo_pool, work)
        self._pending_models.add(fut)

        def done(f: "asyncio.Future") -> None:
            self._pending_models.discard(f)
            if f.cancelled() or epoch != self.epoch:
                return
            if f.exception():
                log.error("emotion failed: %s", f.exception())
                self._emit_later({"type": "error", "stage": "emotion",
                                  "message": f"Ошибка определения эмоции: {f.exception()}"})
                return
            probs, ms = f.result()
            label = max(probs, key=probs.get)
            payload = {"probs": {k: round(v, 3) for k, v in probs.items()}, "label": label, "ms": int(ms)}
            for u in self.utterances:
                if u["id"] == uid:
                    u["emotion"] = payload
                    break
            self._emit_later({"type": "emotion", "id": uid, **payload})

        fut.add_done_callback(done)

    # ----------------------------------------------------------------- export
    def transcript(self) -> str:
        """Текст разговора для LLM: [мм:сс] Имя (эмоция): реплика."""
        ru = {"angry": "раздражение", "sad": "грусть", "neutral": "нейтрально", "positive": "позитив"}
        lines = []
        for u in self.utterances:
            m, s = divmod(int(u["start"]), 60)
            name = self.names.get(u["speaker"]) or f"Собеседник {u['speaker']}"
            emo = f" ({ru.get(u['emotion']['label'], u['emotion']['label'])})" if u.get("emotion") else ""
            mark = " [перебил]" if u.get("interrupted") else ""
            lines.append(f"[{m:02d}:{s:02d}] {name}{emo}{mark}: {u['text']}")
        return "\n".join(lines)

    def counters(self) -> dict:
        return {
            "audio_sec": round(self.samples_in / SAMPLE_RATE, 1),
            "speech_sec": round(self.speech_samples / SAMPLE_RATE, 1),
            "chunks": self.chunks_in,
            "bytes_in": self.bytes_in,
            "queue": self.queue.qsize() + (1 if self.busy else 0),
            "utterances": len(self.utterances),
            "speakers": len(self.clusterer.speakers),
            "empty_segments": self.empty_segments,
            "clipped_chunks": self.clipped,
            "peak_db": round(20 * np.log10(self.peak + 1e-9), 1),
        }
