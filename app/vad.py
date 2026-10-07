"""Потоковый детектор речи на Silero VAD (официальная ONNX-модель из пакета silero-vad).

Модель каждые 32 мс выдаёт вероятность речи. Фрагмент начинается, когда вероятность
поднимается выше порога, и заканчивается после паузы длиной min_silence. Вместе со
звуком фрагмента возвращаются покадровые вероятности — по ним потом точно находятся
паузы между собеседниками внутри одного фрагмента.
"""
from __future__ import annotations

import importlib.util
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

SAMPLE_RATE = 16000
WINDOW = 512               # 32 мс
CONTEXT = 64
FRAME_SEC = WINDOW / SAMPLE_RATE


def default_model_path() -> str:
    """Путь к silero_vad.onnx внутри установленного пакета silero-vad (без его импорта)."""
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("пакет silero-vad не установлен")
    path = os.path.join(list(spec.submodule_search_locations)[0], "data", "silero_vad.onnx")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def make_onnx_session(model_path: str, *, provider: str = "CPUExecutionProvider"):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"ONNX Runtime не поддерживает {provider}")
    session = ort.InferenceSession(model_path, sess_options=opts, providers=[provider])
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"ONNX Runtime не смог включить {provider}: {session.get_providers()}")
    return session


@dataclass
class SpeechSegment:
    start: int                 # номер первого отсчёта от начала сессии
    samples: np.ndarray        # float32, 16 кГц
    probs: np.ndarray          # вероятность речи на каждом кадре 32 мс
    lead: int = 0              # сколько отсчётов тишины добавлено перед речью
    trail: int = 0             # сколько отсчётов тишины оставлено после речи

    @property
    def end(self) -> int:
        return self.start + self.samples.size


class StreamingVad:
    def __init__(
        self,
        session,
        threshold: float = 0.5,
        min_silence: float = 0.45,
        min_speech: float = 0.25,
        max_speech: float = 15.0,
        pad: float = 0.10,
    ):
        self.sess = session
        self.threshold = threshold
        self.neg_threshold = max(0.05, threshold - 0.15)
        self.min_silence = int(min_silence * SAMPLE_RATE)
        self.min_speech = int(min_speech * SAMPLE_RATE)
        self.max_speech = int(max_speech * SAMPLE_RATE)
        self.pad = int(pad * SAMPLE_RATE)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._rest = np.zeros(0, dtype=np.float32)       # хвост, не добравший до целого кадра
        self._pos = 0                                    # сколько отсчётов уже разобрано на кадры
        self._preroll: Deque[np.ndarray] = deque(maxlen=8)   # ~0.25 с до начала речи
        self._frames: List[np.ndarray] = []              # кадры текущего фрагмента
        self._probs: List[float] = []
        self._seg_start = 0
        self._lead = 0
        self._silence_at: Optional[int] = None           # индекс кадра, с которого идёт тишина
        self.speaking = False
        self.last_prob = 0.0

    # ----------------------------------------------------------------- model
    def _prob(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame])[None, :]
        out, self._state = self.sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = frame[-CONTEXT:]
        return float(out[0][0])

    # ---------------------------------------------------------------- stream
    def accept(self, x: np.ndarray) -> List[SpeechSegment]:
        """Принять порцию звука; вернуть завершившиеся фрагменты речи (обычно 0 или 1)."""
        out: List[SpeechSegment] = []
        buf = np.concatenate([self._rest, x.astype(np.float32, copy=False)])
        n = buf.size // WINDOW
        for i in range(n):
            frame = buf[i * WINDOW:(i + 1) * WINDOW]
            p = self._prob(frame)
            self.last_prob = p
            seg = self._step(frame, p)
            if seg is not None:
                out.append(seg)
            self._pos += WINDOW
        self._rest = buf[n * WINDOW:].copy()
        return out

    def _step(self, frame: np.ndarray, p: float) -> Optional[SpeechSegment]:
        if not self.speaking:
            if p >= self.threshold:
                self.speaking = True
                pre = list(self._preroll)[-max(1, self.pad // WINDOW + 1):]
                self._frames = pre + [frame]
                self._probs = [self.neg_threshold] * len(pre) + [p]
                self._seg_start = self._pos - len(pre) * WINDOW
                self._lead = len(pre) * WINDOW
                self._silence_at = None
                self._preroll.clear()
            else:
                self._preroll.append(frame)
            return None

        self._frames.append(frame)
        self._probs.append(p)
        k = len(self._frames) - 1
        if p >= self.threshold:
            self._silence_at = None
        elif p < self.neg_threshold and self._silence_at is None:
            self._silence_at = k

        if self._silence_at is not None and (k - self._silence_at + 1) * WINDOW >= self.min_silence:
            return self._close(self._silence_at)
        if len(self._frames) * WINDOW >= self.max_speech:
            return self._close_long()
        return None

    def _close(self, silence_from: int) -> Optional[SpeechSegment]:
        """Закрыть фрагмент: речь шла до кадра silence_from, дальше тишина."""
        keep = min(len(self._frames), silence_from + self.pad // WINDOW + 1)
        seg = self._make(keep, trail=max(0, keep - silence_from) * WINDOW)
        tail = self._frames[keep:]
        self._preroll.clear()
        self._preroll.extend(tail[-self._preroll.maxlen:])
        self._frames, self._probs = [], []
        self.speaking = False
        self._silence_at = None
        return seg

    def _close_long(self) -> Optional[SpeechSegment]:
        """Речь без пауз дольше max_speech: режем по самому тихому месту второй половины."""
        probs = np.asarray(self._probs)
        half = len(probs) // 2
        cut = half + int(np.argmin(probs[half:]))
        if probs[cut] >= self.threshold or len(probs) - cut < 3:
            cut = len(probs)                       # тихого места нет — режем по текущей точке
        seg = self._make(cut)
        rest_frames, rest_probs = self._frames[cut:], self._probs[cut:]
        self._seg_start += cut * WINDOW
        self._lead = 0
        self._frames, self._probs = rest_frames, rest_probs
        self._silence_at = None
        if not self._frames:
            # разрез пришёлся на текущий кадр: следующий кадр продолжит ту же речь
            self._seg_start = self._pos + WINDOW
        return seg

    def _make(self, n_frames: int, trail: int = 0) -> Optional[SpeechSegment]:
        if n_frames <= 0:
            return None
        samples = np.concatenate(self._frames[:n_frames])
        if samples.size - self._lead - trail < self.min_speech:
            return None
        return SpeechSegment(
            self._seg_start, samples, np.asarray(self._probs[:n_frames], dtype=np.float32), self._lead, trail
        )

    def flush(self) -> List[SpeechSegment]:
        """Микрофон остановлен: вернуть незавершённый фрагмент, если он есть."""
        out: List[SpeechSegment] = []
        if self.speaking and self._frames:
            end = self._silence_at if self._silence_at is not None else len(self._frames)
            keep = min(len(self._frames), end + self.pad // WINDOW + 1)
            seg = self._make(keep, trail=max(0, keep - end) * WINDOW)
            if seg is not None:
                out.append(seg)
        pos = self._pos + self._rest.size
        self.reset()
        self._pos = pos
        return out

    def current(self, max_sec: float = 3.0) -> np.ndarray:
        """Последние секунды ещё не завершённого фрагмента (для оценки голоса на лету)."""
        if not self.speaking or not self._frames:
            return np.zeros(0, dtype=np.float32)
        n = int(max_sec * SAMPLE_RATE / WINDOW) + 1
        return np.concatenate(self._frames[-n:])


def find_pauses(probs: np.ndarray, threshold: float, min_frames: int = 2) -> List[tuple]:
    """Паузы внутри фрагмента: [(начало_сек, конец_сек)] по кадрам с вероятностью ниже порога."""
    out, start = [], None
    for i, p in enumerate(probs):
        if p < threshold:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= min_frames:
                out.append((start * FRAME_SEC, i * FRAME_SEC))
            start = None
    return out
