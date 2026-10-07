"""Загрузка и вызов моделей.

* Детектор речи      — Silero VAD (ONNX, через sherpa-onnx)
* Распознавание      — GigaAM-v3 e2e RNN-T с пунктуацией (ONNX, через sherpa-onnx)
* Голос собеседника  — 3D-Speaker CAM++ (ONNX, через sherpa-onnx)
* Эмоции по тону     — GigaAM-Emo (PyTorch, официальный пакет gigaam)

Модели общие для всех подключений; каждая вызывается из своего однопоточного
пула, поэтому обращения к ним не пересекаются.
"""
from __future__ import annotations

import logging
import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import SAMPLE_RATE, Settings
from .vad import StreamingVad, default_model_path, make_onnx_session

log = logging.getLogger("engines")

EMOTIONS = ("angry", "sad", "neutral", "positive")


class ComponentState:
    """Состояние одного компонента для панели «Модели»."""

    def __init__(self, title: str, model: str):
        self.title = title
        self.model = model
        self.state = "pending"       # pending | loading | ready | error | off
        self.detail = ""
        self.load_sec: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "model": self.model,
            "state": self.state,
            "detail": self.detail,
            "load_sec": self.load_sec,
        }


class Engines:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.components: Dict[str, ComponentState] = {
            "vad": ComponentState("Детектор речи", "Silero VAD (ONNX)"),
            "asr": ComponentState("Распознавание речи", "GigaAM-v3 e2e RNN-T (int8 ONNX)"),
            "spk": ComponentState("Голоса собеседников", "3D-Speaker CAM++ (ONNX)"),
            "emo": ComponentState("Эмоции по тону", "GigaAM-Emo"),
            "llm": ComponentState("Разбор разговора (LLM)", cfg.llm_model),
        }
        self._asr = None
        self._vad_sess = None
        self._spk = None
        self._emo = None
        self._torch = None
        self._spk_lock = threading.Lock()
        self.asr_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self.emo_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="emo")
        self.live_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live")
        self.on_change: Optional[Callable[[], None]] = None
        self.versions: Dict[str, str] = {}

    # ------------------------------------------------------------- lifecycle
    def _set(self, key: str, state: str, detail: str = "", load_sec: Optional[float] = None) -> None:
        c = self.components[key]
        if (c.state, c.detail) == (state, detail) and load_sec is None:
            return                              # ничего не изменилось — не шлём лишних уведомлений
        c.state, c.detail = state, detail
        if load_sec is not None:
            c.load_sec = round(load_sec, 2)
        if self.on_change:
            try:
                self.on_change()
            except Exception:  # pragma: no cover - уведомление не должно ронять загрузку
                log.exception("status callback failed")

    @property
    def core_ready(self) -> bool:
        return all(self.components[k].state == "ready" for k in ("vad", "asr", "spk"))

    def load_core(self) -> None:
        """Загрузить модели распознавания (вызывается в фоновом потоке при старте)."""
        import sherpa_onnx

        self.versions["sherpa-onnx"] = getattr(sherpa_onnx, "__version__", "?")
        self.versions["python"] = platform.python_version()
        self.versions["arch"] = platform.machine()
        cfg = self.cfg

        # --- детектор речи: проверяем, что модель на месте и читается
        try:
            t = time.time()
            self._set("vad", "loading")
            self.new_vad().accept(np.zeros(1600, dtype=np.float32))
            self._set("vad", "ready", f"порог {cfg.vad_threshold}, пауза {cfg.vad_min_silence} с", time.time() - t)
        except Exception as exc:
            log.exception("VAD load failed")
            self._set("vad", "error", str(exc))

        # --- распознавание
        try:
            t = time.time()
            self._set("asr", "loading")
            d = cfg.asr_dir
            self._asr = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=os.path.join(d, "encoder.int8.onnx"),
                decoder=os.path.join(d, "decoder.onnx"),
                joiner=os.path.join(d, "joiner.onnx"),
                tokens=os.path.join(d, "tokens.txt"),
                model_type="nemo_transducer",
                num_threads=cfg.asr_threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=64,
                decoding_method="greedy_search",
            )
            self._set("asr", "ready", f"{cfg.asr_threads} потока CPU", time.time() - t)
        except Exception as exc:
            log.exception("ASR load failed")
            self._set("asr", "error", str(exc))

        # --- эмбеддинги голоса
        try:
            t = time.time()
            self._set("spk", "loading")
            self._spk = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=cfg.spk_model, num_threads=cfg.spk_threads)
            )
            self._set("spk", "ready", f"вектор {self._spk.dim}, до {cfg.max_speakers} голосов", time.time() - t)
        except Exception as exc:
            log.exception("speaker model load failed")
            self._set("spk", "error", str(exc))

    def load_emotions(self, attempts: int = 3, pause: float = 30.0) -> None:
        """Загрузить GigaAM-Emo. При первом запуске веса скачиваются с серверов Сбера.

        Сбой сети не фатален: распознавание работает и без эмоций, а загрузка
        повторяется несколько раз с паузой.
        """
        if not self.cfg.emo_enabled:
            self._set("emo", "off", "отключено в настройках (EMO_ENABLED=false)")
            return
        for attempt in range(1, attempts + 1):
            try:
                t = time.time()
                self._set("emo", "loading", "загрузка весов (при первом запуске — скачивание)")
                import torch
                import gigaam

                torch.set_num_threads(max(1, self.cfg.emo_threads))
                self.versions["torch"] = torch.__version__
                os.makedirs(self.cfg.emo_cache, exist_ok=True)
                model = gigaam.load_model("emo", device="cpu", download_root=self.cfg.emo_cache)
                self._torch, self._emo = torch, model
                # пробный прогон: убеждаемся, что модель действительно отвечает
                self.emotions(np.zeros(SAMPLE_RATE, dtype=np.float32))
                self._set("emo", "ready", f"{self.cfg.emo_threads} потока CPU", time.time() - t)
                return
            except Exception as exc:
                log.exception("emotion model load failed (attempt %d/%d)", attempt, attempts)
                self._emo = None
                if isinstance(exc, AssertionError) and "checksum" in str(exc).lower():
                    # файл скачался повреждённым — удаляем, чтобы следующая попытка начала заново
                    try:
                        os.remove(os.path.join(self.cfg.emo_cache, "emo.ckpt"))
                    except OSError:
                        pass
                reason = f"{type(exc).__name__}: {exc}"[:220]
                if attempt < attempts:
                    self._set("emo", "loading", f"не получилось ({reason}); повтор через {int(pause)} с")
                    time.sleep(pause)
                else:
                    self._set("emo", "error", f"веса не загрузились: {reason}")

    # ------------------------------------------------------------------ calls
    def new_vad(self) -> StreamingVad:
        cfg = self.cfg
        if self._vad_sess is None:
            self._vad_sess = make_onnx_session(default_model_path())
        return StreamingVad(
            self._vad_sess,
            threshold=cfg.vad_threshold,
            min_silence=cfg.vad_min_silence,
            min_speech=cfg.vad_min_speech,
            max_speech=cfg.vad_max_speech,
        )

    def transcribe(self, samples: np.ndarray) -> Tuple[str, List[str], List[float]]:
        """Текст, токены и время появления каждого токена (секунды от начала фрагмента)."""
        st = self._asr.create_stream()
        st.accept_waveform(SAMPLE_RATE, samples)
        self._asr.decode_stream(st)
        r = st.result
        return r.text.strip(), list(r.tokens), [float(x) for x in r.timestamps]

    def embed(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """Нормированный вектор голоса или None, если фрагмент слишком короткий."""
        if self._spk is None or len(samples) < int(0.4 * SAMPLE_RATE):
            return None
        with self._spk_lock:
            st = self._spk.create_stream()
            st.accept_waveform(SAMPLE_RATE, samples)
            st.input_finished()
            if not self._spk.is_ready(st):
                return None
            e = np.asarray(self._spk.compute(st), dtype=np.float32)
        n = float(np.linalg.norm(e))
        return e / n if n > 0 else None

    @property
    def emotions_ready(self) -> bool:
        return self._emo is not None

    def emotions(self, samples: np.ndarray) -> Dict[str, float]:
        """Вероятности эмоций по звуку: angry / sad / neutral / positive.

        Повторяет GigaAMEmo.get_probs из пакета gigaam, но принимает звук из памяти,
        а не путь к файлу — запись на диск не нужна.
        """
        torch, model = self._torch, self._emo
        x = samples[: 12 * SAMPLE_RATE]                 # модель обучалась на коротких фрагментах
        with torch.no_grad():
            wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
            length = torch.full([1], wav.shape[-1])
            encoded, _ = model.forward(wav, length)
            pooled = torch.nn.functional.avg_pool1d(encoded, kernel_size=encoded.shape[-1]).squeeze(-1)
            probs = torch.nn.functional.softmax(model.head(pooled)[0], dim=-1).tolist()
        names = model.id2name
        return {str(names[i]): float(probs[i]) for i in range(len(probs))}

    def status(self) -> dict:
        return {k: c.as_dict() for k, c in self.components.items()}
