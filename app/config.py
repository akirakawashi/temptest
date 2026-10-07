"""Настройки приложения. Все значения берутся из переменных окружения (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _s(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


SAMPLE_RATE = 16000


@dataclass
class Settings:
    # --- модели распознавания (лежат внутри образа) ---
    models_dir: str = field(default_factory=lambda: _s("MODELS_DIR", "/models"))
    asr_threads: int = field(default_factory=lambda: _i("ASR_THREADS", 4))
    spk_threads: int = field(default_factory=lambda: _i("SPK_THREADS", 2))

    # --- детектор речи (Silero VAD) ---
    vad_threshold: float = field(default_factory=lambda: _f("VAD_THRESHOLD", 0.5))
    vad_min_silence: float = field(default_factory=lambda: _f("VAD_MIN_SILENCE", 0.45))
    vad_min_speech: float = field(default_factory=lambda: _f("VAD_MIN_SPEECH", 0.25))
    # модель распознавания принимает не больше 25 с за раз — держим запас
    vad_max_speech: float = field(default_factory=lambda: min(24.0, max(3.0, _f("VAD_MAX_SPEECH", 15.0))))

    # --- разделение собеседников ---
    speaker_threshold: float = field(default_factory=lambda: _f("SPEAKER_THRESHOLD", 0.45))
    # не больше пяти: у каждого собеседника свой цвет в интерфейсе
    max_speakers: int = field(default_factory=lambda: min(5, max(1, _i("MAX_SPEAKERS", 5))))
    split_turns: bool = field(default_factory=lambda: _b("SPLIT_TURNS", True))

    # --- эмоции (GigaAM-Emo, веса скачиваются при первом запуске) ---
    emo_enabled: bool = field(default_factory=lambda: _b("EMO_ENABLED", True))
    emo_threads: int = field(default_factory=lambda: _i("EMO_THREADS", 2))
    emo_cache: str = field(default_factory=lambda: _s("EMO_CACHE", "/cache/gigaam"))

    # --- разбор разговора текстовой LLM (Ollama) ---
    llm_enabled: bool = field(default_factory=lambda: _b("LLM_ENABLED", True))
    ollama_url: str = field(default_factory=lambda: _s("OLLAMA_URL", "http://ollama:11434").rstrip("/"))
    llm_model: str = field(
        default_factory=lambda: _s("LLM_MODEL", "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:Q4_K_M")
    )
    llm_autopull: bool = field(default_factory=lambda: _b("LLM_AUTOPULL", True))
    llm_num_ctx: int = field(default_factory=lambda: _i("LLM_NUM_CTX", 8192))
    llm_keep_alive: str | int = field(default_factory=lambda: _s("LLM_KEEP_ALIVE", "1m"))

    @property
    def asr_dir(self) -> str:
        return os.path.join(self.models_dir, "asr")

    @property
    def spk_model(self) -> str:
        return os.path.join(self.models_dir, "speaker.onnx")


settings = Settings()
