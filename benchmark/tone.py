"""T-one в том же потоке Silero VAD/ASR, что и сокращённый тест GigaAM."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path

import numpy as np

from app.config import SAMPLE_RATE, Settings
from app.engines import Engines
from app.vad import StreamingVad, default_model_path, make_onnx_session
from benchmark.gpu import sherpa_runtime_version
from benchmark.pipeline import FirstLinePipeline, StageTimer, TimedVad
from benchmark.tone_audio import LOWPASS_8K, to_8k

log = logging.getLogger("тестовый-t-one")
MODEL_NAME = "sherpa-onnx-streaming-t-one-russian-2025-09-08"
MODEL_ARCHIVE_SHA256 = "b9c907450e99a6e5049e279bf18368a17db0bdc5e63b7fa978943138debbe3ae"
TONE_RATE = 8000
LEAD_SECONDS = .3
TAIL_SECONDS = 1.0
DELAY_SECONDS = .40


def word_tokens(tokens, timestamps, duration: float) -> tuple[list[str], list[float]]:
    if len(tokens) != len(timestamps):
        raise ValueError("T-one вернул разное число токенов и временных меток")
    converted, stamps, new_word = [], [], True
    for char, stamp in zip(tokens, timestamps):
        if char == " ":
            new_word = True
            continue
        converted.append((" " if new_word else "") + char)
        stamps.append(min(duration, max(0., float(stamp) - LEAD_SECONDS - DELAY_SECONDS)))
        new_word = False
    return converted, stamps


class ToneEngines(Engines):
    def __init__(self, cfg: Settings, model_dir: Path):
        super().__init__(cfg)
        self.model_dir = model_dir
        self.components["asr"].model = "T-one CTC, 8 кГц"
        self.timer = StageTimer()
        self.raw_results = []
        self.model_hashes = {}
        for key in ("spk", "emo", "llm"):
            self._set(key, "off", "Отключено в отдельном тесте T-one")

    @property
    def core_ready(self) -> bool:
        return all(self.components[key].state == "ready" for key in ("vad", "asr"))

    def load_core(self) -> None:
        import onnxruntime as ort
        import sherpa_onnx

        runtime = sherpa_runtime_version(sherpa_onnx)
        self.versions = {"sherpa-onnx": sherpa_onnx.__version__,
                         "sherpa_onnx_runtime": runtime, "onnxruntime-gpu": ort.__version__}
        if "+cuda12.cudnn9" not in sherpa_onnx.__version__:
            raise RuntimeError("Нужен CUDA wheel sherpa-onnx; CPU-замеры вместо GPU не допускаются")
        if runtime != ort.__version__:
            raise RuntimeError(f"Версии ONNX Runtime не совпадают: sherpa={runtime}, Python={ort.__version__}")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime не поддерживает CUDAExecutionProvider")
        for filename in ("model.onnx", "tokens.txt"):
            path = self.model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("rb") as stream:
                self.model_hashes[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
        with Path(default_model_path()).open("rb") as stream:
            self.model_hashes["silero_vad.onnx"] = hashlib.file_digest(stream, "sha256").hexdigest()
        started = time.perf_counter()
        self._vad_sess = make_onnx_session(default_model_path(), provider="CUDAExecutionProvider")
        self.new_vad().accept(np.zeros(1600, dtype=np.float32))
        self._set("vad", "ready", "Silero VAD: CUDAExecutionProvider", time.perf_counter() - started)
        log.info("T-one: Silero VAD подготовлен, провайдеры %s", self._vad_sess.get_providers())
        started = time.perf_counter()
        self._asr = sherpa_onnx.OnlineRecognizer.from_t_one_ctc(
            tokens=str(self.model_dir / "tokens.txt"), model=str(self.model_dir / "model.onnx"),
            num_threads=self.cfg.asr_threads, sample_rate=TONE_RATE,
            decoding_method="greedy_search", provider="cuda", device=0,
        )
        self._set("asr", "ready", "T-one: provider=cuda, device=0, greedy_search", time.perf_counter() - started)
        log.info("T-one: распознаватель создан на CUDA:0; версии %s", self.versions)

    def new_vad(self):
        return TimedVad(StreamingVad(self._vad_sess, threshold=self.cfg.vad_threshold,
            min_silence=self.cfg.vad_min_silence, min_speech=self.cfg.vad_min_speech,
            max_speech=self.cfg.vad_max_speech), self.timer)

    def transcribe(self, samples):
        def recognize():
            started = time.perf_counter()
            audio8k = to_8k(samples)
            conversion_seconds = time.perf_counter() - started
            stream = self._asr.create_stream()
            stream.accept_waveform(TONE_RATE, np.zeros(int(LEAD_SECONDS * TONE_RATE), dtype=np.float32))
            stream.accept_waveform(TONE_RATE, audio8k)
            stream.accept_waveform(TONE_RATE, np.zeros(int(TAIL_SECONDS * TONE_RATE), dtype=np.float32))
            stream.input_finished()
            calls = 0
            while self._asr.is_ready(stream):
                self._asr.decode_stream(stream)
                calls += 1
            result = self._asr.get_result_all(stream)
            tokens, stamps = word_tokens(result.tokens, result.timestamps, len(samples) / SAMPLE_RATE)
            detail = {"text": result.text.strip(), "raw_tokens": list(result.tokens),
                "raw_token_timestamps": [float(t) for t in result.timestamps],
                "converted_tokens": tokens, "corrected_timestamps": stamps,
                "samples_8k": len(audio8k), "conversion_seconds": conversion_seconds,
                "decode_calls": calls, "padding_seconds": LEAD_SECONDS + TAIL_SECONDS}
            self.raw_results.append(detail)
            return result.text.strip(), tokens, stamps
        return self.timer.measure("asr", recognize, audio_seconds=len(samples) / SAMPLE_RATE)

    def placement(self):
        return {"asr": "sherpa-onnx CUDA:0 (float32 ONNX, greedy_search)",
            "vad": self._vad_sess.get_providers(), "speaker": "отключено",
            "emotion": "отключено", "llm": "отключено"}


class TonePipeline(FirstLinePipeline):
    def __init__(self, cfg: Settings, threads: int, timeout: float, model_dir: Path):
        if cfg.emo_enabled or cfg.llm_enabled or cfg.split_turns:
            raise ValueError("Тест T-one требует отключить голоса, эмоции и LLM")
        self.cfg, self.threads, self.timeout = cfg, threads, timeout
        self.llm_runtime = self.llm = None
        self.engines = ToneEngines(cfg, model_dir)
        self.events = []

    async def load(self):
        await asyncio.to_thread(self.engines.load_core)

    async def run(self, samples):
        self.engines.raw_results = []
        result = await super().run(samples)
        result.update(engine="tone", pipeline="vad_tone", raw_asr_results=list(self.engines.raw_results))
        # Сохранить исходный текст модели, включая числа; форматирование
        # реплик/метки времени лежат рядом и не заменяют исходный ответ.
        result["formatted_text"] = result["text"]
        result["text"] = "\n".join(r["text"] for r in self.engines.raw_results if r["text"])
        for detail in self.engines.raw_results:
            log.info("T-one — ответ распознавания: %s", detail)
        if result.get("error"):
            result["error"] = result["error"].replace("Полный цикл GigaAM", "Цикл T-one").replace("полного цикла GigaAM", "цикла T-one")
        return result
