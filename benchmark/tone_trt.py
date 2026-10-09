"""Официальный T-one: TensorRT через внутренний Triton, контекст всей записи."""
from __future__ import annotations

from dataclasses import asdict
import importlib.metadata
import json
import logging
from pathlib import Path
import time

import numpy as np

from benchmark.tone_audio import to_8k

log = logging.getLogger("t-one-tensorrt")
SYSTEMS = {"tone_trt_greedy": "T-one TensorRT без KenLM", "tone_trt_kenlm": "T-one TensorRT с KenLM"}
ENDPOINT = "triton:8001"
MODEL = "streaming_acoustic"
VERSION = "1"
STAGES = {"conversion": "Пересчёт частоты", "acoustic": "Акустика и обмен с Triton",
          "splitter": "Границы фраз", "decoder": "Преобразование в слова"}


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.origin = time.perf_counter()
        self.calls = []

    def measure(self, stage, function):
        started = time.perf_counter()
        failed = False
        try:
            return function()
        except BaseException:
            failed = True
            raise
        finally:
            self.calls.append({"stage": stage, "offset_seconds": started - self.origin,
                "seconds": time.perf_counter() - started, "error": failed})

    def stages(self):
        return {key: {"seconds": sum(c["seconds"] for c in self.calls if c["stage"] == key),
            "calls": sum(c["stage"] == key for c in self.calls),
            "errors": sum(c["stage"] == key and c["error"] for c in self.calls)} for key in STAGES}


def validate_config(response: dict) -> dict:
    config = response.get("config", response)
    instances = config.get("instance_group", [])
    if (config.get("name") != MODEL or config.get("platform") != "tensorrt_plan"
            or config.get("max_batch_size") != 1 or len(instances) != 1
            or instances[0].get("kind") != "KIND_GPU"
            or instances[0].get("gpus") != [0] or instances[0].get("count", 1) != 1):
        raise RuntimeError("Triton должен использовать только TensorRT на GPU:0, один инстанс, batch=1")
    expected = {"signal": ("TYPE_INT32", [2400, 1]), "state": ("TYPE_FP16", [219729])}
    actual = {item["name"]: (item["data_type"], [int(d) for d in item["dims"]]) for item in config.get("input", [])}
    if actual != expected:
        raise RuntimeError(f"Входы Triton отличаются от официальной модели: {actual}")
    if "dynamic_batching" in config:
        raise RuntimeError("В последовательном тесте накопление батча должно быть отключено")
    return config


class TritonAcoustic:
    def __init__(self, client, grpc_module, timer: Timer, timeout: int):
        self.client, self.grpc, self.timer, self.timeout = client, grpc_module, timer, min(timeout, 60)
        self.requests = 0

    def forward(self, audio_chunk, state=None):
        if audio_chunk.shape != (1, 2400, 1) or audio_chunk.dtype != np.int32:
            raise ValueError("Акустике нужен один чанк INT32 [1,2400,1]")
        if state is None:
            state = np.zeros((1, 219729), dtype=np.float16)
        if state.shape != (1, 219729) or state.dtype != np.float16:
            raise ValueError("Состояние акустики должно быть FP16 [1,219729]")

        def infer():
            inputs = []
            for name, array, dtype in (("signal", audio_chunk, "INT32"), ("state", state, "FP16")):
                value = self.grpc.InferInput(name, array.shape, dtype)
                value.set_data_from_numpy(np.ascontiguousarray(array))
                inputs.append(value)
            response = self.client.infer(MODEL, inputs, model_version=VERSION,
                outputs=[self.grpc.InferRequestedOutput("logprobs"), self.grpc.InferRequestedOutput("state_next")],
                client_timeout=self.timeout)
            logprobs, next_state = response.as_numpy("logprobs"), response.as_numpy("state_next")
            if (logprobs is None or next_state is None or logprobs.shape != (1, 10, 35)
                    or next_state.shape != (1, 219729) or logprobs.dtype != np.float32
                    or next_state.dtype != np.float16 or not np.isfinite(logprobs).all()
                    or not np.isfinite(next_state).all()):
                raise RuntimeError("TensorRT вернул повреждённые данные или неверный формат")
            self.requests += 1
            return logprobs, next_state
        return self.timer.measure("acoustic", infer)


class TimedSplitter:
    def __init__(self, splitter, timer):
        self.splitter, self.timer = splitter, timer

    def forward(self, *args, **kwargs):
        return self.timer.measure("splitter", lambda: self.splitter.forward(*args, **kwargs))


class TimedDecoder:
    def __init__(self, decoder, timer):
        self.decoder, self.timer = decoder, timer

    def forward(self, logprobs):
        return self.timer.measure("decoder", lambda: self.decoder.forward(logprobs))


def pauses_from_phrases(phrases, duration):
    # Границы — оценка официального logprob splitter, а не независимый Silero VAD.
    previous, pauses = 0., []
    for phrase in phrases:
        start, end = max(0., min(duration, phrase["start"])), max(0., min(duration, phrase["end"]))
        if start > previous:
            pauses.append({"start": previous, "end": start, "seconds": start - previous})
        previous = max(previous, end)
    if previous < duration:
        pauses.append({"start": previous, "end": duration, "seconds": duration - previous})
    return pauses


class TensorRTPipeline:
    def __init__(self, system: str, cache: Path, timeout: int):
        if system not in SYSTEMS:
            raise ValueError("Неизвестный вариант T-one TensorRT")
        self.system, self.cache, self.timeout = system, cache, timeout
        self.timer = Timer()
        self.client = self.pipeline = None

    def load(self):
        import tone
        import tritonclient.grpc as grpc

        self.client = grpc.InferenceServerClient(ENDPOINT)
        deadline = min(self.timeout, 30)
        if not self.client.is_server_ready(client_timeout=deadline) or not self.client.is_model_ready(MODEL, VERSION, client_timeout=deadline):
            raise RuntimeError("Внутренний Triton или модель не готовы")
        self.config = validate_config(self.client.get_model_config(MODEL, VERSION, as_json=True, client_timeout=deadline))
        self.server_metadata = self.client.get_server_metadata(as_json=True, client_timeout=deadline)
        self.model_metadata = self.client.get_model_metadata(MODEL, VERSION, as_json=True, client_timeout=deadline)
        acoustic = TritonAcoustic(self.client, grpc, self.timer, self.timeout)
        if self.system == "tone_trt_kenlm":
            path = self.cache / "artifacts/kenlm.bin"
            if not path.is_file():
                raise FileNotFoundError("Не скачана официальная KenLM; запустите подготовку с KenLM")
            log.info("Начало загрузки KenLM (5,46 ГБ) в RAM — вне замеров")
            decoder = tone.BeamSearchCTCDecoder.from_local(path)
        else:
            decoder = tone.GreedyCTCDecoder()
        self.pipeline = tone.StreamingCTCPipeline(acoustic,
            TimedSplitter(tone.StreamingLogprobSplitter(), self.timer), TimedDecoder(decoder, self.timer))
        self.versions = {key: importlib.metadata.version(key) for key in
            ("tone", "tritonclient", "numpy", "pyctcdecode", "kenlm", "av", "faster-whisper")}
        self.engine_metadata = json.loads((self.cache / "движок.json").read_text())
        self.artifacts = json.loads((self.cache / "артефакты.json").read_text())
        log.info("%s подготовлен: акустика TensorRT GPU:0; splitter и decoder CPU; batch=1", SYSTEMS[self.system])

    def statistics(self):
        return self.client.get_inference_statistics(MODEL, VERSION, as_json=True, client_timeout=min(self.timeout, 30))

    def run(self, samples):
        self.timer.reset()
        started = self.timer.origin
        acoustic = self.pipeline.model
        acoustic.requests = 0
        raw_phrases, error = [], None
        try:
            def conversion():
                converted = to_8k(samples)
                return (converted * 32768).clip(-32768, 32767).astype(np.int32)
            audio8k = self.timer.measure("conversion", conversion)
            # Официальная реализация: один state на ВСЮ запись, 300 мс padding
            # только по краям записи; is_last дренирует незавершённую фразу.
            raw_phrases = self.pipeline.forward_offline(audio8k)
            status = "ok" if any(p.text.strip() for p in raw_phrases) else "no_speech"
        except Exception as exc:
            status, error = "timeout" if isinstance(exc, TimeoutError) else "error", str(exc)
        elapsed = time.perf_counter() - started
        stages = self.timer.stages()
        exclusive = {key: value["seconds"] for key, value in stages.items()}
        exclusive["other"] = max(0., elapsed - sum(exclusive.values()))
        duration = len(samples) / 16000
        phrases = [{"text": p.text, "start": min(duration, max(0., p.start_time)),
            "end": min(duration, max(0., p.start_time, p.end_time)),
            "raw_start": p.start_time, "raw_end": p.end_time} for p in raw_phrases]
        return {"engine": "tone", "system": self.system, "pipeline": "official_tone_triton_tensorrt",
            "status": status, "error": error, "text": "\n".join(p["text"] for p in phrases if p["text"].strip()),
            "elapsed_seconds": elapsed, "asr_seconds": elapsed, "llm_seconds": 0., "analysis": None,
            "acoustic_rpc_seconds": stages["acoustic"]["seconds"],
            "triton_requests": acoustic.requests, "trt_stages": stages, "stage_calls": list(self.timer.calls),
            "stages": {"asr": {"seconds": elapsed, "calls": 1, "errors": int(error is not None)}},
            "timing": {"exclusive_wall_seconds": exclusive,
                "exclusive_wall_percent": {key: value / elapsed * 100 if elapsed else 0 for key, value in exclusive.items()},
                "method": "Последовательные интервалы perf_counter; acoustic включает gRPC и передачу состояния; сумма 100%"},
            "utterances": phrases, "raw_phrases": [asdict(p) for p in raw_phrases],
            "pauses": pauses_from_phrases(phrases, duration),
            "pause_method": "Официальный logprob splitter, не Silero VAD и не ручная разметка",
            "timestamps": "Тайминги фраз от официального T-one; пословных меток этот вариант не выдаёт",
            "input_samples_16k": len(samples), "input_samples_8k": (len(samples) + 1) // 2,
            "context_resets_per_record": 1, "padding_per_record_seconds": .6}

    def close(self):
        if self.client is not None:
            self.client.close()
