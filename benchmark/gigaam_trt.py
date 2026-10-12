"""GigaAM v3: энкодер через TensorRT в Triton; поиск речи и декодер в клиенте."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
from pathlib import Path
import time

import numpy as np

from app.config import SAMPLE_RATE, Settings
from app.engines import Engines
from app.vad import StreamingVad, default_model_path, make_onnx_session
from benchmark.gigaam_trt_prepare import (FEATURES, HIDDEN, LABEL, MAX_FRAMES, MIN_FRAMES, MODEL, MODEL_NAME,
                                          SYSTEM, TRT_STAGES, VAD_PROVIDERS, vad_provider)
from benchmark.pipeline import FirstLinePipeline, StageTimer, TimedVad

log = logging.getLogger("gigaam-tensorrt")
ENDPOINT, VERSION = "triton:8001", "1"


def validate_config(response: dict) -> dict:
    config = response.get("config", response)
    instances = config.get("instance_group", [])
    if (config.get("name") != MODEL or config.get("platform") != "tensorrt_plan"
            or int(config.get("max_batch_size", 0)) != 0 or len(instances) != 1
            or instances[0].get("kind") != "KIND_GPU"
            or instances[0].get("gpus") != [0] or instances[0].get("count", 1) != 1):
        raise RuntimeError("Triton должен использовать только TensorRT на GPU:0, один инстанс, без батчей")
    expected = {"audio_signal": ("TYPE_FP32", [1, FEATURES, -1]), "length": ("TYPE_INT64", [1])}
    actual = {item["name"]: (item["data_type"], [int(d) for d in item["dims"]]) for item in config.get("input", [])}
    if actual != expected:
        raise RuntimeError(f"Входы Triton отличаются от экспортированной модели: {actual}")
    if "dynamic_batching" in config:
        raise RuntimeError("В последовательном тесте накопление батча должно быть отключено")
    return config


class SubTimer:
    """Части одного вызова распознавания; идут строго друг за другом."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.values = {key: {"seconds": 0.0, "calls": 0, "errors": 0} for key in TRT_STAGES}

    def measure(self, stage: str, call):
        started, failed = time.perf_counter(), False
        try:
            return call()
        except BaseException:
            failed = True
            raise
        finally:
            entry = self.values[stage]
            entry["seconds"] += time.perf_counter() - started
            entry["calls"] += 1
            entry["errors"] += int(failed)


class TritonEncoder:
    def __init__(self, client, grpc_module, timeout: int):
        self.client, self.grpc, self.timeout = client, grpc_module, min(timeout, 60)
        self.requests = 0

    def forward(self, features: np.ndarray, length: int) -> tuple[np.ndarray, int]:
        if features.ndim != 3 or features.shape[:2] != (1, FEATURES) or features.dtype != np.float32:
            raise ValueError("Энкодеру нужны признаки FP32 [1,64,T]")
        signal = self.grpc.InferInput("audio_signal", features.shape, "FP32")
        signal.set_data_from_numpy(features)
        frames = self.grpc.InferInput("length", [1], "INT64")
        frames.set_data_from_numpy(np.array([length], dtype=np.int64))
        response = self.client.infer(MODEL, [signal, frames], model_version=VERSION,
            outputs=[self.grpc.InferRequestedOutput("encoded"), self.grpc.InferRequestedOutput("encoded_len")],
            client_timeout=self.timeout)
        encoded, encoded_len = response.as_numpy("encoded"), response.as_numpy("encoded_len")
        if (encoded is None or encoded_len is None or encoded.ndim != 3 or encoded.shape[:2] != (1, HIDDEN)
                or encoded.dtype != np.float32 or not np.isfinite(encoded).all()):
            raise RuntimeError("TensorRT вернул повреждённые данные или неверный формат")
        count = int(np.asarray(encoded_len).reshape(-1)[0])
        if not 0 < count <= encoded.shape[2]:
            raise RuntimeError("TensorRT вернул неверную длину выхода энкодера")
        self.requests += 1
        return encoded, count


class GreedyDecoder:
    """Жадный разбор RNN-T, как в официальном пакете: до max_symbols символов на кадр."""

    def __init__(self, decoder, joint, blank: int, hidden: int, layers: int, max_symbols: int):
        self.decoder, self.joint, self.blank, self.max_symbols = decoder, joint, blank, max_symbols
        zeros = np.zeros((layers, 1, hidden), dtype=np.float32)
        # Вложение пустого символа в весах нулевое: это тот же старт, что у PyTorch с нулевым входом.
        self.start = self._predict(blank, zeros, zeros)

    def _predict(self, label: int, h: np.ndarray, c: np.ndarray):
        return self.decoder.run(None, {"x": np.array([[label]], dtype=np.int64), "hi": h, "ci": c})

    def decode(self, encoded: np.ndarray, frames: int) -> tuple[list[int], list[int]]:
        ids, emitted = [], []
        dec, h, c = self.start
        for t in range(frames):
            frame = np.ascontiguousarray(encoded[:, :, t:t + 1])
            for _ in range(self.max_symbols):
                token = int(self.joint.run(None, {"enc": frame, "dec": dec.swapaxes(1, 2)})[0][0, 0, 0].argmax())
                if token == self.blank:
                    break
                ids.append(token)
                emitted.append(t)
                dec, h, c = self._predict(token, h, c)
        return ids, emitted


class GigaTrtEngines(Engines):
    def __init__(self, cfg: Settings, cache: Path, timeout: int):
        super().__init__(cfg)
        self.cache, self.timeout = cache, timeout
        self.components["asr"].model = "GigaAM-v3 e2e RNNT (TensorRT, Triton)"
        self.timer, self.sub = StageTimer(), SubTimer()
        self.client = self.encoder = self.decoder = None
        self.skipped_short = 0
        self.vad_provider = vad_provider()
        for key in ("spk", "emo", "llm"):
            self._set(key, "off", "Отключено: только поиск речи и распознавание")

    @property
    def core_ready(self) -> bool:
        return all(self.components[key].state == "ready" for key in ("vad", "asr"))

    def load_core(self) -> None:
        import hydra
        import omegaconf
        import onnxruntime as ort
        import torch
        import tritonclient.grpc as grpc
        from gigaam.decoding import Tokenizer

        started = time.perf_counter()
        self._vad_sess = make_onnx_session(default_model_path(), provider=VAD_PROVIDERS[self.vad_provider])
        self.new_vad().accept(np.zeros(1600, dtype=np.float32))
        self._set("vad", "ready", f"Silero VAD: {self._vad_sess.get_providers()[0]}", time.perf_counter() - started)

        started = time.perf_counter()
        artifacts = self.cache / "artifacts"
        self.manifest = json.loads((self.cache / "артефакты.json").read_text(encoding="utf-8"))
        self.engine_metadata = json.loads((self.cache / "движок.json").read_text(encoding="utf-8"))
        cfg = omegaconf.OmegaConf.load(artifacts / f"{MODEL_NAME}.yaml")
        self._torch = torch
        torch.set_num_threads(max(1, self.cfg.asr_threads))
        self._preprocessor = hydra.utils.instantiate(cfg.preprocessor).eval()
        # Окно и мел-фильтры хранятся в весах обученной модели и отличаются от создаваемых заново.
        self._preprocessor.load_state_dict(torch.load(artifacts / f"{MODEL_NAME}_preprocessor.pt",
                                                      map_location="cpu", weights_only=True))
        self.tokenizer = Tokenizer(list(cfg.decoding.get("vocabulary") or []), str(artifacts / f"{MODEL_NAME}_tokenizer.model"))
        if len(self.tokenizer) != self.manifest["blank_id"]:
            raise RuntimeError("Словарь не соответствует экспортированной модели")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 3
        sessions = [ort.InferenceSession(str(artifacts / f"{MODEL_NAME}_{part}.onnx"), sess_options=options,
                                         providers=["CPUExecutionProvider"]) for part in ("decoder", "joint")]
        self.decoder = GreedyDecoder(*sessions, blank=self.manifest["blank_id"], hidden=self.manifest["pred_hidden"],
                                     layers=self.manifest["pred_rnn_layers"], max_symbols=self.manifest["max_symbols"])
        self.client = grpc.InferenceServerClient(ENDPOINT)
        deadline = min(self.timeout, 30)
        if not self.client.is_server_ready(client_timeout=deadline) or not self.client.is_model_ready(MODEL, VERSION, client_timeout=deadline):
            raise RuntimeError("Внутренний Triton или модель не готовы")
        self.config = validate_config(self.client.get_model_config(MODEL, VERSION, as_json=True, client_timeout=deadline))
        self.server_metadata = self.client.get_server_metadata(as_json=True, client_timeout=deadline)
        self.model_metadata = self.client.get_model_metadata(MODEL, VERSION, as_json=True, client_timeout=deadline)
        self.encoder = TritonEncoder(self.client, grpc, self.timeout)
        self._asr = self.encoder
        self.versions = {key: importlib.metadata.version(key) for key in
                         ("gigaam", "torch", "tritonclient", "onnxruntime-gpu", "numpy", "av", "faster-whisper")}
        self._set("asr", "ready", "Энкодер TensorRT GPU:0; декодер CPU", time.perf_counter() - started)
        log.info("%s подготовлен: энкодер TensorRT GPU:0; признаки и декодер CPU; поиск речи %s",
                 LABEL, self._vad_sess.get_providers()[0])

    def new_vad(self):
        return TimedVad(StreamingVad(self._vad_sess, threshold=self.cfg.vad_threshold,
            min_silence=self.cfg.vad_min_silence, min_speech=self.cfg.vad_min_speech,
            max_speech=self.cfg.vad_max_speech), self.timer)

    def features(self, samples) -> tuple[np.ndarray, int]:
        torch = self._torch
        with torch.inference_mode():
            wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
            values, lengths = self._preprocessor(wav, torch.tensor([wav.shape[-1]]))
        return np.ascontiguousarray(values.numpy(), dtype=np.float32), int(lengths[0])

    def transcribe(self, samples):
        def recognize():
            values, length = self.sub.measure("features", lambda: self.features(samples))
            if length < MIN_FRAMES:
                self.skipped_short += 1
                return "", [], []
            if length > MAX_FRAMES:
                raise RuntimeError(f"Фрагмент речи {length / 100:.1f} с длиннее профиля движка {MAX_FRAMES / 100:.0f} с")
            encoded, frames = self.sub.measure("encoder", lambda: self.encoder.forward(values, length))
            ids, emitted = self.sub.measure("decoder", lambda: self.decoder.decode(encoded, frames))
            shift = len(samples) / SAMPLE_RATE / frames
            return (self.tokenizer.decode(ids).strip(), [self.tokenizer.id_to_str(i) for i in ids],
                    [float(frame * shift) for frame in emitted])
        return self.timer.measure("asr", recognize, audio_seconds=len(samples) / SAMPLE_RATE)

    def placement(self) -> dict:
        return {"asr": "Энкодер TensorRT, Triton GPU:0; признаки и декодер CPU",
                "vad": self._vad_sess.get_providers(), "speaker": "отключено", "emotion": "отключено", "llm": "отключено"}


class GigaTrtPipeline(FirstLinePipeline):
    """Урезанный flow: поиск речи → распознавание → паузы; энкодер в TensorRT."""

    def __init__(self, cfg: Settings, threads: int, timeout: int, cache: Path):
        if cfg.emo_enabled or cfg.llm_enabled or cfg.split_turns:
            raise ValueError("GigaAM TensorRT работает без голосов, эмоций и LLM")
        super().__init__(cfg, GigaTrtEngines(cfg, cache, timeout), timeout)
        self.threads = threads

    async def load(self) -> None:
        await asyncio.to_thread(self.engines.load_core)

    def statistics(self) -> dict:
        return self.engines.client.get_inference_statistics(MODEL, VERSION, as_json=True, client_timeout=min(self.timeout, 30))

    async def run(self, samples) -> dict:
        engines = self.engines
        engines.sub.reset()
        engines.encoder.requests = engines.skipped_short = 0
        result = await super().run(samples)
        stages = {key: dict(value) for key, value in engines.sub.values.items()}
        result.update(engine="gigaam", system=SYSTEM, pipeline="vad_gigaam_tensorrt", trt_stages=stages,
                      triton_requests=engines.encoder.requests, encoder_rpc_seconds=stages["encoder"]["seconds"],
                      skipped_short_segments=engines.skipped_short, vad_provider=engines.vad_provider)
        return result

    def close(self) -> None:
        self.shutdown()
        if self.engines.client is not None:
            self.engines.client.close()
