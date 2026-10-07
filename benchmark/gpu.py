"""CUDA-адаптеры того же речевого конвейера; импорт без загрузки моделей."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import time

import numpy as np

from app.config import SAMPLE_RATE
from app.engines import Engines
from app.vad import StreamingVad, default_model_path, make_onnx_session

log = logging.getLogger("сравнение")


def sherpa_runtime_version(sherpa) -> str:
    """Версия встроенного ORT; старые wheel не экспортируют её в Python."""
    version = getattr(sherpa, "onnxruntime_version", None)
    if version is not None:
        return str(version)

    import ctypes

    version_function = ctypes.CFUNCTYPE(ctypes.c_char_p)

    class OrtApiBase(ctypes.Structure):
        _fields_ = [("get_api", ctypes.c_void_p), ("get_version", version_function)]

    library = ctypes.CDLL(str(Path(sherpa.__file__).parent / "lib" / "libonnxruntime.so"))
    library.OrtGetApiBase.restype = ctypes.POINTER(OrtApiBase)
    return library.OrtGetApiBase().contents.get_version().decode("ascii")


def log_gpu_memory(stage: str) -> None:
    import torch

    free, total = torch.cuda.mem_get_info(0)
    mib = 1024 ** 2
    log.info("GPU после %s: свободно %.0f / %.0f МиБ; PyTorch занял %.0f МиБ, зарезервировал %.0f МиБ",
             stage, free / mib, total / mib, torch.cuda.memory_allocated(0) / mib,
             torch.cuda.memory_reserved(0) / mib)
    reserve = int(os.environ.get("BENCH_GPU_RESERVE_MIB", "0"))
    if free / mib < reserve:
        raise RuntimeError(f"После {stage} свободно {free / mib:.0f} МиБ GPU, нужен резерв {reserve} МиБ. "
                           "Останавливаем только стенд; рабочий Whisper не управляется")


def gpu_info() -> dict:
    import torch
    import ctranslate2

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Стенду нужна ровно одна доступная NVIDIA GPU. Проверьте BENCH_GPU и доступ Docker к CUDA")
    if ctranslate2.get_cuda_device_count() != 1:
        raise RuntimeError("CTranslate2 не видит единственную GPU, выбранную для обеих систем")
    prop = torch.cuda.get_device_properties(0)
    return {"name": prop.name, "total_memory_bytes": prop.total_memory,
            "uuid": str(getattr(prop, "uuid", "не сообщён")), "logical_index": 0,
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


async def require_idle_ollama(url: str, timeout: float) -> None:
    """До Whisper сервер стенда должен быть готов и не держать другую модель."""
    import httpx

    async with asyncio.timeout(timeout), httpx.AsyncClient(base_url=url, timeout=10) as client:
        while True:
            try:
                response = await client.get("/api/ps")
                response.raise_for_status()
            except httpx.HTTPError:
                await asyncio.sleep(1)
                continue
            if response.json().get("models"):
                raise RuntimeError("Перед Whisper в Ollama ещё загружена LLM; GPU должна быть свободна от другой системы")
            return


def configure_whisper_vad() -> list[str]:
    """Сохраняем Silero из faster-whisper 1.2.1, меняем только ONNX-провайдер.

    Фабрика действует только в отдельном процессе Whisper; __call__, веса и
    алгоритм выделения фрагментов остаются из закреплённой версии библиотеки.
    """
    from faster_whisper import vad
    from faster_whisper.utils import get_assets_path

    class CudaSilero(vad.SileroVADModel):
        def __init__(self):
            self.session = make_onnx_session(str(Path(get_assets_path()) / "silero_vad_v6.onnx"),
                                             provider="CUDAExecutionProvider")

    model = CudaSilero()
    vad.get_vad_model = lambda: model
    return model.session.get_providers()


class CudaEngines(Engines):
    """ASR/эмоции — официальный GigaAM CUDA; CAM++ — sherpa-onnx CUDA.

    CPU-int8 ONNX-энкодер исходного монитора здесь не используется. На вход
    исходной Session возвращаются те же токены SentencePiece и их времена.
    """

    def load_core(self) -> None:
        log.info("GigaAM: начало импорта официального пакета")
        import gigaam
        log.info("GigaAM: официальный пакет импортирован; начало импорта sherpa-onnx")
        import sherpa_onnx
        import torch
        import onnxruntime as ort

        native_ort = sherpa_runtime_version(sherpa_onnx)
        self.versions["sherpa-onnx-runtime"] = native_ort
        log.info("GigaAM: sherpa-onnx %s, его ONNX Runtime %s; PyTorch %s",
                 sherpa_onnx.__version__, native_ort, torch.__version__)
        if native_ort != ort.__version__:
            raise RuntimeError(f"Несовместимые ONNX Runtime: sherpa-onnx использует {native_ort}, "
                               f"Python — {ort.__version__}. CUDA-сборки должны использовать одну версию Runtime")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна для GigaAM; переход на CPU запрещён")
        if "+cuda12" not in sherpa_onnx.__version__:
            raise RuntimeError("Для CAM++ требуется CUDA 12/cuDNN 9 сборка sherpa-onnx")
        self._torch = torch
        self._asr_stream = torch.cuda.Stream(device=0)
        torch.set_num_threads(max(1, self.cfg.asr_threads))
        self.versions.update({"sherpa-onnx": sherpa_onnx.__version__, "torch": torch.__version__})
        self.components["asr"].model = "GigaAM-v3 e2e RNNT (PyTorch CUDA, fp16 encoder)"

        t = time.perf_counter()
        log.info("GigaAM VAD: начало загрузки Silero ONNX на CUDA и проверочного вызова")
        self.new_vad().accept(np.zeros(1600, dtype=np.float32))
        self._set("vad", "ready", "CUDAExecutionProvider", time.perf_counter() - t)
        log.info("GigaAM VAD: загрузка и проверка завершены, провайдеры %s", self._vad_sess.get_providers())
        log_gpu_memory("VAD")

        t = time.perf_counter()
        log.info("GigaAM ASR: начало загрузки v3_e2e_rnnt на CUDA, кеш %s", self.cfg.emo_cache)
        self._asr = gigaam.load_model("v3_e2e_rnnt", device="cuda:0", fp16_encoder=True,
                                    use_flash=False, download_root=self.cfg.emo_cache)
        if next(self._asr.parameters()).device.type != "cuda":
            raise RuntimeError("Модель GigaAM ASR не загружена на CUDA")
        self._set("asr", "ready", "CUDA:0, fp16 encoder / fp32 head", time.perf_counter() - t)
        log.info("GigaAM ASR: загрузка завершена")
        log_gpu_memory("ASR")

        t = time.perf_counter()
        log.info("GigaAM CAM++: начало создания SpeakerEmbeddingExtractor, модель %s, provider=cuda", self.cfg.spk_model)
        self._spk = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self.cfg.spk_model, num_threads=self.cfg.spk_threads, provider="cuda"))
        self._set("spk", "ready", "CUDA:0, CAM++", time.perf_counter() - t)
        log.info("GigaAM CAM++: загрузка завершена, размер вектора %d", self._spk.dim)
        log_gpu_memory("CAM++")

    def new_vad(self):
        cfg = self.cfg
        if self._vad_sess is None:
            log.info("GigaAM VAD: создание ONNX-сессии CUDAExecutionProvider")
            self._vad_sess = make_onnx_session(default_model_path(), provider="CUDAExecutionProvider")
            log.info("GigaAM VAD: ONNX-сессия создана")
        return StreamingVad(self._vad_sess, threshold=cfg.vad_threshold, min_silence=cfg.vad_min_silence,
                            min_speech=cfg.vad_min_speech, max_speech=cfg.vad_max_speech)

    def load_emotions(self, attempts: int = 1, pause: float = 0) -> None:
        log.info("GigaAM эмоции: начало загрузки emo на CUDA")
        import gigaam

        t = time.perf_counter()
        self._emo_stream = self._torch.cuda.Stream(device=0)
        self._emo = gigaam.load_model("emo", device="cuda:0", fp16_encoder=True,
                                    use_flash=False, download_root=self.cfg.emo_cache)
        if next(self._emo.parameters()).device.type != "cuda":
            raise RuntimeError("Модель эмоций не загружена на CUDA")
        log.info("GigaAM эмоции: веса загружены; начало проверочного вызова")
        self.emotions(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self._set("emo", "ready", "CUDA:0, fp16 encoder / fp32 head", time.perf_counter() - t)
        log.info("GigaAM эмоции: загрузка и проверка завершены")
        log_gpu_memory("эмоций")

    def transcribe(self, samples):
        from gigaam.timestamps_utils import compute_frame_shift

        torch, model = self._torch, self._asr
        # Синхронизация находится внутри измеряемого вызова: не меряем отправку ядра.
        self._asr_stream.wait_stream(torch.cuda.current_stream(0))
        with torch.cuda.stream(self._asr_stream), torch.inference_mode():
            wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).to("cuda:0").unsqueeze(0)
            length = torch.full([1], wav.shape[-1], device="cuda:0", dtype=torch.int64)
            encoded, encoded_len = model.forward(wav, length)
            encoded = encoded.to(next(model.head.parameters()).dtype)
            text, ids, frames = model.decoding.decode(model.head, encoded, encoded_len)[0]
            shift = compute_frame_shift(len(samples), int(encoded_len[0].item()))
            tokens = [model.decoding.tokenizer.id_to_str(i) for i in ids]
            stamps = [float(frame * shift) for frame in frames]
        self._asr_stream.synchronize()
        return text.strip(), tokens, stamps

    def emotions(self, samples):
        torch, model = self._torch, self._emo
        self._emo_stream.wait_stream(torch.cuda.current_stream(0))
        with torch.cuda.stream(self._emo_stream), torch.inference_mode():
            wav = torch.from_numpy(np.ascontiguousarray(samples[:12 * SAMPLE_RATE], dtype=np.float32))
            wav = wav.to("cuda:0").unsqueeze(0)
            length = torch.full([1], wav.shape[-1], device="cuda:0", dtype=torch.int64)
            encoded, _ = model.forward(wav, length)
            pooled = torch.nn.functional.avg_pool1d(encoded, kernel_size=encoded.shape[-1]).squeeze(-1)
            # GigaAM.forward возвращает fp16 с CUDA, а голова Emo остаётся fp32.
            pooled = pooled.to(next(model.head.parameters()).dtype)
            probs = torch.nn.functional.softmax(model.head(pooled)[0], dim=-1).cpu().tolist()
        self._emo_stream.synchronize()
        return {str(model.id2name[i]): float(p) for i, p in enumerate(probs)}

    def placement(self) -> dict:
        return {"asr": "PyTorch CUDA:0, fp16 encoder / fp32 head",
                "speaker": "sherpa-onnx CUDA:0, fp32 CAM++",
                "emotion": "PyTorch CUDA:0, fp16 encoder / fp32 head",
                "vad": self._vad_sess.get_providers()}
