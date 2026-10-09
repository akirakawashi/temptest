"""Проверка настоящего декодера на синтетическом звуке, без ASR/GPU/весов."""
from __future__ import annotations

import importlib.metadata
import json
import math
import struct
import tempfile
import wave
from pathlib import Path


def decoder_info() -> dict:
    versions = {}
    for name in ("av", "faster-whisper", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "не установлен"
    return {"versions": versions, "decoder": "faster_whisper.audio.decode_audio"}


def synthetic_wav(path: Path, channels: int):
    # 0,25 с, 8 кГц: проверяем открытие, downmix и ресэмплинг до 16 кГц.
    pcm = b"".join(struct.pack("<h", int(4000 * math.sin(2 * math.pi * 440 * i / 8000))) * channels
                   for i in range(2000))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(pcm)


def check_decoder() -> dict:
    import numpy as np
    from faster_whisper.audio import decode_audio

    result = {**decoder_info(), "synthetic_tests": [], "uses_user_audio": False,
              "loads_asr_models": False, "uses_gpu": False}
    with tempfile.TemporaryDirectory(prefix="speech-audio-check-") as temporary:
        for channels in (1, 2):
            path = Path(temporary) / f"synthetic-{channels}ch.wav"
            synthetic_wav(path, channels)
            # Даже если ffmpeg есть в PATH, проверяем именно PyAV/faster-whisper.
            samples = decode_audio(str(path), sampling_rate=16000)
            if samples.ndim != 1 or len(samples) != 4000 or samples.dtype != np.float32:
                raise RuntimeError(f"Декодер вернул неверный формат для {channels} каналов: "
                                   f"shape={samples.shape}, dtype={samples.dtype}")
            if not np.isfinite(samples).all() or not 0 < float(np.max(np.abs(samples))) < 1:
                raise RuntimeError("Декодер вернул повреждённый или пустой сигнал")
            result["synthetic_tests"].append({"source_channels": channels, "source_sample_rate": 8000,
                                             "output_samples": len(samples), "output_sample_rate": 16000})
    return result


if __name__ == "__main__":
    print("Проверка декодирования без моделей и GPU: " +
          json.dumps(check_decoder(), ensure_ascii=False), flush=True)
