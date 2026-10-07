"""Запуск API только с заглушками: без Docker, CUDA и реальных моделей."""
import asyncio
from dataclasses import asdict
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark import compare, whisper_server


@pytest.mark.parametrize("device,free,allowed", [("cuda", 14000, True), ("cpu", 14000, False), ("cuda", 3000, False)])
def test_server_checks_actual_model_placement_memory_and_reports_versions(tmp_path, monkeypatch, device, free, allowed):
    monkeypatch.setattr(sys, "argv", ["whisper_server"])
    monkeypatch.setattr(whisper_server.importlib.metadata, "version", lambda name: whisper_server.EXPECTED[name])
    for key, value in {"WHISPER_DEVICE": "cuda", "WHISPER_COMPUTE_TYPE": "float16", "WHISPER_LOCAL_ONLY": "1",
                       "WHISPER_MODEL": "large-v3", "BENCH_GPU_RESERVE_MIB": "4096"}.items():
        monkeypatch.setenv(key, value)
    source = tmp_path / "api_server.py"
    source.write_text("# Тестовая заглушка\n")
    routes, loads = {}, []

    class App:
        def get(self, path, **kwargs):
            def register(function):
                routes[path] = function
                return function
            return register

    api = SimpleNamespace(app=App(), __file__=str(source), _model=SimpleNamespace(
        model=SimpleNamespace(device=device, compute_type="float16")), _load_model=lambda: loads.append("load"))
    monkeypatch.setitem(sys.modules, "api_server", api)
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 1))
    monkeypatch.setattr(whisper_server.subprocess, "check_output", lambda *a, **k: str(free))

    def run(app, **kwargs):
        assert kwargs["workers"] == 1 and kwargs["port"] == 9000
        api._load_model()

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    if allowed:
        whisper_server.main()
        metadata = asyncio.run(routes["/benchmark/metadata"]())
        assert metadata["device"] == "cuda" and metadata["model_loaded"] is True
        assert metadata["versions"] == whisper_server.EXPECTED
        assert metadata["external_network"] is False and metadata["diarization"] is False
        assert metadata["gpu_free_mib_after_load"] == free
    else:
        with pytest.raises(RuntimeError):
            whisper_server.main()
    assert loads == ["load"]


def test_model_download_does_not_inspect_gpu_or_start_api(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["whisper_server", "--download"])
    monkeypatch.setattr(whisper_server.importlib.metadata, "version", lambda name: whisper_server.EXPECTED[name])
    monkeypatch.setenv("WHISPER_MODEL", "large-v3")
    calls = []
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", SimpleNamespace(
        download_model=lambda *args, **kwargs: calls.append((args, kwargs))))
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace())
    whisper_server.main()
    assert calls == [(("large-v3",), {"cache_dir": "/var/lib/whisper"})]


def test_preparation_without_ffmpeg_creates_shared_pcm_with_safe_clipping(tmp_path, monkeypatch):
    monkeypatch.setattr(compare.shutil, "which", lambda name: None)
    decoded = np.array([-1.0, 0.0, 0.5, 1.0, -0.25], dtype=np.float32)
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", SimpleNamespace(decode_audio=lambda *args, **kwargs: decoded))
    audio = compare.prepare_audio(tmp_path / "запись.mp3", tmp_path / "общий.wav", max_seconds=4 / 16000)
    result = compare.read_audio(audio)
    assert audio.samples == 4 and len(audio.sha256_pcm) == 64
    assert result.tolist() == [-1.0, 0.0, 0.5, 32767 / 32768]
    assert compare.Audio(**asdict(audio)) == audio
