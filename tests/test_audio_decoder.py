"""Реальное PyAV-декодирование синтетического звука; никаких моделей и звонков."""
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
import wave
from types import ModuleType

import numpy as np
import pytest

from benchmark import audio_check, compare, tone_run


@pytest.fixture
def native_decoder(monkeypatch):
    pytest.importorskip("av")
    try:
        distribution = importlib.metadata.distribution("faster-whisper")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Для проверки нативного декодера нужны av и faster-whisper")
    # Импортируем настоящий audio.py закреплённой библиотеки без __init__,
    # который импортирует ASR-классы и CTranslate2. Декодер требует только av/numpy.
    path = distribution.locate_file("faster_whisper/audio.py")
    spec = importlib.util.spec_from_file_location("faster_whisper.audio", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "faster_whisper", ModuleType("faster_whisper"))
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", module)
    return module.decode_audio


def test_build_check_uses_real_av_on_mono_and_stereo(native_decoder):
    result = audio_check.check_decoder()
    assert result["versions"]["av"] == "15.1.0"
    assert result["versions"]["faster-whisper"] == "1.2.1"
    assert [r["source_channels"] for r in result["synthetic_tests"]] == [1, 2]
    assert all(r["output_samples"] == 4000 for r in result["synthetic_tests"])
    assert not result["loads_asr_models"] and not result["uses_gpu"] and not result["uses_user_audio"]


@pytest.mark.parametrize("channels", [1, 2])
def test_real_mp3_preparation_without_ffmpeg_in_container_matches_decoder(native_decoder, tmp_path, monkeypatch, channels):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg нужен только для создания синтетического MP3 в тесте")
    source_wav = tmp_path / "synthetic.wav"
    source_mp3 = tmp_path / "synthetic.mp3"
    audio_check.synthetic_wav(source_wav, channels)
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(source_wav),
                    "-c:a", "libmp3lame", "-b:a", "32k", str(source_mp3)], check=True, timeout=10)
    decoded = native_decoder(str(source_mp3), sampling_rate=16000)
    assert decoded.dtype == np.float32 and decoded.ndim == 1 and len(decoded) > 0
    assert np.isfinite(decoded).all() and np.max(np.abs(decoded)) > .05
    # Саму подготовку выполняет PyAV, как в контейнере пользователя, без ffmpeg.
    monkeypatch.setattr(compare.shutil, "which", lambda name: None)
    output = tmp_path / "prepared.wav"
    result = compare.prepare_audio(source_mp3, output)
    expected_pcm = (decoded * 32768).clip(-32768, 32767).astype("<i2")
    with wave.open(str(output)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        actual = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    np.testing.assert_array_equal(actual, expected_pcm)
    assert result.source_channels == channels and result.samples == len(decoded)


def test_build_check_does_not_hide_incompatible_open_keyword(monkeypatch):
    def fail(*args, **kwargs):
        raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
    module = ModuleType("faster_whisper.audio")
    module.decode_audio = fail
    monkeypatch.setitem(sys.modules, "faster_whisper", ModuleType("faster_whisper"))
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", module)
    with pytest.raises(TypeError, match="metadata_errors"):
        audio_check.check_decoder()


def test_tone_prepares_real_mp3_corpus_and_warmup_in_new_output_directory(native_decoder, tmp_path, monkeypatch):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg нужен только для создания синтетического MP3 в тесте")
    recordings = tmp_path / "audio"
    recordings.mkdir()
    for channels in (1, 2):
        source = tmp_path / f"synthetic-{channels}.wav"
        audio_check.synthetic_wav(source, channels)
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(source), "-c:a", "libmp3lame",
                        "-b:a", "32k", str(recordings / f"synthetic-{channels}.mp3")], check=True, timeout=10)
    output = tmp_path / "results"
    output.mkdir()
    assert not (output / "временные").exists()
    args = tone_run.parser().parse_args([str(recordings), "--out", str(output), "--expected-files", "2"])
    monkeypatch.setattr(compare.shutil, "which", lambda name: None)
    conditions = {"records": [], "vad_settings": dict(tone_run.VAD_SETTINGS)}
    rows = tone_run.prepare(args, conditions)
    assert len(rows) == len(conditions["records"]) == 2
    assert {row["audio"].source_channels for row in rows} == {1, 2}
    assert len(list((output / "временные").glob("*.wav"))) == 3
    for audio in [row["audio"] for row in rows] + [compare.Audio(**conditions["warmup"]["audio"])]:
        samples = compare.read_audio(audio)
        assert len(samples) == audio.samples and audio.samples > 0 and np.isfinite(samples).all()


def test_decoder_versions_are_available_before_preparation_without_model_imports():
    result = audio_check.decoder_info()
    assert set(result["versions"]) == {"av", "faster-whisper", "numpy"}
    assert result["decoder"] == "faster_whisper.audio.decode_audio"
