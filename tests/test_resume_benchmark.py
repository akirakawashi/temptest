"""Продолжение после Whisper: никаких повторных ASR-запросов или настоящих моделей."""
import asyncio
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from benchmark import artifacts, compare, server_run
from test_benchmark import write_wav


def saved_whisper(tmp_path, monkeypatch):
    source, previous = tmp_path / "audio", tmp_path / "previous"
    source.mkdir()
    previous.mkdir()
    write_wav(source / "ncc.wav", 200)
    write_wav(source / "telphin.wav", 300)

    class API:
        def __init__(self, *args, **kwargs):
            self.base_url = args[0]
        async def check(self):
            return {}
        async def health(self):
            return {}
        async def close(self):
            pass
        async def run(self, path, seconds):
            return {"status": "ok", "elapsed_seconds": 1.25, "text": "Результат Whisper",
                    "mode": "test_server_api", "request_started_at": "2026-10-08T14:20:00+03:00"}

    async def no_sleep(*args):
        pass
    monkeypatch.setattr(server_run, "WhisperAPI", API)
    monkeypatch.setattr(server_run.asyncio, "sleep", no_sleep)
    args = compare.parser().parse_args(["--include-first-line", "--audio-dir", str(source),
        "--out", str(previous), "--expected-files", "2"])
    assert asyncio.run(server_run.whisper_phase(args)) == 0
    logs = previous / "логи"
    logs.mkdir()
    (logs / "тестовый-whisper-остановлен.txt").write_text("Whisper остановлен")
    (logs / "gpu.csv").write_text("timestamp, phase\n2026-10-08, Whisper\n")
    (previous / "прогон.log").write_text("Исходный лог Whisper\n")
    artifacts.finalize(previous, 1)  # Как остановка лаунчера на загрузке GigaChat: WAV уже удалены.

    def forbidden(*args, **kwargs):
        raise AssertionError("При продолжении Whisper не должен использоваться")
    monkeypatch.setattr(server_run, "WhisperAPI", forbidden)
    return source, previous


def resume_args(source, previous, output):
    return ["--phase", "gigaam-resume", "--audio-dir", str(source), "--resume-from", str(previous),
            "--out", str(output), "--expected-files", "2"]


def test_resume_preserves_whisper_and_runs_both_gigaam_phases_on_identical_pcm(tmp_path, monkeypatch):
    source, previous = saved_whisper(tmp_path, monkeypatch)
    original = {p.relative_to(previous): p.read_bytes() for p in previous.rglob("*") if p.is_file()}
    output = tmp_path / "resumed"
    assert not (previous / "временные").exists()
    assert compare.main(resume_args(source, previous, output)) == 0
    conditions = json.loads((output / "условия.json").read_text())
    assert conditions["resume"]["pcm_verified"] is True
    assert conditions["state"] == "Whisper завершён"
    assert conditions["phases"]["whisper"]["state"] == "Whisper завершён"
    assert (output / "источник-whisper/логи/gpu.csv").read_bytes() == original[Path("логи/gpu.csv")]
    for directory in ("0001", "0002"):
        assert (output / f"записи/{directory}/whisper.json").read_bytes() == original[Path(f"записи/{directory}/whisper.json")]
    calls = []

    async def worker(request):
        audio, warmup = compare.Audio(**request["audio"]), compare.Audio(**request["warmup"])
        compare.read_audio(audio)
        compare.read_audio(warmup)
        calls.append((request["system"], audio.sha256_pcm, warmup.sha256_pcm))
        return {"result": {"status": "ok", "elapsed_seconds": 2, "asr_seconds": .5,
                           "text": "GigaAM", "stages": {}, "utterances": []},
                "preparation": {"system": request["system"], "gpu": {"uuid": "test"}}}

    monkeypatch.setattr(compare, "run_worker", worker)
    args = resume_args(source, previous, output)
    args[1] = "gigaam"
    assert compare.main(args) == 0
    (output / "логи/тестовая-ollama-остановлена.txt").touch()
    args[1] = "gigaam-first-line"
    assert compare.main(args) == 0
    assert [c[0] for c in calls] == ["gigaam", "gigaam", "gigaam_first_line", "gigaam_first_line"]
    assert [c[1:] for c in calls[:2]] == [c[1:] for c in calls[2:]]
    summary = json.loads((output / "итоги.json").read_text())
    assert summary["matched_comparison"]["records"] == 2
    assert summary["systems"]["whisper"]["processing_seconds"]["mean"] == 1.25
    assert summary["corpus_sha256"] == json.loads((previous / "итоги.json").read_text())["corpus_sha256"]
    assert "замеры Whisper перенесены без изменения" in (output / "отчёт.html").read_text()
    artifacts.finalize(output, 0)
    assert not (output / "временные").exists()
    assert {p.relative_to(previous): p.read_bytes() for p in previous.rglob("*") if p.is_file()} == original


@pytest.mark.parametrize("failure", ["pcm", "count", "names", "whisper", "marker", "gigaam", "phase", "gpu"])
def test_resume_rejects_changed_or_incomplete_corpus_before_inference(tmp_path, monkeypatch, failure):
    source, previous = saved_whisper(tmp_path, monkeypatch)
    if failure == "pcm":
        write_wav(source / "ncc.wav", 201)
    elif failure == "count":
        (source / "ncc.wav").unlink()
    elif failure == "names":
        (source / "ncc.wav").rename(source / "new.wav")
    elif failure == "whisper":
        (previous / "записи/0001/whisper.json").write_text('{"status":"error"}')
    elif failure == "marker":
        (previous / "логи/тестовый-whisper-остановлен.txt").unlink()
    elif failure == "gigaam":
        (previous / "записи/0001/gigaam.json").write_text('{"status":"ok"}')
    elif failure == "phase":
        manifest = previous / "условия.json"
        value = json.loads(manifest.read_text())
        value["phases"]["whisper"]["state"] = "Whisper выполняется"
        compare.write_json(manifest, value)
    else:
        monkeypatch.setenv("BENCH_GPU", "1")
    original = {p.relative_to(previous): p.read_bytes() for p in previous.rglob("*") if p.is_file()}
    args = resume_args(source, previous, tmp_path / "resumed")
    if failure == "pcm":
        assert compare.main(args) == 1
    else:
        with pytest.raises(ValueError):
            compare.main(args)
    assert {p.relative_to(previous): p.read_bytes() for p in previous.rglob("*") if p.is_file()} == original
    assert not list((tmp_path / "resumed").rglob("gigaam.json"))


@pytest.mark.parametrize("cached,pull_exit", [(True, 0), (False, 0), (False, 17)])
def test_llm_preparation_uses_cached_model_without_pull(tmp_path, cached, pull_exit):
    """Исполняем фактическую команду подготовки из лаунчера с поддельным Ollama CLI."""
    script = (Path(__file__).resolve().parents[1] / "benchmark/run.sh").read_text()
    start = script.index("'if ollama show") + 1
    command = script[start:script.index("fi'", start) + 2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    ollama = bin_dir / "ollama"
    ollama.write_text(f'#!/bin/sh\nprintf "%s\\n" "$1" >> "{calls}"\n'
                      f'if [ "$1" = show ]; then exit {0 if cached else 1}; fi\nexit {pull_exit}\n')
    ollama.chmod(0o755)
    result = subprocess.run([shutil.which("sh"), "-c", command],
                            env={"PATH": str(bin_dir), "LLM_MODEL": "test-model"},
                            capture_output=True, text=True, timeout=5)
    assert calls.read_text().splitlines() == (["show"] if cached else ["show", "pull"])
    assert result.returncode == (0 if cached else pull_exit)
