"""Без запросов на рабочий сервер, Docker, скачиваний и GPU inference."""
import ast
import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile

import httpx
import pytest

from benchmark import artifacts, compare, server_run
from benchmark.whisper_api import TEST_URL, WhisperAPI, speech_seconds, validate_url
from test_benchmark import install_fake_runners, write_wav


@pytest.mark.parametrize("phase,count,expected_code", [
    ("gigaam-prepare", "2", 0), ("gigaam-prepare", "3", 1), ("whisper-api", "3", 1),
])
def test_cpu_preparation_and_artifacts_work_with_only_benchmark_package(tmp_path, phase, count, expected_code):
    """Как COPY benchmark в CPU-образе: app отсутствует, моделей и API нет."""
    project = Path(__file__).resolve().parents[1]
    image = tmp_path / "cpu-image"
    shutil.copytree(project / "benchmark", image / "benchmark", ignore=shutil.ignore_patterns("__pycache__"))
    source, output = tmp_path / "audio", tmp_path / "results"
    source.mkdir()
    write_wav(source / "1.wav", 100)
    write_wav(source / "2.wav", 200)
    bootstrap = '''
import importlib.abc, os, runpy, sys, types, wave
sys.path.insert(0, sys.argv.pop(1))
sys.path.append(os.environ["CPU_TEST_NUMPY_PATH"])
module = sys.argv.pop(1)
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"app", "torch", "gigaam", "sherpa_onnx", "onnxruntime", "ctranslate2"}:
            raise ModuleNotFoundError("Недоступно в CPU-контейнере: " + fullname)
sys.meta_path.insert(0, NoModels())
if module == "benchmark.compare":
    import numpy as np
    from benchmark import compare
    compare.shutil.which = lambda name: None
    def decode_audio(path, sampling_rate):
        with wave.open(path, "rb") as wav:
            return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    decoder = types.ModuleType("faster_whisper.audio")
    decoder.decode_audio = decode_audio
    sys.modules["faster_whisper"] = types.ModuleType("faster_whisper")
    sys.modules["faster_whisper.audio"] = decoder
runpy.run_module(module, run_name="__main__")
'''
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "LLM_NUM_CTX": "8192",
           "CPU_TEST_NUMPY_PATH": str(Path(sys.modules["numpy"].__file__).resolve().parents[1]),
           "BENCH_LLM_NUM_BATCH": "64", "BENCH_LLM_KV_CACHE_TYPE": "q8_0"}
    command = [sys.executable, "-I", "-c", bootstrap, str(image), "benchmark.compare",
               "--phase", phase, "--audio-dir", str(source), "--out", str(output),
               "--expected-files", count, "--threads", "1"]
    prepared = subprocess.run(command, cwd=image, env=env, capture_output=True, text=True, timeout=15)
    assert prepared.returncode == expected_code, prepared.stdout + prepared.stderr
    conditions = json.loads((output / "условия.json").read_text())
    assert conditions["gigaam"]["llm_runtime"]["options"]["num_ctx"] == 8192
    if expected_code == 0:
        assert conditions["state"] == "GigaAM подготовлен" and len(conditions["records"]) == 2
        assert len(list((output / "временные").glob("*.wav"))) == 3
    assert not list(output.rglob("gigaam.json")) and not list(output.rglob("whisper.json"))
    finalized = subprocess.run([sys.executable, "-I", "-c", bootstrap, str(image), "benchmark.artifacts",
                                str(output), "--exit-code", str(expected_code)], cwd=image, env=env,
                               capture_output=True, text=True, timeout=15)
    assert finalized.returncode == 0, finalized.stdout + finalized.stderr
    assert not (output / "временные").exists()
    with tarfile.open(output / "диагностика.tar.gz") as archive:
        assert "отчёт.html" in archive.getnames()
        assert all(not name.endswith(".wav") for name in archive.getnames())


def test_gpu_reserve_stops_only_benchmark_model_loading(monkeypatch):
    from types import SimpleNamespace
    from benchmark.gpu import log_gpu_memory

    monkeypatch.setenv("BENCH_GPU_RESERVE_MIB", "4096")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        mem_get_info=lambda _: (3000 * 1024 ** 2, 95830 * 1024 ** 2),
        memory_allocated=lambda _: 1, memory_reserved=lambda _: 1)))
    with pytest.raises(RuntimeError, match="Останавливаем только стенд"):
        log_gpu_memory("загрузки ASR")


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "http://1.1.1.1/v1", "http://user:password@10.0.0.1/v1", "http://10.0.0.1/v1?token=abc"])
def test_external_endpoint_credentials_and_query_are_rejected(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_detected_duration_uses_union_not_sum_of_overlapping_segments():
    assert speech_seconds([{"start": 0, "end": 3, "text": "А"}, {"start": 2, "end": 6, "text": "Б"},
                           {"start": 6, "end": 8, "text": ""}], 5) == 5


@pytest.mark.parametrize("failure", [429, 503, "timeout", "redirect", "malformed"])
def test_api_never_retries_or_follows_redirects_and_redacts_key(tmp_path, failure):
    path = tmp_path / "audio.wav"
    write_wav(path, 100)
    requests = []
    key = "secret-test-key"

    def response(request):
        requests.append(request)
        assert request.headers["authorization"] == f"Bearer {key}"
        if failure == "timeout":
            raise httpx.ReadTimeout("Тест", request=request)
        if failure == "redirect":
            return httpx.Response(307, headers={"location": "https://external.example/v1"})
        if failure == "malformed":
            return httpx.Response(200, json={"wrong": key})
        return httpx.Response(failure, json={"error": key})

    async def scenario():
        client = WhisperAPI("http://10.220.21.2:8002/v1", key, "large-v3", 1200, httpx.MockTransport(response))
        try:
            result = await client.run(path, 0.2)
            assert result["status"] == "error" and result["attempts"] == 1
            assert len(requests) == 1
            assert key not in json.dumps(result)
            assert result["server_may_still_be_processing"] is (failure == "timeout")
        finally:
            await client.close()
    asyncio.run(scenario())


def test_whisper_uses_only_health_models_and_transcription_endpoints(tmp_path):
    path = tmp_path / "audio.wav"
    write_wav(path, 100)
    requests = []

    def response(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "model": "large-v3"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "large-v3"}]})
        body = request.read()
        assert b'verbose_json' in body and b'vad_filter' in body and b'large-v3' in body
        return httpx.Response(200, json={"text": "Тест", "segments": [{"start": 0, "end": 0.2, "text": "Тест"}]})

    async def scenario():
        client = WhisperAPI("http://10.220.21.2:8002/v1", "test-key", "large-v3", 1200, httpx.MockTransport(response))
        try:
            await client.check()
            result = await client.run(path, 0.2)
            assert result["speech_seconds"] == 0.2
            assert result["server_processing_seconds"] is None
        finally:
            await client.close()
    asyncio.run(scenario())
    assert requests == [("GET", "/health"), ("GET", "/v1/models"), ("POST", "/v1/audio/transcriptions")]


def test_wrong_model_is_rejected_before_transcription():
    def response(request):
        assert request.method == "GET"
        return httpx.Response(200, json={"status": "ok", "data": [{"id": "other-model"}]})

    async def scenario():
        client = WhisperAPI("http://10.0.0.1/v1", "key", "large-v3", 10, httpx.MockTransport(response))
        try:
            with pytest.raises(RuntimeError, match="менять/загружать"):
                await client.check()
        finally:
            await client.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [False, True])
def test_all_whisper_requests_precede_gigaam_and_error_blocks_phase(tmp_path, monkeypatch, failure):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    write_wav(source / "1.wav", 100)
    write_wav(source / "2.wav", 200)
    calls = install_fake_runners(monkeypatch)
    api_calls = []

    class API:
        def __init__(self, *args, **kwargs):
            assert args[0] == TEST_URL and args[1] == "" and kwargs["standalone"] is True
            self.base_url = TEST_URL
        async def check(self):
            return {"status": "ok"}
        async def health(self):
            return {"status": "ok"}
        async def close(self):
            pass
        async def run(self, path, duration):
            api_calls.append(path.name)
            return {"status": "error" if failure else "ok", "mode": "test_server_api", "elapsed_seconds": 1,
                    "segments": [], "text": "Тест", "speech_seconds": 0, "error": "Ошибка" if failure else None}

    async def no_sleep(*args):
        pass
    monkeypatch.setattr(server_run, "WhisperAPI", API)
    monkeypatch.setattr(server_run.asyncio, "sleep", no_sleep)
    args = compare.parser().parse_args(["--audio-dir", str(source), "--out", str(output), "--expected-files", "2", "--threads", "1"])
    assert asyncio.run(server_run.whisper_phase(args)) == int(failure)
    assert not calls  # GigaAM не загрузилась во время фазы Whisper.
    if failure:
        assert len(api_calls) == 1
        with pytest.raises(ValueError):
            asyncio.run(server_run.gigaam_phase(args))
    else:
        assert len(api_calls) == 3  # Отдельный прогрев, затем две измеряемые записи.
        assert len(list(output.glob("записи/*/whisper.json"))) == 2
        conditions = json.loads((output / "условия.json").read_text())
        assert conditions["whisper"]["warmup"]["included_in_measurements"] is False
        with pytest.raises(ValueError, match="остановить тестовый Whisper"):
            asyncio.run(server_run.gigaam_phase(args))
        assert not calls
        (output / "логи").mkdir()
        (output / "логи" / "тестовый-whisper-остановлен.txt").write_text("Остановлен")
        assert asyncio.run(server_run.gigaam_phase(args)) == 0
        assert [name for name, _ in calls] == ["gigaam", "gigaam"]
        assert (output / "отчёт.html").is_file()
        artifacts.finalize(output, 0)
        assert not (output / "временные").exists()
        with tarfile.open(output / "диагностика.tar.gz") as archive:
            assert "отчёт.html" in archive.getnames()
            assert all(not name.endswith(".wav") for name in archive.getnames())


def test_incorrect_corpus_size_sends_no_requests(tmp_path, monkeypatch):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    write_wav(source / "1.wav", 100)
    def forbidden(*args):
        raise AssertionError("Клиент не должен создаваться")
    monkeypatch.setattr(server_run, "WhisperAPI", forbidden)
    args = compare.parser().parse_args(["--audio-dir", str(source), "--out", str(output), "--expected-files", "100"])
    assert asyncio.run(server_run.whisper_phase(args)) == 1


@pytest.mark.parametrize("failure", [None, "count", "duration", "warmup"])
def test_gigaam_only_prepares_new_corpus_without_whisper_and_preserves_previous_results(tmp_path, monkeypatch, failure):
    import wave
    from benchmark import pipeline

    monkeypatch.setenv("BENCH_LLM_NUM_BATCH", "64")
    monkeypatch.setenv("BENCH_LLM_KV_CACHE_TYPE", "q8_0")
    source, output = tmp_path / "input", tmp_path / "new-results"
    source.mkdir()
    write_wav(source / "1.wav", 100)
    write_wav(source / "2.wav", 200)
    if failure == "duration":
        with wave.open(str(source / "1.wav"), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b'\0\0' * 32000)
    previous = tmp_path / "old-results" / "записи" / "0001" / "whisper.json"
    previous.parent.mkdir(parents=True)
    previous.write_text('{"text":"Сохранённый результат Whisper"}\n')
    original = previous.read_bytes()
    calls = install_fake_runners(monkeypatch, warm_giga_status="error" if failure == "warmup" else "ok")
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append(args)
        raise AssertionError("Whisper и его API не должны использоваться")

    monkeypatch.setattr(server_run, "WhisperAPI", forbidden)
    monkeypatch.setattr(pipeline, "WhisperPipeline", forbidden)
    args = ["--phase", "gigaam-prepare", "--audio-dir", str(source), "--out", str(output),
            "--expected-files", "3" if failure == "count" else "2", "--threads", "1", "--api-gap", "0"]
    if failure == "duration":
        args.extend(["--max-audio-seconds", "1"])
    code = compare.main(args)
    assert code == int(failure in {"count", "duration"})
    assert not calls
    if code == 0:
        args[1] = "gigaam-only"
        code = compare.main(args)
    assert code == int(failure is not None)
    assert not forbidden_calls
    assert previous.read_bytes() == original
    assert not list(output.rglob("whisper.json"))
    conditions = json.loads((output / "условия.json").read_text())
    assert conditions["mode"] == "gigaam_only" and conditions["whisper"]["enabled"] is False
    runtime = conditions["gigaam"]["llm_runtime"]
    assert runtime["options"]["num_ctx"] == 8192 and runtime["options"]["num_batch"] == 64
    assert runtime["options"]["num_gpu"] == 999 and runtime["kv_cache_type_requested"] == "q8_0"
    if failure:
        assert not calls
        assert conditions["state"].startswith("Остановлен")
    else:
        assert [name for name, _ in calls] == ["gigaam", "gigaam"]
        assert len(list(output.glob("записи/*/gigaam.json"))) == 2
        assert conditions["state"] == "Завершён"
        assert all(record["order"] == ["gigaam"] for record in conditions["records"])
        assert conditions["warmup"]["included_in_measurements"] is False
        assert "GigaAM — полный цикл: 6.000 с" in (output / "отчёт.md").read_text()
        assert "1.400 с" in (output / "графики" / "итоги.svg").read_text()
        assert "3.000 с" in (output / "графики" / "этапы-gigaam.svg").read_text()
        assert "успешных результатов GigaAM: 2" in (output / "отчёт.html").read_text()
        assert "Контекст LLM: 8192; батч: 64; KV-кеш (задано): q8_0" in (output / "отчёт.html").read_text()
        assert all(item["llm_runtime"] == runtime for item in conditions["preparations"])
        with pytest.raises(ValueError, match="повторного запуска"):
            compare.main(args)
        args[1] = "gigaam-prepare"
        with pytest.raises(SystemExit) as error:
            compare.main(args)
        assert error.value.code == 2
    artifacts.finalize(output, code)
    assert not (output / "временные").exists()
    with tarfile.open(output / "диагностика.tar.gz") as archive:
        assert "отчёт.html" in archive.getnames()
        assert not any(name.endswith(("whisper.json", ".wav")) for name in archive.getnames())
    if not failure:
        assert "GigaAM — полный цикл: 6.000 с" in (output / "отчёт.md").read_text()


@pytest.mark.parametrize("failure", [None, "build", "prefetch", "gigaam", "whisper-running", "hup"])
def test_gigaam_only_launcher_starts_no_whisper_and_preserves_old_results(tmp_path, failure):
    folder = tmp_path / "stand"
    (folder / "benchmark").mkdir(parents=True)
    (folder / "audio").mkdir()
    script = Path(__file__).resolve().parents[1] / "benchmark" / "run.sh"
    (folder / "benchmark" / "run.sh").write_bytes(script.read_bytes())
    previous = folder / "benchmark-results" / "previous" / "whisper.json"
    previous.parent.mkdir(parents=True)
    previous.write_text("Сохранённый Whisper")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    command_log = tmp_path / "commands.log"
    ready = tmp_path / "gigaam-ready"
    docker = bin_dir / "docker"
    docker.write_text('#!/usr/bin/env python3\nimport os,sys,time\nfrom pathlib import Path\na=sys.argv[1:]; failure=os.environ["FAILURE"]\n'
                      'with open(os.environ["COMMAND_LOG"],"a") as f: f.write(repr(a)+"\\n")\n'
                      'if a[0]=="compose" and "ps" in a and failure=="whisper-running": print("own-whisper-id")\n'
                      'if a[0]=="compose" and "build" in a and failure=="build": sys.exit(23)\n'
                      'if a[0]=="info": print("/tmp")\n'
                      'if a[0]=="wait":\n'
                      ' if "-gigaam-" in a[-1] and failure=="hup":\n'
                      '  Path(os.environ["READY"]).touch()\n'
                      '  while True: time.sleep(60)\n'
                      ' print(51 if "-prefetch-" in a[-1] and failure=="prefetch" else 47 if "-gigaam-" in a[-1] and failure=="gigaam" else 0)\n'
                      'if a[0]=="logs" and "--follow" in a:\n'
                      ' print("Тест: журнал GigaAM",flush=True)\n'
                      ' while True: time.sleep(60)\n'
                      'sys.exit(0)\n')
    nvidia = bin_dir / "nvidia-smi"
    nvidia.write_text('#!/usr/bin/env python3\nimport sys\na=" ".join(sys.argv)\n'
                      'print("14600" if "--query-gpu=memory.free" in a else "100" if "--query-gpu=utilization.gpu" in a else "2026/10/08, uuid, 100, 1000, 14600, 95830, 80, 32")\n')
    docker.chmod(0o755)
    nvidia.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "COMMAND_LOG": str(command_log),
           "FAILURE": failure or "", "READY": str(ready), "BENCH_GPU_MIN_FREE_MIB": "12288",
           "BENCH_GPU_MAX_UTIL": "100", "BENCH_MIN_RAM_MIB": "0", "BENCH_MIN_DISK_MIB": "0"}
    with (tmp_path / "terminal.log").open("w") as terminal:
        process = subprocess.Popen(["bash", str(folder / "benchmark" / "run.sh"), str(folder / "audio"),
                                    "--gigaam-only", "--expected-files", "2"], env=env, stdin=subprocess.DEVNULL,
                                   stdout=terminal, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if failure == "hup":
                import time
                deadline = time.monotonic() + 10
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert ready.exists()
                os.killpg(process.pid, signal.SIGHUP)
            code = process.wait(timeout=15)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    assert code == {None: 0, "build": 23, "prefetch": 51, "gigaam": 47, "whisper-running": 42, "hup": 129}[failure]
    assert previous.read_text() == "Сохранённый Whisper"
    commands = [ast.literal_eval(line) for line in command_log.read_text().splitlines()]
    for command in commands:
        assert not any(name in str(command) for name in ("whisper-asr", "model-proxy", "vllm", "10.220.21.2"))
        assert not {"--rm", "--remove-orphans", "rm", "down"} & set(command)
        assert not any(name in str(command) for name in ("whisper-download", "whisper-api"))
        if "whisper-client" in command:
            assert command[0] == "compose" and "build" in command
        if "whisper-bench" in command:
            assert command[0] == "compose" and "ps" in command
        if command[0] == "compose" and command[1:] != ["version"]:
            assert command[command.index("--project-name") + 1] == "speech-comparison"
    gigaam_runs = [c for c in commands if c[0] == "compose" and "run" in c and "gigaam-only" in c]
    assert bool(gigaam_runs) is (failure not in {"build", "prefetch", "whisper-running"})
    if gigaam_runs:
        assert "--expected-files" in gigaam_runs[0] and "2" in gigaam_runs[0]
        assert "compare" in gigaam_runs[0] and "--detach" in gigaam_runs[0]
        preparation = next(c for c in commands if "audio-prepare" in c and "run" in c)
        assert "gigaam-prepare" in preparation
        assert commands.index(preparation) < commands.index(gigaam_runs[0])
    logs = list((folder / "benchmark-results").glob("*/логи/запуск.log"))
    assert len(logs) == 1
    if failure == "hup":
        assert "Получен SIGHUP" in logs[0].read_text()
        assert "Завершение: код 129" in logs[0].read_text()


@pytest.mark.parametrize("free, util, min_free, max_util, api_exit, client_build_exit, whisper_start_exit, whisper_stop_exit, whisper_download_exit, ollama_download_start_exit, ignore_log_term, blocked_console", [
    (*case, False, False) for case in [
    (14600, 0, 16384, 10, 0, 0, 0, 0, 0, 0), (90000, 0, 16384, 10, 47, 0, 0, 0, 0, 0),
    (90000, 0, 16384, 10, 0, 0, 0, 0, 0, 0), (14600, 100, 12288, 10, 0, 0, 0, 0, 0, 0),
    (14600, 100, 12288, 100, 0, 0, 0, 0, 0, 0), (3000, 100, 12288, 100, 0, 0, 0, 0, 0, 0),
    (14600, 100, 12288, 100, 0, 23, 0, 0, 0, 0), (14600, 100, 12288, 100, 0, 0, 31, 0, 0, 0),
    (14600, 100, 12288, 100, 0, 0, 0, 32, 0, 0), (14600, 100, 12288, 100, 0, 0, 0, 0, 51, 0),
    (14600, 100, 12288, 100, 0, 0, 0, 0, 0, 52),
    ]
] + [
    pytest.param(14600, 100, 12288, 100, 0, 0, 0, 0, 0, 0, True, False, id="completed-whisper-log-reader-ignores-sigterm"),
    pytest.param(14600, 100, 12288, 100, 0, 0, 0, 0, 0, 0, False, True, id="blocked-console-does-not-stop-gigaam"),
])
def test_launcher_never_controls_production_and_does_not_start_gigaam_after_failure(tmp_path, free, util, min_free, max_util, api_exit, client_build_exit, whisper_start_exit, whisper_stop_exit, whisper_download_exit, ollama_download_start_exit, ignore_log_term, blocked_console):
    folder = tmp_path / "stand"
    (folder / "benchmark").mkdir(parents=True)
    script = Path(__file__).resolve().parents[1] / "benchmark" / "run.sh"
    (folder / "benchmark" / "run.sh").write_bytes(script.read_bytes())
    (folder / "audio").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    command_log = tmp_path / "commands.log"
    log_reader_ready = tmp_path / "log-reader.ready"
    docker = bin_dir / "docker"
    docker.write_text('#!/usr/bin/env python3\nimport os,sys,time,signal\nfrom pathlib import Path\na=sys.argv[1:]\nwith open(os.environ["COMMAND_LOG"],"a") as f: f.write(repr(a)+"\\n")\n'
                      'if a[0] == "compose" and "build" in a and a[-1] == "whisper-client": sys.exit(int(os.environ["CLIENT_BUILD_EXIT"]))\n'
                      'if a[0] == "compose" and "up" in a and a[-1] == "whisper-bench": sys.exit(int(os.environ["WHISPER_START_EXIT"]))\n'
                      'if a[0] == "compose" and "stop" in a and a[-1] == "whisper-bench": sys.exit(int(os.environ["WHISPER_STOP_EXIT"]))\n'
                      'if a[0] == "compose" and "up" in a and a[-1] == "ollama-download": sys.exit(int(os.environ["OLLAMA_DOWNLOAD_START_EXIT"]))\n'
                      'if a[0] == "compose" and "run" in a: assert sys.stdin.buffer.read(1) == b""\n'
                      'if a[0] == "wait":\n'
                      ' if "-client-" in a[-1] and (os.environ["IGNORE_LOG_TERM"] == "1" or os.environ["BLOCKED_CONSOLE"] == "1"):\n'
                      '  while not Path(os.environ["LOG_READER_READY"]).exists(): time.sleep(0.01)\n'
                      ' print(os.environ["WHISPER_DOWNLOAD_EXIT"] if "-whisper-download-" in a[-1] else os.environ["API_EXIT"] if "-client-" in a[-1] else "0"); sys.exit(0)\n'
                      'if a[0] == "logs" and "--follow" in a:\n'
                      ' if os.environ["IGNORE_LOG_TERM"] == "1" and "-client-" in a[-1]:\n'
                      '  signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
                      ' print("Тест: поток логов открыт после завершения задачи", flush=True)\n'
                      ' if os.environ["BLOCKED_CONSOLE"] == "1" and "-client-" in a[-1]: print("И" * 1024 * 1024, flush=True)\n'
                      ' if "-client-" in a[-1] and (os.environ["IGNORE_LOG_TERM"] == "1" or os.environ["BLOCKED_CONSOLE"] == "1"):\n'
                      '  Path(os.environ["LOG_READER_READY"]).write_text(str(os.getpid()))\n'
                      ' while True: time.sleep(60)\n'
                      'if a[0] == "logs": print("Тест: полный журнал завершённого контейнера")\n'
                      'if "info" in a: print("/tmp")\n'
                      'sys.exit(0)\n')
    nvidia = bin_dir / "nvidia-smi"
    nvidia.write_text('#!/usr/bin/env python3\nimport os,sys\na=" ".join(sys.argv)\n'
                      'print(os.environ["FREE_GPU"] if "--query-gpu=memory.free" in a else os.environ["GPU_UTIL"] if "--query-gpu=utilization.gpu" in a else "2026/10/07, uuid, 0, 1000, 90000, 95830, 80, 32")\n')
    docker.chmod(0o755)
    nvidia.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "COMMAND_LOG": str(command_log),
           "IGNORE_LOG_TERM": "1" if ignore_log_term else "0", "LOG_READER_READY": str(log_reader_ready),
           "BLOCKED_CONSOLE": "1" if blocked_console else "0",
           "FREE_GPU": str(free), "GPU_UTIL": str(util), "BENCH_GPU_MIN_FREE_MIB": str(min_free),
           "BENCH_GPU_MAX_UTIL": str(max_util),
           "BENCH_BUILD_NETWORK": "default",
           "BENCH_DOWNLOAD_NETWORK": "host", "BENCH_OLLAMA_DOWNLOAD_PORT": "11435",
           "API_EXIT": str(api_exit), "CLIENT_BUILD_EXIT": str(client_build_exit),
           "WHISPER_START_EXIT": str(whisper_start_exit), "WHISPER_STOP_EXIT": str(whisper_stop_exit),
           "WHISPER_DOWNLOAD_EXIT": str(whisper_download_exit), "OLLAMA_DOWNLOAD_START_EXIT": str(ollama_download_start_exit),
           "BENCH_MIN_RAM_MIB": "0", "BENCH_MIN_DISK_MIB": "0"}
    # Вход родительского терминала остаётся открытым; поток Docker logs не даёт EOF.
    # Для blocked_console stdout — заполненная труба, которую до выхода никто не читает.
    # Завершение контейнера должно перевести прогон дальше независимо от вывода.
    with (tmp_path / "terminal.log").open("w+") as terminal:
        process = subprocess.Popen(["bash", str(folder / "benchmark" / "run.sh"), str(folder / "audio")], env=env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE if blocked_console else terminal,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=15)
        finally:
            process.stdin.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        terminal.seek(0)
        if blocked_console:
            console = process.stdout.read().decode("utf-8", errors="replace")
            process.stdout.close()
        else:
            console = terminal.read()
        launch_log = next((folder / "benchmark-results").glob("*/логи/запуск.log"))
        result = subprocess.CompletedProcess(process.args, returncode, stdout=launch_log.read_text(), stderr="")
    precheck_failed = free < min_free or util > max_util
    assert result.returncode == (42 if precheck_failed else client_build_exit or whisper_download_exit or whisper_start_exit or api_exit or int(bool(whisper_stop_exit)) or ollama_download_start_exit)
    commands = command_log.read_text()
    builds = [command for line in commands.splitlines()
              if (command := ast.literal_eval(line))[0] == "compose" and "build" in command]
    for build in builds:
        assert build[build.index("--project-name") + 1] == "speech-comparison"
        assert build[build.index("--project-directory") + 1] == str(folder)
        assert build[build.index("-f") + 1] == str(folder / "compose.benchmark.yml")
        assert build[build.index("--progress") + 1] == "plain"
        assert build[-1] in {"whisper-client", "compare"}
    assert "'buildx', 'build'" not in commands
    assert "whisper-asr" not in commands and "model-proxy" not in commands and "vllm" not in commands
    if precheck_failed or client_build_exit or whisper_download_exit or whisper_start_exit or api_exit or whisper_stop_exit:
        assert "'build', 'compare'" not in commands and "'up', '-d', 'ollama'" not in commands
        assert len(builds) == (0 if precheck_failed else 1)
    else:
        assert len(builds) == 2
        assert commands.index("whisper-api") < commands.index("'stop', 'whisper-bench'") < commands.index("'build', 'compare'")
        download_up = next(ast.literal_eval(line) for line in commands.splitlines()
                           if "'up'" in line and "'ollama-download'" in line)
        assert "--wait" in download_up and "--wait-timeout" in download_up
        assert "'stop', 'ollama-download'" in commands
        if ollama_download_start_exit:
            assert "'exec', '--interactive=false', '-T', 'ollama-download'" not in commands
            assert "'up', '-d', 'ollama'" not in commands
        else:
            assert commands.index("'build', 'compare'") < commands.index("'up', '-d', 'ollama'")
            assert "'stop', 'ollama'" in commands
            waits = [ast.literal_eval(line) for line in commands.splitlines()
                     if ast.literal_eval(line)[0] == "wait"]
            assert len(waits) == 5
            assert all(task[1].startswith(prefix) for task, prefix in zip(waits, (
                "speech-comparison-whisper-download-", "speech-comparison-client-",
                "speech-comparison-prefetch-", "speech-comparison-gigaam-", "speech-comparison-first-line-"), strict=True))
            assert commands.index("'stop', 'ollama'") < commands.index("gigaam-first-line")
    if whisper_download_exit:
        assert "whisper-api" not in commands and "'up', '-d', '--wait', '--wait-timeout', '600', 'whisper-bench'" not in commands
        download_logs = list((folder / "benchmark-results").glob("*/логи/загрузка-whisper.log"))
        assert len(download_logs) == 1
        assert "полный журнал завершённого контейнера" in download_logs[0].read_text()
    if precheck_failed or client_build_exit:
        assert "whisper-api" not in commands
        assert "file changed as we read it" not in result.stdout + result.stderr
        archives = list((folder / "benchmark-results").glob("*/диагностика.tar.gz"))
        assert len(archives) == 1
        with tarfile.open(archives[0]) as archive:
            assert "./логи/запуск.log" in archive.getnames()
            assert all(not name.endswith(".tar.gz") for name in archive.getnames())
            log_text = archive.extractfile("./логи/запуск.log").read().decode()
            assert ("Стенд не запускается" if precheck_failed else f"Завершение: код {client_build_exit}") in log_text
        assert not list((folder / "benchmark-results").glob("*.tar.part"))
    for line in commands.splitlines():
        command = ast.literal_eval(line)
        assert "--rm" not in command and "--remove-orphans" not in command
        assert "rm" not in command and "down" not in command
        if command[0] == "compose" and "run" in command:
            assert "--detach" in command and "--interactive=false" in command and "-T" in command
    assert "10.220.21.2" not in commands and "WHISPER_API_KEY" not in commands
    if ignore_log_term:
        assert log_reader_ready.is_file()
        with pytest.raises(ProcessLookupError):
            os.kill(int(log_reader_ready.read_text()), 0)
        assert "Docker сообщил о завершении задачи speech-comparison-client-" in result.stdout
    if blocked_console:
        assert log_reader_ready.is_file()
        assert "И" * 1024 * 1024 in result.stdout
        assert len(console) < len(result.stdout)
        assert "Завершение: код 0" in result.stdout


@pytest.mark.parametrize("download_network,batch,cache", [("host", "64", "q8_0"), ("bridge", "128", "f16")])
def test_download_network_never_reaches_audio_processing_containers(tmp_path, download_network, batch, cache):
    """Только разбор Compose: без демона Docker, контейнеров и скачивания моделей."""
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Для разбора Compose нужен CLI Docker")
    project = Path(__file__).resolve().parents[1]
    env = {**os.environ, "BENCH_AUDIO_DIR": str(tmp_path / "audio"), "BENCH_OUT": str(tmp_path / "results"),
           "BENCH_CACHE": str(tmp_path / "cache"), "BENCH_DOWNLOAD_NETWORK": download_network,
           "BENCH_BUILD_NETWORK": "default", "BENCH_OLLAMA_DOWNLOAD_PORT": "11435",
           "BENCH_LLM_NUM_BATCH": batch, "BENCH_LLM_KV_CACHE_TYPE": cache, "LLM_NUM_CTX": "8192",
           "HF_TOKEN": "", "WHISPER_API_BASE_URL": "http://production.example/v1", "WHISPER_API_KEY": "prod-secret"}
    parsed = subprocess.run([docker, "compose", "--env-file", "/dev/null", "--project-name", "speech-comparison",
                             "--project-directory", str(project), "-f", str(project / "compose.benchmark.yml"),
                             "config", "--format", "json"], env=env, capture_output=True, text=True, timeout=15)
    assert parsed.returncode == 0, parsed.stderr
    config = json.loads(parsed.stdout)
    services = config["services"]
    for name in ("whisper-download", "prefetch", "ollama-download"):
        service = services[name]
        assert service["network_mode"] == download_network and not service.get("networks")
        assert not service.get("ports") and not service.get("gpus") and not service.get("deploy")
        assert service["environment"]["NVIDIA_VISIBLE_DEVICES"] == "void"
        assert all(mount["target"] != "/recordings" for mount in service.get("volumes", []))
    for name, network in (("whisper-client", "whisper"), ("whisper-bench", "whisper"), ("compare", "gigaam"), ("ollama", "gigaam")):
        service = services[name]
        assert not service.get("network_mode") and set(service["networks"]) == {network}
        assert config["networks"][network]["internal"] is True
        assert not service.get("ports")
    client = services["whisper-client"]["environment"]
    assert client["WHISPER_API_BASE_URL"] == TEST_URL and "WHISPER_API_KEY" not in client
    assert services["ollama-download"]["environment"]["OLLAMA_HOST"] == "127.0.0.1:11435"
    assert services["ollama-download"]["healthcheck"]["test"] == ["CMD", "ollama", "list"]
    for name in ("audio-prepare", "whisper-client", "compare"):
        assert services[name]["environment"]["BENCH_LLM_NUM_BATCH"] == batch
        assert services[name]["environment"]["BENCH_LLM_KV_CACHE_TYPE"] == cache
        assert services[name]["environment"]["LLM_NUM_CTX"] == "8192"
    assert services["ollama"]["environment"]["OLLAMA_KV_CACHE_TYPE"] == cache
    assert services["ollama"]["environment"]["OLLAMA_FLASH_ATTENTION"] == "1"
    assert "OLLAMA_KV_CACHE_TYPE" not in services["ollama-download"]["environment"]
    first_line = services["first-line"]
    assert first_line["network_mode"] == "none" and not first_line.get("depends_on")
    assert not first_line.get("ports") and not first_line.get("networks")
    assert first_line["deploy"] == services["compare"]["deploy"]
    for key in ("VAD_THRESHOLD", "VAD_MIN_SILENCE", "VAD_MIN_SPEECH", "VAD_MAX_SPEECH", "BENCH_THREADS"):
        assert first_line["environment"][key] == services["compare"]["environment"][key]
    assert all(first_line["environment"][key] == "false" for key in ("EMO_ENABLED", "LLM_ENABLED", "SPLIT_TURNS"))


@pytest.mark.parametrize("url", ["http://10.220.21.2:8002/v1", "http://localhost:9000/v1", "https://api.openai.com/v1"])
def test_standalone_client_rejects_every_endpoint_except_own_service(url):
    with pytest.raises(ValueError, match="подключение к проду запрещено"):
        WhisperAPI(url, "", "large-v3", 10, standalone=True)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_standalone_client_verifies_own_service_and_no_production_credentials(tmp_path, device):
    requests = []
    path = tmp_path / "тест.wav"
    write_wav(path, 100)

    def response(request):
        requests.append(request.url.path)
        assert request.url.host == "whisper-bench"
        assert "authorization" not in request.headers
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "large-v3"}]})
        if request.url.path == "/benchmark/metadata":
            return httpx.Response(200, json={"stand": "speech-comparison", "model": "large-v3", "device": device,
                                            "compute_type": "float16", "model_loaded": True})
        return httpx.Response(200, json={"text": "Тест без ключа", "segments": []})

    async def scenario():
        client = WhisperAPI(TEST_URL, "", "large-v3", 10, httpx.MockTransport(response), standalone=True)
        try:
            if device == "cpu":
                with pytest.raises(RuntimeError, match="CUDA/float16"):
                    await client.check()
            else:
                await client.check()
                result = await client.run(path, 0.2)
                assert result["text"] == "Тест без ключа" and result["mode"] == "test_server_api"
        finally:
            await client.close()
    asyncio.run(scenario())
    assert ("/v1/audio/transcriptions" in requests) is (device == "cuda")
