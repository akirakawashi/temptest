"""Без запросов на рабочий сервер, Docker, скачиваний и GPU inference."""
import ast
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import httpx
import pytest

from benchmark import artifacts, compare, server_run
from benchmark.whisper_api import TEST_URL, WhisperAPI, speech_seconds, validate_url
from test_benchmark import install_fake_runners, write_wav


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


@pytest.mark.parametrize("free, util, min_free, max_util, api_exit, client_build_exit, whisper_start_exit, whisper_stop_exit", [
    (14600, 0, 16384, 10, 0, 0, 0, 0), (90000, 0, 16384, 10, 47, 0, 0, 0), (90000, 0, 16384, 10, 0, 0, 0, 0),
    (14600, 100, 12288, 10, 0, 0, 0, 0), (14600, 100, 12288, 100, 0, 0, 0, 0),
    (3000, 100, 12288, 100, 0, 0, 0, 0), (14600, 100, 12288, 100, 0, 23, 0, 0),
    (14600, 100, 12288, 100, 0, 0, 31, 0), (14600, 100, 12288, 100, 0, 0, 0, 32),
])
def test_launcher_never_controls_production_and_does_not_start_gigaam_after_failure(tmp_path, free, util, min_free, max_util, api_exit, client_build_exit, whisper_start_exit, whisper_stop_exit):
    folder = tmp_path / "stand"
    (folder / "benchmark").mkdir(parents=True)
    script = Path(__file__).resolve().parents[1] / "benchmark" / "run.sh"
    (folder / "benchmark" / "run.sh").write_bytes(script.read_bytes())
    (folder / "audio").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    command_log = tmp_path / "commands.log"
    docker = bin_dir / "docker"
    docker.write_text('#!/usr/bin/env python3\nimport os,sys\na=sys.argv[1:]\nwith open(os.environ["COMMAND_LOG"],"a") as f: f.write(repr(a)+"\\n")\n'
                      'if a[0] == "compose" and "build" in a and a[-1] == "whisper-client": sys.exit(int(os.environ["CLIENT_BUILD_EXIT"]))\n'
                      'if a[0] == "compose" and "up" in a and a[-1] == "whisper-bench": sys.exit(int(os.environ["WHISPER_START_EXIT"]))\n'
                      'if a[0] == "compose" and "stop" in a and a[-1] == "whisper-bench": sys.exit(int(os.environ["WHISPER_STOP_EXIT"]))\n'
                      'if "info" in a: print("/tmp")\n'
                      'sys.exit(int(os.environ["API_EXIT"]) if "whisper-api" in a else 0)\n')
    nvidia = bin_dir / "nvidia-smi"
    nvidia.write_text('#!/usr/bin/env python3\nimport os,sys\na=" ".join(sys.argv)\n'
                      'print(os.environ["FREE_GPU"] if "--query-gpu=memory.free" in a else os.environ["GPU_UTIL"] if "--query-gpu=utilization.gpu" in a else "2026/10/07, uuid, 0, 1000, 90000, 95830, 80, 32")\n')
    docker.chmod(0o755)
    nvidia.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "COMMAND_LOG": str(command_log),
           "FREE_GPU": str(free), "GPU_UTIL": str(util), "BENCH_GPU_MIN_FREE_MIB": str(min_free),
           "BENCH_GPU_MAX_UTIL": str(max_util),
           "BENCH_BUILD_NETWORK": "default",
           "API_EXIT": str(api_exit), "CLIENT_BUILD_EXIT": str(client_build_exit),
           "WHISPER_START_EXIT": str(whisper_start_exit), "WHISPER_STOP_EXIT": str(whisper_stop_exit),
           "BENCH_MIN_RAM_MIB": "0", "BENCH_MIN_DISK_MIB": "0"}
    result = subprocess.run(["bash", str(folder / "benchmark" / "run.sh"), str(folder / "audio")], env=env,
                            capture_output=True, text=True, timeout=15)
    precheck_failed = free < min_free or util > max_util
    assert result.returncode == (42 if precheck_failed else client_build_exit or whisper_start_exit or api_exit or int(bool(whisper_stop_exit)))
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
    if precheck_failed or client_build_exit or whisper_start_exit or api_exit or whisper_stop_exit:
        assert "'build', 'compare'" not in commands and "'up', '-d', 'ollama'" not in commands
        assert len(builds) == (0 if precheck_failed else 1)
    else:
        assert len(builds) == 2
        assert commands.index("whisper-api") < commands.index("'stop', 'whisper-bench'") < commands.index("'build', 'compare'") < commands.index("'up', '-d', 'ollama'")
        assert "'stop', 'ollama'" in commands and "'stop', 'ollama-download'" in commands
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
    assert "10.220.21.2" not in commands and "WHISPER_API_KEY" not in commands


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
