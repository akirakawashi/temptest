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
import time

import httpx
import pytest

from benchmark import artifacts, compare, server_run, summary
from benchmark.whisper_api import TEST_URL, WhisperAPI, speech_seconds, validate_url
from test_benchmark import write_wav


def test_client_and_artifacts_work_with_only_benchmark_package(tmp_path):
    """Как COPY benchmark в образе клиента Whisper: app отсутствует, моделей и API нет."""
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
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "CPU_TEST_NUMPY_PATH": str(Path(sys.modules["numpy"].__file__).resolve().parents[1])}
    # Число записей не совпадает с ожидаемым: клиент останавливается до первого запроса к серверу.
    command = [sys.executable, "-I", "-c", bootstrap, str(image), "benchmark.compare",
               "--audio-dir", str(source), "--out", str(output), "--expected-files", "3"]
    prepared = subprocess.run(command, cwd=image, env=env, capture_output=True, text=True, timeout=15)
    assert prepared.returncode == 1, prepared.stdout + prepared.stderr
    conditions = json.loads((output / "условия.json").read_text())
    assert conditions["mode"] == "whisper_only" and conditions["planned_systems"] == ["whisper"]
    assert conditions["state"].startswith("Остановлен") and not list(output.rglob("whisper.json"))
    finalized = subprocess.run([sys.executable, "-I", "-c", bootstrap, str(image), "benchmark.artifacts",
                                str(output), "--exit-code", "1"], cwd=image, env=env,
                               capture_output=True, text=True, timeout=15)
    assert finalized.returncode == 0, finalized.stdout + finalized.stderr
    assert not (output / "временные").exists()
    with tarfile.open(output / "диагностика.tar.gz") as archive:
        assert "отчёт.html" in archive.getnames()
        assert all(not name.endswith(".wav") for name in archive.getnames())


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


@pytest.mark.parametrize("failure", [None, "warmup", "record"])
def test_whisper_corpus_is_warmed_up_measured_and_stops_on_first_error(tmp_path, monkeypatch, failure):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    write_wav(source / "1.wav", 100)
    write_wav(source / "2.wav", 200)
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
            failed = failure == "warmup" or (failure == "record" and len(api_calls) == 2)
            return {"status": "error" if failed else "ok", "mode": "test_server_api", "elapsed_seconds": 1,
                    "segments": [], "text": "Тест", "speech_seconds": 0, "error": "Ошибка" if failed else None}

    async def no_sleep(*args):
        pass
    monkeypatch.setattr(server_run, "WhisperAPI", API)
    monkeypatch.setattr(server_run.asyncio, "sleep", no_sleep)
    args = compare.parser().parse_args(["--audio-dir", str(source), "--out", str(output), "--expected-files", "2"])
    assert asyncio.run(server_run.whisper_phase(args)) == int(failure is not None)
    conditions = json.loads((output / "условия.json").read_text())
    result = json.loads((output / "итоги.json").read_text())
    assert conditions["mode"] == "whisper_only" and list(result["systems"]) == ["whisper"]
    assert conditions["phases"]["whisper"]["state"] == conditions["state"]
    statuses = result["systems"]["whisper"]["statuses"]
    if failure == "warmup":
        assert len(api_calls) == 1 and not list(output.glob("записи/*/whisper.json"))
        assert conditions["state"].startswith("Остановлен") and statuses["pending"] == 2
    elif failure == "record":
        # Прогрев и первая запись; вторая запись не отправляется.
        assert len(api_calls) == 2 and len(list(output.glob("записи/*/whisper.json"))) == 1
        assert conditions["state"].startswith("Остановлен") and statuses["error"] == statuses["pending"] == 1
    else:
        assert len(api_calls) == 3  # Отдельный прогрев, затем две измеряемые записи.
        assert len(list(output.glob("записи/*/whisper.json"))) == 2
        assert conditions["state"] == "Завершён" and statuses["ok"] == 2
        assert conditions["whisper"]["warmup"]["included_in_measurements"] is False
        assert result["systems"]["whisper"]["records_per_15_minutes"] == 900
        markdown, page = (output / "отчёт.md").read_text(), (output / "отчёт.html").read_text()
        assert "Успешных: 2." in markdown and "Whisper: 2.000 с; среднее 1.000 с" in markdown
        assert "Успешно: <b>2 из 2</b>" in page and "GigaAM" not in page + markdown
        header = (output / "сводка.csv").read_text(encoding="utf-8-sig").splitlines()[0]
        assert "Whisper, с" in header and "GigaAM" not in header and "Первая линия" not in header
    artifacts.finalize(output, int(failure is not None))
    assert not (output / "временные").exists()
    assert json.loads((output / "условия.json").read_text())["launcher_exit_code"] == int(failure is not None)
    with tarfile.open(output / "диагностика.tar.gz") as archive:
        assert "отчёт.html" in archive.getnames()
        assert all(not name.endswith(".wav") for name in archive.getnames())


def test_summary_marks_whisper_stages_as_unmeasured(tmp_path):
    row = {"directory": "записи/0001", "audio": compare.Audio("test.wav", "prepared.wav", 10, 160000, "test", .1),
           "whisper": {"status": "ok", "elapsed_seconds": 1}}
    result = summary.write_summary(tmp_path, [row], {"state": "Завершён", "mode": "whisper_only", "planned_systems": ["whisper"]})
    stages = result["systems"]["whisper"]["stages"]
    assert set(stages) == {"vad", "asr"}
    for stage in stages.values():
        assert stage["enabled"] is True and stage["measured"] is False
        assert stage["seconds"] == {"count": 0}
        assert stage["calls"] is None and stage["operation_time_vs_total_percent"] is None
    assert result["matched_comparison"] == {"records": 1, "processing_seconds": {"whisper": 1}}
    header = (tmp_path / "замеры.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert "LLM" not in header and "Голоса" not in header and "Эмоции" not in header


def test_existing_run_is_rejected_before_any_request(tmp_path, monkeypatch):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (output / "условия.json").write_text("{}")
    def forbidden(*args, **kwargs):
        raise AssertionError("Клиент не должен создаваться")
    monkeypatch.setattr(server_run, "WhisperAPI", forbidden)
    with pytest.raises(SystemExit) as error:
        compare.main(["--audio-dir", str(source), "--out", str(output)])
    assert error.value.code == 2
    assert (output / "условия.json").read_text() == "{}"


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


def fake_stand(tmp_path):
    """Копия run.sh в пустой папке и подменные docker/nvidia-smi, которые только записывают команды."""
    folder = tmp_path / "stand"
    (folder / "benchmark").mkdir(parents=True)
    script = Path(__file__).resolve().parents[1] / "benchmark" / "run.sh"
    (folder / "benchmark" / "run.sh").write_bytes(script.read_bytes())
    (folder / "audio").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text('#!/usr/bin/env python3\nimport os,sys,time,signal\nfrom pathlib import Path\na=sys.argv[1:]\nwith open(os.environ["COMMAND_LOG"],"a") as f: f.write(repr(a)+"\\n")\n'
                      'if a[0] == "compose" and "build" in a and a[-1] == "whisper-client": sys.exit(int(os.environ["CLIENT_BUILD_EXIT"]))\n'
                      'if a[0] == "compose" and "up" in a and a[-1] == "whisper-bench": sys.exit(int(os.environ["WHISPER_START_EXIT"]))\n'
                      'if a[0] == "compose" and "stop" in a and a[-1] == "whisper-bench": sys.exit(int(os.environ["WHISPER_STOP_EXIT"]))\n'
                      'if a[0] == "compose" and "run" in a: assert sys.stdin.buffer.read(1) == b""\n'
                      'if a[0] == "wait":\n'
                      ' if "-client-" in a[-1] and os.environ["HANG_CLIENT"] == "1":\n'
                      '  Path(os.environ["CLIENT_WAITING"]).touch()\n'
                      '  while True: time.sleep(60)\n'
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
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "COMMAND_LOG": str(tmp_path / "commands.log"),
           "IGNORE_LOG_TERM": "0", "BLOCKED_CONSOLE": "0", "LOG_READER_READY": str(tmp_path / "log-reader.ready"),
           "HANG_CLIENT": "0", "CLIENT_WAITING": str(tmp_path / "client.waiting"),
           "FREE_GPU": "14600", "GPU_UTIL": "100", "BENCH_GPU_MIN_FREE_MIB": "12288", "BENCH_GPU_MAX_UTIL": "100",
           "BENCH_BUILD_NETWORK": "default", "BENCH_DOWNLOAD_NETWORK": "host",
           "API_EXIT": "0", "CLIENT_BUILD_EXIT": "0", "WHISPER_START_EXIT": "0", "WHISPER_STOP_EXIT": "0",
           "WHISPER_DOWNLOAD_EXIT": "0", "BENCH_MIN_RAM_MIB": "0", "BENCH_MIN_DISK_MIB": "0"}
    return folder, env


@pytest.mark.parametrize("free, util, min_free, max_util, api_exit, client_build_exit, whisper_start_exit, whisper_stop_exit, whisper_download_exit, ignore_log_term, blocked_console", [
    (*case, False, False) for case in [
    (14600, 0, 16384, 10, 0, 0, 0, 0, 0), (90000, 0, 16384, 10, 47, 0, 0, 0, 0),
    (90000, 0, 16384, 10, 0, 0, 0, 0, 0), (14600, 100, 12288, 10, 0, 0, 0, 0, 0),
    (14600, 100, 12288, 100, 0, 0, 0, 0, 0), (3000, 100, 12288, 100, 0, 0, 0, 0, 0),
    (14600, 100, 12288, 100, 0, 23, 0, 0, 0), (14600, 100, 12288, 100, 0, 0, 31, 0, 0),
    (14600, 100, 12288, 100, 0, 0, 0, 32, 0), (14600, 100, 12288, 100, 0, 0, 0, 0, 51),
    ]
] + [
    pytest.param(14600, 100, 12288, 100, 0, 0, 0, 0, 0, True, False, id="completed-whisper-log-reader-ignores-sigterm"),
    pytest.param(14600, 100, 12288, 100, 0, 0, 0, 0, 0, False, True, id="blocked-console-does-not-stop-the-run"),
])
def test_launcher_runs_only_own_whisper_and_never_controls_production(tmp_path, free, util, min_free, max_util, api_exit, client_build_exit, whisper_start_exit, whisper_stop_exit, whisper_download_exit, ignore_log_term, blocked_console):
    folder, env = fake_stand(tmp_path)
    command_log, log_reader_ready = tmp_path / "commands.log", tmp_path / "log-reader.ready"
    env.update({"IGNORE_LOG_TERM": "1" if ignore_log_term else "0", "BLOCKED_CONSOLE": "1" if blocked_console else "0",
                "FREE_GPU": str(free), "GPU_UTIL": str(util), "BENCH_GPU_MIN_FREE_MIB": str(min_free),
                "BENCH_GPU_MAX_UTIL": str(max_util), "API_EXIT": str(api_exit), "CLIENT_BUILD_EXIT": str(client_build_exit),
                "WHISPER_START_EXIT": str(whisper_start_exit), "WHISPER_STOP_EXIT": str(whisper_stop_exit),
                "WHISPER_DOWNLOAD_EXIT": str(whisper_download_exit)})
    # Вход родительского терминала остаётся открытым; поток Docker logs не даёт EOF.
    # Для blocked_console stdout — заполненная труба, которую до выхода никто не читает.
    # Завершение контейнера должно перевести прогон дальше независимо от вывода.
    with (tmp_path / "terminal.log").open("w+") as terminal:
        process = subprocess.Popen(["bash", str(folder / "benchmark" / "run.sh"), str(folder / "audio"), "--expected-files", "2"],
                                   env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE if blocked_console else terminal,
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
        launch_log = next((folder / "benchmark-results").glob("*/логи/запуск.log")).read_text()
    precheck_failed = free < min_free or util > max_util
    assert returncode == (42 if precheck_failed else client_build_exit or whisper_download_exit or whisper_start_exit or api_exit or int(bool(whisper_stop_exit)))
    text = command_log.read_text()
    commands = [ast.literal_eval(line) for line in text.splitlines()]
    builds = [command for command in commands if command[0] == "compose" and "build" in command]
    for build in builds:
        assert build[build.index("--project-name") + 1] == "speech-comparison"
        assert build[build.index("--project-directory") + 1] == str(folder)
        assert build[build.index("-f") + 1] == str(folder / "compose.benchmark.yml")
        assert build[build.index("--progress") + 1] == "plain"
        assert build[-1] == "whisper-client"
    assert len(builds) == (0 if precheck_failed else 1)
    assert "'buildx', 'build'" not in text
    for word in ("whisper-asr", "model-proxy", "vllm", "10.220.21.2", "WHISPER_API_KEY", "ollama", "gigaam", "first-line", "prefetch"):
        assert word not in text
    for command in commands:
        assert not {"--rm", "--remove-orphans", "rm", "down"} & set(command)
        if command[0] == "compose" and command[1:] != ["version"]:
            assert command[command.index("--project-name") + 1] == "speech-comparison"
        if command[0] == "compose" and "run" in command:
            assert "--detach" in command and "--interactive=false" in command and "-T" in command
    waits = [command[1] for command in commands if command[0] == "wait"]
    client_runs = [command for command in commands if command[0] == "compose" and "run" in command and "whisper-client" in command]
    reached_download = not (precheck_failed or client_build_exit)
    reached_client = reached_download and not (whisper_download_exit or whisper_start_exit)
    assert [name.rsplit("-", 3)[0] for name in waits] == (["speech-comparison-whisper-download"] if reached_download else []) + (["speech-comparison-client"] if reached_client else [])
    assert len(client_runs) == int(reached_client)
    if reached_client:
        client = client_runs[0]
        assert client[-6:] == ["--expected-files", "2", "--audio-dir", "/recordings", "--out", "/results/" + waits[1].split("-client-")[1]]
        assert "--phase" not in client
        start = next(c for c in commands if c[0] == "compose" and "up" in c)
        assert start[-5:] == ["-d", "--wait", "--wait-timeout", "600", "whisper-bench"]
        stop = next(c for c in commands if c[0] == "compose" and "stop" in c)
        assert stop[-1] == "whisper-bench"
        assert commands.index(start) < commands.index(client) < commands.index(stop)
        artifacts_run = next(c for c in commands if c[0] == "run")
        assert "benchmark.artifacts" in artifacts_run and "NVIDIA_VISIBLE_DEVICES=void" in artifacts_run
        assert artifacts_run[-1] == str(returncode) and commands.index(stop) < commands.index(artifacts_run)
    if whisper_download_exit:
        assert not any(c[0] == "compose" and "up" in c for c in commands)
        download_logs = list((folder / "benchmark-results").glob("*/логи/загрузка-whisper.log"))
        assert len(download_logs) == 1
        assert "полный журнал завершённого контейнера" in download_logs[0].read_text()
    if precheck_failed or client_build_exit:
        assert "file changed as we read it" not in launch_log
        archives = list((folder / "benchmark-results").glob("*/диагностика.tar.gz"))
        assert len(archives) == 1
        with tarfile.open(archives[0]) as archive:
            assert "./логи/запуск.log" in archive.getnames()
            assert all(not name.endswith(".tar.gz") for name in archive.getnames())
            log_text = archive.extractfile("./логи/запуск.log").read().decode()
            assert ("Стенд не запускается" if precheck_failed else f"Завершение: код {client_build_exit}") in log_text
        assert not list((folder / "benchmark-results").glob("*.tar.part"))
    if returncode == 0:
        assert "Результаты:" in launch_log and "Завершение: код 0" in launch_log
        assert next((folder / "benchmark-results").glob("*/логи/этап.txt")).read_text().strip() == "Whisper завершён"
    if ignore_log_term:
        assert log_reader_ready.is_file()
        with pytest.raises(ProcessLookupError):
            os.kill(int(log_reader_ready.read_text()), 0)
        assert "Docker сообщил о завершении задачи speech-comparison-client-" in launch_log
    if blocked_console:
        assert log_reader_ready.is_file()
        assert "И" * 1024 * 1024 in launch_log
        assert len(console) < len(launch_log)
        assert "Завершение: код 0" in launch_log


@pytest.mark.parametrize("flag", ["--gigaam-only", "--resume-gigaam", "--skip-full-gigaam"])
def test_launcher_refuses_removed_gigaam_modes_before_touching_docker(tmp_path, flag):
    folder, env = fake_stand(tmp_path)
    refused = subprocess.run(["bash", str(folder / "benchmark" / "run.sh"), str(folder / "audio"), flag],
                             env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
    assert refused.returncode == 2 and "run-gigaam-trt.sh" in refused.stderr
    assert not (tmp_path / "commands.log").exists() and not (folder / "benchmark-results").exists()


def test_closed_terminal_stops_own_whisper_and_is_a_failed_run(tmp_path):
    folder, env = fake_stand(tmp_path)
    waiting = tmp_path / "client.waiting"
    env["HANG_CLIENT"] = "1"
    with (tmp_path / "terminal.log").open("w") as terminal:
        process = subprocess.Popen(["bash", str(folder / "benchmark" / "run.sh"), str(folder / "audio")], env=env,
                                   stdin=subprocess.DEVNULL, stdout=terminal, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            while not waiting.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert waiting.exists()
            os.killpg(process.pid, signal.SIGHUP)
            code = process.wait(timeout=15)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    assert code == 129
    launch_log = next((folder / "benchmark-results").glob("*/логи/запуск.log")).read_text()
    assert "Получен SIGHUP" in launch_log and "Завершение: код 129" in launch_log
    commands = [ast.literal_eval(line) for line in (tmp_path / "commands.log").read_text().splitlines()]
    assert any(c[0] == "compose" and "stop" in c and c[-1] == "whisper-bench" for c in commands)
    assert any(c[0] == "stop" and c[-1].startswith("speech-comparison-client-") for c in commands)


@pytest.mark.parametrize("download_network", ["host", "bridge"])
def test_download_network_never_reaches_audio_processing_containers(tmp_path, download_network):
    """Только разбор Compose: без демона Docker, контейнеров и скачивания моделей."""
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Для разбора Compose нужен CLI Docker")
    project = Path(__file__).resolve().parents[1]
    env = {**os.environ, "BENCH_AUDIO_DIR": str(tmp_path / "audio"), "BENCH_OUT": str(tmp_path / "results"),
           "BENCH_EXPECTED_FILES": "2",
           "BENCH_CACHE": str(tmp_path / "cache"), "BENCH_DOWNLOAD_NETWORK": download_network,
           "BENCH_BUILD_NETWORK": "default",
           "WHISPER_API_BASE_URL": "http://production.example/v1", "WHISPER_API_KEY": "prod-secret"}
    parsed = subprocess.run([docker, "compose", "--env-file", "/dev/null", "--project-name", "speech-comparison",
                             "--project-directory", str(project), "-f", str(project / "compose.benchmark.yml"),
                             "config", "--format", "json"], env=env, capture_output=True, text=True, timeout=15)
    assert parsed.returncode == 0, parsed.stderr
    config = json.loads(parsed.stdout)
    services = config["services"]
    assert set(services) == {"whisper-client", "whisper-bench", "whisper-download"}
    assert set(config["networks"]) == {"whisper"} and not config.get("volumes")
    download = services["whisper-download"]
    assert download["network_mode"] == download_network and not download.get("networks")
    assert not download.get("ports") and not download.get("gpus") and not download.get("deploy")
    assert download["environment"]["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert all(mount["target"] != "/recordings" for mount in download.get("volumes", []))
    for name in ("whisper-client", "whisper-bench"):
        service = services[name]
        assert not service.get("network_mode") and set(service["networks"]) == {"whisper"}
        assert not service.get("ports")
    assert config["networks"]["whisper"]["internal"] is True
    client = services["whisper-client"]["environment"]
    assert client["WHISPER_API_BASE_URL"] == TEST_URL and "WHISPER_API_KEY" not in client
    assert client["BENCH_EXPECTED_FILES"] == "2" and not any("LLM" in key for key in client)
    assert "prod-secret" not in parsed.stdout and "production.example" not in parsed.stdout
    assert not services["whisper-client"].get("deploy") and services["whisper-bench"]["deploy"]


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
