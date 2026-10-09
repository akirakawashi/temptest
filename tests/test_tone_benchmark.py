"""T-one: заглушки моделей и Docker; настоящий прогон и скачивания не нужны."""
import asyncio
import builtins
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import wave
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from test_benchmark import FakeEngines, ScriptedVad, speech

from app.config import Settings
from app.vad import StreamingVad
from benchmark import pipeline, tone, tone_run, tone_worker
from benchmark.compare import Audio

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("length", [0, 1, 16, 127, 16000])
def test_resampling_preserves_correct_duration_even_for_short_input(length):
    result = tone.to_8k(np.zeros(length, dtype=np.float32))
    assert len(result) == (length + 1) // 2
    assert result.dtype == np.float32 and result.flags.c_contiguous


def test_resampling_matches_source_filter_and_suppresses_aliasing():
    times = np.arange(16000) / 16000
    low = np.sin(2 * np.pi * 1000 * times).astype(np.float32)
    high = np.sin(2 * np.pi * 6000 * times).astype(np.float32)
    assert np.max(np.abs(tone.to_8k(high)[100:-100])) < .001
    assert np.max(np.abs(tone.to_8k(low)[100:-100])) > .99
    expected = np.convolve(low, tone.LOWPASS_8K, mode="same")[::2]
    np.testing.assert_array_equal(tone.to_8k(low), expected)


def test_word_tokens_keep_numbers_and_correct_time():
    tokens, stamps = tone.word_tokens(list("355 рублей"), np.arange(10) / 10, .15)
    assert "".join(tokens).strip() == "355 рублей"
    assert stamps[0] == 0 and max(stamps) == .15
    with pytest.raises(ValueError):
        tone.word_tokens(["1"], [], 2)


def test_stream_is_drained_and_padding_is_part_of_asr():
    class Recognizer:
        remaining = 3
        def create_stream(self):
            return SimpleNamespace(accept_waveform=lambda rate, x: chunks.append((rate, len(x))),
                                   input_finished=lambda: finished.append(True))
        def is_ready(self, stream):
            assert finished
            return self.remaining > 0
        def decode_stream(self, stream):
            self.remaining -= 1
        def get_result_all(self, stream):
            assert self.remaining == 0
            return SimpleNamespace(text="триста пятьдесят пять рублей", tokens=list("355 рублей"), timestamps=[.8] * 10)
    chunks, finished = [], []
    engine = tone.ToneEngines(Settings(emo_enabled=False, llm_enabled=False, split_turns=False), Path("/unused"))
    engine._asr = Recognizer()
    try:
        text, tokens, stamps = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == "триста пятьдесят пять рублей"
        assert "".join(tokens).strip() == "355 рублей"
        assert chunks == [(8000, 2400), (8000, 8000), (8000, 8000)]
        assert engine.raw_results[0]["decode_calls"] == 3
        assert engine.raw_results[0]["padding_seconds"] == 1.3
        assert engine.timer.snapshot()["asr"]["calls"] == 1
        assert stamps[0] == pytest.approx(.1)
    finally:
        for pool in (engine.asr_pool, engine.emo_pool, engine.live_pool):
            pool.shutdown()


@pytest.mark.parametrize("cuda_wheel,runtime,provider,works", [
    (True, "1.24.4", True, True), (False, "1.24.4", True, False),
    (True, "1.23.2", True, False), (True, "1.24.4", False, False),
])
def test_load_requires_cuda_and_matching_native_runtimes(monkeypatch, tmp_path, cuda_wheel, runtime, provider, works):
    factory_calls = []
    for name in ("model.onnx", "tokens.txt", "vad.onnx"):
        (tmp_path / name).write_bytes(b"test")
    fake_vad = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"], run=lambda *args: None)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(
        __version__="1.13.4+cuda12.cudnn9" if cuda_wheel else "1.13.4",
        onnxruntime_version=runtime, OnlineRecognizer=SimpleNamespace(
            from_t_one_ctc=lambda **kwargs: factory_calls.append(kwargs) or object())))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(__version__="1.24.4",
        get_available_providers=lambda: ["CUDAExecutionProvider"] if provider else ["CPUExecutionProvider"]))
    monkeypatch.setattr(tone, "default_model_path", lambda: str(tmp_path / "vad.onnx"))
    monkeypatch.setattr(tone, "make_onnx_session", lambda path, *, provider: fake_vad)
    monkeypatch.setattr(tone.ToneEngines, "new_vad", lambda self: SimpleNamespace(accept=lambda x: []))
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name not in {"gigaam", "torch", "faster_whisper"}
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    engine = tone.ToneEngines(Settings(emo_enabled=False, llm_enabled=False, split_turns=False), tmp_path)
    try:
        if works:
            engine.load_core()
            assert factory_calls[0]["provider"] == "cuda" and factory_calls[0]["device"] == 0
            assert factory_calls[0]["sample_rate"] == 8000 and engine.core_ready
            assert set(engine.model_hashes) == {"model.onnx", "tokens.txt", "silero_vad.onnx"}
            assert all(engine.components[k].state == "off" for k in ("spk", "emo", "llm"))
        else:
            with pytest.raises(RuntimeError):
                engine.load_core()
            assert not factory_calls
    finally:
        for pool in (engine.asr_pool, engine.emo_pool, engine.live_pool):
            pool.shutdown()


def test_pipeline_has_no_speaker_emotion_or_llm_and_preserves_raw_text():
    class Engines(FakeEngines):
        def __init__(self):
            super().__init__()
            self.emotions_ready = False
        def new_vad(self):
            return pipeline.TimedVad(StreamingVad(ScriptedVad()), self.timer)
        def transcribe(self, samples):
            result = super().transcribe(samples)
            self.raw_results.append({"text": "триста пятьдесят пять рублей"})
            return result
        def embed(self, samples):
            raise AssertionError("Голоса не должны вызываться")
        def emotions(self, samples):
            raise AssertionError("Эмоции не должны вызываться")
    async def scenario():
        runner = tone.TonePipeline(Settings(emo_enabled=False, llm_enabled=False, split_turns=False), 1, 3, Path("/unused"))
        runner.shutdown()
        runner.engines = Engines()
        try:
            result = await runner.run(speech())
            assert result["status"] == "ok" and result["text"] == "триста пятьдесят пять рублей"
            assert result["pipeline"] == "vad_tone"
            assert result["formatted_text"] == "Добрый день."
            assert result["analysis"] is None and result["llm_seconds"] == 0 and runner.llm is None
            assert result["enabled_stages"] == ["vad", "asr"]
            assert result["stages"]["speaker"]["calls"] == result["stages"]["emotion"]["calls"] == 0
            assert sum(result["timing"]["exclusive_wall_percent"].values()) == pytest.approx(100)
            assert result["utterances"][0]["speaker"] is None and result["pauses"]
            silent = await runner.run(np.zeros(16000, dtype=np.float32))
            assert silent["status"] == "no_speech" and silent["asr_seconds"] == 0 and silent["text"] == ""
        finally:
            runner.shutdown()
    asyncio.run(scenario())


def args_for(tmp_path, count=2):
    folder = tmp_path / "audio"
    folder.mkdir()
    output = tmp_path / "results"
    output.mkdir()
    for index in range(count):
        with wave.open(str(folder / f"{index}.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setframerate(16000)
            wav.setsampwidth(2)
            wav.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    return tone_run.parser().parse_args([str(folder), "--out", str(output), "--expected-files", str(count), "--gap", "0"])


def fake_prepare(source, target, max_seconds=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    with wave.open(str(target)) as wav:
        raw = wav.readframes(wav.getnframes())
    return Audio(str(source), str(target), 1., 16000, hashlib.sha256(raw).hexdigest(), .05, 1, 16000, source.stat().st_size)


def test_reference_reorders_and_rejects_wrong_corpus():
    files = [Path("a.wav"), Path("b.wav")]
    reference = {"records": [{"directory": "записи/0001", "audio": {"source": "/recordings/b.wav"}},
                             {"directory": "записи/0002", "audio": {"source": "/recordings/a.wav"}}]}
    assert [p.name for p, _, _ in tone_run.reference_files(files, reference, 2)] == ["b.wav", "a.wav"]
    with pytest.raises(ValueError):
        tone_run.reference_files(files, reference, 100)
    reference["records"][0]["directory"] = "../../outside"
    with pytest.raises(ValueError):
        tone_run.reference_files(files, reference, 2)


def test_changed_pcm_is_rejected_before_worker(monkeypatch, tmp_path):
    args = args_for(tmp_path)
    old = tmp_path / "old"
    old.mkdir()
    reference = {"records": [{"directory": f"записи/{i + 1:04d}", "audio": {
        "source": f"/recordings/{i}.wav", "samples": 16000, "sha256_pcm": "changed"}} for i in range(2)]}
    (old / "условия.json").write_text(json.dumps(reference))
    args.reference_run = old
    monkeypatch.setattr(tone_run, "prepare_audio", fake_prepare)
    async def forbidden(request):
        raise AssertionError("Распознавание не должно начинаться")
    monkeypatch.setattr(tone_run, "run_worker", forbidden)
    original = (old / "условия.json").read_bytes()
    assert asyncio.run(tone_run.run(args)) == 1
    assert (old / "условия.json").read_bytes() == original
    conditions = json.loads((args.out / "условия.json").read_text())
    assert conditions["state"].startswith("Остановлен") and not conditions["reference"]["pcm_verified"]


@pytest.mark.parametrize("status,expected_code", [("ok", 0), ("no_speech", 0), ("error", 1)])
def test_corpus_writes_real_statuses_summary_and_shareable_archive(monkeypatch, tmp_path, status, expected_code):
    args = args_for(tmp_path)
    monkeypatch.setattr(tone_run, "prepare_audio", fake_prepare)
    async def run_worker(request):
        return {"result": {"status": status, "text": "триста пятьдесят пять рублей" if status == "ok" else "",
            "elapsed_seconds": .5, "asr_seconds": .4, "llm_seconds": 0, "error": "test" if status == "error" else None,
            "stages": {"asr": {"seconds": .4, "calls": 1, "errors": 0}, "vad": {"seconds": .1, "calls": 1, "errors": 0}}},
            "preparation": {"system": "tone", "load_seconds": 1, "warmup_seconds": .2, "worker_wall_seconds": 2}}
    monkeypatch.setattr(tone_run, "run_worker", run_worker)
    code = asyncio.run(tone_run.run(args))
    assert code == expected_code
    summary = json.loads((args.out / "итоги.json").read_text())
    assert summary["records"] == 2 and summary["systems"]["tone"]["statuses"][status] == 2
    assert set(summary["systems"]) == {"tone"}
    assert summary["systems"]["tone"]["stages"]["llm"]["enabled"] is False
    assert (args.out / "отчёт.html").is_file() and (args.out / "замеры.csv").is_file()
    (args.out / "model.onnx").write_bytes(b"exclude")
    (args.out / "original.wav").write_bytes(b"exclude")
    tone_run.finalize(args.out, code)
    assert not (args.out / "временные").exists()
    with tarfile.open(args.out / "диагностика.tar.gz") as archive:
        names = archive.getnames()
        assert "записи/0001/tone.json" in names and "условия.json" in names
        assert not any(p.endswith((".wav", ".onnx", ".part", ".tar.gz")) for p in names)


def test_preparation_failure_stops_and_marks_unprocessed_files(monkeypatch, tmp_path):
    args = args_for(tmp_path)
    monkeypatch.setattr(tone_run, "prepare_audio", fake_prepare)
    async def fail(request):
        return {"result": {"status": "error", "error": "CUDA недоступна", "text": "", "elapsed_seconds": None,
                            "preparation_failed": True}, "preparation": {"system": "tone"}}
    monkeypatch.setattr(tone_run, "run_worker", fail)
    assert asyncio.run(tone_run.run(args)) == 1
    summary = json.loads((args.out / "итоги.json").read_text())
    assert summary["systems"]["tone"]["statuses"]["error"] == 1
    assert summary["systems"]["tone"]["statuses"]["pending"] == 1


def test_worker_subprocess_saves_native_output_and_rejects_cpu_fallback(monkeypatch, tmp_path):
    class Process:
        pid = 123
        returncode = 0
        def __init__(self):
            self.stdout = asyncio.StreamReader()
            self.stdout.feed_data(b"Native: Fallback to cpu!\n")
            self.stdout.feed_eof()
        async def wait(self):
            return 0
    async def start(*args, **kwargs):
        assert args[4] == "benchmark.tone_worker" and kwargs["start_new_session"] is True
        tone_run.save_json(tmp_path / "tone-ответ.json", {"result": {"status": "ok", "elapsed_seconds": 1},
                                                       "preparation": {"system": "tone"}})
        return Process()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    payload = asyncio.run(tone_run.run_worker({"output": str(tmp_path), "timeout": 2}))
    assert payload["result"]["status"] == "error" and payload["result"]["preparation_failed"]
    assert "CPU" in payload["result"]["error"]
    assert (tmp_path / "tone-процесс.log").read_text() == "Native: Fallback to cpu!\n"


def test_worker_separates_warmup_and_measurement(monkeypatch, tmp_path):
    calls = []
    class Runner:
        def __init__(self, cfg, threads, timeout, model_dir):
            assert not cfg.emo_enabled and not cfg.llm_enabled and not cfg.split_turns
            self.engines = SimpleNamespace(placement=lambda: {"asr": "cuda"}, model_hashes={"model.onnx": "x"}, versions={})
        async def load(self):
            calls.append("load")
        async def run(self, samples):
            calls.append("run")
            return {"status": "ok", "elapsed_seconds": .2, "asr_seconds": .1, "text": "355"}
        def shutdown(self):
            calls.append("shutdown")
    monkeypatch.setattr(tone, "TonePipeline", Runner)
    monkeypatch.setattr("benchmark.gpu.gpu_info", lambda: {"uuid": "test"})
    monkeypatch.setattr("benchmark.gpu.log_gpu_memory", lambda stage: {"free_mib": 5000})
    monkeypatch.setattr(tone_worker, "read_audio", lambda audio: np.zeros(16000, dtype=np.float32))
    audio = asdict(Audio("source", "prepared", 1, 16000, "sha", .1))
    payload = asyncio.run(tone_worker.execute({"audio": audio, "warmup": audio, "output": str(tmp_path),
        "threads": 4, "timeout": 3, "model_dir": "/models/tone", "vad_settings": tone_run.VAD_SETTINGS}))
    assert calls == ["load", "run", "run", "shutdown"]
    assert payload["preparation"]["warmup_seconds"] == .2
    assert payload["preparation"]["included_in_measurements"] is False
    assert payload["result"]["elapsed_seconds"] == .2
    assert (tmp_path / "tone-прогрев.json").is_file()


def test_compose_has_only_tone_no_network_ports_or_production_api():
    config = yaml.safe_load((ROOT / "compose.tone-benchmark.yml").read_text())
    assert set(config["services"]) == {"tone"}
    service = config["services"]["tone"]
    assert config["name"] == "speech-comparison-tone" and service["network_mode"] == "none"
    assert "ports" not in service and "depends_on" not in service and "env_file" not in service
    assert service["volumes"][0].endswith(":/recordings:ro")
    assert service["volumes"][2].endswith(":/reference:ro")
    assert "WHISPER_API_BASE_URL" not in service["environment"]


@pytest.mark.parametrize("free_mib,base_exists,run_code", [(20000, True, 0), (2000, True, 42), (20000, False, 0)])
def test_launcher_controls_only_its_unique_containers(tmp_path, free_mib, base_exists, run_code):
    binary = tmp_path / "bin"
    binary.mkdir()
    audio = tmp_path / "audio"
    audio.mkdir()
    journal = tmp_path / "commands.jsonl"
    script = f"""#!{sys.executable}
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['FAKE_COMMANDS'],'a') as f: f.write(json.dumps([Path(sys.argv[0]).name,*args])+'\\n')
if Path(sys.argv[0]).name=='nvidia-smi':
    if any('memory.free,utilization' in a for a in args): print('{free_mib}, 0')
    else: print('2026/10/09 12:00:00, GPU-test, 0, 70000, 20000, 90000, 90, 30')
elif args[:2]==['image','inspect'] and args[-1]=='speech-comparison:4.0.0-gigaam-cuda':
    sys.exit({0 if base_exists else 1})
elif args[0]=='info': print(os.environ['FAKE_ROOT'])
elif args[0]=='inspect': print('false 0')
elif args[0]=='logs': print('Fake T-one completed')
"""
    for name in ("docker", "nvidia-smi"):
        path = binary / name
        path.write_text(script)
        path.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + ":" + os.environ["PATH"], "FAKE_COMMANDS": str(journal),
           "FAKE_ROOT": str(tmp_path), "BENCH_OUT": str(tmp_path / "out"), "BENCH_CACHE": str(tmp_path / "cache"),
           "BENCH_GPU_MAX_UTIL": "100", "BENCH_MIN_DISK_MIB": "0", "BENCH_MIN_RAM_MIB": "0"}
    completed = subprocess.run(["bash", str(ROOT / "benchmark/run-tone.sh"), str(audio)],
                               env=env, capture_output=True, text=True, timeout=15, check=False)
    assert completed.returncode == run_code, completed.stdout + completed.stderr
    commands = [json.loads(line) for line in journal.read_text().splitlines()]
    docker = [cmd[1:] for cmd in commands if cmd[0] == "docker"]
    if run_code == 42:
        assert not any(cmd[0] == "compose" for cmd in docker)
        assert list((tmp_path / "out").glob("tone-*/диагностика.tar.gz"))
        return
    assert any("--detach" in cmd and "--no-deps" in cmd for cmd in docker)
    assert not any(any(word in cmd for word in ("down", "prune", "rm", "up")) for cmd in docker)
    assert not any(any(word in " ".join(cmd) for word in ("whisper-asr", "model-proxy", "vllm-", "ollama")) for cmd in docker)
    assert all(cmd[-1].startswith("speech-comparison-tone-tone-") for cmd in docker if cmd[0] == "stop")
    for cmd in docker:
        if cmd[0] == "compose":
            assert cmd[cmd.index("--project-name") + 1] == "speech-comparison-tone"
    assert any("--finalize" in cmd and "NVIDIA_VISIBLE_DEVICES=void" in cmd for cmd in docker)
    assert any("build" in cmd and "compare" in cmd for cmd in docker) is not base_exists


def test_worker_timeout_kills_only_its_process_group_and_keeps_log(monkeypatch, tmp_path):
    killed = []
    class Process:
        pid = 987654
        returncode = None
        def __init__(self):
            self.stdout = asyncio.StreamReader()
            self.stdout.feed_data(b"Native inference started\n")
            self.stdout.feed_eof()
        async def wait(self):
            self.returncode = -9
            return -9
    async def start(*args, **kwargs):
        return Process()
    async def timeout(awaitable, seconds):
        awaitable.close()
        raise TimeoutError
    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    monkeypatch.setattr(asyncio, "wait_for", timeout)
    monkeypatch.setattr(tone_run.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    payload = asyncio.run(tone_run.run_worker({"output": str(tmp_path), "timeout": 1}))
    assert killed == [(987654, 9)]
    assert payload["result"]["status"] == "timeout" and payload["result"]["elapsed_seconds"] is None
    assert (tmp_path / "tone-процесс.log").read_text() == "Native inference started\n"


def test_reference_matching_pcm_and_settings_are_verified(monkeypatch, tmp_path):
    args = args_for(tmp_path)
    args.reference_run = tmp_path / "reference"
    args.reference_run.mkdir()
    pcm = hashlib.sha256(np.zeros(16000, dtype="<i2").tobytes()).hexdigest()
    settings = {**tone_run.VAD_SETTINGS, "vad_threshold": .6, "asr_threads": args.threads}
    audio = asdict(Audio("/recordings/1.wav", "/old/1.wav", 1, 16000, pcm, .1))
    reference = {"records": [{"directory": "записи/0001", "audio": audio},
        {"directory": "записи/0002", "audio": {**audio, "source": "/recordings/0.wav"}}],
        "warmup": {"audio": audio}, "preparations": [{"system": "gigaam_first_line", "settings": settings}]}
    tone_run.save_json(args.reference_run / "условия.json", reference)
    monkeypatch.setattr(tone_run, "prepare_audio", fake_prepare)
    conditions = {"records": [], "vad_settings": dict(tone_run.VAD_SETTINGS)}
    rows = tone_run.prepare(args, conditions)
    assert [Path(r["audio"].source).name for r in rows] == ["1.wav", "0.wav"]
    assert conditions["reference"]["pcm_verified"] and conditions["warmup"]["matches_reference"]
    assert conditions["vad_settings"]["vad_threshold"] == .6
    assert Path(conditions["warmup"]["audio"]["source"]).name == "1.wav"


def test_finalize_needs_only_standard_library_and_no_app_or_models(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    (output / "failure.log").write_text("Сборка не завершилась")
    completed = subprocess.run(["python3", "-S", "-m", "benchmark.tone_run", "--finalize", str(output), "42"],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=10, check=False)
    assert completed.returncode == 0, completed.stderr
    with tarfile.open(output / "диагностика.tar.gz") as archive:
        assert archive.getnames() == ["failure.log"]


def test_source_package_includes_separate_tone_launcher_and_image():
    from benchmark.package import source_files
    paths = {str(p.relative_to(ROOT)) for p in source_files()}
    assert {"benchmark/run-tone.sh", "benchmark/tone_worker.py", "benchmark/tone_run.py",
            "Dockerfile.tone-benchmark", "compose.tone-benchmark.yml", "benchmark/TONE.md"} <= paths
