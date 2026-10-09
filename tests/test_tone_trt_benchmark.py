"""TensorRT-стенд: настоящий официальный splitter/decoder, GPU и Docker — заглушки."""
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import yaml

from benchmark import tone_trt, tone_trt_prepare as preparation, tone_trt_run, tone_trt_worker
from benchmark.compare import Audio

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {"name": "streaming_acoustic", "platform": "tensorrt_plan", "max_batch_size": 1,
        "instance_group": [{"kind": "KIND_GPU", "gpus": [0], "count": 1}],
        "input": [{"name": "signal", "data_type": "TYPE_INT32", "dims": ["2400", "1"]},
                  {"name": "state", "data_type": "TYPE_FP16", "dims": ["219729"]}]}


@pytest.mark.parametrize("change", ["platform", "kind", "gpu", "batch", "dims", "queue"])
def test_cpu_wrong_gpu_batch_and_wrong_model_are_rejected(change):
    config = configuration()
    assert tone_trt.validate_config({"config": config}) is config
    if change == "platform": config["platform"] = "onnxruntime_onnx"
    if change == "kind": config["instance_group"][0]["kind"] = "KIND_CPU"
    if change == "gpu": config["instance_group"][0]["gpus"] = [1]
    if change == "batch": config["max_batch_size"] = 16
    if change == "dims": config["input"][0]["dims"] = ["4800", "1"]
    if change == "queue": config["dynamic_batching"] = {}
    with pytest.raises(RuntimeError):
        tone_trt.validate_config(config)


class Input:
    def __init__(self, name, shape, dtype):
        self.name, self.shape, self.dtype = name, shape, dtype
    def set_data_from_numpy(self, data):
        self.data = data.copy()


def fake_grpc():
    return SimpleNamespace(InferInput=Input, InferRequestedOutput=lambda name: name)


def fake_client():
    class Client:
        def __init__(self):
            self.states, self.signals, self.requests = [], [], []
        def infer(self, model, inputs, **kwargs):
            arrays = {item.name: item.data for item in inputs}
            self.states.append(arrays["state"].copy())
            self.signals.append(arrays["signal"].copy())
            self.requests.append((model, kwargs))
            logprobs = np.full((1, 10, 35), -20., dtype=np.float32)
            logprobs[..., -1] = 0.
            if len(self.states) == 2:
                logprobs[0, :5, -1] = -20.
                logprobs[0, :3, 4] = 0.  # д
                logprobs[0, 3:5, 0] = 0.  # а
            values = {"logprobs": logprobs, "state_next": arrays["state"] + np.float16(1)}
            return SimpleNamespace(as_numpy=lambda name: values[name])
        def close(self): pass
    return Client()


def test_real_official_pipeline_keeps_state_and_drains_final_phrase():
    official = pytest.importorskip("tone")
    runner = tone_trt.TensorRTPipeline("tone_trt_greedy", Path("/unused"), 5)
    client = fake_client()
    acoustic = tone_trt.TritonAcoustic(client, fake_grpc(), runner.timer, 5)
    runner.pipeline = official.StreamingCTCPipeline(acoustic,
        tone_trt.TimedSplitter(official.StreamingLogprobSplitter(), runner.timer),
        tone_trt.TimedDecoder(official.GreedyCTCDecoder(), runner.timer))
    result = runner.run(np.full(16000, .1, dtype=np.float32))
    assert result["status"] == "ok" and result["text"] == "да"
    assert len(client.states) == 6  # 1 с записи + 0,6 с padding, кратно 300 мс
    assert all(np.all(state == i) for i, state in enumerate(client.states))
    assert np.all(client.signals[0] == 0) and np.all(client.signals[-1] == 0)
    assert result["context_resets_per_record"] == 1 and result["padding_per_record_seconds"] == .6
    assert result["trt_stages"]["acoustic"]["calls"] == result["triton_requests"] == 6
    assert result["trt_stages"]["decoder"]["calls"] == 1
    assert result["elapsed_seconds"] == result["asr_seconds"]
    assert sum(result["timing"]["exclusive_wall_percent"].values()) == pytest.approx(100)
    assert sum(result["timing"]["exclusive_wall_seconds"].values()) == pytest.approx(result["elapsed_seconds"])
    assert result["llm_seconds"] == 0 and result["analysis"] is None
    assert all(0 <= p["start"] <= p["end"] <= 1 for p in result["utterances"])
    assert client.requests[0][1]["model_version"] == "1"
    second = runner.run(np.zeros(16000, dtype=np.float32))
    assert second["status"] == "no_speech"
    assert np.all(client.states[6] == 0)  # Контекст между РАЗНЫМИ звонками сбрасывается.


def test_bad_acoustic_output_is_an_error_without_cpu_fallback():
    client = SimpleNamespace(infer=lambda *a, **kw: SimpleNamespace(as_numpy=lambda name: None))
    timer = tone_trt.Timer()
    model = tone_trt.TritonAcoustic(client, fake_grpc(), timer, 1)
    with pytest.raises(RuntimeError):
        model.forward(np.zeros((1, 2400, 1), dtype=np.int32))
    assert model.requests == 0 and timer.stages()["acoustic"]["errors"] == 1


def test_real_grpc_config_json_matches_validated_shapes():
    protobuf = pytest.importorskip("tritonclient.grpc.service_pb2")
    from google.protobuf.json_format import ParseDict, MessageToDict
    response = protobuf.ModelConfigResponse()
    ParseDict({"config": configuration()}, response)
    decoded = MessageToDict(response, preserving_proto_field_name=True)
    assert tone_trt.validate_config(decoded)["platform"] == "tensorrt_plan"


def test_server_statistics_exclude_previous_calls_and_detect_restart():
    def snapshot(count, ns):
        return {"model_stats": [{"name": "streaming_acoustic", "version": "1", "inference_stats": {
            "success": {"count": str(count), "ns": str(ns)},
            "compute_infer": {"count": str(count), "ns": str(ns)}}}]}
    delta = tone_trt_worker.statistics_delta(snapshot(10, 1000), snapshot(16, 7000), 6)
    assert delta["matches_client_requests"] and delta["compute_infer"]["seconds"] == .000006
    with pytest.raises(RuntimeError, match="сбросились"):
        tone_trt_worker.statistics_delta(snapshot(16, 7000), snapshot(1, 100), 6)
    with pytest.raises(RuntimeError, match="отличается"):
        tone_trt_worker.statistics_delta(snapshot(10, 1000), snapshot(17, 7000), 6)
    assert not tone_trt_worker.statistics_delta({}, {}, 6)["available"]


def test_download_checks_hash_and_reuses_only_verified_files(monkeypatch, tmp_path):
    blocks = {"model.onnx": b"official model", "kenlm.bin": b"official lm"}
    monkeypatch.setattr(preparation, "ARTIFACTS", {k: (len(v), hashlib.sha256(v).hexdigest()) for k, v in blocks.items()})
    requested = []
    class Response:
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size):
            data, self.data = self.data, b""
            return data
    def urlopen(url, **kwargs):
        requested.append(url)
        return Response(blocks[url.rsplit("/", 1)[1]])
    monkeypatch.setattr(preparation.urllib.request, "urlopen", urlopen)
    preparation.download(tmp_path, True)
    preparation.download(tmp_path, True)
    assert len(requested) == 2 and all(preparation.HF_REF in url for url in requested)
    assert not list(tmp_path.rglob("*.part"))
    (tmp_path / "artifacts/model.onnx").write_bytes(b"corrupt")
    preparation.download(tmp_path, False)
    assert len(requested) == 3 and (tmp_path / "artifacts/model.onnx").read_bytes() == blocks["model.onnx"]


def test_corrupt_download_never_replaces_cached_model(monkeypatch, tmp_path):
    folder = tmp_path / "artifacts"
    folder.mkdir()
    (folder / "model.onnx").write_bytes(b"previous")
    monkeypatch.setattr(preparation, "ARTIFACTS", {"model.onnx": (4, hashlib.sha256(b"good").hexdigest())})
    from io import BytesIO
    monkeypatch.setattr(preparation.urllib.request, "urlopen", lambda *a, **kw: BytesIO(b"bad!"))
    with pytest.raises(RuntimeError, match="SHA256"):
        preparation.download(tmp_path, False)
    assert (folder / "model.onnx").read_bytes() == b"previous" and not list(tmp_path.rglob("*.part"))


def test_engine_cache_changes_with_gpu_or_build_settings_and_preserves_old_engine_on_failure(monkeypatch, tmp_path):
    cache, output = tmp_path / "cache", tmp_path / "results"
    (cache / "artifacts").mkdir(parents=True)
    output.mkdir()
    (cache / "artifacts/model.onnx").write_bytes(b"official")
    monkeypatch.setattr(preparation, "ARTIFACTS", {"model.onnx": (8, hashlib.sha256(b"official").hexdigest())})
    device = {"name": "H100", "uuid": "GPU-1", "driver_version": "610", "free_mib": 18000}
    monkeypatch.setattr(preparation, "gpu_snapshot", lambda: dict(device))
    monkeypatch.setattr(preparation, "trt_version", lambda: "10.11.0")
    builds = []
    def run(command, **kwargs):
        builds.append(command)
        path = Path(next(c.split("=", 1)[1] for c in command if c.startswith("--saveEngine=")))
        path.write_bytes(f"engine-{len(builds)}".encode())
    monkeypatch.setattr(preparation.subprocess, "run", run)
    preparation.compile_engine(cache, output, 1024, 10)
    preparation.compile_engine(cache, output, 1024, 10)
    assert len(builds) == 1 and json.loads((cache / "движок.json").read_text())["reused"]
    device["uuid"] = "GPU-2"
    preparation.compile_engine(cache, output, 1024, 10)
    assert len(builds) == 2
    assert "--stronglyTyped" in builds[0] and "--skipInference" in builds[0]
    assert "--memPoolSize=workspace:1024" in builds[0]
    assert all("signal:1x2400x1" in c for c in builds[0] if "Shapes=" in c)
    old = (cache / "repository/streaming_acoustic/1/model.plan").read_bytes()
    def fail(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 10)
    monkeypatch.setattr(preparation.subprocess, "run", fail)
    with pytest.raises(subprocess.TimeoutExpired):
        preparation.compile_engine(cache, output, 512, 10)
    assert (cache / "repository/streaming_acoustic/1/model.plan").read_bytes() == old
    assert not list(cache.rglob("*.part"))


def fake_audio(source, target, max_seconds=None):
    target.write_bytes(source.read_bytes())
    pcm = np.zeros(16000, dtype="<i2").tobytes()
    return Audio(str(source), str(target), 1., 16000, hashlib.sha256(pcm).hexdigest(), .01, 1, 16000, source.stat().st_size)


def args_for(tmp_path, without_kenlm=False):
    audio, output, cache = tmp_path / "audio", tmp_path / "out", tmp_path / "cache"
    audio.mkdir(); output.mkdir(); cache.mkdir()
    for name in ("a.wav", "b.wav"):
        with wave.open(str(audio / name), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    (cache / "движок.json").write_text(json.dumps({"fingerprint": {"gpu": {"uuid": "GPU-1"}}}))
    (cache / "артефакты.json").write_text("{}")
    from benchmark.tone_run import parser
    args = parser().parse_args([str(audio), "--out", str(output), "--model-dir", str(cache),
                               "--expected-files", "2", "--gap", "0"])
    args.without_kenlm = without_kenlm
    return args


@pytest.mark.parametrize("without_kenlm,fail_first", [(False, False), (True, False), (False, True)])
def test_corpus_preserves_each_variant_and_pending_files(monkeypatch, tmp_path, without_kenlm, fail_first):
    args = args_for(tmp_path, without_kenlm)
    monkeypatch.setattr("benchmark.tone_run.prepare_audio", fake_audio)
    monkeypatch.setattr(tone_trt_run, "gpu_snapshot", lambda: {"uuid": "GPU-1"})
    calls = []
    async def worker(request):
        calls.append(request["system"])
        failed = fail_first and len(calls) == 1
        result = {"status": "error" if failed else "ok", "text": "355 рублей", "elapsed_seconds": .4,
            "asr_seconds": .4, "error": "load failure" if failed else None, "preparation_failed": failed,
            "stages": {"asr": {"seconds": .4, "calls": 1, "errors": 0}},
            "trt_stages": {key: {"seconds": .1, "calls": 1, "errors": 0} for key in tone_trt.STAGES}}
        return {"result": result, "preparation": {"system": request["system"], "load_seconds": 1., "warmup_seconds": .5}}
    monkeypatch.setattr(tone_trt_run, "run_worker", worker)
    code = asyncio.run(tone_trt_run.run(args))
    assert code == int(fail_first)
    summary = json.loads((args.out / "итоги.json").read_text())
    if fail_first:
        assert summary["systems"]["tone_trt_greedy"]["statuses"]["pending"] == 1
        assert summary["systems"]["tone_trt_kenlm"]["statuses"]["pending"] == 2
    else:
        assert calls == ["tone_trt_greedy"] * 2 + ([] if without_kenlm else ["tone_trt_kenlm"] * 2)
        for entry in summary["systems"].values():
            assert entry["statuses"]["ok"] == 2 and not entry["stages"]["vad"]["enabled"]
    tone_trt_run.finalize(args.out, code)
    assert not (args.out / "временные").exists()
    with tarfile.open(args.out / "диагностика.tar.gz") as archive:
        names = archive.getnames()
        assert "условия.json" in names and "записи/0001/tone_trt_greedy.json" in names
        assert not any(n.endswith((".wav", ".onnx", ".plan", ".bin")) for n in names)


def test_worker_keeps_preparation_and_warmup_outside_measurement(monkeypatch, tmp_path):
    calls = []
    class Pipeline:
        versions = {}; server_metadata = {}; model_metadata = {}; config = {}; engine_metadata = {"engine_sha256": "verified"}
        def __init__(self, *args): pass
        def load(self): calls.append("load")
        def run(self, samples):
            calls.append("recognize")
            return {"status": "ok", "elapsed_seconds": .5, "asr_seconds": .5,
                    "text": "355 рублей", "triton_requests": 2, "trt_stages": {}, "raw_phrases": []}
        def statistics(self): calls.append("statistics"); return {}
        def close(self): calls.append("close")
    monkeypatch.setattr(tone_trt, "TensorRTPipeline", Pipeline)
    monkeypatch.setattr(tone_trt_worker, "gpu_snapshot", lambda: {"free_mib": 18000})
    monkeypatch.setattr(tone_trt_worker, "read_audio", lambda x: np.zeros(16000, dtype=np.float32))
    audio = asdict(Audio("input", "prepared", 1., 16000, "sha", .01))
    payload = tone_trt_worker.execute({"system": "tone_trt_greedy", "output": str(tmp_path),
        "model_dir": "/cache", "timeout": 3, "audio": audio, "warmup": audio, "protocol": {}})
    assert calls == ["load", "recognize", "statistics", "recognize", "statistics", "close"]
    assert payload["result"]["elapsed_seconds"] == .5
    assert not payload["preparation"]["included_in_measurements"]
    assert (tmp_path / "tone_trt_greedy-прогрев.json").is_file()


def test_compose_keeps_audio_off_download_and_export_and_all_ports_private():
    config = yaml.safe_load((ROOT / "compose.tone-trt-benchmark.yml").read_text())
    assert config["name"] == "speech-comparison-tone-trt"
    assert set(config["services"]) == {"download", "export", "triton", "client"}
    assert config["networks"]["tone"]["internal"]
    for service in config["services"].values():
        assert "ports" not in service and "env_file" not in service
        assert not any("docker.sock" in v for v in service.get("volumes", []))
    for name in ("download", "export", "triton"):
        assert not any("recordings" in v or "BENCH_AUDIO_DIR" in v for v in config["services"][name]["volumes"])
    assert "deploy" not in config["services"]["download"]
    assert config["services"]["export"]["network_mode"] == "none"
    assert config["services"]["client"]["volumes"][0].endswith(":ro")
    assert config["services"]["client"]["environment"]["NVIDIA_DRIVER_CAPABILITIES"] == "utility"


def test_source_bundle_includes_new_image_compose_and_launcher():
    from benchmark.package import source_files
    names = {str(p.relative_to(ROOT)) for p in source_files()}
    assert {"Dockerfile.tone-trt-benchmark", "compose.tone-trt-benchmark.yml", "benchmark/run-tone-trt.sh",
            "benchmark/TONE-TRT.md", "benchmark/tone_trt.py", "benchmark/tone_trt_prepare.py"} <= names


def test_finalize_works_without_dependencies_or_models(tmp_path):
    (tmp_path / "launch.log").write_text("Сборка прервана")
    result = subprocess.run([sys.executable, "-S", "-m", "benchmark.tone_trt_run", "--finalize", str(tmp_path), "1"],
        cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "диагностика.tar.gz").is_file()


@pytest.mark.parametrize("free_mib,client_code,hang_logs,image_cached,pull_denied", [
    (18000, 0, False, True, False),
    (18000, 1, True, True, False),
    (2000, 0, False, False, False),
    (18000, 0, False, False, False),
    (18000, 0, False, False, True),
])
def test_launcher_stops_only_exact_owned_ids_and_never_hangs_on_finished_logs(
        tmp_path, free_mib, client_code, hang_logs, image_cached, pull_denied):
    binary, audio = tmp_path / "bin", tmp_path / "audio"
    binary.mkdir(); audio.mkdir()
    commands, containers = tmp_path / "commands.jsonl", tmp_path / "containers.json"
    image = tmp_path / "triton-image-present"
    if image_cached:
        image.touch()
    script = f'''#!{sys.executable}
import fcntl,hashlib,json,os,signal,sys,time
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['FAKE_COMMANDS'],'a') as f:
    fcntl.flock(f,fcntl.LOCK_EX); f.write(json.dumps([Path(sys.argv[0]).name,*args])+'\\n')
if Path(sys.argv[0]).name=='nvidia-smi':
    if any('memory.free,utilization' in a for a in args): print('{free_mib}, 100')
    else: print('2026/10/09 12:00:00, GPU-test, 100, 77000, {free_mib}, 95000, 200, 45')
    sys.exit(0)
path=Path(os.environ['FAKE_CONTAINERS'])
image=Path(os.environ['FAKE_IMAGE'])
if args[:2]==['image','inspect']:
    if 'nvcr.io/nvidia/tritonserver:25.06-py3' in args and not image.exists(): sys.exit(1)
    print('[]'); sys.exit(0)
if args[0]=='pull':
    if {pull_denied!r}:
        print('unexpected status: 403 Forbidden',file=sys.stderr); sys.exit(1)
    image.touch(); sys.exit(0)
with open(str(path)+'.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    registry=json.loads(path.read_text()) if path.exists() else {{}}
    if args[0]=='info': print(os.environ['FAKE_ROOT'])
    elif args[0]=='ps': pass
    elif args[0]=='compose' and 'run' in args:
        name=args[args.index('--name')+1]; cid=hashlib.sha256(name.encode()).hexdigest()
        registry[cid]={{'name':name,'running':'-server-' in name,'code': {client_code} if '-client-' in name else 0}}
        print(cid)
    elif args[0]=='inspect':
        key=args[-1]; cid=next((k for k,v in registry.items() if k==key or v['name']==key),None)
        if '--format' not in args: print(json.dumps(registry.get(cid,{{}})))
        else:
            fmt=args[args.index('--format')+1]; row=registry.get(cid,{{}})
            if fmt=='{{{{.Id}}}}': print(cid)
            elif '.Config.Labels' in fmt: print('speech-comparison-tone-trt')
            elif '.State.Health' in fmt: print('true healthy')
            else: print(('true' if row.get('running') else 'false')+' '+str(row.get('code',0)))
    elif args[0]=='stop':
        if args[-1] not in registry: sys.exit(3)
        registry[args[-1]]['running']=False
    path.write_text(json.dumps(registry))
if args[0]=='logs':
    print('Журнал нашего контейнера',flush=True)
    if '--follow' in args and {hang_logs!r}:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        time.sleep(120)
'''
    for name in ("docker", "nvidia-smi"):
        executable = binary / name
        executable.write_text(script)
        executable.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + ":" + os.environ["PATH"],
        "FAKE_COMMANDS": str(commands), "FAKE_CONTAINERS": str(containers), "FAKE_ROOT": str(tmp_path),
        "FAKE_IMAGE": str(image),
        "BENCH_OUT": str(tmp_path / "results"), "BENCH_CACHE": str(tmp_path / "cache"),
        "BENCH_GPU_MAX_UTIL": "100", "BENCH_MIN_RAM_MIB": "0", "BENCH_MIN_DISK_MIB": "0"}
    launched = subprocess.run(["bash", str(ROOT / "benchmark/run-tone-trt.sh"), str(audio)],
        env=env, capture_output=True, text=True, timeout=20)
    expected = 42 if free_mib < 12288 else (1 if pull_denied else client_code)
    assert launched.returncode == expected, launched.stdout + launched.stderr
    journal = [json.loads(line) for line in commands.read_text().splitlines()]
    docker = [cmd[1:] for cmd in journal if cmd[0] == "docker"]
    assert not any(any(word in cmd for word in ("rm", "prune", "up", "down")) for cmd in docker)
    assert not any(any(word in ' '.join(cmd) for word in ("whisper-asr", "model-proxy", "vllm-", "ollama")) for cmd in docker)
    for cmd in docker:
        if cmd[0] == "compose":
            assert cmd[cmd.index("--project-name") + 1] == "speech-comparison-tone-trt"
    registry = json.loads(containers.read_text()) if containers.exists() else {}
    stops = [cmd[-1] for cmd in docker if cmd[0] == "stop"]
    assert set(stops) <= set(registry)
    assert all(not registry[cid]["running"] for cid in stops)
    assert list((tmp_path / "results").glob("tone-trt-*/диагностика.tar.gz"))
    pulls = [cmd for cmd in docker if cmd[0] == "pull"]
    assert not any(cmd[0] == "compose" and "pull" in cmd for cmd in docker)
    if free_mib < 12288:
        assert not stops and not any(cmd[0] == "compose" for cmd in docker)
        assert not pulls
    elif pull_denied:
        assert not stops and not registry and not any(cmd[0] == "compose" for cmd in docker)
        assert pulls == [["pull", "nvcr.io/nvidia/tritonserver:25.06-py3"]]
        assert "403 Forbidden" in launched.stdout + launched.stderr
        assert "Обработка записей не начиналась" in launched.stdout
        with tarfile.open(next((tmp_path / "results").glob("tone-trt-*/диагностика.tar.gz"))) as archive:
            assert any(name.endswith("запуск.log") for name in archive.getnames())
    else:
        assert len(stops) == 4
        runs = [cmd for cmd in docker if cmd[0] == "compose" and "run" in cmd]
        assert len(runs) == 4 and all("--no-deps" in cmd for cmd in runs)
        assert all(cmd[cmd.index("--pull") + 1] == "never" for cmd in runs)
        assert len(pulls) == (0 if image_cached else 1)
        if pulls:
            build_index = next(i for i, cmd in enumerate(docker) if cmd[0] == "compose" and "build" in cmd)
            assert docker.index(pulls[0]) < build_index
