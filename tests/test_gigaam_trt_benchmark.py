"""GigaAM TensorRT на заглушках: без весов, Docker, Triton и GPU."""
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

from app.config import Settings
from benchmark import gigaam_trt, gigaam_trt_prepare as preparation, gigaam_trt_run, gigaam_trt_worker, pipeline
from benchmark.compare import Audio

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {"config": {"name": "gigaam_encoder", "platform": "tensorrt_plan",
        "input": [{"name": "audio_signal", "data_type": "TYPE_FP32", "dims": ["1", "64", "-1"]},
                  {"name": "length", "data_type": "TYPE_INT64", "dims": ["1"]}],
        "instance_group": [{"kind": "KIND_GPU", "gpus": [0], "count": 1}]}}


@pytest.mark.parametrize("change", ["platform", "kind", "gpu", "batch", "dims", "queue", "name"])
def test_cpu_wrong_gpu_batches_and_wrong_model_are_rejected(change):
    config = configuration()
    assert gigaam_trt.validate_config(config)["platform"] == "tensorrt_plan"
    inner = config["config"]
    if change == "platform": inner["platform"] = "onnxruntime_onnx"
    if change == "kind": inner["instance_group"][0]["kind"] = "KIND_CPU"
    if change == "gpu": inner["instance_group"][0]["gpus"] = [1]
    if change == "batch": inner["max_batch_size"] = 8
    if change == "dims": inner["input"][0]["dims"] = ["1", "80", "-1"]
    if change == "queue": inner["dynamic_batching"] = {}
    if change == "name": inner["name"] = "streaming_acoustic"
    with pytest.raises(RuntimeError):
        gigaam_trt.validate_config(config)


def test_generated_triton_config_matches_validated_shapes():
    text = preparation.model_config()
    assert 'platform: "tensorrt_plan"' in text and "max_batch_size: 0" in text
    assert "dims: [1, 64, -1]" in text and "dims: [1, 768, -1]" in text
    assert "KIND_GPU" in text and "KIND_CPU" not in text and "dynamic_batching" not in text


class ScriptedSessions:
    """Декодер помнит число выданных символов; joint выдаёт символы кадра по сценарию."""

    def __init__(self, script, blank=9):
        self.script, self.blank, self.frame, self.calls = script, blank, -1, []
        self.decoder, self.joint = SimpleNamespace(run=self.predict), SimpleNamespace(run=self.join)

    def predict(self, _, inputs):
        self.calls.append(int(inputs["x"][0, 0]))
        count = inputs["hi"] + 1
        return [np.full((1, 1, 2), float(count[0, 0, 0]), dtype=np.float32), count, inputs["ci"]]

    def join(self, _, inputs):
        frame, emitted = int(inputs["enc"][0, 0, 0]), int(inputs["dec"][0, 0, 0]) - 1
        planned = self.script[frame]
        offset = emitted - sum(len(self.script[i]) for i in range(frame))
        token = planned[offset] if 0 <= offset < len(planned) else self.blank
        logits = np.zeros((1, 1, 1, self.blank + 1), dtype=np.float32)
        logits[0, 0, 0, token] = 1
        return [logits]


def encoded_frames(count):
    return np.arange(count, dtype=np.float32).reshape(1, 1, count).repeat(2, axis=1)


def test_greedy_decoder_emits_tokens_with_their_frames_and_feeds_labels_back():
    fake = ScriptedSessions({0: [], 1: [4, 5], 2: [], 3: [6]})
    decoder = gigaam_trt.GreedyDecoder(fake.decoder, fake.joint, blank=9, hidden=2, layers=1, max_symbols=10)
    assert decoder.decode(encoded_frames(4), 4) == ([4, 5, 6], [1, 1, 3])
    # Первый вызов — пустой символ с нулевым состоянием, дальше каждый выданный символ.
    assert fake.calls == [9, 4, 5, 6]
    # Длина выхода энкодера ограничивает разбор: кадр 3 не читается.
    fake = ScriptedSessions({0: [], 1: [4, 5], 2: [], 3: [6]})
    decoder = gigaam_trt.GreedyDecoder(fake.decoder, fake.joint, blank=9, hidden=2, layers=1, max_symbols=10)
    assert decoder.decode(encoded_frames(4), 3) == ([4, 5], [1, 1])


def test_greedy_decoder_stops_at_symbol_limit_per_frame():
    fake = ScriptedSessions({0: [1, 2, 3, 4]})
    decoder = gigaam_trt.GreedyDecoder(fake.decoder, fake.joint, blank=9, hidden=2, layers=1, max_symbols=2)
    assert decoder.decode(encoded_frames(1), 1) == ([1, 2], [0, 0])


class Input:
    def __init__(self, name, shape, datatype):
        self.name, self.shape, self.datatype = name, shape, datatype

    def set_data_from_numpy(self, value):
        self.value = value


def fake_grpc():
    return SimpleNamespace(InferInput=Input, InferRequestedOutput=lambda name: name)


def fake_client(encoded, length):
    calls = []
    def infer(model, inputs, **kwargs):
        calls.append((model, {i.name: (i.datatype, tuple(i.shape)) for i in inputs}, kwargs["outputs"]))
        return SimpleNamespace(as_numpy=lambda name: {"encoded": encoded, "encoded_len": length}[name])
    return SimpleNamespace(infer=infer, calls=calls)


def test_encoder_sends_one_fragment_and_counts_requests():
    client = fake_client(np.zeros((1, 768, 25), dtype=np.float32), np.array([25], dtype=np.int32))
    encoder = gigaam_trt.TritonEncoder(client, fake_grpc(), 600)
    encoded, frames = encoder.forward(np.zeros((1, 64, 100), dtype=np.float32), 100)
    assert encoded.shape == (1, 768, 25) and frames == 25 and encoder.requests == 1 and encoder.timeout == 60
    model, inputs, outputs = client.calls[0]
    assert model == "gigaam_encoder" and outputs == ["encoded", "encoded_len"]
    assert inputs == {"audio_signal": ("FP32", (1, 64, 100)), "length": ("INT64", (1,))}


@pytest.mark.parametrize("encoded,length", [
    (np.full((1, 768, 5), np.nan, dtype=np.float32), np.array([5], dtype=np.int32)),
    (np.zeros((1, 512, 5), dtype=np.float32), np.array([5], dtype=np.int32)),
    (np.zeros((1, 768, 5), dtype=np.float16), np.array([5], dtype=np.int32)),
    (np.zeros((1, 768, 5), dtype=np.float32), np.array([6], dtype=np.int32)),
    (np.zeros((1, 768, 5), dtype=np.float32), np.array([0], dtype=np.int32)),
    (None, np.array([5], dtype=np.int32)),
])
def test_bad_encoder_output_is_an_error_without_cpu_fallback(encoded, length):
    encoder = gigaam_trt.TritonEncoder(fake_client(encoded, length), fake_grpc(), 5)
    with pytest.raises(RuntimeError):
        encoder.forward(np.zeros((1, 64, 100), dtype=np.float32), 100)
    assert encoder.requests == 0
    with pytest.raises(ValueError):
        encoder.forward(np.zeros((1, 80, 100), dtype=np.float32), 100)


def engines_with_fakes(tmp_path, frames=100, encoded=25):
    engines = gigaam_trt.GigaTrtEngines(Settings(emo_enabled=False, llm_enabled=False, split_turns=False), tmp_path, 5)
    engines.features = lambda samples: (np.zeros((1, 64, frames), dtype=np.float32), frames)
    engines.encoder = SimpleNamespace(requests=0, forward=lambda values, length: (np.zeros((1, 768, encoded), dtype=np.float32), encoded))
    engines.decoder = SimpleNamespace(decode=lambda values, count: ([7, 8], [0, 10]))
    engines.tokenizer = SimpleNamespace(decode=lambda ids: " Добрый день. ", id_to_str=lambda i: {7: "▁Добрый", 8: "▁день."}[i])
    return engines


def test_transcribe_returns_text_tokens_and_times_and_measures_parts(tmp_path):
    engines = engines_with_fakes(tmp_path)
    try:
        text, tokens, stamps = engines.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == "Добрый день." and tokens == ["▁Добрый", "▁день."]
        assert stamps == pytest.approx([0., 10 * (1 / 25)])
        assert engines.timer.snapshot()["asr"]["calls"] == 1
        assert [engines.sub.values[key]["calls"] for key in ("features", "encoder", "decoder")] == [1, 1, 1]
        assert engines.core_ready is False and engines.components["spk"].state == "off"
    finally:
        for pool in (engines.asr_pool, engines.emo_pool, engines.live_pool):
            pool.shutdown()


def test_too_short_fragment_is_skipped_and_too_long_is_an_error(tmp_path):
    engines = engines_with_fakes(tmp_path, frames=preparation.MIN_FRAMES - 1)
    called = []
    engines.encoder.forward = lambda *args: called.append(args)
    try:
        assert engines.transcribe(np.zeros(1000, dtype=np.float32)) == ("", [], [])
        assert engines.skipped_short == 1 and not called
        engines.features = lambda samples: (np.zeros((1, 64, 8), dtype=np.float32), preparation.MAX_FRAMES + 1)
        with pytest.raises(RuntimeError, match="длиннее профиля"):
            engines.transcribe(np.zeros(1000, dtype=np.float32))
        assert not called and engines.timer.snapshot()["asr"]["errors"] == 1
    finally:
        for pool in (engines.asr_pool, engines.emo_pool, engines.live_pool):
            pool.shutdown()


def test_vad_runs_on_cpu_by_default_and_only_known_providers_are_allowed(monkeypatch):
    monkeypatch.delenv("BENCH_VAD_PROVIDER", raising=False)
    assert preparation.vad_provider() == "cpu"
    monkeypatch.setenv("BENCH_VAD_PROVIDER", "CUDA")
    assert preparation.vad_provider() == "cuda"
    monkeypatch.setenv("BENCH_VAD_PROVIDER", "tensorrt")
    with pytest.raises(ValueError):
        preparation.vad_provider()


def test_pipeline_refuses_speakers_emotions_and_llm(tmp_path):
    with pytest.raises(ValueError):
        gigaam_trt.GigaTrtPipeline(Settings(emo_enabled=True, llm_enabled=False, split_turns=False), 1, 3, tmp_path)
    with pytest.raises(ValueError):
        gigaam_trt.GigaTrtPipeline(Settings(emo_enabled=False, llm_enabled=True, split_turns=False), 1, 3, tmp_path)


def test_pipeline_keeps_the_reduced_flow_and_reports_tensorrt_parts(tmp_path, monkeypatch):
    from test_benchmark import ScriptedVad, speech
    from app.vad import StreamingVad

    async def scenario():
        runner = gigaam_trt.GigaTrtPipeline(Settings(emo_enabled=False, llm_enabled=False, split_turns=False), 1, 3, tmp_path)
        engines = runner.engines
        fakes = engines_with_fakes(tmp_path)
        engines.features, engines.encoder, engines.decoder, engines.tokenizer = fakes.features, fakes.encoder, fakes.decoder, fakes.tokenizer
        for pool in (fakes.asr_pool, fakes.emo_pool, fakes.live_pool):
            pool.shutdown()
        engines.new_vad = lambda: pipeline.TimedVad(StreamingVad(ScriptedVad()), engines.timer)
        for key in ("vad", "asr"):
            engines._set(key, "ready")
        engines.embed = lambda samples: (_ for _ in ()).throw(AssertionError("Голоса отключены"))
        engines.emotions = lambda samples: (_ for _ in ()).throw(AssertionError("Эмоции отключены"))
        try:
            result = await runner.run(speech())
            assert result["status"] == "ok" and result["system"] == "gigaam_trt"
            assert result["pipeline"] == "vad_gigaam_tensorrt" and result["enabled_stages"] == ["vad", "asr"]
            assert result["text"] == "Добрый день." and not {"llm_seconds", "analysis"} & set(result)
            assert result["trt_stages"]["encoder"]["calls"] == result["stages"]["asr"]["calls"] >= 1
            assert result["vad_provider"] == "cpu" and result["skipped_short_segments"] == 0
            assert result["pauses"] and result["speech_segments"]
            assert sum(result["timing"]["exclusive_wall_percent"].values()) == pytest.approx(100)
            silent = await runner.run(np.zeros(16000, dtype=np.float32))
            assert silent["status"] == "no_speech" and silent["trt_stages"]["encoder"]["calls"] == 0
        finally:
            runner.shutdown()
    asyncio.run(scenario())


def test_server_statistics_use_the_encoder_model_and_detect_restart():
    def stats(count, ns):
        return {"model_stats": [{"name": "gigaam_encoder", "version": "1", "inference_stats": {
            key: {"count": count, "ns": ns} for key in ("success", "fail", "queue", "compute_input", "compute_infer", "compute_output")}}]}
    delta = gigaam_trt_worker.statistics_delta(stats(10, 1000), stats(13, 4000), 3, "gigaam_encoder")
    assert delta["success"] == {"calls": 3, "seconds": 3e-6} and delta["matches_client_requests"]
    assert not gigaam_trt_worker.statistics_delta(stats(10, 1000), stats(13, 4000), 3, "streaming_acoustic")["available"]
    with pytest.raises(RuntimeError, match="отличается"):
        gigaam_trt_worker.statistics_delta(stats(10, 1000), stats(13, 4000), 4, "gigaam_encoder")
    with pytest.raises(RuntimeError, match="сбросились"):
        gigaam_trt_worker.statistics_delta(stats(10, 1000), stats(2, 100), 3, "gigaam_encoder")


def prepared_cache(tmp_path):
    cache, output = tmp_path / "cache", tmp_path / "results"
    (cache / "artifacts").mkdir(parents=True)
    output.mkdir()
    name = "v3_e2e_rnnt_encoder.onnx"
    (cache / "artifacts" / name).write_bytes(b"encoder")
    (cache / "артефакты.json").write_text(json.dumps({"files": {name: {"sha256": hashlib.sha256(b"encoder").hexdigest()}}}))
    return cache, output


def test_engine_cache_changes_with_gpu_and_preserves_old_engine_on_failure(monkeypatch, tmp_path):
    cache, output = prepared_cache(tmp_path)
    device = {"name": "RTX", "uuid": "GPU-1", "driver_version": "610", "free_mib": 10000}
    monkeypatch.setattr(preparation, "gpu_snapshot", lambda: dict(device))
    monkeypatch.setattr(preparation, "trt_version", lambda: "10.11.0")
    builds = []
    def run(command, **kwargs):
        builds.append(command)
        Path(next(c.split("=", 1)[1] for c in command if c.startswith("--saveEngine="))).write_bytes(f"engine-{len(builds)}".encode())
    monkeypatch.setattr(preparation.subprocess, "run", run)
    preparation.compile_engine(cache, output, 4096, 10)
    preparation.compile_engine(cache, output, 4096, 10)
    assert len(builds) == 1 and json.loads((cache / "движок.json").read_text())["reused"]
    assert "--fp16" in builds[0] and "--skipInference" in builds[0] and "--memPoolSize=workspace:4096" in builds[0]
    assert [c.split("=")[1] for c in builds[0] if "Shapes=" in c] == [
        "audio_signal:1x64x16,length:1", "audio_signal:1x64x800,length:1", "audio_signal:1x64x3200,length:1"]
    assert (cache / "repository/gigaam_encoder/config.pbtxt").read_text() == preparation.model_config()
    assert json.loads((output / "tensorRT-подготовка.json").read_text())["included_in_measurements"] is False
    device["uuid"] = "GPU-2"
    preparation.compile_engine(cache, output, 4096, 10)
    assert len(builds) == 2
    old = (cache / "repository/gigaam_encoder/1/model.plan").read_bytes()
    def fail(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 10)
    monkeypatch.setattr(preparation.subprocess, "run", fail)
    with pytest.raises(subprocess.TimeoutExpired):
        preparation.compile_engine(cache, output, 2048, 10)
    assert (cache / "repository/gigaam_encoder/1/model.plan").read_bytes() == old
    assert not list(cache.rglob("*.part"))


def test_engine_is_not_built_from_a_missing_or_changed_export(monkeypatch, tmp_path):
    cache, output = prepared_cache(tmp_path)
    monkeypatch.setattr(preparation.subprocess, "run", lambda *a, **k: pytest.fail("trtexec не должен запускаться"))
    (cache / "artifacts/v3_e2e_rnnt_encoder.onnx").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="повреждён"):
        preparation.compile_engine(cache, output, 4096, 10)
    (cache / "артефакты.json").unlink()
    with pytest.raises(RuntimeError, match="экспорт"):
        preparation.compile_engine(cache, output, 4096, 10)


def fake_audio(source, target, max_seconds=None):
    target.write_bytes(source.read_bytes())
    pcm = np.zeros(16000, dtype="<i2").tobytes()
    return Audio(str(source), str(target), 1., 16000, hashlib.sha256(pcm).hexdigest(), .01, 1, 16000, source.stat().st_size)


def args_for(tmp_path):
    audio, output, cache = tmp_path / "audio", tmp_path / "out", tmp_path / "cache"
    audio.mkdir(); output.mkdir(); cache.mkdir()
    for name in ("a.wav", "b.wav"):
        with wave.open(str(audio / name), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    (cache / "движок.json").write_text(json.dumps({"fingerprint": {"gpu": {"uuid": "GPU-1"}}}))
    (cache / "артефакты.json").write_text("{}")
    from benchmark.trt_common import parser
    return parser().parse_args([str(audio), "--out", str(output), "--model-dir", str(cache)])


@pytest.mark.parametrize("fail_first", [False, True])
def test_corpus_keeps_results_reports_and_pending_files(monkeypatch, tmp_path, fail_first):
    args = args_for(tmp_path)
    monkeypatch.delenv("BENCH_VAD_PROVIDER", raising=False)
    monkeypatch.setattr("benchmark.trt_common.prepare_audio", fake_audio)
    monkeypatch.setattr(gigaam_trt_run, "gpu_snapshot", lambda: {"uuid": "GPU-1"})
    calls = []
    async def worker(request):
        calls.append(request["system"])
        assert request["vad_settings"]["vad_max_speech"] == 15. and request["model_dir"] == str(args.model_dir)
        failed = fail_first and len(calls) == 1
        stages = {key: {"seconds": .2, "calls": 1, "errors": 0} for key in pipeline.STAGES}
        result = {"status": "error" if failed else "ok", "text": "Добрый день.", "elapsed_seconds": .4, "asr_seconds": .2,
            "error": "load failure" if failed else None, "preparation_failed": failed, "stages": stages,
            "pause_seconds": .5, "triton_requests": 1,
            "utterances": [{"text": "Добрый день.", "start": 0., "end": 1.}], "stage_calls": [{"stage": "asr"}],
            "timing": {"exclusive_wall_seconds": {"vad": .1, "asr": .1, "parallel": .2, "other": 0},
                       "exclusive_wall_percent": {"vad": 25., "asr": 25., "parallel": 50., "other": 0.}},
            "trt_stages": {key: {"seconds": .05, "calls": 1, "errors": 0} for key in preparation.TRT_STAGES}}
        return {"result": result, "preparation": {"system": request["system"], "load_seconds": 1., "warmup_seconds": .5}}
    monkeypatch.setattr(gigaam_trt_run, "run_worker", worker)
    code = asyncio.run(gigaam_trt_run.run(args))
    assert code == int(fail_first)
    conditions = json.loads((args.out / "условия.json").read_text())
    summary = json.loads((args.out / "итоги.json").read_text())
    entry = summary["systems"]["gigaam_trt"]
    assert conditions["mode"] == "gigaam_tensorrt" and conditions["vad_provider"] == "cpu"
    assert conditions["protocol"]["disabled_stages"] == ["speaker", "emotion", "llm"]
    assert "gap_seconds" not in conditions["protocol"]
    assert entry["label"] == "GigaAM TensorRT"
    # Файл записи хранит подробности целиком, хотя в памяти прогона остаётся только нужное сводке.
    saved = json.loads((args.out / "записи/0001/gigaam_trt.json").read_text())
    assert saved["utterances"] and saved["stage_calls"]
    if fail_first:
        assert calls == ["gigaam_trt"] and entry["statuses"]["pending"] == 1 and conditions["state"].startswith("Остановлен")
    else:
        assert calls == ["gigaam_trt"] * 2 and entry["statuses"]["ok"] == 2 and conditions["state"] == "Завершён"
        assert entry["stages"]["vad"]["measured"] and entry["stages"]["asr"]["measured"]
        assert set(entry["stages"]) == {"vad", "asr"} and set(entry["mean_record_wall_percent"]) == {"vad", "asr", "parallel", "other"}
        assert entry["trt_stages"]["encoder"]["calls"] == 2 and entry["trt_stages"]["encoder"]["seconds"]["total"] == pytest.approx(.1)
        header = (args.out / "замеры.csv").read_text(encoding="utf-8-sig").splitlines()[0]
        assert "Энкодер и gRPC, с" in header and "Паузы, с" in header and "Вычисление энкодера внутри Triton, с" in header
        assert "LLM" not in header and "Голоса" not in header
        page = (args.out / "отчёт.html").read_text()
        assert "GigaAM TensorRT" in page and "Из чего складывается распознавание" in page and "CPU" in "".join(summary["notes"])
    gigaam_trt_run.finalize(args.out, code)
    assert not (args.out / "временные").exists()
    with tarfile.open(args.out / "диагностика.tar.gz") as archive:
        names = archive.getnames()
        assert "условия.json" in names and "записи/0001/gigaam_trt.json" in names
        assert not any(n.endswith((".wav", ".onnx", ".plan", ".ckpt")) for n in names)


def test_worker_keeps_load_and_warmup_outside_measurement(monkeypatch, tmp_path):
    calls = []
    class Pipeline:
        def __init__(self, cfg, threads, timeout, cache):
            assert not cfg.emo_enabled and not cfg.llm_enabled and not cfg.split_turns and cfg.vad_max_speech == 15.
            self.engines = SimpleNamespace(versions={}, server_metadata={}, model_metadata={}, config={},
                engine_metadata={"engine_sha256": "verified"}, placement=lambda: {"speaker": "отключено"})
        async def load(self): calls.append("load")
        async def run(self, samples):
            calls.append("recognize")
            return {"status": "ok", "elapsed_seconds": .5, "asr_seconds": .2, "text": "Добрый день.",
                    "triton_requests": 2, "trt_stages": {}, "utterances": [{"start": 0., "end": 1., "text": "Добрый день."}]}
        def statistics(self): calls.append("statistics"); return {}
        def close(self): calls.append("close")
    monkeypatch.setattr(gigaam_trt, "GigaTrtPipeline", Pipeline)
    monkeypatch.setattr(gigaam_trt_worker, "gpu_snapshot", lambda: {"free_mib": 10000})
    monkeypatch.setattr(gigaam_trt_worker, "read_audio", lambda x: np.zeros(16000, dtype=np.float32))
    audio = asdict(Audio("input", "prepared", 1., 16000, "sha", .01))
    payload = asyncio.run(gigaam_trt_worker.execute({"system": "gigaam_trt", "output": str(tmp_path), "threads": 1,
        "model_dir": "/cache", "timeout": 3, "audio": audio, "warmup": audio, "protocol": {},
        "vad_settings": {"vad_threshold": .5, "vad_min_silence": .45, "vad_min_speech": .25, "vad_max_speech": 15.}}))
    assert calls == ["load", "recognize", "statistics", "recognize", "statistics", "close"]
    assert payload["result"]["elapsed_seconds"] == .5 and payload["result"]["status"] == "ok"
    assert not payload["preparation"]["included_in_measurements"] and payload["preparation"]["warmup_seconds"] == .5
    assert (tmp_path / "gigaam_trt-прогрев.json").is_file()


def test_run_and_prepare_import_without_models_or_numeric_libraries(tmp_path):
    """Завершение на хосте и сборка в образе Triton идут на чистом Python."""
    (tmp_path / "launch.log").write_text("Сборка прервана")
    result = subprocess.run([sys.executable, "-S", "-m", "benchmark.gigaam_trt_run", "--finalize", str(tmp_path), "1"],
        cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "диагностика.tar.gz").is_file()
    result = subprocess.run([sys.executable, "-S", "-c", "import benchmark.gigaam_trt_prepare as p; print(p.MODEL)"],
        cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stdout.strip() == "gigaam_encoder", result.stderr


def test_compose_keeps_audio_off_preparation_and_all_ports_private():
    config = yaml.safe_load((ROOT / "compose.gigaam-trt-benchmark.yml").read_text())
    assert config["name"] == "speech-comparison-gigaam-trt"
    assert set(config["services"]) == {"export", "compile", "triton", "client"}
    assert config["networks"]["gigaam"]["internal"]
    for service in config["services"].values():
        assert "ports" not in service and "env_file" not in service
        assert not any("docker.sock" in v for v in service.get("volumes", []))
    for name in ("export", "compile", "triton"):
        assert not any("recordings" in v or "BENCH_AUDIO_DIR" in v for v in config["services"][name]["volumes"])
    assert "deploy" not in config["services"]["export"]
    assert config["services"]["compile"]["network_mode"] == "none"
    assert config["services"]["client"]["volumes"][0].endswith(":ro")
    assert config["services"]["client"]["environment"]["BENCH_VAD_PROVIDER"] == "${BENCH_VAD_PROVIDER:-cpu}"
    assert not any("ollama" in json.dumps(service).lower() for service in config["services"].values())


@pytest.mark.parametrize("free_mib,client_code,image_cached,base_built", [
    (18000, 0, True, True), (18000, 7, True, True), (2000, 0, True, True), (18000, 0, False, True), (18000, 0, True, False)])
def test_launcher_prepares_in_order_and_stops_only_its_own_containers(tmp_path, free_mib, client_code, image_cached, base_built):
    binary, audio = tmp_path / "bin", tmp_path / "audio"
    binary.mkdir(); audio.mkdir()
    commands, containers = tmp_path / "commands.jsonl", tmp_path / "containers.json"
    script = f'''#!{sys.executable}
import fcntl,hashlib,json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['FAKE_COMMANDS'],'a') as f:
    fcntl.flock(f,fcntl.LOCK_EX); f.write(json.dumps([Path(sys.argv[0]).name,*args])+'\\n')
if Path(sys.argv[0]).name=='nvidia-smi':
    if any('memory.free,utilization' in a for a in args): print('{free_mib}, 100')
    else: print('2026/10/12 16:00:00, GPU-test, 100, 2000, {free_mib}, 12282, 100, 45')
    sys.exit(0)
if args[:2]==['image','inspect']:
    if 'nvcr.io/nvidia/tritonserver:25.06-py3' in args and not {image_cached!r}: sys.exit(1)
    if args[2:]==['speech-comparison:4.0.0-gigaam-cuda'] and not {base_built!r}: sys.exit(1)
    print('[]'); sys.exit(0)
if args[0]=='pull':
    print('unexpected status: 403 Forbidden',file=sys.stderr); sys.exit(1)
path=Path(os.environ['FAKE_CONTAINERS'])
with open(str(path)+'.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    registry=json.loads(path.read_text()) if path.exists() else {{}}
    if args[0]=='info': print(os.environ['FAKE_ROOT'])
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
            elif '.Config.Labels' in fmt: print('speech-comparison-gigaam-trt')
            elif '.State.Health' in fmt: print('true healthy')
            else: print(('true' if row.get('running') else 'false')+' '+str(row.get('code',0)))
    elif args[0]=='stop':
        if args[-1] not in registry: sys.exit(3)
        registry[args[-1]]['running']=False
    path.write_text(json.dumps(registry))
if args[0]=='logs': print('Журнал нашего контейнера',flush=True)
'''
    for name in ("docker", "nvidia-smi"):
        executable = binary / name
        executable.write_text(script)
        executable.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + ":" + os.environ["PATH"], "FAKE_COMMANDS": str(commands),
        "FAKE_CONTAINERS": str(containers), "FAKE_ROOT": str(tmp_path),
        "BENCH_OUT": str(tmp_path / "results"), "BENCH_CACHE": str(tmp_path / "cache"),
        "BENCH_GPU_MAX_UTIL": "100", "BENCH_MIN_RAM_MIB": "0", "BENCH_MIN_DISK_MIB": "0"}
    env.pop("BENCH_VAD_PROVIDER", None)
    env.pop("BENCH_GPU_MIN_FREE_MIB", None)
    launched = subprocess.run(["bash", str(ROOT / "benchmark/run-gigaam-trt.sh"), str(audio), "--max-audio-seconds", "7200"],
        env=env, capture_output=True, text=True, timeout=30)
    expected = 42 if free_mib < 12288 else 1 if not image_cached else client_code
    assert launched.returncode == expected, launched.stdout + launched.stderr
    journal = [json.loads(line) for line in commands.read_text().splitlines()]
    docker = [cmd[1:] for cmd in journal if cmd[0] == "docker"]
    assert not any(any(word in cmd for word in ("rm", "prune", "up", "down")) for cmd in docker)
    assert not any(any(word in " ".join(cmd) for word in ("whisper-asr", "model-proxy", "vllm-", "ollama")) for cmd in docker)
    for cmd in docker:
        if cmd[0] == "compose":
            assert cmd[cmd.index("--project-name") + 1] == "speech-comparison-gigaam-trt"
    registry = json.loads(containers.read_text()) if containers.exists() else {}
    stops = [cmd[-1] for cmd in docker if cmd[0] == "stop"]
    assert set(stops) <= set(registry) and all(not registry[cid]["running"] for cid in stops)
    assert list((tmp_path / "results").glob("gigaam-trt-*/диагностика.tar.gz"))
    runs = [cmd for cmd in docker if cmd[0] == "compose" and "run" in cmd]
    if free_mib < 12288 or not image_cached:
        assert not runs and not stops
        if not image_cached:
            assert "Обработка записей не начиналась" in launched.stdout and "fetch_triton_image.py" in launched.stdout
        return
    order = [next(part for part in ("export", "compile", "triton", "client") if part in cmd[cmd.index("--name") + 2:][:1]) for cmd in runs]
    assert order == ["export", "compile", "triton", "client"]
    # Общий образ библиотек собирается только когда его нет, своим файлом и раньше клиентского образа.
    builds = [cmd for cmd in docker if cmd[0] == "compose" and "build" in cmd]
    assert [cmd[-1] for cmd in builds] == ([] if base_built else ["base"]) + ["client"]
    assert all(cmd[cmd.index("-f") + 1].endswith("compose.gigaam-trt-benchmark.yml" if cmd[-1] == "client" else "compose.trt-base.yml") for cmd in builds)
    assert all("--pull" in cmd and cmd[cmd.index("--pull") + 1] == "never" for cmd in runs)
    client = runs[-1]
    assert client[-4:] == ["--out", "/results", "--max-audio-seconds", "7200"] and "/recordings" in client
    assert "поиск речи: cpu" in launched.stdout


@pytest.mark.parametrize("script", ["run-gigaam-trt.sh", "run-tone-trt.sh"])
@pytest.mark.parametrize("flag", [["--gap", "1"], ["--expected-files", "3878"], ["--gap=0"]])
def test_launchers_refuse_removed_limits_before_touching_docker(tmp_path, script, flag):
    audio = tmp_path / "audio"
    audio.mkdir()
    env = {**os.environ, "BENCH_OUT": str(tmp_path / "results"), "BENCH_CACHE": str(tmp_path / "cache")}
    refused = subprocess.run(["bash", str(ROOT / "benchmark" / script), str(audio), *flag],
        env=env, capture_output=True, text=True, timeout=15)
    assert refused.returncode == 2 and "больше нет" in refused.stderr
    assert not (tmp_path / "results").exists() and not (tmp_path / "cache").exists()
