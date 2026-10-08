"""Три режима на заглушках; без скачивания весов, Docker и GPU inference."""
import asyncio
from dataclasses import asdict
import builtins
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from benchmark import artifacts, compare, pipeline, server_run, summary, worker
from test_benchmark import FakeEngines, ScriptedVad, speech, write_wav
from app.vad import StreamingVad


def test_first_line_loads_only_vad_and_asr_models(monkeypatch):
    cfg = Settings(emo_enabled=False, llm_enabled=False, split_turns=False)
    loaded = []
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "sherpa_onnx":
            raise AssertionError("CAM++ не должен импортироваться для первой линии")
        return real_import(name, *args, **kwargs)
    def load_model(name, **kwargs):
        loaded.append(name)
        assert kwargs["device"] == "cuda:0" and kwargs["fp16_encoder"] is True
        return SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type="cuda"))]))
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setitem(sys.modules, "gigaam", SimpleNamespace(load_model=load_model))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test", set_num_threads=lambda n: None,
        cuda=SimpleNamespace(is_available=lambda: True, Stream=lambda **kwargs: object())))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(__version__="test"))
    monkeypatch.setattr("benchmark.gpu.log_gpu_memory", lambda *args: None)
    def new_vad(self):
        self._vad_sess = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"])
        return SimpleNamespace(accept=lambda x: [])
    monkeypatch.setattr(pipeline.TimedEngines, "new_vad", new_vad)
    async def scenario():
        runner = pipeline.FirstLinePipeline(cfg, 1, 3)
        try:
            await runner.load()
            assert loaded == ["v3_e2e_rnnt"]
            assert all(runner.engines.components[k].state == "off" for k in ("spk", "emo", "llm"))
            assert runner.engines._spk is None and runner.engines._emo is None and runner.llm is None
            assert runner.engines.placement()["speaker"] == "отключено"
        finally:
            runner.shutdown()
    asyncio.run(scenario())


def test_first_line_runs_vad_asr_without_any_speaker_emotion_or_llm_calls(monkeypatch):
    class Engines(FakeEngines):
        def __init__(self):
            super().__init__()
            self.emotions_ready = False
        def new_vad(self):
            return pipeline.TimedVad(StreamingVad(ScriptedVad()), self.timer)
        def embed(self, samples):
            raise AssertionError("Голоса отключены")
        def emotions(self, samples):
            raise AssertionError("Эмоции отключены")
    def forbidden(*args, **kwargs):
        raise AssertionError("LLM первой линии не должна создаваться")
    monkeypatch.setattr(pipeline, "LLM", forbidden)

    async def scenario():
        cfg = Settings(emo_enabled=False, llm_enabled=False, split_turns=False)
        runner = pipeline.FirstLinePipeline(cfg, 1, 3)
        runner.shutdown()
        runner.engines = Engines()
        try:
            for _ in range(2):
                result = await runner.run(speech())
                assert result["status"] == "ok" and result["pipeline"] == "first_line"
                assert result["text"] == "Добрый день."
                assert result["utterances"][0]["speaker"] is None
                assert result["utterances"][0]["id"] == 1
                assert result["utterances"][0]["word_timestamps"]
                assert result["analysis"] is None and result["llm_seconds"] == 0
                assert result["enabled_stages"] == ["vad", "asr"]
                assert result["stages"]["speaker"]["calls"] == result["stages"]["emotion"]["calls"] == 0
                assert result["speech_segments"] and len(result["pauses"]) == 2
                assert {call["stage"] for call in result["stage_calls"]} == {"vad", "asr"}
                assert sum(result["timing"]["exclusive_wall_percent"].values()) == pytest.approx(100)
            silent = await runner.run(np.zeros(16000, dtype=np.float32))
            assert silent["status"] == "no_speech" and silent["asr_seconds"] == 0
            assert silent["pauses"] == [{"start": 0., "end": 1., "seconds": 1.}]
        finally:
            runner.shutdown()
    asyncio.run(scenario())


def test_wall_percentages_count_concurrency_once():
    result = pipeline.wall_time_breakdown([
        {"stage": "vad", "offset_seconds": 0, "seconds": 2},
        {"stage": "asr", "offset_seconds": 1, "seconds": 2},
        {"stage": "llm", "offset_seconds": 3, "seconds": 1},
    ], 5)
    seconds = result["exclusive_wall_seconds"]
    assert seconds["vad"] == seconds["asr"] == seconds["parallel"] == seconds["llm"] == seconds["other"] == 1
    assert sum(seconds.values()) == 5
    assert sum(result["exclusive_wall_percent"].values()) == pytest.approx(100)


def test_summary_distinguishes_unmeasured_whisper_stages_and_disabled_models(tmp_path):
    row = {"directory": "записи/0001", "audio": compare.Audio("test.wav", "prepared.wav", 10, 160000, "test", .1),
           "whisper": {"status": "ok", "elapsed_seconds": 1},
           "gigaam": {"status": "error", "elapsed_seconds": 5, "llm_seconds": 1,
                      "stage_calls": [{"stage": "llm", "seconds": 1, "error": True}]}}
    conditions = {"state": "Остановлен из-за ошибки", "planned_systems": list(summary.SYSTEMS)}
    result = summary.write_summary(tmp_path, [row], conditions)
    for stage in ("asr", "vad"):
        whisper = result["systems"]["whisper"]["stages"][stage]
        assert whisper["enabled"] is True and whisper["measured"] is False
        assert whisper["seconds"] == {"count": 0}
        assert whisper["calls"] is None and whisper["operation_time_vs_total_percent"] is None
    llm = result["systems"]["gigaam"]["stages"]["llm"]
    assert llm["calls"] == llm["errors"] == 1
    assert result["matched_comparison"]["records"] == 0
    row["gigaam_first_line"] = {"status": "ok", "elapsed_seconds": 2, "asr_seconds": 1, "llm_seconds": 0,
        "stages": {k: {"seconds": 1 if k in {"asr", "vad"} else 0,
                        "calls": 1 if k in {"asr", "vad"} else 0, "errors": 0} for k in pipeline.STAGES}}
    result = summary.write_summary(tmp_path, [row], conditions)
    first = result["systems"]["gigaam_first_line"]["stages"]
    assert first["asr"]["measured"] is True
    for stage in ("speaker", "emotion", "llm"):
        assert first[stage]["enabled"] is False and first[stage]["measured"] is False
        assert first[stage]["seconds"] == {"count": 0}


def test_first_line_worker_never_contacts_ollama(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    write_wav(source, 200)
    audio = compare.prepare_audio(source, tmp_path / "prepared.wav")
    class First:
        def __init__(self, cfg, *args):
            assert not cfg.emo_enabled and not cfg.llm_enabled and not cfg.split_turns
            self.llm_runtime = None
            self.engines = SimpleNamespace(versions={}, placement=lambda: {"speaker": "отключено"})
        async def load(self):
            pass
        async def run(self, samples):
            return {"status": "ok", "elapsed_seconds": .1, "asr_seconds": .05, "text": "Текст", "analysis": None}
        def shutdown(self):
            pass
    async def forbidden(*args):
        raise AssertionError("Не должно быть обращений к Ollama")
    monkeypatch.setattr(pipeline, "FirstLinePipeline", First)
    monkeypatch.setattr("benchmark.gpu.require_idle_ollama", forbidden)
    monkeypatch.setattr("benchmark.gpu.gpu_info", lambda: {"uuid": "test"})
    monkeypatch.setattr("benchmark.gpu.log_gpu_memory", lambda *args: {"free_mib": 90000})
    result = asyncio.run(worker.execute({"system": "gigaam_first_line", "audio": asdict(audio), "warmup": asdict(audio),
        "output": str(tmp_path), "threads": 1, "timeout": 3}))
    assert result["result"]["status"] == "ok"
    assert result["preparation"]["llm_runtime"] is None
    assert "ollama_loaded_models" not in result["preparation"]
    assert result["preparation"]["process_usage"]["peak_rss_mib"] > 0


@pytest.mark.parametrize("failure", [None, "full", "first"])
def test_three_phases_use_identical_corpus_and_keep_separate_results(tmp_path, monkeypatch, failure):
    source, output = tmp_path / "audio", tmp_path / "results"
    source.mkdir()
    output.mkdir()
    write_wav(source / "ncc.wav", 200)
    write_wav(source / "telphin.wav", 300)
    calls = []
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
            calls.append(("whisper", path.name))
            return {"status": "ok", "elapsed_seconds": 1, "text": "Whisper", "mode": "test_server_api"}
    async def run_worker(request):
        system = request["system"]
        compare.read_audio(compare.Audio(**request["audio"]))
        calls.append((system, request["audio"]["sha256_pcm"]))
        failed = failure == ("first" if system == "gigaam_first_line" else "full")
        return {"result": {"status": "error" if failed else "ok", "error": "Тест" if failed else None,
            "elapsed_seconds": 2 if system == "gigaam_first_line" else 3, "asr_seconds": .5,
            "text": system, "utterances": [], "stages": {}, "llm_seconds": 0 if system == "gigaam_first_line" else 1},
            "preparation": {"system": system, "gpu": {"uuid": "test"}, "included_in_measurements": False}}
    async def no_sleep(*args):
        pass
    monkeypatch.setattr(server_run, "WhisperAPI", API)
    monkeypatch.setattr(compare, "run_worker", run_worker)
    monkeypatch.setattr(server_run.asyncio, "sleep", no_sleep)
    args = compare.parser().parse_args(["--include-first-line", "--audio-dir", str(source), "--out", str(output), "--expected-files", "2"])
    assert asyncio.run(server_run.whisper_phase(args)) == 0
    logs = output / "логи"
    logs.mkdir()
    (logs / "тестовый-whisper-остановлен.txt").touch()
    original_whisper = (output / "записи/0001/whisper.json").read_bytes()
    assert asyncio.run(server_run.gigaam_phase(args)) == int(failure == "full")
    if failure == "full":
        with pytest.raises(ValueError):
            asyncio.run(server_run.first_line_phase(args))
        assert not any(name == "gigaam_first_line" for name, _ in calls)
    else:
        assert json.loads((output / "условия.json").read_text())["state"] == "Полный GigaAM завершён"
        with pytest.raises(ValueError, match="остановить тестовую Ollama"):
            asyncio.run(server_run.first_line_phase(args))
        (logs / "тестовая-ollama-остановлена.txt").touch()
        assert asyncio.run(server_run.first_line_phase(args)) == int(failure == "first")
        assert (output / "записи/0001/whisper.json").read_bytes() == original_whisper
        full = [hash_ for name, hash_ in calls if name == "gigaam"]
        reduced = [hash_ for name, hash_ in calls if name == "gigaam_first_line"]
        assert full[:len(reduced)] == reduced
        with pytest.raises(ValueError, match="повторного запуска"):
            asyncio.run(server_run.first_line_phase(args))
    conditions = json.loads((output / "условия.json").read_text())
    summary = json.loads((output / "итоги.json").read_text())
    assert conditions["planned_systems"] == ["whisper", "gigaam", "gigaam_first_line"]
    assert summary["matched_comparison"]["records"] == (0 if failure else 2)
    assert (output / "замеры.csv").is_file()
    assert "GigaAM первая линия" in (output / "отчёт.html").read_text()
    artifacts.finalize(output, int(failure is not None))
    assert not (output / "временные").exists()
    assert (output / "диагностика.tar.gz").is_file()
