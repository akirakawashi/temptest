"""Проверки стенда с заглушками: модели, Docker и записи пользователя не запускаются."""
import asyncio
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from app.config import Settings
from app.llm import LLM
from app.session import Session
from app.vad import StreamingVad, make_onnx_session
from benchmark import compare, pipeline, worker


@dataclass
class FakeSegment:
    text: str
    start: float
    end: float


def test_whisper_times_lazy_segment_generation(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: clock[0])
    arguments = {}

    class Model:
        def transcribe(self, samples, **kwargs):
            arguments.update(kwargs)
            clock[0] += 2.0

            def generate():
                clock[0] += 3.0
                yield FakeSegment(" Добрый день.", 0.0, 1.0)
                clock[0] += 5.0
                yield FakeSegment(" Как дела?", 1.0, 2.0)

            return generate(), SimpleNamespace()

    whisper = pipeline.WhisperPipeline("large-v3", "/unused", 4)
    whisper.model = Model()
    result = whisper.run(np.zeros(1600, dtype=np.float32))
    assert result["elapsed_seconds"] == 10.0
    assert result["text"] == "Добрый день. Как дела?"
    assert len(result["segments"]) == 2 and result["status"] == "ok"
    assert arguments["language"] == "ru" and arguments["vad_filter"] is True


def test_asr_timer_counts_empty_answers_and_failed_calls(monkeypatch):
    ticks = iter([0, 0.123456789, 1, 1.25])
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: next(ticks))
    timer = pipeline.StageTimer()
    assert timer.measure("asr", lambda: ("", [], [])) == ("", [], [])

    def fail():
        raise RuntimeError("Ошибка заглушки")

    with pytest.raises(RuntimeError):
        timer.measure("asr", fail)
    assert timer.snapshot()["asr"] == {"seconds": pytest.approx(0.373456789), "calls": 2, "errors": 1}
    timer.reset()
    assert timer.snapshot()["asr"]["seconds"] == 0


@pytest.mark.parametrize("available, selected, succeeds", [
    (["CPUExecutionProvider"], ["CPUExecutionProvider"], False),
    (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"], False),
    (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"], True),
])
def test_onnx_gpu_request_cannot_silently_fall_back_to_cpu(monkeypatch, available, selected, succeeds):
    session = SimpleNamespace(get_providers=lambda: selected)
    runtime = SimpleNamespace(SessionOptions=SimpleNamespace, get_available_providers=lambda: available,
                              InferenceSession=lambda *args, **kwargs: session)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    if succeeds:
        assert make_onnx_session("unused.onnx", provider="CUDAExecutionProvider") is session
    else:
        with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
            make_onnx_session("unused.onnx", provider="CUDAExecutionProvider")


class ScriptedVad:
    def run(self, _, inputs):
        p = float(np.max(np.abs(inputs["input"][0, 64:])) > 0.01)
        return np.array([[p]], dtype=np.float32), inputs["state"]


class FakeEngines:
    def __init__(self, *, gate=None, entered=None, emotion_error=False):
        self.components = {"vad": SimpleNamespace(state="ready")}
        self.asr_pool = ThreadPoolExecutor(max_workers=1)
        self.emo_pool = ThreadPoolExecutor(max_workers=1)
        self.live_pool = ThreadPoolExecutor(max_workers=1)
        self.timer = pipeline.StageTimer()
        self.emotions_ready = True
        self.gate, self.entered, self.emotion_error = gate, entered, emotion_error

    def new_vad(self):
        return StreamingVad(ScriptedVad())

    def transcribe(self, samples):
        return self.timer.measure("asr", lambda: ("Добрый день.", [" Добрый", " день", "."], [0.1, 0.3, 0.5]))

    def embed(self, samples):
        return np.ones(192, dtype=np.float32) / np.sqrt(192)

    def emotions(self, samples):
        if self.entered:
            self.entered.set()
        if self.gate:
            assert self.gate.wait(3), "Тест не освободил заглушку эмоций"
        if self.emotion_error:
            raise RuntimeError("Тестовая ошибка эмоций")
        return {"neutral": 1.0}

    def shutdown(self):
        for pool in (self.asr_pool, self.emo_pool, self.live_pool):
            pool.shutdown(wait=True)


def speech():
    return np.concatenate([np.zeros(8000), np.full(12800, 0.1), np.zeros(9600)]).astype(np.float32)


def test_session_waits_for_emotion_model_and_event_delivery():
    async def scenario():
        model_entered, model_release = threading.Event(), threading.Event()
        event_entered, event_release = asyncio.Event(), asyncio.Event()
        engines = FakeEngines(gate=model_release, entered=model_entered)
        events = []

        async def emit(event):
            if event["type"] == "emotion":
                event_entered.set()
                await event_release.wait()
            events.append(event)

        session = Session(engines, Settings(), emit)
        session.start()
        try:
            await session.feed((speech() * 32768).astype("<i2").tobytes())
            await session.flush()
            assert await asyncio.to_thread(model_entered.wait, 3)
            idle = asyncio.create_task(session.wait_idle())
            await asyncio.sleep(0)
            assert not idle.done()
            model_release.set()
            await asyncio.wait_for(event_entered.wait(), 3)
            assert not idle.done()
            event_release.set()
            await asyncio.wait_for(idle, 3)
            assert [e["type"] for e in events if e["type"] in {"utterance", "emotion"}] == ["utterance", "emotion"]
            assert session.utterances[0]["emotion"]["label"] == "neutral"
            assert session.queue._unfinished_tasks == 0
        finally:
            model_release.set()
            event_release.set()
            await session.close()
            engines.shutdown()

    asyncio.run(scenario())


class FakeLLM:
    def __init__(self, event=None):
        self.called = 0
        self.event = event or {"type": "llm", "state": "done", "completed": True,
                               "result": {"summary": "Тестовый разговор"}}

    async def analyze(self, transcript, emit, **kwargs):
        self.called += 1
        assert "нейтрально" in transcript  # эмоция готова до запроса LLM
        assert kwargs["options"] == {"num_thread": 2, "num_gpu": 999, "seed": 42}
        await emit(self.event)


def fake_giga(*, emotion_error=False, llm_event=None):
    giga = pipeline.GigaPipeline(Settings(), 2, 5)
    giga.shutdown()  # пулы настоящих Engines не запускаются и больше не нужны
    giga.engines = FakeEngines(emotion_error=emotion_error)
    giga.llm = FakeLLM(llm_event)
    return giga


def test_full_giga_waits_for_emotions_and_resets_speakers_per_file():
    async def scenario():
        giga = fake_giga()
        try:
            first = await giga.run(speech())
            second = await giga.run(speech())
            assert first["status"] == second["status"] == "ok"
            for result in (first, second):
                assert result["utterances"][0]["id"] == 1
                assert result["utterances"][0]["speaker"] == 1
                assert result["utterances"][0]["emotion"]["label"] == "neutral"
                assert result["elapsed_seconds"] >= result["asr_seconds"]
                assert result["stages"]["asr"]["calls"] == 1
                assert result["analysis"]["result"]
            assert giga.llm.called == 2
        finally:
            giga.shutdown()

    asyncio.run(scenario())


def test_emotion_failure_cannot_be_reported_as_full_success():
    async def scenario():
        giga = fake_giga(emotion_error=True)
        try:
            result = await giga.run(speech())
            assert result["status"] == "error"
            assert "Тестовая ошибка эмоций" in result["error"]
            assert any(e.get("stage") == "emotion" for e in result["events"])
            assert giga.llm.called == 0
        finally:
            giga.shutdown()

    asyncio.run(scenario())


def test_full_giga_result_is_not_ready_before_llm_finishes():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        giga = fake_giga()

        class DelayedLLM(FakeLLM):
            async def analyze(self, transcript, emit, **kwargs):
                entered.set()
                await release.wait()
                await super().analyze(transcript, emit, **kwargs)

        giga.llm = DelayedLLM()
        task = asyncio.create_task(giga.run(speech()))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert not task.done()
            release.set()
            result = await asyncio.wait_for(task, 3)
            assert result["status"] == "ok" and result["analysis"]["result"]
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            giga.shutdown()

    asyncio.run(scenario())


def test_silence_is_not_a_successful_full_llm_cycle():
    async def scenario():
        giga = fake_giga()
        try:
            result = await giga.run(np.zeros(16000, dtype=np.float32))
            assert result["status"] == "no_speech"
            assert result["asr_seconds"] == 0
            assert result["analysis"] is None and giga.llm.called == 0
        finally:
            giga.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("event", [
    {"type": "llm", "state": "done", "completed": False, "result": {"summary": "Оборвано"}},
    {"type": "llm", "state": "done", "completed": True, "result": None, "raw": "не JSON"},
    {"type": "llm", "state": "error", "message": "Ошибка заглушки LLM"},
])
def test_bad_llm_response_is_a_failed_full_cycle(event):
    async def scenario():
        giga = fake_giga(llm_event=event)
        try:
            result = await giga.run(speech())
            assert result["status"] == "error"
            assert result["analysis"]
        finally:
            giga.shutdown()

    asyncio.run(scenario())


def test_llm_request_has_cpu_thread_limit_and_keeps_original_options(monkeypatch):
    bodies = []

    class Response:
        status_code = 200

        async def aiter_lines(self):
            yield json.dumps({"message": {"content": '{"summary": "Готово"}'}, "done": True})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, method, url, *, json):
            bodies.append(json)
            return Response()

    monkeypatch.setattr("app.llm.httpx.AsyncClient", Client)

    async def scenario():
        engines = SimpleNamespace(components={"llm": SimpleNamespace(state="ready")})
        model = LLM(Settings(llm_keep_alive="-1"), engines)
        events = []

        async def emit(event):
            events.append(event)

        await model.analyze("Тест", emit, options={"num_thread": 2, "num_gpu": 999, "seed": 42})
        assert bodies[0]["options"]["num_thread"] == 2
        assert bodies[0]["options"]["num_gpu"] == 999
        assert bodies[0]["options"]["temperature"] == 0.2
        assert bodies[0]["keep_alive"] == -1
        assert isinstance(bodies[0]["keep_alive"], int)
        assert events[-1]["completed"] is True

    asyncio.run(scenario())


def write_wav(path, level):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(np.full(3200, level, dtype="<i2").tobytes())


def install_fake_runners(monkeypatch, *, fail_whisper=False, warm_giga_status="ok"):
    calls = []
    active_models = []

    def prepare(source, target, **kwargs):
        shutil.copyfile(source, target)
        with wave.open(str(target), "rb") as wav:
            count = wav.getnframes()
            pcm = wav.readframes(count)
        return compare.Audio(str(source), str(target), count / 16000, count, hashlib.sha256(pcm).hexdigest(), 0.5)

    class Whisper:
        def __init__(self, *args):
            self.count = 0
            self.vad_providers = ["CUDAExecutionProvider"]

        def load(self):
            assert not active_models, "Предыдущая система осталась загружена"
            active_models.append("whisper")

        def run(self, samples):
            self.count += 1
            if self.count > 1:
                calls.append(("whisper", samples.copy()))
            failed = fail_whisper and self.count > 1
            return {"status": "error" if failed else "ok", "elapsed_seconds": 1.0,
                    "text": "Текст Whisper", "segments": [], "error": "Тестовая ошибка" if failed else None}

        def shutdown(self):
            active_models.remove("whisper")

    class Giga:
        def __init__(self, *args):
            self.engines = SimpleNamespace(versions={}, placement=lambda: {"asr": "Тестовая CUDA"})
            self.count = 0

        async def load(self):
            assert not active_models, "Предыдущая система осталась загружена"
            active_models.append("gigaam")

        async def gpu_model_info(self):
            return [{"name": "Тестовая LLM", "size": 1024, "size_vram": 1024}]

        async def unload_llm(self):
            pass

        async def run(self, samples):
            self.count += 1
            if self.count > 1:
                calls.append(("gigaam", samples.copy()))
            return {"status": warm_giga_status if self.count == 1 else "ok", "elapsed_seconds": 3.0, "asr_seconds": 0.7,
                    "stages": {"asr": {"seconds": 0.7, "calls": 1, "errors": 0}},
                    "llm_seconds": 1.5, "text": "Текст GigaAM", "utterances": [], "events": [],
                    "analysis": {"result": {"summary": "Итог"}, "truncated": False, "load_sec": 0},
                    "error": 'Ollama ответила 400: missing unit in duration' if warm_giga_status == "error" else None}

        def shutdown(self):
            active_models.remove("gigaam")

    monkeypatch.setattr(compare, "prepare_audio", prepare)
    monkeypatch.setattr(pipeline, "WhisperPipeline", Whisper)
    monkeypatch.setattr(pipeline, "GigaPipeline", Giga)
    monkeypatch.setattr("benchmark.gpu.gpu_info", lambda: {"name": "Тестовая GPU", "uuid": "test"})
    monkeypatch.setattr("benchmark.gpu.log_gpu_memory", lambda *args: None)
    async def idle(*args):
        pass
    monkeypatch.setattr("benchmark.gpu.require_idle_ollama", idle)
    monkeypatch.setattr(compare, "run_worker", worker.execute)
    return calls


def test_runner_uses_identical_input_and_alternates_order(tmp_path, monkeypatch):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    write_wav(source / "1.wav", 100)
    write_wav(source / "2.wav", 200)
    calls = install_fake_runners(monkeypatch)
    args = compare.parser().parse_args(["--audio-dir", str(source), "--out", str(output), "--threads", "1"])
    assert asyncio.run(compare.run(args)) == 0
    measured = calls  # заглушки записывают только измеряемые вызовы
    assert [name for name, _ in measured] == ["whisper", "gigaam", "gigaam", "whisper"]
    np.testing.assert_array_equal(measured[0][1], measured[1][1])
    np.testing.assert_array_equal(measured[2][1], measured[3][1])
    with (output / "сводка.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file, delimiter=";"))
    assert len(rows) == 2
    assert rows[0]["GigaAM распознавание, с"] == "0,700000"
    assert rows[0]["GigaAM полностью, с"] == "3,000000"
    assert rows[0]["Whisper — статус"] == "Успешно"
    assert (output / "0001" / "whisper.txt").read_text().strip() == "Текст Whisper"
    assert json.loads((output / "условия.json").read_text())["state"] == "Завершён"
    preparations = json.loads((output / "условия.json").read_text())["preparations"]
    assert len(preparations) == 4
    assert all(p["included_in_measurements"] is False for p in preparations)


@pytest.mark.parametrize("warm_status, expected_advice", [("error", False), ("no_speech", True)])
def test_warmup_file_advice_is_only_shown_when_speech_is_missing(tmp_path, monkeypatch, warm_status, expected_advice):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    write_wav(source / "1.wav", 100)
    calls = install_fake_runners(monkeypatch, warm_giga_status=warm_status)
    args = compare.parser().parse_args(["--audio-dir", str(source), "--out", str(output), "--threads", "1"])
    assert asyncio.run(compare.run(args)) == 1
    assert [name for name, _ in calls] == ["whisper"]
    result = json.loads((output / "0001" / "gigaam.json").read_text())
    assert ("--warmup-file" in result["error"]) is expected_advice
    assert result["elapsed_seconds"] is None and result["preparation_failed"] is True


def test_runner_stops_on_error_without_starting_other_system(tmp_path, monkeypatch):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    write_wav(source / "1.wav", 100)
    write_wav(source / "2.wav", 200)
    calls = install_fake_runners(monkeypatch, fail_whisper=True)
    args = compare.parser().parse_args(["--audio-dir", str(source), "--out", str(output), "--threads", "1"])
    assert asyncio.run(compare.run(args)) == 1
    assert [name for name, _ in calls] == ["whisper"]
    assert "Полностью успешных пар: 0" in (output / "отчёт.md").read_text()
    result = json.loads((output / "0001" / "whisper.json").read_text())
    assert result["status"] == "error"
    assert not (output / "0001" / "gigaam.json").exists()


def test_existing_output_is_rejected_before_any_model_load(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    with pytest.raises(SystemExit) as exc:
        compare.main(["--audio-dir", str(source), "--out", str(output)])
    assert exc.value.code == 2


def test_worker_failure_preserves_stderr_trace_and_early_metadata(tmp_path, monkeypatch):
    original_spawn = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        assert args[1:3] == ("-X", "faulthandler")
        # Только обычный Python: без моделей, GPU и вызовов пользователя.
        return await original_spawn(sys.executable, "-X", "faulthandler", "-c",
            "import faulthandler,sys; print('Нативная ошибка', file=sys.stderr); "
            "faulthandler.dump_traceback(); sys.exit(7)", **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    compare.write_json(tmp_path / "gigaam-подготовка.json", {"phase": "Перед загрузкой моделей", "versions": {"test": "1"}})
    payload = asyncio.run(compare.run_worker({"system": "gigaam", "output": str(tmp_path), "timeout": 1}))
    assert payload["result"]["status"] == "error"
    assert "кодом 7" in payload["result"]["error"]
    assert payload["preparation"]["phase"] == "Перед загрузкой моделей"
    output = Path(payload["result"]["process_log"]).read_text(encoding="utf-8")
    assert "Нативная ошибка" in output and 'File "<string>"' in output


@pytest.mark.parametrize("size_vram", [0, 512, None, 2048, 1024])
def test_llm_requires_full_gpu_placement(monkeypatch, size_vram):
    cfg = Settings()

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, path):
            assert path == "/api/ps"
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                "models": [{"name": cfg.llm_model, "size": 1024, "size_vram": size_vram}]
            })

    monkeypatch.setattr("httpx.AsyncClient", Client)

    async def scenario():
        giga = pipeline.GigaPipeline(cfg, 2, 5)
        try:
            if size_vram == 1024:
                assert (await giga.gpu_model_info())[0]["size_vram"] == 1024
            else:
                with pytest.raises(RuntimeError, match="целиком на GPU"):
                    await giga.gpu_model_info()
        finally:
            giga.shutdown()

    asyncio.run(scenario())
