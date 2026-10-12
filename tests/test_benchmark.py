"""Проверки стенда с заглушками: модели, Docker и записи пользователя не запускаются."""
import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from app.config import Settings
from app.session import Session
from app.vad import StreamingVad, make_onnx_session
from benchmark import pipeline


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
    def __init__(self, *, gate=None, entered=None):
        self.components = {"vad": SimpleNamespace(state="ready")}
        self.asr_pool = ThreadPoolExecutor(max_workers=1)
        self.emo_pool = ThreadPoolExecutor(max_workers=1)
        self.live_pool = ThreadPoolExecutor(max_workers=1)
        self.timer = pipeline.StageTimer()
        self.emotions_ready = True
        self.gate, self.entered = gate, entered

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
        return {"neutral": 1.0}

    def shutdown(self):
        for pool in (self.asr_pool, self.emo_pool, self.live_pool):
            pool.shutdown(wait=True)


def speech():
    return np.concatenate([np.zeros(8000), np.full(12800, 0.1), np.zeros(9600)]).astype(np.float32)


def write_wav(path, level):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(np.full(3200, level, dtype="<i2").tobytes())


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


class ReducedEngines(FakeEngines):
    """Движок урезанного flow: голоса и эмоции вызывать нельзя."""

    def __init__(self):
        super().__init__()
        self.emotions_ready = False

    def new_vad(self):
        return pipeline.TimedVad(StreamingVad(ScriptedVad()), self.timer)

    def embed(self, samples):
        raise AssertionError("Голоса отключены")

    def emotions(self, samples):
        raise AssertionError("Эмоции отключены")


def test_reduced_flow_runs_only_vad_and_asr_and_reports_pauses():
    async def scenario():
        cfg = Settings(emo_enabled=False, llm_enabled=False, split_turns=False)
        runner = pipeline.FirstLinePipeline(cfg, ReducedEngines(), 3)
        try:
            for _ in range(2):
                result = await runner.run(speech())
                assert result["status"] == "ok" and result["error"] is None
                assert result["text"] == "Добрый день."
                assert result["utterances"][0]["speaker"] is None
                assert result["utterances"][0]["id"] == 1
                assert result["utterances"][0]["word_timestamps"]
                assert result["enabled_stages"] == ["vad", "asr"] and set(result["stages"]) == {"vad", "asr"}
                assert result["speech_segments"] and len(result["pauses"]) == 2
                assert {call["stage"] for call in result["stage_calls"]} == {"vad", "asr"}
                assert set(result["timing"]["exclusive_wall_seconds"]) == {"vad", "asr", "parallel", "other"}
                assert sum(result["timing"]["exclusive_wall_percent"].values()) == pytest.approx(100)
                assert not {"llm_seconds", "analysis"} & set(result)
            silent = await runner.run(np.zeros(16000, dtype=np.float32))
            assert silent["status"] == "no_speech" and silent["asr_seconds"] == 0
            assert silent["pauses"] == [{"start": 0., "end": 1., "seconds": 1.}]
        finally:
            runner.shutdown()
    asyncio.run(scenario())


def test_reduced_flow_reports_recognition_failure_as_error():
    class Broken(ReducedEngines):
        def transcribe(self, samples):
            return self.timer.measure("asr", lambda: (_ for _ in ()).throw(RuntimeError("Модель недоступна")))

    async def scenario():
        runner = pipeline.FirstLinePipeline(Settings(emo_enabled=False, llm_enabled=False, split_turns=False), Broken(), 3)
        try:
            result = await runner.run(speech())
            assert result["status"] == "error" and "Модель недоступна" in result["error"]
            assert result["stages"]["asr"]["errors"] == 1 and result["utterances"] == []
        finally:
            runner.shutdown()
    asyncio.run(scenario())


def test_wall_percentages_count_concurrency_once():
    result = pipeline.wall_time_breakdown([
        {"stage": "vad", "offset_seconds": 0, "seconds": 2},
        {"stage": "asr", "offset_seconds": 1, "seconds": 2},
    ], 4)
    seconds = result["exclusive_wall_seconds"]
    assert seconds == {"vad": 1, "asr": 1, "parallel": 1, "other": 1}
    assert sum(result["exclusive_wall_percent"].values()) == pytest.approx(100)
