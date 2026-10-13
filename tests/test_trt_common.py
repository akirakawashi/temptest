"""Общая часть прогонов TensorRT: корпус, эталонный прогон и рабочий процесс. Без моделей и Docker."""
import asyncio
import hashlib
import json
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from benchmark import trt_common
from benchmark.compare import Audio


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
    return trt_common.parser().parse_args([str(folder), "--out", str(output)])


def fake_prepare(source, target, max_seconds=None):
    target.write_bytes(source.read_bytes())
    with wave.open(str(target)) as wav:
        raw = wav.readframes(wav.getnframes())
    return Audio(str(source), str(target), 1., 16000, hashlib.sha256(raw).hexdigest(), .05, 1, 16000, source.stat().st_size)


def test_reference_reorders_and_rejects_wrong_corpus():
    files = [Path("a.wav"), Path("b.wav")]
    reference = {"records": [{"directory": "записи/0001", "audio": {"source": "/recordings/b.wav"}},
                             {"directory": "записи/0002", "audio": {"source": "/recordings/a.wav"}}]}
    assert [p.name for p, _, _ in trt_common.reference_files(files, reference)] == ["b.wav", "a.wav"]
    with pytest.raises(ValueError, match="не совпадают"):
        trt_common.reference_files([*files, Path("c.wav")], reference)
    with pytest.raises(ValueError, match="не совпадают"):
        trt_common.reference_files(files[:1], reference)
    reference["records"][0]["directory"] = "../../outside"
    with pytest.raises(ValueError):
        trt_common.reference_files(files, reference)


def test_any_number_of_records_is_accepted_and_empty_folder_is_not():
    files = [Path(f"{index:04d}.wav") for index in range(3878)]
    selected = trt_common.reference_files(files, None)
    assert len(selected) == 3878 and selected[0][1] == "записи/0001" and selected[-1][1] == "записи/3878"
    assert len({directory for _, directory, _ in selected}) == 3878
    with pytest.raises(ValueError, match="нет аудиозаписей"):
        trt_common.reference_files([], None)


def test_removed_limits_are_not_accepted_as_parameters(tmp_path):
    for flag in (["--expected-files", "100"], ["--gap", "5"]):
        with pytest.raises(SystemExit):
            trt_common.parser().parse_args([str(tmp_path), "--out", str(tmp_path), *flag])


def test_summary_row_drops_per_record_details_and_keeps_what_reports_need():
    result = {"status": "ok", "elapsed_seconds": 1.5, "asr_seconds": 1., "text": "Добрый день.", "error": None,
              "stages": {"asr": {"seconds": 1., "calls": 1, "errors": 0}}, "timing": {"exclusive_wall_seconds": {}},
              "trt_stages": {}, "triton_requests": 3, "triton_statistics_delta": {}, "pause_seconds": .5,
              "stage_calls": [{}] * 1000, "speech_segments": [{}], "events": [{}], "utterances": [{}],
              "raw_phrases": [{}], "pauses": [{}], "counters": {}}
    row = trt_common.summary_row(result)
    assert set(row) == {"status", "elapsed_seconds", "asr_seconds", "text", "error", "stages", "timing",
                        "trt_stages", "triton_requests", "triton_statistics_delta", "pause_seconds"}
    assert len(result["stage_calls"]) == 1000  # полный результат не меняется: он уходит в файл записи


def test_changed_pcm_is_rejected_and_reference_stays_untouched(monkeypatch, tmp_path):
    args = args_for(tmp_path)
    old = tmp_path / "old"
    old.mkdir()
    reference = {"records": [{"directory": f"записи/{i + 1:04d}", "audio": {
        "source": f"/recordings/{i}.wav", "samples": 16000, "sha256_pcm": "changed"}} for i in range(2)]}
    (old / "условия.json").write_text(json.dumps(reference))
    args.reference_run = old
    monkeypatch.setattr(trt_common, "prepare_audio", fake_prepare)
    original = (old / "условия.json").read_bytes()
    conditions = {"records": [], "vad_settings": dict(trt_common.VAD_SETTINGS)}
    with pytest.raises(ValueError, match="отличается от исходного теста"):
        trt_common.prepare(args, conditions)
    assert (old / "условия.json").read_bytes() == original
    assert not conditions["reference"]["pcm_verified"] and "warmup" not in conditions


def test_reference_matching_pcm_and_settings_are_verified(monkeypatch, tmp_path):
    args = args_for(tmp_path)
    args.reference_run = tmp_path / "reference"
    args.reference_run.mkdir()
    pcm = hashlib.sha256(np.zeros(16000, dtype="<i2").tobytes()).hexdigest()
    settings = {**trt_common.VAD_SETTINGS, "vad_threshold": .6, "asr_threads": args.threads}
    audio = asdict(Audio("/recordings/1.wav", "/old/1.wav", 1, 16000, pcm, .1))
    reference = {"records": [{"directory": "записи/0001", "audio": audio},
        {"directory": "записи/0002", "audio": {**audio, "source": "/recordings/0.wav"}}],
        "warmup": {"audio": audio}, "preparations": [{"system": "gigaam_first_line", "settings": settings}]}
    trt_common.save_json(args.reference_run / "условия.json", reference)
    monkeypatch.setattr(trt_common, "prepare_audio", fake_prepare)
    conditions = {"records": [], "vad_settings": dict(trt_common.VAD_SETTINGS)}
    rows = trt_common.prepare(args, conditions)
    assert [Path(r["audio"].source).name for r in rows] == ["1.wav", "0.wav"]
    assert conditions["reference"]["pcm_verified"] and conditions["warmup"]["matches_reference"]
    assert conditions["vad_settings"]["vad_threshold"] == .6
    assert Path(conditions["warmup"]["audio"]["source"]).name == "1.wav"


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
        assert args[4] == "benchmark.gigaam_trt_worker" and kwargs["start_new_session"] is True
        trt_common.save_json(tmp_path / "gigaam_trt-ответ.json", {"result": {"status": "ok", "elapsed_seconds": 1},
                                                                 "preparation": {"system": "gigaam_trt"}})
        return Process()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    payload = asyncio.run(trt_common.run_worker({"system": "gigaam_trt", "output": str(tmp_path), "timeout": 2}))
    assert payload["result"]["status"] == "error" and payload["result"]["preparation_failed"]
    assert "CPU" in payload["result"]["error"]
    assert (tmp_path / "gigaam_trt-процесс.log").read_text() == "Native: Fallback to cpu!\n"


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
        assert args[4] == "benchmark.tone_trt_worker"
        return Process()
    async def timeout(awaitable, seconds):
        awaitable.close()
        raise TimeoutError
    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    monkeypatch.setattr(asyncio, "wait_for", timeout)
    monkeypatch.setattr(trt_common.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    payload = asyncio.run(trt_common.run_worker({"system": "tone_trt_greedy", "output": str(tmp_path), "timeout": 1}))
    assert killed == [(987654, 9)]
    assert payload["result"]["status"] == "timeout" and payload["result"]["elapsed_seconds"] is None
    assert (tmp_path / "tone_trt_greedy-процесс.log").read_text() == "Native inference started\n"


def test_unknown_worker_is_rejected_before_any_process_starts(monkeypatch, tmp_path):
    async def forbidden(*args, **kwargs):
        raise AssertionError("Процесс не должен запускаться")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    with pytest.raises(ValueError, match="Неизвестный рабочий процесс"):
        asyncio.run(trt_common.run_worker({"system": "gigaam", "output": str(tmp_path), "timeout": 1}))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("parallel", [1, 3, 5])
def test_rows_are_processed_in_order_with_bounded_concurrency(parallel):
    active, peak, started = 0, 0, []

    async def handle(index, row):
        nonlocal active, peak
        started.append(index)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.01)
        active -= 1
        row["done"] = True

    rows = [{} for _ in range(12)]
    asyncio.run(trt_common.process_rows(rows, parallel, handle))
    assert peak == parallel and started == list(range(1, 13)) and all(row["done"] for row in rows)


def test_failure_stops_new_records_but_lets_running_ones_finish():
    started, finished = [], []

    async def handle(index, row):
        started.append(index)
        await asyncio.sleep(.01 if index != 2 else 0)
        if index == 2:
            raise RuntimeError("Нарушен резерв GPU")
        finished.append(index)

    with pytest.raises(RuntimeError, match="резерв"):
        asyncio.run(trt_common.process_rows([{} for _ in range(20)], 3, handle))
    assert started == [1, 2, 3] and finished == [1, 3]


def test_loop_statistics_subtract_server_counters_and_survive_their_absence():
    before = {"success": {"calls": 10, "seconds": 1.}, "queue": {"calls": 10, "seconds": .5}}
    after = {"success": {"calls": 60, "seconds": 3.5}, "queue": {"calls": 60, "seconds": 2.}}
    loop = trt_common.loop_statistics(5, 12., before, after)
    assert loop == {"parallel": 5, "loop_seconds": 12., "triton": {"success": {"calls": 50, "seconds": 2.5}, "queue": {"calls": 50, "seconds": 1.5}}}
    assert trt_common.loop_statistics(1, 3., None, after)["triton"] is None
    # Вне контейнера клиента сервера моделей нет: счётчики просто отсутствуют.
    assert trt_common.triton_statistics("gigaam_encoder") is None
