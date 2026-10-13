"""Общее для прогонов TensorRT: подготовка корпуса и отдельный процесс на каждую запись."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from benchmark.compare import FORMATS, prepare_audio

log = logging.getLogger("прогон-tensorrt")
VAD_SETTINGS = {"vad_threshold": .5, "vad_min_silence": .45, "vad_min_speech": .25, "vad_max_speech": 15.}
WORKERS = {"tone_trt_greedy": "benchmark.tone_trt_worker", "tone_trt_kenlm": "benchmark.tone_trt_worker",
           "gigaam_trt": "benchmark.gigaam_trt_worker", "gigaam_trt_full": "benchmark.gigaam_trt_worker"}
# Подробности записи лежат в её файле; сводке и отчёту они не нужны, а на тысячах записей занимают гигабайты памяти.
HEAVY = ("stage_calls", "speech_segments", "events", "utterances", "raw_phrases", "pauses", "counters")
# Отчёт переписывается целиком, поэтому не после каждой записи: на тысячах записей это съедало бы время потоков.
REPORT_INTERVAL = 5.


def save_json(path: Path, value):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def summary_row(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in HEAVY}


async def process_rows(rows: list[dict], parallel: int, handle) -> None:
    """Записи идут по порядку, не больше parallel одновременно; после сбоя новые не начинаются."""
    queue, failures = iter(enumerate(rows, 1)), []

    async def stream():
        for index, row in queue:
            if failures:
                return
            try:
                await handle(index, row)
            except Exception as exc:
                failures.append(exc)
                return

    tasks = [asyncio.create_task(stream()) for _ in range(parallel)]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    if failures:
        raise failures[0]


def triton_statistics(model: str) -> dict | None:
    """Счётчики модели за всё время работы сервера; None, если сервер или его клиент недоступны."""
    try:
        import tritonclient.grpc as grpc

        client = grpc.InferenceServerClient("triton:8001")
        try:
            stats = client.get_inference_statistics(model, "1", as_json=True, client_timeout=30)
        finally:
            client.close()
        row = next(item["inference_stats"] for item in stats["model_stats"] if item["name"] == model)
        return {key: {"calls": int(row.get(key, {}).get("count", 0)), "seconds": int(row.get(key, {}).get("ns", 0)) / 1e9}
                for key in ("success", "fail", "queue", "compute_input", "compute_infer", "compute_output")}
    except Exception:
        return None


def loop_statistics(parallel: int, seconds: float, before: dict | None, after: dict | None) -> dict:
    """Итог прохода по корпусу: время по часам и работа сервера моделей за этот проход."""
    result = {"parallel": parallel, "loop_seconds": seconds, "triton": None}
    if before and after:
        result["triton"] = {key: {"calls": after[key]["calls"] - before[key]["calls"],
                                  "seconds": after[key]["seconds"] - before[key]["seconds"]} for key in after}
    return result


def reference_files(files: list[Path], reference: dict | None):
    if not files:
        raise ValueError("В папке нет аудиозаписей; распознавание не начиналось")
    if reference is None:
        return [(source, f"записи/{i:04d}", None) for i, source in enumerate(files, 1)]
    records = reference.get("records", [])
    names = {p.name: p for p in files}
    old_names = [Path(r["audio"]["source"]).name for r in records]
    if len(names) != len(files) or len(set(old_names)) != len(records) or set(old_names) != set(names):
        raise ValueError("Имена записей нового и исходного прогонов не совпадают либо повторяются")
    result, directories = [], set()
    for record, name in zip(records, old_names):
        directory = record["directory"]
        if not re.fullmatch(r"записи/[0-9]{4,}", directory) or directory in directories:
            raise ValueError("Некорректные или повторяющиеся ID в исходном прогоне")
        directories.add(directory)
        result.append((names[name], directory, record["audio"]))
    return result


def prepare(args, conditions: dict) -> list[dict]:
    files = sorted(p for p in args.audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in FORMATS)
    reference = None
    if args.reference_run:
        source = args.reference_run / "условия.json"
        original = source.read_bytes()
        reference = json.loads(original)
        conditions["reference"] = {"path": str(args.reference_run),
            "manifest_sha256": hashlib.sha256(original).hexdigest(), "pcm_verified": False,
            "note": "Тесты выполнены в разное время; прежние замеры не пересчитывались"}
        save_json(args.out / "исходные-условия.json", reference)
        # Эталоном может быть сохранённый прогон прежнего GigaAM через PyTorch: берём его настройки поиска речи.
        settings = next((p["settings"] for p in reference.get("preparations", [])
                         if p.get("system") == "gigaam_first_line" and p.get("settings")), None)
        if settings:
            conditions["vad_settings"] = {key: settings[key] for key in VAD_SETTINGS}
            if settings.get("asr_threads", args.threads) != args.threads:
                raise ValueError("Число потоков отличается от исходного GigaAM; установите BENCH_THREADS как в исходном прогоне")
    selected = reference_files(files, reference)
    (args.out / "временные").mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (source, directory, old) in enumerate(selected, 1):
        target = args.out / "временные" / f"{Path(directory).name}.wav"
        log.info("Подготовка аудио %d/%d: %s — вне замеров", index, len(selected), source)
        audio = prepare_audio(source, target)
        if audio.seconds > args.max_audio_seconds:
            raise ValueError(f"Запись длиннее {args.max_audio_seconds} с: {source}")
        if old and (audio.sha256_pcm != old["sha256_pcm"] or audio.samples != old["samples"]):
            raise ValueError(f"PCM записи {source.name} отличается от исходного теста; замеры не начинались")
        (args.out / directory).mkdir(parents=True)
        row = {"directory": directory, "audio": audio}
        rows.append(row)
        conditions["records"].append({"directory": directory, "audio": asdict(audio)})
        save_json(args.out / directory / "аудио.json", asdict(audio))
    warm_source, warm_seconds = args.warmup_file or selected[0][0], args.warmup_seconds
    old_warm = None
    if reference and not args.warmup_file:
        old_warm = reference.get("warmup", reference.get("whisper", {}).get("warmup", {})).get("audio")
        if old_warm:
            matches = [p for p in files if p.name == Path(old_warm["source"]).name]
            if len(matches) != 1:
                raise ValueError("Не найдена исходная запись прогрева; укажите --warmup-file")
            warm_source, warm_seconds = matches[0], old_warm["seconds"]
    if not warm_source.resolve().is_relative_to(args.audio_dir.resolve()):
        raise ValueError("Запись прогрева должна находиться в папке аудио")
    warm = prepare_audio(warm_source, args.out / "временные" / "прогрев.wav", max_seconds=warm_seconds)
    if old_warm and warm.sha256_pcm != old_warm["sha256_pcm"]:
        raise ValueError("PCM исходного прогрева отличается; замеры не начинались")
    conditions["warmup"] = {"audio": asdict(warm), "included_in_measurements": False,
                            "matches_reference": bool(old_warm)}
    if reference:
        conditions["reference"]["pcm_verified"] = True
    return rows


async def run_worker(request: dict) -> dict:
    directory = Path(request["output"])
    system = request["system"]
    if system not in WORKERS:
        raise ValueError("Неизвестный рабочий процесс")
    task = directory / f"{system}-задание.json"
    response = directory / f"{system}-ответ.json"
    process_log = directory / f"{system}-процесс.log"
    save_json(task, request)
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(sys.executable, "-X", "faulthandler",
        "-m", WORKERS[system], str(task), start_new_session=True,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

    async def capture():
        with process_log.open("wb") as stream:
            while chunk := await process.stdout.read(65536):
                stream.write(chunk)
                stream.flush()
                sys.stdout.write(chunk.decode("utf-8", "replace"))
                sys.stdout.flush()

    capture_task = asyncio.create_task(capture())
    timed_out = False
    try:
        await asyncio.wait_for(process.wait(), request["timeout"] * 3 + 180)
    except (TimeoutError, asyncio.CancelledError) as exc:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()
        if isinstance(exc, asyncio.CancelledError):
            raise
        timed_out = True
    finally:
        await capture_task
    prep_path = directory / f"{system}-подготовка.json"
    preparation = json.loads(prep_path.read_text()) if prep_path.is_file() else {"system": system}
    if response.is_file() and not timed_out:
        payload = json.loads(response.read_text())
    else:
        payload = {"result": {"status": "timeout" if timed_out else "error", "text": "",
            "elapsed_seconds": None, "preparation_failed": True,
            "error": f"Рабочий процесс {system} завершён: код {process.returncode}; таймаут {timed_out}; журнал {process_log}"},
            "preparation": preparation}
    payload["preparation"].update(worker_wall_seconds=time.perf_counter() - started,
        process_exit_code=process.returncode, process_log=str(process_log))
    native_log = process_log.read_text(encoding="utf-8", errors="replace").lower()
    if "fallback to cpu" in native_log or "fall back to cpu" in native_log:
        payload["result"].update(status="error", error="Модель перешла на CPU; результат исключён из GPU-замеров", preparation_failed=True)
    if process.returncode and payload["result"]["status"] in {"ok", "no_speech"}:
        payload["result"].update(status="error", error=f"Ненулевой код рабочего процесса: {process.returncode}", preparation_failed=True)
    save_json(prep_path, payload["preparation"])
    return payload


def parser():
    cli = argparse.ArgumentParser()
    cli.add_argument("audio_dir", type=Path)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--reference-run", type=Path)
    cli.add_argument("--model-dir", type=Path, default=Path("/cache"))
    cli.add_argument("--threads", type=int, default=int(os.environ.get("BENCH_THREADS", "4")))
    cli.add_argument("--parallel", type=int, default=1, help="Сколько записей обрабатывать одновременно")
    cli.add_argument("--timeout", type=int, default=3600)
    cli.add_argument("--warmup-file", type=Path)
    cli.add_argument("--warmup-seconds", type=float, default=30.)
    cli.add_argument("--max-audio-seconds", type=float, default=1800.)
    return cli
