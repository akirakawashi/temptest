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
           "gigaam_trt": "benchmark.gigaam_trt_worker"}


def save_json(path: Path, value):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def reference_files(files: list[Path], reference: dict | None, expected: int):
    if len(files) != expected:
        raise ValueError(f"Найдено {len(files)} аудиозаписей, ожидалось {expected}; распознавание не начиналось")
    if reference is None:
        return [(source, f"записи/{i:04d}", None) for i, source in enumerate(files, 1)]
    records = reference.get("records", [])
    if len(records) != expected:
        raise ValueError("В исходном прогоне другое число записей")
    names = {p.name: p for p in files}
    old_names = [Path(r["audio"]["source"]).name for r in records]
    if len(names) != expected or len(set(old_names)) != expected or set(old_names) != set(names):
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
    selected = reference_files(files, reference, args.expected_files)
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
    cli.add_argument("--expected-files", type=int, default=int(os.environ.get("BENCH_EXPECTED_FILES", "100")))
    cli.add_argument("--threads", type=int, default=int(os.environ.get("BENCH_THREADS", "4")))
    cli.add_argument("--timeout", type=int, default=3600)
    cli.add_argument("--gap", type=float, default=5.)
    cli.add_argument("--warmup-file", type=Path)
    cli.add_argument("--warmup-seconds", type=float, default=30.)
    cli.add_argument("--max-audio-seconds", type=float, default=1800.)
    return cli
