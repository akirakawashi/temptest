"""Весь корпус через собственный тестовый Whisper: один запрос за раз, без повторов."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import wave

from benchmark import compare
from benchmark.whisper_api import TEST_URL, WhisperAPI

log = logging.getLogger("серверный-прогон")


def load_rows(output: Path, conditions: dict) -> list[dict]:
    rows = []
    for item in conditions["records"]:
        row = {**item, "audio": compare.Audio(**item["audio"])}
        path = output / row["directory"] / "whisper.json"
        if path.is_file():
            row["whisper"] = json.loads(path.read_text(encoding="utf-8"))
        rows.append(row)
    return rows


def update(output: Path, rows: list, conditions: dict):
    from benchmark.artifacts import render

    compare.write_json(output / "условия.json", conditions)
    compare.save_reports(output, rows, conditions["state"])
    render(output, rows, conditions)


async def whisper_phase(args) -> int:
    wall_started = time.perf_counter()
    files = sorted(p.resolve() for p in args.audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in compare.FORMATS)
    conditions = {"schema_version": 3, "mode": "whisper_only", "planned_systems": ["whisper"], "state": "Подготовка",
                  "started_at": datetime.now().astimezone().isoformat(), "host": compare.host_info(),
                  "selected_gpu": os.environ.get("BENCH_GPU", "0"), "input": "mono_16000_pcm_s16le",
                  "records": [], "preparations": [],
                  "whisper": {"url": TEST_URL,
                              "model": os.environ.get("BENCH_WHISPER_MODEL", "large-v3"),
                              "image": os.environ.get("BENCH_WHISPER_IMAGE"),
                              "model_load_per_corpus": True, "warmup_per_corpus": True,
                              "concurrency": 1, "retries": 0, "gap_seconds": args.api_gap,
                              "timeout_seconds": args.api_timeout,
                              "versions_source": "Фактические версии тестового контейнера из /benchmark/metadata"},
                  "order": "Все записи последовательно через собственный тестовый Whisper",
                  "timers": {"whisper": "Полный HTTP-запрос после прогрева, загрузка и прогрев исключены"},
                  "guard_policy": {key: os.environ.get(key) for key in
                                   ("BENCH_GPU_MIN_FREE_MIB", "BENCH_GPU_MAX_UTIL", "BENCH_GPU_RESERVE_MIB", "BENCH_MIN_RAM_MIB")},
                  "production_service_management": False, "production_requests": False,
                  "containers_retained": True, "whisper_external_network": False}
    rows, client = [], None
    try:
        if len(files) != args.expected_files:
            raise ValueError(f"Найдено {len(files)} записей, ожидалось {args.expected_files}; ни одного запроса не отправлено")
        client = WhisperAPI(conditions["whisper"]["url"], "", conditions["whisper"]["model"],
                            args.api_timeout, standalone=True)
        conditions["whisper"]["url"] = client.base_url
        log.info("Корпус: %d файлов. Whisper: один запрос за раз, пауза %.1f с, повторов нет", len(files), args.api_gap)
        for index, source in enumerate(files):
            directory = f"записи/{index + 1:04d}"
            (args.out / directory).mkdir(parents=True)
            target = args.out / "временные" / f"{index + 1:04d}.wav"
            target.parent.mkdir(exist_ok=True)
            audio = compare.prepare_audio(source, target)
            if audio.seconds > args.max_audio_seconds or target.stat().st_size > 64 * 1024 ** 2:
                raise ValueError(f"{source.name}: превышен предел {args.max_audio_seconds} с / 64 МиБ; запросов ещё не было")
            row = {"directory": directory, "audio": audio, "whisper": {"mode": "test_server_api", "status": "pending"}}
            rows.append(row)
            conditions["records"].append({"directory": directory, "audio": asdict(audio)})
            log.info("Вход %d: %s; %.3f с, отсчётов %d, SHA256 PCM %s", index + 1, source, audio.seconds, audio.samples, audio.sha256_pcm)
            log.info("Исходный файл: каналов %s, частота %s Гц, размер %s байт; вход Whisper — моно 16000 Гц PCM16",
                     audio.source_channels, audio.source_sample_rate, audio.source_bytes)
        update(args.out, rows, conditions)
        conditions["whisper"]["preflight"] = await client.check()
        warm = compare.prepare_audio(args.warmup_file or files[0], args.out / "временные" / "whisper-прогрев.wav",
                                     max_seconds=args.warmup_seconds)
        log.info("Прогрев тестового Whisper на %.3f с; загрузка модели и прогрев вне таблицы замеров", warm.seconds)
        warm_result = await client.run(Path(warm.prepared), warm.seconds)
        compare.write_json(args.out / "whisper-прогрев.json", warm_result)
        conditions["whisper"]["warmup"] = {"audio": asdict(warm), "included_in_measurements": False,
                                           "elapsed_seconds": warm_result.get("elapsed_seconds"), "status": warm_result["status"]}
        if warm_result["status"] != "ok" or not warm_result.get("text", "").strip():
            raise RuntimeError(f"Прогрев Whisper не завершён: {warm_result.get('error') or 'речь не найдена'}. "
                               "Выберите запись с речью через --warmup-file")
        log.info("Прогрев Whisper завершён за %.3f с; начинаем измеряемые запросы", warm_result["elapsed_seconds"])
        conditions["state"] = "Whisper выполняется"
        for index, row in enumerate(rows):
            if index:
                log.info("Пауза %.1f с перед следующим запросом Whisper", args.api_gap)
                await asyncio.sleep(args.api_gap)
                await client.health()
            log.info("Тестовый Whisper %d/%d: %s", index + 1, len(rows), row["audio"].source)
            # Контрольный SHA256 не требует numpy/CUDA и выполняется вне HTTP-таймера.
            path = Path(row["audio"].prepared)
            with wave.open(str(path), "rb") as wav:
                digest = hashlib.sha256(wav.readframes(wav.getnframes())).hexdigest()
            if digest != row["audio"].sha256_pcm:
                raise ValueError("Подготовленный WAV изменился; запрос не отправлен")
            result = await client.run(path, row["audio"].seconds)
            row["whisper"] = result
            compare.save_result(args.out, row, result)
            update(args.out, rows, conditions)
            if result["status"] != "ok":
                raise RuntimeError("Whisper: прекращаем все последующие запросы")
        conditions["state"] = "Завершён"
        return 0
    except Exception as exc:
        conditions["state"] = "Остановлен из-за ошибки Whisper/подготовки"
        log.exception("%s", exc)
        return 1
    finally:
        if client:
            await client.close()
        conditions["whisper_finished_at"] = datetime.now().astimezone().isoformat()
        conditions["phases"] = {"whisper": {"started_at": conditions["started_at"],
            "finished_at": conditions["whisper_finished_at"], "wall_seconds": time.perf_counter() - wall_started,
            "includes_preparation_pauses_and_report_writes": True, "state": conditions["state"]}}
        update(args.out, rows, conditions)
