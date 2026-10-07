"""Две фазы: собственный тестовый Whisper, затем полный цикл GigaAM."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
from pathlib import Path

from benchmark import compare
from benchmark.whisper_api import TEST_URL, WhisperAPI

log = logging.getLogger("серверный-прогон")


def load_rows(output: Path, conditions: dict) -> list[dict]:
    rows = []
    for item in conditions["records"]:
        row = {**item, "audio": compare.Audio(**item["audio"])}
        for system in ("whisper", "gigaam"):
            path = output / row["directory"] / f"{system}.json"
            if path.is_file():
                row[system] = json.loads(path.read_text(encoding="utf-8"))
        rows.append(row)
    return rows


def update(output: Path, rows: list, conditions: dict):
    from benchmark.artifacts import render

    compare.write_json(output / "условия.json", conditions)
    compare.save_reports(output, rows, conditions["state"], mode="standalone-api")
    render(output, rows, conditions)


async def whisper_phase(args) -> int:
    files = sorted(p.resolve() for p in args.audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in compare.FORMATS)
    conditions = {"mode": "isolated_whisper_then_gigaam", "state": "Подготовка",
                  "started_at": datetime.now().astimezone().isoformat(), "host": compare.host_info(),
                  "selected_gpu": os.environ.get("BENCH_GPU", "0"), "records": [], "preparations": [],
                  "whisper": {"url": TEST_URL,
                              "model": os.environ.get("BENCH_WHISPER_MODEL", "large-v3"),
                              "image": os.environ.get("BENCH_WHISPER_IMAGE"),
                              "model_load_per_corpus": True, "warmup_per_corpus": True,
                              "concurrency": 1, "retries": 0, "gap_seconds": args.api_gap,
                              "timeout_seconds": args.api_timeout,
                              "versions_source": "Фактические версии тестового контейнера из /benchmark/metadata"},
                  "order": "Все записи через тестовый Whisper; его остановка; все записи через GigaAM",
                  "timers": {"whisper": "Полный HTTP-запрос после прогрева, загрузка и прогрев исключены",
                             "gigaam": "Полный локальный цикл и отдельно сумма ASR; загрузка и прогрев исключены"},
                  "gigaam": {"model_load_per_record": True, "warmup_per_record": True,
                             "preparation_included_in_measurements": False},
                  "production_service_management": False, "production_requests": False,
                  "containers_retained": True,
                  "whisper_external_network": False, "gigaam_external_network": False}
    conditions["guard_policy"] = {key: os.environ.get(key) for key in
                                  ("BENCH_GPU_MIN_FREE_MIB", "BENCH_GPU_MAX_UTIL", "BENCH_GPU_RESERVE_MIB", "BENCH_MIN_RAM_MIB")}
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
            row = {"directory": directory, "audio": audio, "order": ["whisper", "gigaam"],
                   "whisper": {"mode": "test_server_api", "status": "pending"}}
            rows.append(row)
            conditions["records"].append({"directory": directory, "audio": asdict(audio), "order": row["order"]})
            log.info("Вход %d: %s; %.3f с, отсчётов %d, SHA256 PCM %s", index + 1, source, audio.seconds, audio.samples, audio.sha256_pcm)
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
            import hashlib
            import wave

            path = Path(row["audio"].prepared)
            with wave.open(str(path), "rb") as wav:
                digest = hashlib.sha256(wav.readframes(wav.getnframes())).hexdigest()
            if digest != row["audio"].sha256_pcm:
                raise ValueError("Подготовленный WAV изменился; запрос не отправлен")
            result = await client.run(path, row["audio"].seconds)
            row["whisper"] = result
            compare.save_result(args.out, row, "whisper", result)
            update(args.out, rows, conditions)
            if result["status"] != "ok":
                raise RuntimeError("Whisper: прекращаем все последующие запросы и фазу GigaAM")
        conditions["state"] = "Whisper завершён"
        return 0
    except Exception as exc:
        conditions["state"] = "Остановлен из-за ошибки Whisper/подготовки"
        log.exception("%s", exc)
        return 1
    finally:
        if client:
            await client.close()
        conditions["whisper_finished_at"] = datetime.now().astimezone().isoformat()
        update(args.out, rows, conditions)


async def gigaam_phase(args) -> int:
    conditions = json.loads((args.out / "условия.json").read_text(encoding="utf-8"))
    rows = load_rows(args.out, conditions)
    if conditions.get("state") != "Whisper завершён" or not rows or any(row.get("whisper", {}).get("status") != "ok" for row in rows):
        raise ValueError("GigaAM разрешена только после полностью успешной фазы Whisper; повторного запуска фазы нет")
    if conditions.get("mode") == "isolated_whisper_then_gigaam":
        if not (args.out / "логи" / "тестовый-whisper-остановлен.txt").is_file():
            raise ValueError("Перед GigaAM запуск должен остановить тестовый Whisper")
        conditions["whisper_stopped_before_gigaam"] = True
    conditions["state"] = "GigaAM выполняется"
    try:
        warm_source = args.warmup_file or Path(rows[0]["audio"].source)
        warm = compare.prepare_audio(warm_source, args.out / "временные" / "прогрев.wav", max_seconds=args.warmup_seconds)
        conditions["warmup"] = {"audio": asdict(warm), "included_in_measurements": False}
        for index, row in enumerate(rows):
            log.info("GigaAM %d/%d: %s; тестовый Whisper уже остановлен, продовые контейнеры не управляются", index + 1, len(rows), row["audio"].source)
            payload = await compare.run_worker({"system": "gigaam", "audio": asdict(row["audio"]),
                "warmup": asdict(warm), "output": str(args.out / row["directory"]), "log_dir": str(args.out),
                "threads": args.threads, "timeout": args.timeout, "whisper_model": args.whisper_model,
                "whisper_cache": args.whisper_cache, "beam_size": args.beam_size})
            preparation = {"directory": row["directory"], **payload["preparation"]}
            conditions["preparations"].append(preparation)
            if preparation.get("gpu"):
                if conditions.get("gpu") and conditions["gpu"] != preparation["gpu"]:
                    raise RuntimeError("GPU изменилась; стенд остановлен")
                conditions["gpu"] = preparation["gpu"]
            row["gigaam"] = payload["result"]
            compare.save_result(args.out, row, "gigaam", payload["result"])
            update(args.out, rows, conditions)
            if payload["result"]["status"] in {"error", "timeout"}:
                raise RuntimeError("GigaAM: остановка после ошибки, следующие записи не запускаются")
        conditions["state"] = "Завершён"
        return 0
    except Exception as exc:
        conditions["state"] = "Остановлен из-за ошибки GigaAM"
        log.exception("%s", exc)
        return 1
    finally:
        conditions["finished_at"] = datetime.now().astimezone().isoformat()
        update(args.out, rows, conditions)


def main(args, cli) -> int:
    if not args.audio_dir.is_dir() or args.audio_dir == args.out or args.audio_dir in args.out.parents:
        cli.error("Нужна папка аудио; результаты должны находиться вне неё")
    if args.api_gap < 5:
        cli.error("Пауза между запросами Whisper должна быть не меньше 5 секунд")
    if args.phase == "whisper-api" and (args.out / "условия.json").exists():
        cli.error("Прогон уже существует; выберите новую папку")
    if args.phase == "gigaam" and not (args.out / "условия.json").is_file():
        cli.error("Нет результатов фазы Whisper")
    args.out.mkdir(parents=True, exist_ok=True)
    compare.setup_logging(args.out)
    try:
        return asyncio.run(whisper_phase(args) if args.phase == "whisper-api" else gigaam_phase(args))
    except KeyboardInterrupt:
        log.warning("Прерван пользователем; запрос Whisper не повторяется, сервер может продолжать обработку")
        return 130
