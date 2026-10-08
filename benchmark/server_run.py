"""Собственный Whisper и полный GigaAM; GigaAM также запускается отдельно."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time

from benchmark import compare
from benchmark.llm_settings import llm_runtime
from benchmark.whisper_api import TEST_URL, WhisperAPI

log = logging.getLogger("серверный-прогон")


def load_rows(output: Path, conditions: dict) -> list[dict]:
    rows = []
    for item in conditions["records"]:
        row = {**item, "audio": compare.Audio(**item["audio"])}
        for system in ("whisper", "gigaam", "gigaam_first_line"):
            path = output / row["directory"] / f"{system}.json"
            if path.is_file():
                row[system] = json.loads(path.read_text(encoding="utf-8"))
        rows.append(row)
    return rows


def update(output: Path, rows: list, conditions: dict):
    from benchmark.artifacts import render

    compare.write_json(output / "условия.json", conditions)
    mode = "gigaam-only" if conditions.get("mode") == "gigaam_only" else "standalone-api"
    compare.save_reports(output, rows, conditions["state"], mode=mode)
    render(output, rows, conditions)


async def whisper_phase(args) -> int:
    wall_started = time.perf_counter()
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
                             "preparation_included_in_measurements": False,
                             "llm_runtime": llm_runtime(args.threads)},
                  "production_service_management": False, "production_requests": False,
                  "containers_retained": True,
                  "whisper_external_network": False, "gigaam_external_network": False}
    conditions["schema_version"] = 2
    conditions["planned_systems"] = ["whisper", "gigaam"] + (["gigaam_first_line"] if args.include_first_line else [])
    conditions["input"] = "mono_16000_pcm_s16le"
    conditions["first_line"] = {"enabled": args.include_first_line, "enabled_stages": ["vad", "asr"],
        "disabled_stages": ["speaker", "emotion", "llm"], "channel_policy": "Общий моно WAV, без восстановления ролей по каналам",
        "model_load_per_record": True, "warmup_per_record": True, "llm_requests": False,
        "pause_method": "Промежутки вне фрагментов VAD, включая начало и конец; оценка детектора"}
    if args.include_first_line:
        conditions["order"] += "; остановка Ollama; все записи через GigaAM первой линии без GigaChat"
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
            row = {"directory": directory, "audio": audio, "order": list(conditions["planned_systems"]),
                   "whisper": {"mode": "test_server_api", "status": "pending"}}
            rows.append(row)
            conditions["records"].append({"directory": directory, "audio": asdict(audio), "order": row["order"]})
            log.info("Вход %d: %s; %.3f с, отсчётов %d, SHA256 PCM %s", index + 1, source, audio.seconds, audio.samples, audio.sha256_pcm)
            log.info("Исходный файл: каналов %s, частота %s Гц, размер %s байт; общий вход всех систем — моно 16000 Гц PCM16",
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
        conditions.setdefault("phases", {})["whisper"] = {"started_at": conditions["started_at"],
            "finished_at": conditions["whisper_finished_at"], "wall_seconds": time.perf_counter() - wall_started,
            "includes_preparation_pauses_and_report_writes": True, "state": conditions["state"]}
        update(args.out, rows, conditions)


async def gigaam_phase(args) -> int:
    conditions = json.loads((args.out / "условия.json").read_text(encoding="utf-8"))
    rows = load_rows(args.out, conditions)
    if args.phase == "gigaam-only":
        if (conditions.get("mode") != "gigaam_only" or conditions.get("state") != "GigaAM подготовлен"
                or len(rows) != args.expected_files or any("gigaam" in row for row in rows)):
            raise ValueError("Для отдельного GigaAM нужен новый подготовленный корпус; повторного запуска фазы нет")
    elif conditions.get("state") != "Whisper завершён" or not rows or any(row.get("whisper", {}).get("status") != "ok" for row in rows):
        raise ValueError("GigaAM разрешена только после полностью успешной фазы Whisper; повторного запуска фазы нет")
    if conditions.get("mode") == "isolated_whisper_then_gigaam":
        if not (args.out / "логи" / "тестовый-whisper-остановлен.txt").is_file():
            raise ValueError("Перед GigaAM запуск должен остановить тестовый Whisper")
        conditions["whisper_stopped_before_gigaam"] = True
    return await run_gigaam(args, rows, conditions)


async def first_line_phase(args) -> int:
    conditions = json.loads((args.out / "условия.json").read_text(encoding="utf-8"))
    rows = load_rows(args.out, conditions)
    if ("gigaam_first_line" not in conditions.get("planned_systems", [])
            or conditions.get("state") != "Полный GigaAM завершён"
            or len(rows) != args.expected_files
            or any(row.get("gigaam", {}).get("status") not in {"ok", "no_speech"} for row in rows)
            or any("gigaam_first_line" in row for row in rows)):
        raise ValueError("Первая линия разрешена только после завершения полного GigaAM; повторного запуска фазы нет")
    if not (args.out / "логи" / "тестовая-ollama-остановлена.txt").is_file():
        raise ValueError("Перед первой линией запуск должен остановить тестовую Ollama")
    conditions["ollama_stopped_before_first_line"] = True
    return await run_gigaam(args, rows, conditions, system="gigaam_first_line")


async def gigaam_prepare_phase(args) -> int:
    """Подготовка того же PCM в CPU-образе Whisper, без модели и запросов ASR."""
    conditions = {"mode": "gigaam_only", "state": "Подготовка GigaAM",
                  "started_at": datetime.now().astimezone().isoformat(), "host": compare.host_info(),
                  "selected_gpu": os.environ.get("BENCH_GPU", "0"), "records": [], "preparations": [],
                  "whisper": {"enabled": False}, "order": "Только полный цикл GigaAM",
                  "input": "mono_16000_pcm_s16le", "concurrency": 1,
                  "timers": {"gigaam": "Полный локальный цикл и отдельно сумма ASR; подготовка, загрузка и прогрев исключены"},
                  "gigaam": {"model_load_per_record": True, "warmup_per_record": True,
                             "preparation_included_in_measurements": False,
                             "llm_runtime": llm_runtime(args.threads)},
                  "guard_policy": {key: os.environ.get(key) for key in
                                   ("BENCH_GPU_MIN_FREE_MIB", "BENCH_GPU_MAX_UTIL", "BENCH_GPU_RESERVE_MIB", "BENCH_MIN_RAM_MIB")},
                  "production_service_management": False, "production_requests": False,
                  "containers_retained": True, "gigaam_external_network": False}
    rows = []
    try:
        files = sorted(p.resolve() for p in args.audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in compare.FORMATS)
        if len(files) != args.expected_files:
            raise ValueError(f"Найдено {len(files)} записей, ожидалось {args.expected_files}; обработка GigaAM не начиналась")
        temporary = args.out / "временные"
        temporary.mkdir(exist_ok=True)
        log.info("Только GigaAM: %d файлов; Whisper и его API не используются; подготовка WAV вне замеров", len(files))
        for index, source in enumerate(files):
            directory = f"записи/{index + 1:04d}"
            (args.out / directory).mkdir(parents=True)
            target = temporary / f"{index + 1:04d}.wav"
            audio = compare.prepare_audio(source, target)
            if audio.seconds > args.max_audio_seconds or target.stat().st_size > 64 * 1024 ** 2:
                raise ValueError(f"{source.name}: превышен предел {args.max_audio_seconds} с / 64 МиБ; обработка GigaAM не начиналась")
            row = {"directory": directory, "audio": audio, "order": ["gigaam"]}
            rows.append(row)
            conditions["records"].append({"directory": directory, "audio": asdict(audio), "order": row["order"]})
            log.info("Вход %d: %s; %.3f с, отсчётов %d, SHA256 PCM %s", index + 1, source, audio.seconds, audio.samples, audio.sha256_pcm)
        warm = compare.prepare_audio(args.warmup_file or files[0], temporary / "прогрев.wav", max_seconds=args.warmup_seconds)
        conditions["warmup"] = {"audio": asdict(warm), "included_in_measurements": False}
        conditions["state"] = "GigaAM подготовлен"
        conditions["audio_prepared_at"] = datetime.now().astimezone().isoformat()
        update(args.out, rows, conditions)
    except Exception as exc:
        conditions["state"] = "Остановлен из-за ошибки подготовки GigaAM"
        conditions["finished_at"] = datetime.now().astimezone().isoformat()
        log.exception("%s", exc)
        update(args.out, rows, conditions)
        return 1
    return 0


async def gigaam_resume_phase(args) -> int:
    """Новый отчёт с прежними замерами Whisper и проверенным восстановленным PCM."""
    source_run = args.resume_from.resolve()
    manifest = source_run / "условия.json"
    original_manifest = manifest.read_bytes()
    conditions = json.loads(original_manifest)
    saved_records = conditions.get("records", [])
    whisper_phase_state = conditions.get("phases", {}).get("whisper", {}).get("state", conditions.get("state"))
    if (conditions.get("mode") != "isolated_whisper_then_gigaam"
            or whisper_phase_state != "Whisper завершён"
            or conditions.get("planned_systems") != ["whisper", "gigaam", "gigaam_first_line"]
            or len(saved_records) != args.expected_files):
        raise ValueError("Для продолжения нужен полностью завершённый Whisper из прогона трёх систем")
    if not (source_run / "логи" / "тестовый-whisper-остановлен.txt").is_file():
        raise ValueError("В предыдущем прогоне нет подтверждения остановки тестового Whisper")
    if conditions.get("selected_gpu", "0") != os.environ.get("BENCH_GPU", "0"):
        raise ValueError("Для продолжения нужно выбрать ту же GPU, что использовалась для Whisper")
    files = sorted(p.resolve() for p in args.audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in compare.FORMATS)
    if len(files) != len(saved_records):
        raise ValueError("Корпус изменился: число исходных записей отличается от замеров Whisper")
    saved_rows = []
    for index, (item, source) in enumerate(zip(saved_records, files, strict=True)):
        directory = f"записи/{index + 1:04d}"
        old_audio = compare.Audio(**item["audio"])
        if item["directory"] != directory or Path(old_audio.source).resolve() != source:
            raise ValueError("Корпус изменился: имена или порядок записей отличаются от замеров Whisper")
        if any((source_run / directory / f"{system}.json").exists() for system in ("gigaam", "gigaam_first_line")):
            raise ValueError("Продолжение предназначено для остановки перед первым замером GigaAM")
        whisper_path = source_run / directory / "whisper.json"
        if whisper_path.is_symlink():
            raise ValueError("Результат Whisper должен быть обычным файлом")
        result = json.loads(whisper_path.read_text(encoding="utf-8"))
        if result.get("status") != "ok" or result.get("elapsed_seconds") is None:
            raise ValueError(f"Нет успешного замера Whisper для {source.name}")
        saved_rows.append((directory, old_audio, result))
    conditions["resume"] = {"source_run": source_run.name,
        "source_manifest_sha256": hashlib.sha256(original_manifest).hexdigest(),
        "started_at": datetime.now().astimezone().isoformat(), "host": compare.host_info(),
        "selected_gpu": os.environ.get("BENCH_GPU", "0"), "whisper_reused": True,
        "whisper_timestamps_preserved": True, "pcm_verified": False,
        "source_logs": "источник-whisper/логи", "source_guard_policy": conditions.get("guard_policy")}
    conditions["guard_policy"] = {key: os.environ.get(key) for key in
        ("BENCH_GPU_MIN_FREE_MIB", "BENCH_GPU_MAX_UTIL", "BENCH_GPU_RESERVE_MIB", "BENCH_MIN_RAM_MIB")}
    for key in ("launcher_exit_code", "finished_at"):
        conditions.pop(key, None)
    conditions["state"] = "Восстановление входа после Whisper"
    conditions["records"] = []
    rows = []
    started = time.perf_counter()
    try:
        temporary = args.out / "временные"
        temporary.mkdir()
        log.info("Продолжение %s: %d замеров Whisper сохраняются; модель и API Whisper не используются",
                 source_run.name, len(saved_rows))
        for index, (directory, old_audio, result) in enumerate(saved_rows):
            target = temporary / f"{index + 1:04d}.wav"
            audio = compare.prepare_audio(Path(old_audio.source), target)
            if (audio.sha256_pcm != old_audio.sha256_pcm or audio.samples != old_audio.samples
                    or audio.seconds != old_audio.seconds):
                raise ValueError(f"PCM изменился: {Path(audio.source).name}; GigaAM не запускается")
            if audio.seconds > args.max_audio_seconds or target.stat().st_size > 64 * 1024 ** 2:
                raise ValueError(f"{Path(audio.source).name}: превышен предел длительности/размера входа")
            row = {"directory": directory, "audio": audio, "order": conditions["planned_systems"], "whisper": result}
            rows.append(row)
            conditions["records"].append({"directory": directory, "audio": asdict(audio), "order": row["order"]})
            log.info("Восстановлен вход %d/%d: %s; %.3f с; SHA256 PCM совпал: %s",
                     index + 1, len(saved_rows), audio.source, audio.seconds, audio.sha256_pcm)
        old_warm = compare.Audio(**conditions["whisper"]["warmup"]["audio"])
        warm_source = Path(old_warm.source).resolve()
        if not warm_source.is_relative_to(args.audio_dir.resolve()):
            raise ValueError("Запись прогрева Whisper находится вне подключённой папки аудио")
        warm = compare.prepare_audio(warm_source, temporary / "прогрев.wav", max_seconds=old_warm.seconds)
        if warm.sha256_pcm != old_warm.sha256_pcm or warm.samples != old_warm.samples:
            raise ValueError("PCM прогрева изменился; GigaAM не запускается")
        conditions["warmup"] = {"audio": asdict(warm), "included_in_measurements": False,
                                "restored_from_whisper_warmup": True}
        for row in rows:
            directory = args.out / row["directory"]
            directory.mkdir(parents=True)
            for name in ("whisper.json", "whisper.txt"):
                original = source_run / row["directory"] / name
                if original.is_file() and not original.is_symlink():
                    shutil.copyfile(original, directory / name)
        provenance = args.out / "источник-whisper"
        provenance.mkdir()
        (provenance / "условия.json").write_bytes(original_manifest)
        for name in ("прогон.log", "whisper-прогрев.json"):
            original = source_run / name
            if original.is_file() and not original.is_symlink():
                shutil.copyfile(original, provenance / name)
        original_logs = source_run / "логи"
        for original in original_logs.rglob("*"):
            if (original.is_file() and not original.is_symlink()
                    and original.resolve().is_relative_to(original_logs.resolve())
                    and original.suffix in {".log", ".txt", ".csv"}):
                target = provenance / "логи" / original.relative_to(original_logs)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
        logs = args.out / "логи"
        logs.mkdir(exist_ok=True)
        (logs / "тестовый-whisper-остановлен.txt").write_text(
            f"Whisper не запускался повторно; исходный прогон {source_run.name}; проверка стенда выполнена перед восстановлением\n",
            encoding="utf-8")
        conditions["resume"]["pcm_verified"] = True
        conditions["resume"]["preparation_seconds"] = time.perf_counter() - started
        conditions["state"] = "Whisper завершён"
        log.info("Восстановление завершено: все %d записей и прогрев совпали с входом Whisper; продолжаем полный GigaAM",
                 len(rows))
        update(args.out, rows, conditions)
        return 0
    except Exception as exc:
        conditions["state"] = "Остановлен из-за ошибки восстановления входа Whisper"
        log.exception("%s", exc)
        update(args.out, rows, conditions)
        return 1


async def run_gigaam(args, rows: list, conditions: dict, *, system="gigaam") -> int:
    first_line = system == "gigaam_first_line"
    title = "GigaAM первая линия" if first_line else "GigaAM"
    if not first_line:
        conditions["gigaam"]["llm_runtime"] = llm_runtime(args.threads)
    conditions["state"] = f"{title} выполняется"
    phase_started = datetime.now().astimezone().isoformat()
    wall_started = time.perf_counter()
    update(args.out, rows, conditions)
    try:
        if conditions.get("mode") == "gigaam_only":
            warm = compare.Audio(**conditions["warmup"]["audio"])
        else:
            if "warmup" in conditions:
                warm = compare.Audio(**conditions["warmup"]["audio"])
            else:
                warm_source = args.warmup_file or Path(rows[0]["audio"].source)
                warm = compare.prepare_audio(warm_source, args.out / "временные" / "прогрев.wav", max_seconds=args.warmup_seconds)
                conditions["warmup"] = {"audio": asdict(warm), "included_in_measurements": False}
        for index, row in enumerate(rows):
            whisper_state = "Whisper не запускался" if conditions.get("mode") == "gigaam_only" else "тестовый Whisper уже остановлен"
            log.info("%s %d/%d: %s; %s, продовые контейнеры не управляются", title, index + 1, len(rows), row["audio"].source, whisper_state)
            payload = await compare.run_worker({"system": system, "audio": asdict(row["audio"]),
                "warmup": asdict(warm), "output": str(args.out / row["directory"]), "log_dir": str(args.out),
                "threads": args.threads, "timeout": args.timeout, "whisper_model": args.whisper_model,
                "whisper_cache": args.whisper_cache, "beam_size": args.beam_size})
            preparation = {"directory": row["directory"], **payload["preparation"]}
            conditions["preparations"].append(preparation)
            if preparation.get("gpu"):
                if conditions.get("gpu") and conditions["gpu"] != preparation["gpu"]:
                    raise RuntimeError("GPU изменилась; стенд остановлен")
                conditions["gpu"] = preparation["gpu"]
            row[system] = payload["result"]
            compare.save_result(args.out, row, system, payload["result"])
            update(args.out, rows, conditions)
            if payload["result"]["status"] in {"error", "timeout"}:
                raise RuntimeError(f"{title}: остановка после ошибки, следующие записи не запускаются")
        conditions["state"] = ("Полный GigaAM завершён" if not first_line and
                               "gigaam_first_line" in conditions.get("planned_systems", []) else "Завершён")
        return 0
    except Exception as exc:
        conditions["state"] = f"Остановлен из-за ошибки {title}"
        log.exception("%s", exc)
        return 1
    finally:
        conditions["finished_at"] = datetime.now().astimezone().isoformat()
        conditions.setdefault("phases", {})[system] = {"started_at": phase_started,
            "finished_at": conditions["finished_at"], "wall_seconds": time.perf_counter() - wall_started,
            "includes_preparation_and_report_writes": True, "state": conditions["state"]}
        update(args.out, rows, conditions)


def main(args, cli) -> int:
    if not args.audio_dir.is_dir() or args.audio_dir == args.out or args.audio_dir in args.out.parents:
        cli.error("Нужна папка аудио; результаты должны находиться вне неё")
    if args.phase == "whisper-api" and args.api_gap < 5:
        cli.error("Пауза между запросами Whisper должна быть не меньше 5 секунд")
    if args.phase in {"whisper-api", "gigaam-prepare", "gigaam-resume"} and (args.out / "условия.json").exists():
        cli.error("Прогон уже существует; выберите новую папку")
    if args.phase == "gigaam-resume":
        if not args.resume_from or not (args.resume_from / "условия.json").is_file():
            cli.error("--resume-from должен указывать на папку завершённого Whisper")
        if args.resume_from.resolve() == args.out or args.resume_from.resolve() in args.out.parents:
            cli.error("Продолжение должно создавать отдельную папку вне предыдущего прогона")
        if args.warmup_file:
            cli.error("Продолжение использует исходный прогрев Whisper; --warmup-file менять нельзя")
    if args.phase in {"gigaam", "gigaam-only", "gigaam-first-line"} and not (args.out / "условия.json").is_file():
        cli.error("Нет подготовленного корпуса" if args.phase == "gigaam-only" else "Нет результатов фазы Whisper")
    args.out.mkdir(parents=True, exist_ok=True)
    compare.setup_logging(args.out)
    try:
        phase = {"whisper-api": whisper_phase, "gigaam": gigaam_phase, "gigaam-only": gigaam_phase,
                 "gigaam-prepare": gigaam_prepare_phase, "gigaam-resume": gigaam_resume_phase,
                 "gigaam-first-line": first_line_phase}[args.phase]
        return asyncio.run(phase(args))
    except KeyboardInterrupt:
        if args.phase == "whisper-api":
            log.warning("Прерван пользователем; запрос Whisper не повторяется, сервер может продолжать обработку")
        else:
            log.warning("Прерван пользователем; следующие записи GigaAM не запускаются")
        return 130
