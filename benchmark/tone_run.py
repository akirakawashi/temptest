"""Отдельный корпус T-one: подготовка, изолированные процессы и отчёт."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import logging
import os
import re
import shutil
import signal
import sys
import tarfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from benchmark.audio_check import decoder_info
from benchmark.compare import FORMATS, Audio, host_info, prepare_audio, setup_logging
from benchmark.summary import write_summary

log = logging.getLogger("прогон-t-one")
VAD_SETTINGS = {"vad_threshold": .5, "vad_min_silence": .45, "vad_min_speech": .25, "vad_max_speech": 15.}


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
    system = request.get("system", "tone")
    if system not in {"tone", "tone_trt_greedy", "tone_trt_kenlm"}:
        raise ValueError("Неизвестный рабочий процесс T-one")
    module = "benchmark.tone_worker" if system == "tone" else "benchmark.tone_trt_worker"
    task = directory / f"{system}-задание.json"
    response = directory / f"{system}-ответ.json"
    process_log = directory / f"{system}-процесс.log"
    save_json(task, request)
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(sys.executable, "-X", "faulthandler",
        "-m", module, str(task), start_new_session=True,
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
            "error": f"Рабочий процесс T-one завершён: код {process.returncode}; таймаут {timed_out}; журнал {process_log}"},
            "preparation": preparation}
    payload["preparation"].update(worker_wall_seconds=time.perf_counter() - started,
        process_exit_code=process.returncode, process_log=str(process_log))
    native_log = process_log.read_text(encoding="utf-8", errors="replace").lower()
    if "fallback to cpu" in native_log or "fall back to cpu" in native_log:
        payload["result"].update(status="error", error="sherpa-onnx перешёл на CPU; результат исключён из GPU-замеров", preparation_failed=True)
    if process.returncode and payload["result"]["status"] in {"ok", "no_speech"}:
        payload["result"].update(status="error", error=f"Ненулевой код рабочего процесса: {process.returncode}", preparation_failed=True)
    save_json(prep_path, payload["preparation"])
    return payload


def update(output: Path, rows: list[dict], conditions: dict):
    save_json(output / "условия.json", conditions)
    summary = write_summary(output, rows, conditions)
    stats = summary["systems"]["tone"]
    e = html.escape
    processing = stats["processing_seconds"]
    mean = processing.get("mean")
    capacity = stats["records_per_15_minutes"]
    intro = (f"Успешно {stats['statuses']['ok']} из {len(rows)}; состояние: {conditions['state']}. "
        f"Средняя обработка: {mean:.3f} с; за 15 минут около {int(capacity)} записей."
        if mean else f"Успешных замеров пока нет; состояние: {conditions['state']}.")
    lines = ["# Отдельный тест T-one", "", intro, "", *[f"- {note}" for note in summary["notes"]], ""]
    maximum = max((row.get("tone", {}).get("elapsed_seconds") or 0 for row in rows), default=1) or 1
    cells, bars = [], []
    for row in rows:
        result = row.get("tone", {})
        seconds = result.get("elapsed_seconds")
        elapsed = f"{seconds:.3f}" if seconds is not None else "—"
        asr = f"{result['asr_seconds']:.3f}" if result.get("asr_seconds") is not None else "—"
        status = result.get("status", "pending")
        name = Path(row["audio"].source).name
        cells.append(f"<tr><td>{e(name)}</td><td>{row['audio'].seconds:.2f}</td><td>{elapsed}</td>"
                     f"<td>{asr}</td><td>{e(status)}</td><td><details><summary>Текст</summary>"
                     f"<pre>{e(result.get('text', ''))}</pre></details></td></tr>")
        if status == "ok":
            bars.append(f"<div class='bar'><span>{e(Path(row['directory']).name)}</span>"
                        f"<i style='width:{seconds / maximum * 80:.2f}%'></i> {elapsed} с</div>")
        lines.extend([f"## {name}", f"Статус: {status}; весь цикл {elapsed} с; ASR {asr} с.", "",
                      result.get("text", ""), ""])
    parts = []
    titles = {"vad": "Выделение речи", "asr": "Распознавание", "parallel": "Одновременно", "other": "Остальная обработка"}
    for key, value in stats.get("exclusive_wall_percent", {}).items():
        if value > 0:
            parts.append(f"<p>{e(titles.get(key, key))}: {value:.1f}%</p>")
    document = """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>T-one — результаты теста</title><style>body{background:#0b1018;color:#dbe9f6;font:16px/1.6 system-ui;max-width:1400px;margin:auto;padding:24px}section{background:#14202c;border:1px solid #304353;padding:20px;margin:24px 0;border-radius:12px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #304353}pre{white-space:pre-wrap;max-width:600px}.bar{display:flex;gap:10px;align-items:center;font-size:12px}.bar i{height:8px;background:#6ae0ba}.bar span{width:40px;flex-shrink:0}.scroll{overflow:auto}</style>"""
    document += f"<h1>T-one — отдельный тест распознавания</h1><p>{e(intro)}</p><section><h2>Что измеряли</h2>"
    document += "".join(f"<p>{e(note)}</p>" for note in summary["notes"])
    document += "</section><section><h2>На что ушло время</h2>" + "".join(parts) + "</section>"
    document += "<section><h2>Обработка каждой записи</h2>" + "".join(bars) + "</section>"
    document += "<section class='scroll'><h2>Все записи и тексты</h2><table><tr><th>Запись</th><th>Аудио, с</th><th>Весь цикл, с</th><th>ASR, с</th><th>Статус</th><th>Расшифровка</th></tr>" + "".join(cells) + "</table></section></html>"
    (output / "отчёт.html").write_text(document, encoding="utf-8")
    (output / "отчёт.md").write_text("\n".join(lines), encoding="utf-8")


async def run(args):
    from benchmark.tone import MODEL_ARCHIVE_SHA256, MODEL_NAME

    conditions = {"schema_version": 2, "mode": "tone_only", "planned_systems": ["tone"],
        "state": "Подготовка", "started_at": datetime.now().astimezone().isoformat(),
        "host": host_info(), "selected_gpu": os.environ.get("BENCH_GPU", "0"), "records": [], "preparations": [],
        "vad_settings": dict(VAD_SETTINGS), "production_service_management": False, "production_requests": False,
        "audio_decoder": decoder_info(),
        "guard_policy": {key: os.environ.get(key) for key in (
            "BENCH_GPU_MIN_FREE_MIB", "BENCH_GPU_MAX_UTIL", "BENCH_GPU_RESERVE_MIB")},
        "source_sha256": {str(path.relative_to(Path(__file__).resolve().parent.parent)):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent / "app")
            for path in sorted(folder.glob("*.py"))},
        "protocol": {"concurrency": 1, "model_load_per_record": True, "warmup_per_record": True,
            "gap_seconds": args.gap, "threads": args.threads, "timeout_seconds_per_phase": args.timeout,
            "preparation_included_in_measurements": False, "asr_includes_16k_to_8k": True,
            "lead_padding_seconds": .3, "tail_padding_seconds": 1., "timestamp_correction_seconds": .4,
            "asr": "T-one CTC, float32 ONNX, CUDA:0, greedy_search", "input": "mono PCM16 16000 Hz",
            "model_dir": str(args.model_dir), "model_name": MODEL_NAME,
            "model_archive_sha256": MODEL_ARCHIVE_SHA256,
            "disabled_stages": ["speaker", "emotion", "llm"]}}
    rows, code, started = [], 1, time.perf_counter()
    try:
        save_json(args.out / "условия.json", conditions)
        log.info("Декодер аудио перед подготовкой: %s", conditions["audio_decoder"])
        rows = prepare(args, conditions)
        conditions["state"] = "T-one выполняется"
        update(args.out, rows, conditions)
        for index, row in enumerate(rows, 1):
            log.info("T-one %d/%d: %s", index, len(rows), row["audio"].source)
            request = {"system": "tone", "threads": args.threads, "timeout": args.timeout,
                "model_dir": str(args.model_dir), "vad_settings": conditions["vad_settings"],
                "audio": asdict(row["audio"]), "warmup": conditions["warmup"]["audio"],
                "output": str(args.out / row["directory"]), "log_dir": str(args.out)}
            payload = await run_worker(request)
            row["tone"] = payload["result"]
            conditions["preparations"].append({"directory": row["directory"], **payload["preparation"]})
            save_json(args.out / row["directory"] / "tone.json", row["tone"])
            (args.out / row["directory"] / "tone.txt").write_text(row["tone"].get("text", "") + "\n", encoding="utf-8")
            update(args.out, rows, conditions)
            if row["tone"].get("preparation_failed") or "резерв" in str(row["tone"].get("error")).lower():
                raise RuntimeError(row["tone"]["error"])
            if index < len(rows) and args.gap:
                log.info("Пауза %.1f с перед следующей записью — вне замеров", args.gap)
                await asyncio.sleep(args.gap)
        code = int(any(r["tone"]["status"] not in {"ok", "no_speech"} for r in rows))
        conditions["state"] = "Завершён с ошибками" if code else "Завершён"
    except (Exception, asyncio.CancelledError) as exc:
        conditions["state"] = "Остановлен: " + (str(exc) or "прерывание")
        log.exception("Прогон T-one остановлен")
    finally:
        conditions["finished_at"] = datetime.now().astimezone().isoformat()
        conditions["wall_seconds_including_preparation_and_pauses"] = time.perf_counter() - started
        if not rows:
            rows = [{**record, "audio": Audio(**record["audio"])} for record in conditions["records"]]
        update(args.out, rows, conditions)
    return code


def finalize(output: Path, exit_code: int):
    manifest = output / "условия.json"
    if manifest.is_file():
        conditions = json.loads(manifest.read_text())
        conditions["launcher_exit_code"] = exit_code
        if exit_code and not conditions["state"].startswith("Остановлен") and conditions["state"] != "Завершён с ошибками":
            conditions["state"] = "Остановлен запуском стенда; причина в логи/запуск.log"
        rows = []
        for record in conditions["records"]:
            row = {**record, "audio": Audio(**record["audio"])}
            path = output / row["directory"] / "tone.json"
            if path.is_file():
                row["tone"] = json.loads(path.read_text())
            rows.append(row)
        update(output, rows, conditions)
    shutil.rmtree(output / "временные", ignore_errors=True)
    path = output / "диагностика.tar.gz"
    temporary = output / "диагностика.part"
    with tarfile.open(temporary, "w:gz") as archive:
        for item in sorted(output.rglob("*")):
            if item.is_file() and not item.is_symlink() and item.suffix.lower() in {".json", ".txt", ".log", ".csv", ".md", ".html"}:
                archive.add(item, arcname=str(item.relative_to(output)), recursive=False)
    temporary.replace(path)
    print(f"Архив для передачи: {path}; без аудио и моделей, с текстами разговоров", flush=True)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("audio_dir", type=Path)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--reference-run", type=Path)
    cli.add_argument("--model-dir", type=Path, default=Path("/models/tone"))
    cli.add_argument("--expected-files", type=int, default=int(os.environ.get("BENCH_EXPECTED_FILES", "100")))
    cli.add_argument("--threads", type=int, default=int(os.environ.get("BENCH_THREADS", "4")))
    cli.add_argument("--timeout", type=int, default=3600)
    cli.add_argument("--gap", type=float, default=5.)
    cli.add_argument("--warmup-file", type=Path)
    cli.add_argument("--warmup-seconds", type=float, default=30.)
    cli.add_argument("--max-audio-seconds", type=float, default=1800.)
    return cli


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--finalize":
        finalize(Path(sys.argv[2]), int(sys.argv[3]))
        return 0
    cli = parser()
    args = cli.parse_args()
    if (min(args.expected_files, args.threads, args.timeout, args.warmup_seconds, args.max_audio_seconds) <= 0
            or not 0 <= args.gap <= 3600):
        cli.error("Число файлов/потоков и времена должны быть положительными; пауза 0–3600 с")
    if args.threads != int(os.environ.get("BENCH_THREADS", str(args.threads))):
        cli.error("Задайте число потоков через BENCH_THREADS, чтобы CPU-квота контейнера совпадала с моделью")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "условия.json").exists():
        cli.error("Папка уже содержит прогон; укажите новую папку")
    setup_logging(args.out)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
