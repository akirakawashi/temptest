"""Отдельный прогон GigaAM TensorRT: поиск речи, распознавание и паузы. Без управления продом."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
import hashlib
import html
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

from benchmark.audio_check import decoder_info
from benchmark.compare import Audio, host_info, setup_logging
from benchmark.gigaam_trt_prepare import LABEL, MAX_FRAMES, MIN_FRAMES, SYSTEM, TRT_STAGES, vad_provider
from benchmark.summary import write_summary
from benchmark.trt_common import VAD_SETTINGS, parser, prepare, run_worker, save_json, summary_row
from benchmark.tone_trt_prepare import gpu_snapshot

log = logging.getLogger("прогон-gigaam-trt")
WALL = {"vad": "Поиск речи", "asr": "Распознавание", "parallel": "Одновременно", "other": "Остальная обработка"}


def update(output: Path, rows: list[dict], conditions: dict) -> None:
    save_json(output / "условия.json", conditions)
    summary = write_summary(output, rows, conditions)
    stats = summary["systems"][SYSTEM]
    escape = html.escape
    mean = stats["processing_seconds"].get("mean")
    processed = sum(stats["statuses"][key] for key in ("ok", "no_speech"))
    caption = (f"В среднем {mean:.3f} с на запись; около {int(stats['records_per_15_minutes'])} записей за 15 минут"
               if mean else "Успешных замеров с речью пока нет")
    cards = [f"<section><h2>{escape(LABEL)}</h2><p>Завершено {processed} из {len(rows)}. {caption}.</p>",
             "<p>Энкодер работает на GPU через TensorRT. Признаки и декодер считаются на CPU.</p>"]
    for key, value in stats.get("mean_record_wall_percent", {}).items():
        if key in WALL:
            cards.append(f"<div class='part'><span>{escape(WALL[key])}</span><i style='width:{value:.3f}%'></i><b>{value:.1f}%</b></div>")
    cards.append("<p>Проценты усреднены по записям. Поиск речи и распознавание идут одновременно; пересечение показано отдельно.</p></section>")
    parts = stats.get("trt_stages", {})
    if parts:
        cards.append("<section><h2>Из чего складывается распознавание</h2>")
        maximum = max((value["seconds"].get("total", 0) for value in parts.values()), default=1) or 1
        for key, title in TRT_STAGES.items():
            seconds = parts[key]["seconds"].get("total", 0)
            cards.append(f"<div class='part'><span>{escape(title)}</span><i style='width:{seconds / maximum * 75:.2f}%'></i><b>{seconds:.1f} с</b></div>")
        cards.append("<p>Сумма по всем записям. Обмен данными с Triton входит во время энкодера.</p></section>")
    successful = [row for row in rows if row.get(SYSTEM, {}).get("status") == "ok"]
    maximum = max((row[SYSTEM]["elapsed_seconds"] for row in successful), default=1) or 1
    cards.append("<section><h2>Время каждой записи</h2>")
    for row in successful:
        seconds = row[SYSTEM]["elapsed_seconds"]
        cards.append(f"<div class='part'><span>{escape(Path(row['directory']).name)}</span>"
                     f"<i style='width:{seconds / maximum * 75:.2f}%'></i><b>{seconds:.2f} с</b></div>")
    cards.append("</section>")
    tables, markdown = [], ["# GigaAM TensorRT", "", conditions["state"], "", caption + ".", ""]
    for row in rows:
        result = row.get(SYSTEM, {})
        seconds, recognition = result.get("elapsed_seconds"), result.get("asr_seconds")
        shown = f"{seconds:.3f}" if seconds is not None else "—"
        asr = f"{recognition:.3f}" if recognition is not None else "—"
        name, text = Path(row["audio"].source).name, result.get("text", "")
        tables.append(f"<tr><td>{escape(Path(row['directory']).name)}</td><td>{escape(name)}</td><td>{row['audio'].seconds:.2f}</td>"
            f"<td>{shown}</td><td>{asr}</td><td>{escape(result.get('status', 'pending'))}</td>"
            f"<td><details><summary>Текст</summary><pre>{escape(text)}</pre></details></td></tr>")
        markdown.extend([f"## {name}", f"{result.get('status', 'pending')}; весь цикл {shown} с; распознавание {asr} с", "", text, ""])
    notes = "".join(f"<p>{escape(note)}</p>" for note in summary["notes"])
    document = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>GigaAM TensorRT — результаты</title><style>body{background:#0b1018;color:#dbe9f6;font:16px/1.6 system-ui;max-width:1400px;margin:auto;padding:24px}section{background:#14202c;border:1px solid #304353;padding:20px;margin:24px 0;border-radius:12px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #304353}pre{white-space:pre-wrap;max-width:650px}.part{display:flex;gap:12px;align-items:center;font-size:13px}.part i{height:10px;background:#6ae0ba;min-width:0}.part span{min-width:210px}.part b{white-space:nowrap}.scroll{overflow:auto}@media(max-width:600px){body{padding:12px}.part span{min-width:120px}}</style>'''
    document += f"<h1>GigaAM TensorRT</h1><p>{escape(conditions['state'])}</p><section><h2>Условия теста</h2>{notes}</section>"
    document += "".join(cards) + ("<section class='scroll'><h2>Все тексты и замеры</h2><table><tr><th>ID</th><th>Запись</th>"
        "<th>Аудио, с</th><th>Весь цикл, с</th><th>Распознавание, с</th><th>Статус</th><th>Текст</th></tr>" + "".join(tables) + "</table></section></html>")
    (output / "отчёт.html").write_text(document, encoding="utf-8")
    (output / "отчёт.md").write_text("\n".join(markdown), encoding="utf-8")


async def run(args) -> int:
    provider = vad_provider()
    conditions = {"schema_version": 3, "mode": "gigaam_tensorrt", "planned_systems": [SYSTEM],
        "state": "Подготовка", "started_at": datetime.now().astimezone().isoformat(),
        "host": host_info(), "selected_gpu": os.environ.get("BENCH_GPU", "0"),
        "production_service_management": False, "production_requests": False,
        "records": [], "preparations": [], "vad_settings": dict(VAD_SETTINGS), "vad_provider": provider,
        "audio_decoder": decoder_info(),
        "guard_policy": {key: os.environ.get(key) for key in (
            "BENCH_GPU_MIN_FREE_MIB", "BENCH_GPU_MAX_UTIL", "BENCH_GPU_RESERVE_MIB")},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))},
        "protocol": {"concurrency": 1, "batch_size": 1, "model_server_load_per_record": False,
            "client_load_per_record": True, "warmup_per_record": True,
            "threads": args.threads, "timeout_seconds_per_phase": args.timeout,
            "rpc_timeout_seconds": min(args.timeout, 60),
            "preparation_included_in_measurements": False, "input": "mono PCM16 16000 Hz",
            "asr": "GigaAM v3_e2e_rnnt: энкодер TensorRT через внутренний Triton, жадный декодер RNN-T на CPU",
            "endpoint": "triton:8001", "encoder": "TensorRT GPU:0, fp16",
            "features": "Лог-мел признаки на CPU с окном и фильтрами из весов модели",
            "decoder": "ONNX Runtime CPU, жадный разбор",
            "vad": f"Silero VAD, {'CPU' if provider == 'cpu' else 'GPU'}; порции по 100 мс",
            "segment_seconds": [MIN_FRAMES / 100, MAX_FRAMES / 100],
            "asr_seconds": "Сумма вызовов распознавания: признаки, энкодер с gRPC, декодер",
            "disabled_stages": ["speaker", "emotion", "llm"], "artifacts_path": str(args.model_dir)}}
    rows, code, started = [], 1, time.perf_counter()
    try:
        save_json(args.out / "условия.json", conditions)
        conditions["gpu"] = gpu_snapshot()
        for name in ("движок.json", "артефакты.json"):
            conditions[name] = json.loads((args.model_dir / name).read_text(encoding="utf-8"))
        if conditions["движок.json"]["fingerprint"]["gpu"]["uuid"] != conditions["gpu"]["uuid"]:
            raise RuntimeError("Движок собран на другой GPU; нужна новая подготовка")
        rows = prepare(args, conditions)
        conditions["state"] = LABEL + " выполняется"
        update(args.out, rows, conditions)
        for index, row in enumerate(rows, 1):
            log.info("%s %d/%d: %s", LABEL, index, len(rows), row["audio"].source)
            payload = await run_worker({"system": SYSTEM, "threads": args.threads, "timeout": args.timeout,
                "model_dir": str(args.model_dir), "vad_settings": conditions["vad_settings"],
                "audio": asdict(row["audio"]), "warmup": conditions["warmup"]["audio"],
                "output": str(args.out / row["directory"]), "log_dir": str(args.out), "protocol": conditions["protocol"]})
            result = payload["result"]
            conditions["preparations"].append({"directory": row["directory"], **payload["preparation"]})
            save_json(args.out / row["directory"] / f"{SYSTEM}.json", result)
            (args.out / row["directory"] / f"{SYSTEM}.txt").write_text(result.get("text", "") + "\n", encoding="utf-8")
            row[SYSTEM] = summary_row(result)
            update(args.out, rows, conditions)
            if row[SYSTEM].get("preparation_failed") or "резерв" in str(row[SYSTEM].get("error")).lower():
                raise RuntimeError(row[SYSTEM]["error"])
        code = int(any(row[SYSTEM]["status"] not in {"ok", "no_speech"} for row in rows))
        conditions["state"] = "Завершён с ошибками" if code else "Завершён"
    except (Exception, asyncio.CancelledError) as exc:
        conditions["state"] = "Остановлен: " + (str(exc) or "прерывание")
        log.exception("Прогон GigaAM TensorRT остановлен")
    finally:
        conditions.update(finished_at=datetime.now().astimezone().isoformat(),
            wall_seconds_including_preparation_and_pauses=time.perf_counter() - started)
        if not rows:
            rows = [{**record, "audio": Audio(**record["audio"])} for record in conditions["records"]]
        update(args.out, rows, conditions)
    return code


def finalize(output: Path, exit_code: int) -> None:
    manifest = output / "условия.json"
    if manifest.is_file():
        conditions = json.loads(manifest.read_text(encoding="utf-8"))
        conditions["launcher_exit_code"] = exit_code
        if exit_code and not conditions["state"].startswith("Остановлен") and conditions["state"] != "Завершён с ошибками":
            conditions["state"] = "Остановлен запуском стенда; причина в логи/запуск.log"
        rows = []
        for record in conditions["records"]:
            row = {**record, "audio": Audio(**record["audio"])}
            path = output / row["directory"] / f"{SYSTEM}.json"
            if path.is_file():
                row[SYSTEM] = summary_row(json.loads(path.read_text(encoding="utf-8")))
            rows.append(row)
        update(output, rows, conditions)
    shutil.rmtree(output / "временные", ignore_errors=True)
    temporary = output / "диагностика.part"
    with tarfile.open(temporary, "w:gz") as archive:
        for item in sorted(output.rglob("*")):
            if item.is_file() and not item.is_symlink() and item.suffix.lower() in {".json", ".txt", ".log", ".csv", ".md", ".html"}:
                archive.add(item, arcname=str(item.relative_to(output)), recursive=False)
    temporary.replace(output / "диагностика.tar.gz")
    print(f"Архив: {output / 'диагностика.tar.gz'}; без аудио и моделей, с текстами разговоров", flush=True)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--finalize":
        finalize(Path(sys.argv[2]), int(sys.argv[3]))
        return 0
    cli = parser()
    cli.description = __doc__
    args = cli.parse_args()
    if min(args.threads, args.timeout, args.warmup_seconds, args.max_audio_seconds) <= 0:
        cli.error("Число потоков и времена положительные")
    if args.threads != int(os.environ.get("BENCH_THREADS", str(args.threads))):
        cli.error("Число потоков задаётся через BENCH_THREADS")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "условия.json").exists():
        cli.error("Укажите новую папку результатов")
    setup_logging(args.out)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
