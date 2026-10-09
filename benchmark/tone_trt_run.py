"""Отдельный прогон официального T-one TensorRT, без управления продом."""
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
from benchmark.summary import write_summary
from benchmark.tone_run import prepare, run_worker, save_json, VAD_SETTINGS
from benchmark.tone_trt_prepare import TONE_REF, HF_REF, gpu_snapshot

log = logging.getLogger("прогон-t-one-trt")
LABELS = {"tone_trt_greedy": "T-one TensorRT без KenLM", "tone_trt_kenlm": "T-one TensorRT с KenLM"}
STAGES = {"conversion": "Пересчёт частоты", "acoustic": "Акустика и обмен с Triton",
          "splitter": "Границы фраз", "decoder": "Получение слов", "other": "Остальная обработка"}


def update(output, rows, conditions):
    save_json(output / "условия.json", conditions)
    summary = write_summary(output, rows, conditions)
    escape = html.escape
    cards, tables, markdown = [], [], ["# T-one TensorRT", "", conditions["state"], ""]
    for system in conditions["planned_systems"]:
        stats = summary["systems"][system]
        mean = stats["processing_seconds"].get("mean")
        capacity = stats["records_per_15_minutes"]
        processed = sum(stats["statuses"][key] for key in ("ok", "no_speech"))
        label = LABELS[system]
        caption = (f"В среднем {mean:.3f} с; около {int(capacity)} записей за 15 минут"
                   if mean else "Успешных замеров с речью пока нет")
        cards.append(f"<section><h2>{escape(label)}</h2><p>Завершено {processed} из {len(rows)}. {caption}.</p>")
        cards.append("<p>Акустика работает на GPU. Разделение фраз и декодер работают на CPU.</p>")
        for key, value in stats.get("mean_record_wall_percent", {}).items():
            cards.append(f"<div class='part'><span>{escape(STAGES.get(key, key))}</span>"
                         f"<i style='width:{value:.3f}%'></i><b>{value:.1f}%</b></div>")
        cards.append("<p>Проценты усреднены по записям. Передача данных в Triton включена в акустику.</p></section>")
        successful = [r for r in rows if r.get(system, {}).get("status") == "ok"]
        maximum = max((r[system]["elapsed_seconds"] for r in successful), default=1) or 1
        cards.append(f"<section><h2>Время каждой записи — {escape(label)}</h2>")
        for row in successful:
            seconds = row[system]["elapsed_seconds"]
            cards.append(f"<div class='part'><span>{escape(Path(row['directory']).name)}</span>"
                         f"<i style='width:{seconds / maximum * 75:.2f}%'></i><b>{seconds:.2f} с</b></div>")
        cards.append("</section>")
        for row in rows:
            result = row.get(system, {})
            seconds = result.get("elapsed_seconds")
            shown = f"{seconds:.3f}" if seconds is not None else "—"
            name, text = Path(row["audio"].source).name, result.get("text", "")
            tables.append(f"<tr><td>{escape(Path(row['directory']).name)}</td><td>{escape(name)}</td>"
                f"<td>{escape(label)}</td><td>{shown}</td><td>{escape(result.get('status', 'pending'))}</td>"
                f"<td><details><summary>Текст</summary><pre>{escape(text)}</pre></details></td></tr>")
            markdown.extend([f"## {label}: {name}", f"{result.get('status', 'pending')}; {shown} с", "", text, ""])
    notes = "".join(f"<p>{escape(note)}</p>" for note in summary["notes"])
    document = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>T-one TensorRT — результаты</title><style>body{background:#0b1018;color:#dbe9f6;font:16px/1.6 system-ui;max-width:1400px;margin:auto;padding:24px}section{background:#14202c;border:1px solid #304353;padding:20px;margin:24px 0;border-radius:12px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #304353}pre{white-space:pre-wrap;max-width:650px}.part{display:flex;gap:12px;align-items:center;font-size:13px}.part i{height:10px;background:#6ae0ba;min-width:0}.part span{min-width:170px}.part b{white-space:nowrap}.scroll{overflow:auto}@media(max-width:600px){body{padding:12px}.part span{min-width:110px}}</style>'''
    document += f"<h1>T-one TensorRT</h1><p>{escape(conditions['state'])}</p><section><h2>Условия теста</h2>{notes}</section>"
    document += "".join(cards) + "<section class='scroll'><h2>Все тексты и замеры</h2><table><tr><th>ID</th><th>Запись</th><th>Режим</th><th>Обработка, с</th><th>Статус</th><th>Текст</th></tr>" + "".join(tables) + "</table></section></html>"
    (output / "отчёт.html").write_text(document, encoding="utf-8")
    (output / "отчёт.md").write_text("\n".join(markdown), encoding="utf-8")


async def run(args):
    systems = ["tone_trt_greedy"] + ([] if args.without_kenlm else ["tone_trt_kenlm"])
    conditions = {"schema_version": 3, "mode": "tone_tensorrt", "planned_systems": systems,
        "state": "Подготовка", "started_at": datetime.now().astimezone().isoformat(),
        "host": host_info(), "selected_gpu": os.environ.get("BENCH_GPU", "0"),
        "production_service_management": False, "production_requests": False,
        "records": [], "preparations": [], "vad_settings": dict(VAD_SETTINGS),
        "audio_decoder": decoder_info(), "tone_ref": TONE_REF, "hf_ref": HF_REF,
        "guard_policy": {key: os.environ.get(key) for key in (
            "BENCH_GPU_MIN_FREE_MIB", "BENCH_GPU_MAX_UTIL", "BENCH_GPU_RESERVE_MIB")},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))},
        "protocol": {"concurrency": 1, "batch_size": 1, "model_server_load_per_record": False,
            "decoder_load_per_record": True, "warmup_per_record": True, "gap_seconds": args.gap,
            "threads": args.threads, "timeout_seconds_per_phase": args.timeout,
            "rpc_timeout_seconds": min(args.timeout, 60),
            "preparation_included_in_measurements": False, "input": "mono PCM16 16000 Hz",
            "asr": "Официальный T-one TensorRT через внутренний Triton; полный контекст записи",
            "endpoint": "triton:8001", "acoustic": "TensorRT GPU:0, stronglyTyped",
            "splitter": "Официальный logprob splitter CPU", "decoder": "greedy / KenLM beam=200 CPU",
            "asr_seconds": "Весь ASR-цикл: пересчёт частоты, акустика/gRPC, splitter, decoder, цикл чанков",
            "context_resets_per_record": 1, "padding_per_record_seconds": .6,
            "disabled_stages": ["external_vad", "speaker", "emotion", "llm"],
            "artifacts_path": str(args.model_dir)}}
    rows, code, started = [], 1, time.perf_counter()
    try:
        save_json(args.out / "условия.json", conditions)
        conditions["gpu"] = gpu_snapshot()
        for name in ("движок.json", "артефакты.json"):
            conditions[name] = json.loads((args.model_dir / name).read_text())
        if conditions["движок.json"]["fingerprint"]["gpu"]["uuid"] != conditions["gpu"]["uuid"]:
            raise RuntimeError("Движок собран на другой GPU; нужна новая подготовка")
        rows = prepare(args, conditions)
        # Унаследованные параметры Silero нужны только для сверки источника:
        # официальный T-one не режет аудио ими и не сбрасывает контекст.
        conditions["reference_vad_settings_used_for_segmentation"] = False
        for system in systems:
            conditions["state"] = LABELS[system] + " выполняется"
            update(args.out, rows, conditions)
            for index, row in enumerate(rows, 1):
                log.info("%s %d/%d: %s", LABELS[system], index, len(rows), row["audio"].source)
                payload = await run_worker({"system": system, "threads": args.threads, "timeout": args.timeout,
                    "model_dir": str(args.model_dir), "audio": asdict(row["audio"]),
                    "warmup": conditions["warmup"]["audio"], "output": str(args.out / row["directory"]),
                    "log_dir": str(args.out), "protocol": conditions["protocol"]})
                row[system] = payload["result"]
                conditions["preparations"].append({"directory": row["directory"], **payload["preparation"]})
                save_json(args.out / row["directory"] / f"{system}.json", row[system])
                (args.out / row["directory"] / f"{system}.txt").write_text(row[system].get("text", "") + "\n", encoding="utf-8")
                update(args.out, rows, conditions)
                if row[system].get("preparation_failed") or "резерв" in str(row[system].get("error")).lower():
                    raise RuntimeError(row[system]["error"])
                if index < len(rows) and args.gap:
                    log.info("Пауза %.1f с перед следующей записью — вне замеров", args.gap)
                    await asyncio.sleep(args.gap)
        code = int(any(r[s]["status"] not in {"ok", "no_speech"} for r in rows for s in systems))
        conditions["state"] = "Завершён с ошибками" if code else "Завершён"
    except (Exception, asyncio.CancelledError) as exc:
        conditions["state"] = "Остановлен: " + (str(exc) or "прерывание")
        log.exception("Прогон TensorRT остановлен")
    finally:
        conditions.update(finished_at=datetime.now().astimezone().isoformat(),
            wall_seconds_including_preparation_and_pauses=time.perf_counter() - started)
        if not rows:
            rows = [{**r, "audio": Audio(**r["audio"])} for r in conditions["records"]]
        update(args.out, rows, conditions)
    return code


def finalize(output: Path, exit_code: int):
    manifest = output / "условия.json"
    if manifest.is_file():
        conditions = json.loads(manifest.read_text())
        conditions["launcher_exit_code"] = exit_code
        if exit_code and not conditions["state"].startswith("Остановлен"):
            conditions["state"] = "Остановлен запуском стенда; причина в логи/запуск.log"
        rows = []
        for record in conditions["records"]:
            row = {**record, "audio": Audio(**record["audio"])}
            for system in conditions["planned_systems"]:
                path = output / row["directory"] / f"{system}.json"
                if path.is_file():
                    row[system] = json.loads(path.read_text())
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


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--finalize":
        finalize(Path(sys.argv[2]), int(sys.argv[3]))
        return 0
    from benchmark.tone_run import parser

    cli = parser()
    cli.description = __doc__
    cli.set_defaults(model_dir=Path("/cache"))
    cli.add_argument("--without-kenlm", action="store_true", help="Только TensorRT greedy, без второго прогона")
    args = cli.parse_args()
    if (min(args.expected_files, args.threads, args.timeout, args.warmup_seconds, args.max_audio_seconds) <= 0
            or not 0 <= args.gap <= 3600):
        cli.error("Число файлов/потоков и времена положительные; пауза 0–3600 с")
    if args.threads != int(os.environ.get("BENCH_THREADS", str(args.threads))):
        cli.error("Число потоков задаётся через BENCH_THREADS")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "условия.json").exists():
        cli.error("Укажите новую папку результатов")
    setup_logging(args.out)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
