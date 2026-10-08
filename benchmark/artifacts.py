"""Графики без внешних библиотек и единый архив диагностики без аудио/весов."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
import json
from pathlib import Path
import shutil
import tarfile


COLORS = ["#3868cb", "#19976d", "#d57725"]


def gpu_chart(output: Path) -> str | None:
    path = output / "логи" / "gpu.csv"
    if not path.is_file():
        return None
    points = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for row in csv.DictReader(stream):
            row = {k.strip(): v.strip() for k, v in row.items() if k and isinstance(v, str)}
            try:
                points.append((float(row["utilization_percent"]), float(row["used_mib"]) / 1024, float(row["total_mib"]) / 1024))
            except (KeyError, ValueError):
                continue
    if not points:
        return None
    stride = max(1, len(points) // 1000)
    sampled = points[::stride]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="470" viewBox="0 0 1100 470">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="30" font-family="sans-serif" font-size="18">Общая GPU: стенд и все рабочие процессы сервера</text>']
    for metric, maximum, top, title in [(0, 100, 70, "Вычислительная загрузка, %"), (1, max(p[2] for p in points), 270, "Занятая VRAM, ГиБ")]:
        parts.append(f'<text x="20" y="{top - 10}" font-family="sans-serif" font-size="14">{title}; шкала 0–{maximum:.1f}</text>')
        coords = ' '.join(f'{60 + i / max(1, len(sampled) - 1) * 1000:.1f},{top + 150 - p[metric] / maximum * 150:.1f}' for i, p in enumerate(sampled))
        parts.append(f'<rect x="60" y="{top}" width="1000" height="150" fill="#f4f6fa"/><polyline points="{coords}" fill="none" stroke="{COLORS[metric]}" stroke-width="1.5"/>')
    parts.append('<text x="20" y="458" font-family="sans-serif" font-size="12">Слева начало, справа конец прогона. Исходные метки времени и фазы: логи/gpu.csv; интервал около 2 с.</text>')
    return '\n'.join(parts + ['</svg>'])


def svg_bars(title: str, entries: list[tuple], *, grouped=False, legend=None, unit="с") -> str:
    height = 95 + len(entries) * (45 if grouped else 35)
    values = [v for _, series in entries for v in series if v is not None]
    maximum = max(values, default=1) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="30" font-size="19" font-family="sans-serif">{escape(title)}</text>']
    if grouped:
        for index, name in enumerate(legend or ["Whisper API", "GigaAM ASR", "GigaAM весь цикл"]):
            parts.append(f'<text x="{20 + index * 260}" y="55" fill="{COLORS[index]}" font-family="sans-serif" font-size="14">{name}</text>')
    for index, (name, series) in enumerate(entries):
        top = 75 + index * (45 if grouped else 35)
        parts.append(f'<text x="15" y="{top + 13}" font-family="sans-serif" font-size="12">{escape(name[:49])}</text>')
        for sub, value in enumerate(series):
            if value is None:
                continue
            y = top + sub * 11 if grouped else top
            width = value / maximum * 600
            parts.append(f'<rect x="380" y="{y}" width="{width:.2f}" height="{8 if grouped else 18}" fill="{COLORS[sub if grouped else index % 3]}"/>')
            parts.append(f'<text x="{387 + width:.2f}" y="{y + 9 if grouped else y + 14}" font-family="sans-serif" font-size="11">{value:.3f} {unit}</text>')
    return "\n".join(parts + ["</svg>"])


def render(output: Path, rows: list, conditions: dict):
    from benchmark.summary import write_summary

    summary = write_summary(output, rows, conditions)
    if "gigaam_first_line" in conditions.get("planned_systems", []):
        render_three_systems(output, rows, conditions, summary)
        return
    folder = output / "графики"
    folder.mkdir(exist_ok=True)
    gigaam_only = conditions.get("mode") == "gigaam_only"
    pairs = [r for r in rows if all(r.get(s, {}).get("status") == "ok" for s in ("whisper", "gigaam"))]
    measured = [r for r in rows if r.get("gigaam", {}).get("status") == "ok"] if gigaam_only else pairs
    metrics = [("gigaam", "asr_seconds"), ("gigaam", "elapsed_seconds")]
    legend = ["GigaAM ASR", "GigaAM весь цикл"]
    if not gigaam_only:
        metrics.insert(0, ("whisper", "elapsed_seconds"))
        legend.insert(0, "Whisper API")
    totals = [sum(r[s][key] for r in measured) for s, key in metrics]
    title = f"Время GigaAM по успешным записям: {len(measured)}" if gigaam_only else f"Время по успешным парам: {len(pairs)}"
    total_svg = svg_bars(title, list(zip(legend, [[v] for v in totals])))
    recording_svg = svg_bars("Время обработки каждой записи", [
        (f"{Path(r['directory']).name} · {Path(r['audio'].source).name}",
         [r.get(s, {}).get(key) if r.get(s, {}).get("status") == "ok" else None for s, key in
          metrics]) for r in rows], grouped=True, legend=legend)
    names = {"vad": "VAD", "asr": "ASR", "speaker": "CAM++", "emotion": "Эмоции"}
    stages = [(name, [sum(r["gigaam"].get("stages", {}).get(key, {}).get("seconds", 0) for r in measured)]) for key, name in names.items()]
    stages.append(("LLM", [sum(r["gigaam"].get("llm_seconds", 0) for r in measured)]))
    stages_svg = svg_bars("Вызовы этапов GigaAM: времена пересекаются, сумма ≠ полное время", stages)
    charts = {"итоги.svg": total_svg, "по-записям.svg": recording_svg, "этапы-gigaam.svg": stages_svg}
    gpu_svg = gpu_chart(output)
    if gpu_svg:
        charts["gpu.svg"] = gpu_svg
    for name, contents in charts.items():
        (folder / name).write_text(contents, encoding="utf-8")
    percent = ""
    if not gigaam_only and totals[0] > 0:
        percent = (f"GigaAM ASR: изменение времени {(totals[1] / totals[0] - 1) * 100:+.1f}%. "
                   f"GigaAM весь цикл: {(totals[2] / totals[0] - 1) * 100:+.1f}% относительно Whisper API.")
    heading = "Полный цикл GigaAM" if gigaam_only else "Сравнение Whisper API и GigaAM"
    count = f"успешных результатов GigaAM: {len(measured)}" if gigaam_only else f"успешных пар: {len(pairs)}"
    method = ('Whisper в этом прогоне не запускается; его результаты сохраняются в предыдущей папке. '
              'GigaAM: полный локальный конвейер и отдельно ASR после прогрева. '
              'Записи можно сопоставить с прежними результатами по имени и SHA256 PCM.' if gigaam_only else
              'Whisper: HTTP-запрос включает передачу аудио, очередь и ответ. GigaAM: локальный конвейер после прогрева. '
              'Тестовый Whisper останавливается перед GigaAM.')
    runtime = conditions.get("gigaam", {}).get("llm_runtime", {})
    options = runtime.get("options", {})
    llm_note = (f'<p>Контекст LLM: {options.get("num_ctx")}; батч: {options.get("num_batch")}; '
                f'KV-кеш (задано): {escape(str(runtime.get("kv_cache_type_requested")))}. '
                'Настройки LLM влияют на время полного цикла и результат текстового разбора.</p>'
                if runtime else '')
    html = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Результаты систем речи</title>'
            '<style>body{font-family:sans-serif;margin:24px;color:#17233a}svg{max-width:100%;height:auto}.chart{overflow:auto}p{max-width:1000px}</style>'
            f'<h1>{heading}</h1><p>Состояние: {escape(conditions["state"])}. '
            f'Записей: {len(rows)}, {count}.</p><p>{percent}</p>'
            f'<p>{method} Продовые контейнеры не управляются; GPU общая. '
            'Скорость не является оценкой точности.</p>'
            '<p>GPU общая с рабочими сервисами. Их нагрузка влияет на время обеих фаз; '
            'график GPU показывает суммарную загрузку всех процессов.</p>'
            + llm_note
            + ''.join(f'<div class="chart">{svg}</div>' for svg in charts.values()) + '</html>')
    (output / "отчёт.html").write_text(html, encoding="utf-8")


def render_three_systems(output: Path, rows: list, conditions: dict, summary: dict):
    from benchmark.summary import SYSTEMS

    folder = output / "графики"
    folder.mkdir(exist_ok=True)
    def stat(system, key):
        return summary["systems"][system]["processing_seconds"].get(key)
    charts = {
        "среднее-время.svg": svg_bars("Среднее измеренное время успешной записи", [(name, [stat(s, "mean")]) for s, name in SYSTEMS.items()]),
        "за-15-минут.svg": svg_bars("За 15 минут: по среднему времени этого смешанного корпуса", [(name, [summary["systems"][s]["records_per_15_minutes"]]) for s, name in SYSTEMS.items()], unit="записей"),
        "по-записям.svg": svg_bars("Три режима: обработка каждой записи", [
            (f"{Path(r['directory']).name} · {Path(r['audio'].source).name}",
             [r.get(s, {}).get("elapsed_seconds") if r.get(s, {}).get("status") == "ok" else None for s in SYSTEMS]) for r in rows],
            grouped=True, legend=list(SYSTEMS.values())),
    }
    names = {"vad": "Поиск речи", "asr": "Распознавание", "speaker": "Голоса", "emotion": "Эмоции", "llm": "GigaChat",
             "parallel": "Одновременная работа этапов", "other": "Очередь, подача звука и вспомогательная работа"}
    for system, filename in (("gigaam", "gigaam"), ("gigaam_first_line", "первая-линия")):
        entry = summary["systems"][system]
        charts[f"этапы-{filename}.svg"] = svg_bars(f"{SYSTEMS[system]}: вызовы этапов, время может пересекаться", [
            (names[k], [v["seconds"].get("mean")]) for k, v in entry["stages"].items() if v["enabled"]])
        if "mean_record_wall_percent" in entry:
            charts[f"проценты-{filename}.svg"] = svg_bars(f"{SYSTEMS[system]}: средние доли времени записи, сумма 100%", [
                (names[key], [value]) for key, value in entry["mean_record_wall_percent"].items() if value > 0], unit="%")
    gpu = gpu_chart(output)
    if gpu:
        charts["gpu.svg"] = gpu
    for filename, svg in charts.items():
        (folder / filename).write_text(svg, encoding="utf-8")
    cards = []
    for system, name in SYSTEMS.items():
        entry = summary["systems"][system]
        average, total, rate = stat(system, "mean"), stat(system, "total"), entry["records_per_15_minutes"]
        cards.append(f'<article><h2>{name}</h2><p>Успешно: <b>{entry["statuses"]["ok"]} из {len(rows)}</b></p>'
                     f'<p>Среднее: <b>{average:.2f} с</b>; сумма: {total:.2f} с.</p><p>За 15 минут: около <b>{rate:.0f} записей</b>.</p></article>'
                     if average is not None else f'<article><h2>{name}</h2><p>Измерения ещё не получены.</p></article>')
    texts = []
    for row in rows:
        texts.append(f'<details><summary>{escape(Path(row["audio"].source).name)}</summary><div class="cards">')
        for system, name in SYSTEMS.items():
            result = row.get(system, {})
            plain = "\n".join(u["text"] for u in result.get("utterances", [])) or result.get("text", "")
            texts.append(f'<article><h3>{name}</h3><p>Статус: {escape(result.get("status", "pending"))}</p><pre>{escape(plain)}</pre></article>')
        texts.append('</div></details>')
    changes = summary["matched_comparison"].get("time_change_vs_whisper_percent", {})
    comparison = " ".join(f'{SYSTEMS[system]}: {value:+.1f}% времени относительно Whisper.' for system, value in changes.items() if system != "whisper")
    resume_note = (f'<p>Продолжение прогона {escape(conditions["resume"]["source_run"])}: замеры Whisper перенесены без изменения, '
                   'полный GigaAM и первая линия измеряются позже. Входной звук проверен по SHA256. Исходные логи сохранены в источник-whisper.</p>'
                   if conditions.get("resume") else '')
    html = ('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
        '<title>Три режима обработки разговоров</title><style>'
        'body{font:16px/1.6 sans-serif;margin:24px;background:#0b1018;color:#edf3ff}.cards{display:flex;gap:16px;flex-wrap:wrap}'
        'article{background:#131c29;border:1px solid #34445a;padding:18px;border-radius:12px;flex:1;min-width:240px}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 sans-serif}svg{max-width:100%;height:auto;border-radius:12px}'
        '.chart{margin:24px 0}details{border:1px solid #34445a;border-radius:12px;padding:16px;margin:12px 0}summary{cursor:pointer}'
        '</style><h1>Whisper, полный GigaAM и первая линия</h1>'
        f'<p>Состояние: {escape(conditions["state"])}. Корпус: {len(rows)} записей. Общая длительность: {summary["audio_seconds"] / 60:.1f} мин.</p>'
        '<p>Первая линия: только поиск речи и распознавание. Голоса, эмоции и GigaChat отключены; её контейнер не имеет сети.</p>'
        '<p>Во всех трёх режимах используется одна моно копия каждой записи. Роли из исходных каналов здесь не восстанавливаются.</p>'
        '<p>Загрузка, прогрев, тестовые паузы и запись отчёта исключены из времени обработки. За 15 минут — расчёт для похожего набора после подготовки моделей.</p>'
        + resume_note
        + '<div class="cards">' + ''.join(cards) + '</div>'
        + f'<p>Для сравнения полностью успешно обработано всеми тремя режимами: {summary["matched_comparison"]["records"]}. {comparison}</p>'
        + '<p>GPU общая с рабочими сервисами. Их нагрузка влияет на время. Скорость не показывает точность распознавания.</p>'
        + ''.join(f'<div class="chart">{svg}</div>' for svg in charts.values())
        + '<h2>Тексты трёх систем</h2><p>Различия нужно проверять по записи: эталонные расшифровки в этом тесте не задавались.</p>'
        + ''.join(texts) + '<p>Подробности: итоги.json, условия.json, замеры.csv, сводка.csv и журналы каждого рабочего процесса.</p></html>')
    (output / "отчёт.html").write_text(html, encoding="utf-8")


def finalize(output: Path, exit_code: int):
    from benchmark.compare import save_reports, write_json
    from benchmark.server_run import load_rows

    manifest = output / "условия.json"
    if manifest.is_file():
        conditions = json.loads(manifest.read_text(encoding="utf-8"))
        conditions["launcher_exit_code"] = exit_code
        conditions["finished_at"] = datetime.now().astimezone().isoformat()
        if exit_code and not conditions.get("state", "").startswith("Остановлен"):
            conditions["state"] = "Остановлен запуском стенда; причина в логи/запуск.log"
        rows = load_rows(output, conditions)
        write_json(manifest, conditions)
        mode = ("gigaam-only" if conditions.get("mode") == "gigaam_only" else
                "standalone-api" if conditions.get("mode") == "isolated_whisper_then_gigaam" else "api")
        save_reports(output, rows, conditions["state"], mode=mode)
        render(output, rows, conditions)
    temporary = output / "временные"
    if temporary.is_dir():
        shutil.rmtree(temporary)
    path = output / "диагностика.tar.gz"
    with tarfile.open(path.with_suffix(".part"), "w:gz") as archive:
        for file in sorted(output.rglob("*")):
            if file.is_file() and not file.is_symlink() and file not in {path, path.with_suffix(".part")}:
                if file.suffix.lower() in {".json", ".txt", ".log", ".csv", ".md", ".html", ".svg"}:
                    archive.add(file, arcname=str(file.relative_to(output)), recursive=False)
    path.with_suffix(".part").replace(path)
    print(f"Архив для передачи: {path} (без аудио и моделей; содержит тексты разговоров)", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--exit-code", type=int, default=0)
    args = cli.parse_args()
    finalize(args.output, args.exit_code)
