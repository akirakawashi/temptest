"""Отчёт Whisper: графики без внешних библиотек и архив диагностики без аудио/весов."""
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


def svg_bars(title: str, entries: list[tuple], *, unit="с") -> str:
    height = 75 + len(entries) * 35
    maximum = max((value for _, value in entries if value is not None), default=1) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="30" font-size="19" font-family="sans-serif">{escape(title)}</text>']
    for index, (name, value) in enumerate(entries):
        top = 55 + index * 35
        parts.append(f'<text x="15" y="{top + 13}" font-family="sans-serif" font-size="12">{escape(name[:49])}</text>')
        if value is None:
            continue
        width = value / maximum * 600
        parts.append(f'<rect x="380" y="{top}" width="{width:.2f}" height="18" fill="{COLORS[0]}"/>')
        parts.append(f'<text x="{387 + width:.2f}" y="{top + 14}" font-family="sans-serif" font-size="11">{value:.3f} {unit}</text>')
    return "\n".join(parts + ["</svg>"])


def render(output: Path, rows: list, conditions: dict):
    from benchmark.summary import write_summary

    entry = write_summary(output, rows, conditions)["systems"]["whisper"]
    folder = output / "графики"
    folder.mkdir(exist_ok=True)
    charts = {"по-записям.svg": svg_bars("Whisper: обработка каждой записи", [
        (f"{Path(r['directory']).name} · {Path(r['audio'].source).name}",
         r.get("whisper", {}).get("elapsed_seconds") if r.get("whisper", {}).get("status") == "ok" else None) for r in rows])}
    gpu = gpu_chart(output)
    if gpu:
        charts["gpu.svg"] = gpu
    for filename, svg in charts.items():
        (folder / filename).write_text(svg, encoding="utf-8")
    average, total = entry["processing_seconds"].get("mean"), entry["processing_seconds"].get("total")
    card = (f'<article><h2>Whisper</h2><p>Успешно: <b>{entry["statuses"]["ok"]} из {len(rows)}</b></p>'
            f'<p>Среднее: <b>{average:.2f} с</b>; сумма: {total:.2f} с.</p>'
            f'<p>За 15 минут: около <b>{entry["records_per_15_minutes"]:.0f} записей</b>.</p></article>'
            if average is not None else '<article><h2>Whisper</h2><p>Измерения ещё не получены.</p></article>')
    texts = []
    for row in rows:
        result = row.get("whisper", {})
        texts.append(f'<details><summary>{escape(Path(row["audio"].source).name)}</summary>'
                     f'<p>Статус: {escape(result.get("status", "pending"))}</p><pre>{escape(result.get("text", ""))}</pre></details>')
    html = ('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
        '<title>Whisper — результаты теста</title><style>'
        'body{font:16px/1.6 sans-serif;margin:24px;background:#0b1018;color:#edf3ff}.cards{display:flex;gap:16px;flex-wrap:wrap}'
        'article{background:#131c29;border:1px solid #34445a;padding:18px;border-radius:12px;flex:1;min-width:240px}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 sans-serif}svg{max-width:100%;height:auto;border-radius:12px}'
        '.chart{margin:24px 0}details{border:1px solid #34445a;border-radius:12px;padding:16px;margin:12px 0}summary{cursor:pointer}'
        '</style><h1>Whisper на корпусе записей</h1>'
        f'<p>Состояние: {escape(conditions["state"])}. Корпус: {len(rows)} записей. Общая длительность: {sum(r["audio"].seconds for r in rows) / 60:.1f} мин.</p>'
        '<p>Собственный тестовый Whisper; время — весь HTTP-запрос: передача аудио, обработка и ответ. Продовые контейнеры не управляются.</p>'
        '<p>Используется одна моно копия каждой записи. Роли из исходных каналов здесь не восстанавливаются.</p>'
        '<p>Загрузка, прогрев, тестовые паузы и запись отчёта исключены из времени обработки. За 15 минут — расчёт для похожего набора после подготовки модели.</p>'
        f'<div class="cards">{card}</div>'
        '<p>GPU общая с рабочими сервисами. Их нагрузка влияет на время. Скорость не показывает точность распознавания.</p>'
        + ''.join(f'<div class="chart">{svg}</div>' for svg in charts.values())
        + '<h2>Тексты</h2><p>Правильность нужно проверять по записи: эталонные расшифровки в этом тесте не задавались.</p>'
        + ''.join(texts) + '<p>Подробности: итоги.json, условия.json, замеры.csv, сводка.csv и журналы.</p></html>')
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
        save_reports(output, rows, conditions["state"])
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
