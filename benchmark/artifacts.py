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


def svg_bars(title: str, entries: list[tuple], *, grouped=False) -> str:
    height = 95 + len(entries) * (45 if grouped else 35)
    values = [v for _, series in entries for v in series if v is not None]
    maximum = max(values, default=1) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="30" font-size="19" font-family="sans-serif">{escape(title)}</text>']
    if grouped:
        for index, name in enumerate(["Whisper API", "GigaAM ASR", "GigaAM весь цикл"]):
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
            parts.append(f'<text x="{387 + width:.2f}" y="{y + 9 if grouped else y + 14}" font-family="sans-serif" font-size="11">{value:.3f} с</text>')
    return "\n".join(parts + ["</svg>"])


def render(output: Path, rows: list, conditions: dict):
    folder = output / "графики"
    folder.mkdir(exist_ok=True)
    pairs = [r for r in rows if all(r.get(s, {}).get("status") == "ok" for s in ("whisper", "gigaam"))]
    totals = [sum(r[s][key] for r in pairs) for s, key in
              [("whisper", "elapsed_seconds"), ("gigaam", "asr_seconds"), ("gigaam", "elapsed_seconds")]]
    total_svg = svg_bars(f"Время по успешным парам: {len(pairs)}", list(zip(
        ["Whisper API", "GigaAM ASR", "GigaAM весь цикл"], [[v] for v in totals])))
    recording_svg = svg_bars("Время обработки каждой записи", [
        (f"{Path(r['directory']).name} · {Path(r['audio'].source).name}",
         [r.get(s, {}).get(key) if r.get(s, {}).get("status") == "ok" else None for s, key in
          [("whisper", "elapsed_seconds"), ("gigaam", "asr_seconds"), ("gigaam", "elapsed_seconds")]]) for r in rows], grouped=True)
    names = {"vad": "VAD", "asr": "ASR", "speaker": "CAM++", "emotion": "Эмоции"}
    stages = [(name, [sum(r["gigaam"].get("stages", {}).get(key, {}).get("seconds", 0) for r in pairs)]) for key, name in names.items()]
    stages.append(("LLM", [sum(r["gigaam"].get("llm_seconds", 0) for r in pairs)]))
    stages_svg = svg_bars("Вызовы этапов GigaAM: времена пересекаются, сумма ≠ полное время", stages)
    charts = {"итоги.svg": total_svg, "по-записям.svg": recording_svg, "этапы-gigaam.svg": stages_svg}
    gpu_svg = gpu_chart(output)
    if gpu_svg:
        charts["gpu.svg"] = gpu_svg
    for name, contents in charts.items():
        (folder / name).write_text(contents, encoding="utf-8")
    percent = ""
    if totals[0] > 0:
        percent = (f"GigaAM ASR: изменение времени {(totals[1] / totals[0] - 1) * 100:+.1f}%. "
                   f"GigaAM весь цикл: {(totals[2] / totals[0] - 1) * 100:+.1f}% относительно Whisper API.")
    html = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Сравнение систем речи</title>'
            '<style>body{font-family:sans-serif;margin:24px;color:#17233a}svg{max-width:100%;height:auto}.chart{overflow:auto}p{max-width:1000px}</style>'
            f'<h1>Сравнение Whisper API и GigaAM</h1><p>Состояние: {escape(conditions["state"])}. '
            f'Записей: {len(rows)}, успешных пар: {len(pairs)}.</p><p>{percent}</p>'
            '<p>Whisper: HTTP-запрос включает передачу аудио, очередь и ответ. GigaAM: локальный конвейер после прогрева. '
            'Тестовый Whisper останавливается перед GigaAM. Продовые контейнеры не управляются; GPU общая. '
            'Скорость не является оценкой точности.</p>'
            '<p>GPU общая с рабочими сервисами. Их нагрузка влияет на время обеих фаз; '
            'график GPU показывает суммарную загрузку всех процессов.</p>'
            + ''.join(f'<div class="chart">{svg}</div>' for svg in charts.values()) + '</html>')
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
        save_reports(output, rows, conditions["state"], mode="standalone-api" if conditions.get("mode") == "isolated_whisper_then_gigaam" else "api")
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
