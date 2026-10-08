"""Сводка измерений для повторяемого отчёта; только стандартная библиотека."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics

SYSTEMS = {"whisper": "Whisper", "gigaam": "GigaAM полностью", "gigaam_first_line": "GigaAM первая линия"}


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)
    return ({"count": len(values), "total": sum(values), "mean": statistics.mean(values),
             "median": statistics.median(values), "p90": percentile(.9), "p95": percentile(.95),
             "min": min(values), "max": max(values)} if values else {"count": 0})


def write_summary(output: Path, rows: list, conditions: dict) -> dict:
    systems = conditions.get("planned_systems") or (["gigaam"] if conditions.get("mode") == "gigaam_only" else ["whisper", "gigaam"])
    corpus = [{"source": r["audio"].source, "samples": r["audio"].samples, "sha256_pcm": r["audio"].sha256_pcm} for r in rows]
    summary = {"schema_version": 2, "state": conditions["state"], "records": len(rows),
        "audio_seconds": sum(r["audio"].seconds for r in rows),
        "corpus_sha256": hashlib.sha256(json.dumps(corpus, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "systems": {}, "matched_comparison": {},
        "notes": ["Whisper — HTTP-запрос; GigaAM — локальная обработка. Подготовка и прогрев исключены.",
                  "Один и тот же моно WAV для всех режимов, роли исходных каналов не восстанавливаются.",
                  "Загрузка общей GPU включает продовые процессы; влияние на время не исключено.",
                  "Паузы — оценка VAD; скорость и различие текстов не определяют точность.",
                  "Суммы вызовов моделей могут пересекаться. exclusive_wall_percent учитывает пересечения отдельно."]}
    if conditions.get("resume"):
        summary["resume"] = conditions["resume"]
        summary["notes"].append("Замеры Whisper перенесены из предыдущего запуска без изменения; PCM восстановлен и проверен по SHA256. Даты замеров различаются.")
    preparations = {(p.get("directory"), p.get("system")): p for p in conditions.get("preparations", [])}
    csv_rows = []
    for system in systems:
        counts = {key: 0 for key in ("ok", "no_speech", "error", "timeout", "pending")}
        good = []
        for row in rows:
            result = row.get(system, {})
            counts[result.get("status", "pending")] += 1
            if result.get("status") == "ok" and result.get("elapsed_seconds") is not None:
                good.append(row)
            preparation = preparations.get((row["directory"], system), {})
            item = {"ID": Path(row["directory"]).name, "Режим": SYSTEMS[system], "Запись": row["audio"].source,
                "SHA256 PCM": row["audio"].sha256_pcm, "Аудио, с": row["audio"].seconds,
                "Статус": result.get("status", "pending"), "Обработка, с": result.get("elapsed_seconds"),
                "ASR, с": result.get("asr_seconds"), "LLM, с": result.get("llm_seconds"),
                "Загрузка моделей, с": preparation.get("load_seconds"), "Прогрев, с": preparation.get("warmup_seconds"),
                "Рабочий процесс целиком, с": preparation.get("worker_wall_seconds"),
                "Начало замера": preparation.get("measurement_started_at", result.get("request_started_at")),
                "Конец замера": preparation.get("measurement_finished_at", result.get("request_finished_at")), "Ошибка": result.get("error")}
            for key, title in (("vad", "VAD"), ("asr", "ASR"), ("speaker", "Голоса"), ("emotion", "Эмоции")):
                values = result.get("stages", {}).get(key, {})
                for field, label in (("seconds", "с"), ("calls", "вызовов"), ("errors", "ошибок")):
                    item[f"{title}, {label}"] = values.get(field)
            csv_rows.append(item)
        entry = {"label": SYSTEMS[system], "statuses": counts,
                 "processing_seconds": distribution([r[system]["elapsed_seconds"] for r in good]),
                 "audio_seconds_successful": sum(r["audio"].seconds for r in good),
                 "load_seconds": distribution([p["load_seconds"] for p in preparations.values() if p.get("system") == system and "load_seconds" in p]),
                 "warmup_seconds": distribution([p["warmup_seconds"] for p in preparations.values() if p.get("system") == system and "warmup_seconds" in p]),
                 "stages": {}}
        mean = entry["processing_seconds"].get("mean")
        entry["records_per_15_minutes"] = 900 / mean if mean and mean > 0 else None
        for stage in ("vad", "asr", "speaker", "emotion", "llm"):
            enabled = stage in {"vad", "asr"} or system == "gigaam"
            stage_rows = [r[system] for r in good if enabled and ((stage == "llm" and "llm_seconds" in r[system])
                          or (stage != "llm" and stage in r[system].get("stages", {})))]
            values = [result["llm_seconds"] if stage == "llm" else result["stages"][stage]["seconds"] for result in stage_rows]
            all_stage_rows = [r.get(system, {}) for r in rows if enabled and ((stage == "llm" and "llm_seconds" in r.get(system, {}))
                              or (stage != "llm" and stage in r.get(system, {}).get("stages", {})))]
            llm_calls = [call for result in all_stage_rows for call in result.get("stage_calls", []) if call["stage"] == "llm"]
            entry["stages"][stage] = {"seconds": distribution(values),
                "calls": (len(llm_calls) if stage == "llm" else
                          sum(result["stages"][stage]["calls"] for result in all_stage_rows)) if all_stage_rows else None,
                "errors": (sum(bool(call.get("error")) for call in llm_calls) if stage == "llm" else
                           sum(result["stages"][stage]["errors"] for result in all_stage_rows)) if all_stage_rows else None,
                "enabled": enabled, "measured": bool(stage_rows)}
            total_processing = entry["processing_seconds"].get("total", 0)
            entry["stages"][stage]["operation_time_vs_total_percent"] = sum(values) / total_processing * 100 if values and total_processing else None
        timed = [r[system]["timing"] for r in good if "timing" in r[system]]
        if timed:
            keys = timed[0]["exclusive_wall_seconds"]
            total = sum(sum(t["exclusive_wall_seconds"].values()) for t in timed)
            entry["exclusive_wall_percent"] = {key: sum(t["exclusive_wall_seconds"][key] for t in timed) / total * 100 if total else 0 for key in keys}
            entry["mean_record_wall_percent"] = {key: statistics.mean(t["exclusive_wall_percent"][key] for t in timed) for key in keys}
        summary["systems"][system] = entry
    matched = [r for r in rows if all(r.get(s, {}).get("status") == "ok" for s in systems)]
    summary["matched_comparison"]["records"] = len(matched)
    summary["matched_comparison"]["processing_seconds"] = {s: sum(r[s]["elapsed_seconds"] for r in matched) for s in systems}
    baseline = summary["matched_comparison"]["processing_seconds"].get("whisper", 0)
    if baseline:
        summary["matched_comparison"]["time_change_vs_whisper_percent"] = {
            s: (seconds / baseline - 1) * 100 for s, seconds in summary["matched_comparison"]["processing_seconds"].items()}
    (output / "итоги.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if csv_rows:
        with (output / "замеры.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(csv_rows)
    return summary
