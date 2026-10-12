"""Корпус через собственный тестовый Whisper; подготовка звука общая для всех тестов стенда."""
from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import wave


FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".wma"}
STATUS = {
    "ok": "Успешно", "error": "Ошибка", "timeout": "Превышено время",
    "no_speech": "Речь не найдена", "pending": "Не запускался",
}
log = logging.getLogger("сравнение")


class RussianFormatter(logging.Formatter):
    """Русские сообщения нашего приложения; технические причины сохраняются дословно."""
    MESSAGES = {
        "VAD load failed": "Ошибка загрузки детектора речи",
        "ASR load failed": "Ошибка загрузки распознавания GigaAM",
        "speaker model load failed": "Ошибка загрузки модели собеседников",
        "emotion model load failed (attempt %d/%d)": "Ошибка загрузки эмоций (попытка %d/%d)",
        "status callback failed": "Ошибка уведомления о состоянии моделей",
        "segment processing failed": "Ошибка обработки фрагмента",
        "emotion failed: %s": "Ошибка определения эмоции: %s",
        "LLM prepare failed": "Ошибка подготовки LLM",
        "LLM analyze failed": "Ошибка разбора разговора LLM",
    }
    LEVELS = {"INFO": "ИНФО", "WARNING": "ПРЕДУПРЕЖДЕНИЕ", "ERROR": "ОШИБКА", "DEBUG": "ОТЛАДКА"}

    def format(self, record):
        from copy import copy

        localized = copy(record)
        localized.msg = self.MESSAGES.get(record.msg, record.msg) if isinstance(record.msg, str) else record.msg
        localized.levelname = self.LEVELS.get(record.levelname, record.levelname)
        return super().format(localized)


def setup_logging(output: Path) -> None:
    formatter = RussianFormatter("%(asctime)s | %(levelname)s | %(name)s | PID %(process)d | %(message)s")
    handlers = [logging.FileHandler(output / "прогон.log", encoding="utf-8"), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    # Журнал стенда содержит этапы и ошибки; низкоуровневые HTTP-запросы избыточны.
    for name in ("httpx", "httpcore", "faster_whisper", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass
class Audio:
    source: str
    prepared: str
    seconds: float
    samples: int
    sha256_pcm: str
    preparation_seconds: float
    source_channels: int | None = None
    source_sample_rate: int | None = None
    source_bytes: int | None = None


def source_audio_info(source: Path) -> tuple[int | None, int | None]:
    try:
        if source.suffix.lower() == ".wav":
            with wave.open(str(source), "rb") as stream:
                return stream.getnchannels(), stream.getframerate()
        import av

        with av.open(str(source)) as container:
            stream = container.streams.audio[0]
            return stream.codec_context.channels, stream.codec_context.sample_rate
    except (ImportError, OSError, IndexError, ValueError, wave.Error) as exc:
        log.warning("Не удалось получить каналы и частоту исходной записи %s: %s", source.name, exc)
        return None, None


def prepare_audio(source: Path, target: Path, *, max_seconds: float | None = None) -> Audio:
    """Единый WAV для всех систем; преобразование не входит в их таймеры."""
    started = time.perf_counter()
    source_channels, source_rate = source_audio_info(source)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-vn"]
    if max_seconds is not None:
        command.extend(["-t", str(max_seconds)])
    command.extend(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)])
    if shutil.which("ffmpeg"):
        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if completed.returncode:
            detail = completed.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"Не удалось подготовить запись {source}: {detail}")
    else:
        # PyAV поставляется с faster-whisper в готовом CUDA-образе.
        from faster_whisper.audio import decode_audio

        log.info("Подготовка %s: PyAV из образа Whisper, моно PCM16 16 кГц; вне замеров", source)
        samples = decode_audio(str(source), sampling_rate=16000)
        if max_seconds is not None:
            samples = samples[:int(max_seconds * 16000)]
        with wave.open(str(target), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes((samples * 32768).clip(-32768, 32767).astype("<i2").tobytes())
    with wave.open(str(target), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (1, 16000, 2):
            raise ValueError("Подготовленный звук должен быть моно, 16 кГц, PCM 16 бит")
        samples = wav.getnframes()
        digest = hashlib.sha256()
        actual_bytes = 0
        while block := wav.readframes(16000 * 60):
            digest.update(block)
            actual_bytes += len(block)
    if samples <= 0 or actual_bytes != samples * 2:
        raise ValueError(f"Пустая или повреждённая запись: {source}")
    return Audio(str(source), str(target), samples / 16000, samples,
                 digest.hexdigest(), time.perf_counter() - started,
                 source_channels, source_rate, source.stat().st_size)


def read_audio(audio: Audio):
    import numpy as np

    with wave.open(audio.prepared, "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
    if hashlib.sha256(pcm).hexdigest() != audio.sha256_pcm:
        raise ValueError("Подготовленный звук изменился после проверки контрольной суммы")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def host_info() -> dict:
    result = {
        "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }
    for key, path in {
        "cpu_quota": "/sys/fs/cgroup/cpu.max", "cpu_set": "/sys/fs/cgroup/cpuset.cpus.effective",
        "memory_limit": "/sys/fs/cgroup/memory.max",
    }.items():
        try:
            result[key] = Path(path).read_text().strip()
        except OSError:
            result[key] = None
    try:
        result["cpu_model"] = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                   if line.startswith("model name"))
    except (OSError, StopIteration):
        result["cpu_model"] = platform.processor()
    return result


def csv_row(row: dict) -> dict:
    audio, whisper = row["audio"], row.get("whisper", {})

    def number(value):
        return "" if value is None else f"{value:.6f}".replace(".", ",")

    elapsed = whisper.get("elapsed_seconds")
    return {
        "Запись": audio.source,
        "Длительность, с": number(audio.seconds), "Общая подготовка, с": number(audio.preparation_seconds),
        "Whisper, с": number(elapsed),
        "Whisper — сегменты с текстом, с": number(whisper.get("speech_seconds")),
        "Whisper — HTTP": whisper.get("http_status", ""),
        "Whisper — попыток": whisper.get("attempts", ""),
        "Whisper / длительность": number(elapsed / audio.seconds) if whisper.get("status") == "ok" and elapsed is not None else "",
        "Whisper — статус": STATUS[whisper.get("status", "pending")],
        "Причина": whisper.get("error") or "",
        "Результаты": row["directory"],
    }


def save_reports(output: Path, rows: list[dict], state: str) -> None:
    if rows:
        csv_rows = [csv_row(row) for row in rows]
        with (output / "сводка.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(csv_rows)
    measured = [row for row in rows if row.get("whisper", {}).get("status") == "ok"]
    lines = ["# Whisper на корпусе записей", "", f"Состояние прогона: **{state}**.", "",
             "Одна NVIDIA GPU, один файл за раз. Подготовка звука, загрузка модели, прогрев и запись отчётов исключены из замеров.",
             "Whisper — собственная копия закреплённого образа рабочего сервиса; модель CUDA/float16 и фактические версии проверены через API.",
             "Время — весь HTTP-запрос: передача WAV, обработка и получение ответа. Чистое время распознавания сервер не сообщает.",
             "Тестовый Whisper использует штатный VAD образа; распознавание — GPU, подготовка звука и часть вспомогательных операций — CPU.",
             "Продовые контейнеры и их API не используются. GPU общая: посторонняя нагрузка влияет на результат.", "",
             f"Записей в корпусе: {len(rows)}. Успешных: {len(measured)}.", "",
             "| Запись | Аудио, с | Whisper, с | Статус |",
             "|---|---:|---:|---|"]
    for row in rows:
        whisper = row.get("whisper", {})
        name = Path(row["audio"].source).name.replace("|", "\\|").replace("\n", " ")
        elapsed = whisper.get("elapsed_seconds")
        lines.append(f"| [{name}]({row['directory']}/) | {row['audio'].seconds:.3f} | "
                     f"{'—' if elapsed is None else f'{elapsed:.3f}'} | {STATUS[whisper.get('status', 'pending')]} |")
    if measured:
        seconds = sum(row["whisper"]["elapsed_seconds"] for row in measured)
        audio_seconds = sum(row["audio"].seconds for row in measured)
        lines.extend(["", "Итог только по успешным записям:", "",
                      f"- Whisper: {seconds:.3f} с; среднее {seconds / len(measured):.3f} с; "
                      f"время / длительность аудио = {seconds / audio_seconds:.4f}; "
                      f"за 15 минут ≈ {900 * len(measured) / seconds:.1f} записей этого корпуса."])
    lines.extend(["", "Время строки с ошибкой — время до отказа, в итог оно не входит.",
                  "Подробный журнал — `прогон.log`, параметры и версии — `условия.json`, таблица — `сводка.csv`.", ""])
    (output / "отчёт.md").write_text("\n".join(lines), encoding="utf-8")


def save_result(output: Path, row: dict, result: dict) -> None:
    directory = output / row["directory"]
    write_json(directory / "whisper.json", result)
    (directory / "whisper.txt").write_text(result.get("text", "") + "\n", encoding="utf-8")
    elapsed = result.get("elapsed_seconds")
    log.info("Whisper: %s; время %s", STATUS[result["status"]],
             "не измерено" if elapsed is None else f"{elapsed:.3f} с")
    if result.get("error"):
        log.error("%s", result["error"])
    if result.get("response_error"):
        log.error("Whisper — ответ при ошибке:\n%s", result["response_error"])
    log.info("Whisper — текст:\n%s", result.get("text", ""))
    for segment in result.get("segments", []):
        log.info("Whisper — сегмент: %s", json.dumps(segment, ensure_ascii=False))
        if isinstance(segment, dict) and isinstance(segment.get("compression_ratio"), (int, float)) and segment["compression_ratio"] > 2.4:
            log.warning("Whisper — высокая повторяемость текста: compression_ratio=%s; нужна проверка аудио", segment["compression_ratio"])
    log.info("Whisper API — HTTP %s, попыток %s, найденные сегменты %.3f с, ответ %s байт; чистое ASR-время сервер не сообщал",
             result.get("http_status"), result.get("attempts"), result.get("speech_seconds", 0), result.get("response_bytes"))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Значение должно быть положительным")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--audio-dir", type=Path, required=True, help="Папка записей; вложенные папки тоже обрабатываются")
    result.add_argument("--out", type=Path, default=Path("benchmark-results") / datetime.now().strftime("%Y%m%d-%H%M%S-%f"), help="Новая папка отчёта")
    result.add_argument("--warmup-file", type=Path, help="Отдельная запись с речью для прогрева; по умолчанию первая запись корпуса")
    result.add_argument("--warmup-seconds", type=positive_int, default=30)
    result.add_argument("--expected-files", type=positive_int, default=int(os.environ.get("BENCH_EXPECTED_FILES", "100")))
    result.add_argument("--api-gap", type=float, default=float(os.environ.get("BENCH_API_GAP", "5")))
    result.add_argument("--api-timeout", type=positive_int, default=int(os.environ.get("BENCH_API_TIMEOUT", "1200")))
    result.add_argument("--max-audio-seconds", type=positive_int, default=1800)
    return result


def main(argv: list[str] | None = None) -> int:
    from benchmark.server_run import whisper_phase

    cli = parser()
    args = cli.parse_args(argv)
    args.audio_dir, args.out = args.audio_dir.resolve(), args.out.resolve()
    if not args.audio_dir.is_dir() or args.audio_dir == args.out or args.audio_dir in args.out.parents:
        cli.error("Нужна папка аудио; результаты должны находиться вне неё")
    if args.api_gap < 5:
        cli.error("Пауза между запросами Whisper должна быть не меньше 5 секунд")
    if (args.out / "условия.json").exists():
        cli.error("Прогон уже существует; выберите новую папку")
    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(args.out)
    try:
        return asyncio.run(whisper_phase(args))
    except KeyboardInterrupt:
        log.warning("Прерван пользователем; запрос Whisper не повторяется, сервер может продолжать обработку")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
