"""Каждая запись: Whisper (текст) и GigaAM (полный цикл). Запуск — вручную."""
from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import wave


FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".wma"}
STATUS = {
    "ok": "Успешно", "error": "Ошибка", "timeout": "Превышено время",
    "no_speech": "Речь не найдена; LLM не запускалась", "pending": "Не запускался",
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


def prepare_audio(source: Path, target: Path, *, max_seconds: float | None = None) -> Audio:
    """Единый WAV для обеих систем; преобразование не входит в их таймеры."""
    started = time.perf_counter()
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
                 digest.hexdigest(), time.perf_counter() - started)


def read_audio(audio: Audio):
    import numpy as np

    with wave.open(audio.prepared, "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
    if hashlib.sha256(pcm).hexdigest() != audio.sha256_pcm:
        raise ValueError("Подготовленный звук изменился после проверки контрольной суммы")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def system_order(index: int) -> tuple[str, str]:
    return ("whisper", "gigaam") if index % 2 == 0 else ("gigaam", "whisper")


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
    audio = row["audio"]
    whisper, giga = row.get("whisper", {}), row.get("gigaam", {})
    duration = audio.seconds

    def number(value):
        return "" if value is None else f"{value:.6f}".replace(".", ",")

    def ratio(result, field):
        value = result.get(field)
        return number(value / duration) if result.get("status") == "ok" and value is not None else ""

    return {
        "Запись": audio.source,
        "Длительность, с": number(duration), "Общая подготовка, с": number(audio.preparation_seconds),
        "Порядок": " → ".join(row["order"]),
        "Whisper, с": number(whisper.get("elapsed_seconds")),
        "Whisper — сегменты с текстом, с": number(whisper.get("speech_seconds")),
        "Whisper — HTTP": whisper.get("http_status", ""),
        "Whisper — попыток": whisper.get("attempts", ""),
        "GigaAM распознавание, с": number(giga.get("asr_seconds")),
        "GigaAM полностью, с": number(giga.get("elapsed_seconds")),
        "GigaAM VAD, с": number(giga.get("stages", {}).get("vad", {}).get("seconds")),
        "GigaAM CAM++, с": number(giga.get("stages", {}).get("speaker", {}).get("seconds")),
        "GigaAM эмоции, с": number(giga.get("stages", {}).get("emotion", {}).get("seconds")),
        "GigaAM LLM, с": number(giga.get("llm_seconds")),
        "Whisper / длительность": ratio(whisper, "elapsed_seconds"),
        "GigaAM распознавание / длительность": ratio(giga, "asr_seconds"),
        "GigaAM полностью / длительность": ratio(giga, "elapsed_seconds"),
        "Whisper — статус": STATUS[whisper.get("status", "pending")],
        "GigaAM — статус": STATUS[giga.get("status", "pending")],
        "LLM — текст сокращён": "да" if (giga.get("analysis") or {}).get("truncated") else "нет" if giga.get("analysis") else "",
        "LLM — загрузка в замере, с": number((giga.get("analysis") or {}).get("load_sec")),
        "Причина": "; ".join(r["error"] for r in (whisper, giga) if r.get("error")),
        "Результаты": row["directory"],
    }


def save_reports(output: Path, rows: list[dict], state: str, *, mode: str | None = None) -> None:
    if rows:
        csv_rows = [csv_row(row) for row in rows]
        with (output / "сводка.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(csv_rows)
    pairs = [row for row in rows if all(row.get(s, {}).get("status") == "ok" for s in ("whisper", "gigaam"))]
    api_mode = mode in {"api", "standalone-api"} or any(row.get("whisper", {}).get("mode") in {"existing_server_api", "test_server_api"} for row in rows)
    standalone = mode == "standalone-api" or any(row.get("whisper", {}).get("mode") == "test_server_api" for row in rows)
    lines = ["# Сравнение Whisper и полного цикла GigaAM", "", f"Состояние прогона: **{state}**.", "",
             "Одна NVIDIA GPU, один файл за раз. У GigaAM в полный цикл входят VAD, распознавание, голоса, эмоции и LLM.",
             "Общая подготовка, загрузка моделей, прогрев и запись отчётов исключены из замеров.",
             "Whisper — faster-whisper large-v3 на CUDA/float16 (если модель не изменена параметром запуска).",
             "GigaAM ASR — официальный v3_e2e_rnnt, CUDA, fp16 encoder / fp32 head.",
             "Каждый замер выполняется в отдельном процессе после прогрева; другая система выгружена из GPU.", "",
             f"Записей в корпусе: {len(rows)}. Полностью успешных пар: {len(pairs)}.", "",
             "| Запись | Аудио, с | Whisper, с | GigaAM ASR, с | GigaAM всё, с | Статус Whisper / GigaAM |",
             "|---|---:|---:|---:|---:|---|"]

    def fmt(result, key):
        value = result.get(key)
        return "—" if value is None else f"{value:.3f}"

    for row in rows:
        w, g = row.get("whisper", {}), row.get("gigaam", {})
        name = Path(row["audio"].source).name.replace("|", "\\|").replace("\n", " ")
        status = f"{STATUS[w.get('status', 'pending')]} / {STATUS[g.get('status', 'pending')]}"
        lines.append(f"| [{name}]({row['directory']}/) | {row['audio'].seconds:.3f} | {fmt(w, 'elapsed_seconds')} | "
                     f"{fmt(g, 'asr_seconds')} | {fmt(g, 'elapsed_seconds')} | {status} |")
    if pairs:
        audio_seconds = sum(row["audio"].seconds for row in pairs)
        lines.extend(["", "Итоги только по полностью успешным парам:", ""])
        for title, system, key in (("Whisper", "whisper", "elapsed_seconds"),
                                   ("GigaAM — распознавание", "gigaam", "asr_seconds"),
                                   ("GigaAM — полный цикл", "gigaam", "elapsed_seconds")):
            elapsed = sum(row[system][key] for row in pairs)
            lines.append(f"- {title}: {elapsed:.3f} с; время / длительность аудио = {elapsed / audio_seconds:.4f}.")
    lines.extend(["", "Времена строк с ошибкой — время до отказа, они не входят в итоговое сравнение.",
                  "Времена отдельных этапов GigaAM могут пересекаться: их сумма не равна полному времени.",
                  "Подробный журнал — `прогон.log`, параметры и версии — `условия.json`, таблица — `сводка.csv`.", ""])
    if api_mode:
        lines[lines.index("Whisper — faster-whisper large-v3 на CUDA/float16 (если модель не изменена параметром запуска).")] = (
            "Whisper — существующий сервис через API; модель и доступные сведения /health записаны в условия.json.")
        lines[lines.index("Каждый замер выполняется в отдельном процессе после прогрева; другая система выгружена из GPU.")] = (
            "Whisper: существующий сервис через API, его модель остаётся в GPU; затем весь корпус GigaAM в отдельных процессах.")
        lines.extend(["Whisper — время HTTP-запроса, включая загрузку WAV, очередь, распознавание и получение ответа.",
                      "Встроенные параметры и провайдер VAD рабочего Whisper стенд не меняет и не определяет по /health.",
                      "Прогрев GigaAM исключён; дополнительного распознавания для прогрева рабочего Whisper нет.",
                      "Это сравнение рабочего сервиса с локальным полным конвейером. Посторонняя нагрузка и порядок фаз влияют на результат.", ""])
    if standalone:
        replacements = {
            "Whisper — существующий сервис через API; модель и доступные сведения /health записаны в условия.json.":
                "Whisper — собственная копия закреплённого образа рабочего сервиса; модель CUDA/float16 и фактические версии проверены через API.",
            "Whisper: существующий сервис через API, его модель остаётся в GPU; затем весь корпус GigaAM в отдельных процессах.":
                "Весь корпус Whisper после прогрева; затем тестовый Whisper остановлен, весь корпус GigaAM в отдельных процессах. Контейнеры сохраняются.",
            "Встроенные параметры и провайдер VAD рабочего Whisper стенд не меняет и не определяет по /health.":
                "Тестовый Whisper использует штатный VAD образа; распознавание — GPU, подготовка звука и часть вспомогательных операций — CPU.",
            "Прогрев GigaAM исключён; дополнительного распознавания для прогрева рабочего Whisper нет.":
                "Загрузка моделей и прогрев обеих систем исключены из измеряемых времён; GigaAM ASR измеряется отдельно от полного цикла.",
            "Это сравнение рабочего сервиса с локальным полным конвейером. Посторонняя нагрузка и порядок фаз влияют на результат.":
                "Продовые контейнеры и их API не используются. GPU общая: посторонняя нагрузка и порядок фаз влияют на результат.",
        }
        lines = [replacements.get(line, line) for line in lines]
    (output / "отчёт.md").write_text("\n".join(lines), encoding="utf-8")


def save_result(output: Path, row: dict, system: str, result: dict) -> None:
    directory = output / row["directory"]
    write_json(directory / f"{system}.json", result)
    (directory / f"{system}.txt").write_text(result.get("text", "") + "\n", encoding="utf-8")
    elapsed = result.get("elapsed_seconds")
    log.info("%s: %s; время %s", system, STATUS[result["status"]],
             "не измерено" if elapsed is None else f"{elapsed:.3f} с")
    if result.get("error"):
        log.error("%s", result["error"])
    if result.get("response_error"):
        log.error("Whisper — ответ при ошибке:\n%s", result["response_error"])
    if system == "gigaam":
        STAGES = {"vad": "Выделение речи", "asr": "Распознавание речи", "speaker": "Разделение собеседников", "emotion": "Определение эмоций"}

        for stage, values in result.get("stages", {}).items():
            log.info("GigaAM — %s: %.6f с, вызовов %d, ошибок %d", STAGES[stage], values["seconds"], values["calls"], values["errors"])
        log.info("GigaAM — разбор LLM: %.6f с", result.get("llm_seconds", 0))
        for call in result.get("stage_calls", []):
            log.info("GigaAM — вызов модели: %s", json.dumps(call, ensure_ascii=False))
        emotions = {"angry": "раздражение", "sad": "грусть", "neutral": "нейтрально", "positive": "позитив"}
        for utterance in result.get("utterances", []):
            emotion = utterance.get("emotion") or {}
            log.info("GigaAM — реплика %d, %.2f–%.2f с, собеседник %s, эмоция: %s: %s",
                     utterance["id"], utterance["start"], utterance["end"], utterance["speaker"],
                     emotions.get(emotion.get("label"), "не определена"), utterance["text"])
        for event in result.get("events", []):
            if event["type"] == "error":
                log.error("GigaAM — событие ошибки: %s", event["message"])
            elif event["type"] == "relabel":
                log.info("GigaAM — объединены собеседники %s → %s", event["from"], event["to"])
        analysis = result.get("analysis") or {}
        if analysis.get("result"):
            log.info("GigaAM — результат разбора LLM:\n%s", json.dumps(analysis["result"], ensure_ascii=False, indent=2))
        if analysis.get("truncated"):
            log.warning("GigaAM — середина разговора сокращена для контекста LLM; это отмечено в сводке")
    else:
        log.info("Whisper — текст:\n%s", result.get("text", ""))
        for segment in result.get("segments", []):
            log.info("Whisper — сегмент: %s", json.dumps(segment, ensure_ascii=False))
            if isinstance(segment, dict) and isinstance(segment.get("compression_ratio"), (int, float)) and segment["compression_ratio"] > 2.4:
                log.warning("Whisper — высокая повторяемость текста: compression_ratio=%s; нужна проверка аудио", segment["compression_ratio"])
        if result.get("mode") in {"existing_server_api", "test_server_api"}:
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
    result.add_argument("--threads", type=positive_int, default=int(os.environ.get("BENCH_THREADS", "4")), help="Потоки CPU для подготовки и вспомогательных операций")
    result.add_argument("--whisper-model", default="large-v3", help="Модель faster-whisper или локальный путь к весам")
    result.add_argument("--whisper-cache", default="/cache/whisper")
    result.add_argument("--beam-size", type=positive_int, default=5)
    result.add_argument("--timeout", type=positive_int, default=3600, help="Максимальное ожидание подготовки и одного прогона, секунды")
    result.add_argument("--warmup-file", type=Path, help="Отдельная запись с речью для прогрева; по умолчанию первая запись корпуса")
    result.add_argument("--warmup-seconds", type=positive_int, default=30)
    result.add_argument("--phase", choices=["local", "whisper-api", "gigaam"], default="local")
    result.add_argument("--expected-files", type=positive_int, default=int(os.environ.get("BENCH_EXPECTED_FILES", "100")))
    result.add_argument("--api-gap", type=float, default=float(os.environ.get("BENCH_API_GAP", "5")))
    result.add_argument("--api-timeout", type=positive_int, default=int(os.environ.get("BENCH_API_TIMEOUT", "1200")))
    result.add_argument("--max-audio-seconds", type=positive_int, default=1800)
    return result


async def run_worker(request: dict) -> dict:
    directory = Path(request["output"])
    request_path = directory / f"{request['system']}-задание.json"
    response_path = directory / f"{request['system']}-ответ.json"
    process_log_path = directory / f"{request['system']}-процесс.log"
    write_json(request_path, request)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-X", "faulthandler", "-m", "benchmark.worker", str(request_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

    async def capture_output():
        with process_log_path.open("wb") as stream:
            while chunk := await process.stdout.read(65536):
                stream.write(chunk)
                stream.flush()
                sys.stdout.write(chunk.decode("utf-8", "replace"))
                sys.stdout.flush()

    output_task = asyncio.create_task(capture_output())
    try:
        # Три фазы: загрузка, прогрев, замер. Зависший native-код прекращаем
        # перед запуском другой системы, чтобы измерения не пересекались.
        await asyncio.wait_for(process.wait(), timeout=request["timeout"] * 3 + 180)
    except (TimeoutError, asyncio.CancelledError) as exc:
        if process.returncode is None:
            process.kill()
        await process.wait()
        if isinstance(exc, asyncio.CancelledError):
            raise
        return {"result": {"status": "timeout", "elapsed_seconds": None, "text": "",
                           "process_log": str(process_log_path),
                           "error": "Превышен предел ожидания рабочего процесса; процесс остановлен"},
                "preparation": {"system": request["system"], "included_in_measurements": False}}
    finally:
        await output_task
    if process.returncode or not response_path.is_file():
        detail = f"кодом {process.returncode}"
        if process.returncode is not None and process.returncode < 0:
            try:
                detail += f" ({signal.Signals(-process.returncode).name})"
            except ValueError:
                pass
        preparation_path = directory / f"{request['system']}-подготовка.json"
        preparation = (json.loads(preparation_path.read_text(encoding="utf-8")) if preparation_path.is_file()
                       else {"system": request["system"], "included_in_measurements": False})
        return {"result": {"status": "error", "elapsed_seconds": None, "text": "",
                           "process_log": str(process_log_path),
                           "error": f"Процесс {request['system']} завершился с {detail}; стек и журнал: {process_log_path}"},
                "preparation": preparation}
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    request_path.unlink(missing_ok=True)
    response_path.unlink(missing_ok=True)
    return payload


async def run(args) -> int:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)

    files = sorted(p.resolve() for p in args.audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in FORMATS)
    if not files:
        log.error("В папке %s нет поддерживаемых записей", args.audio_dir)
        return 1
    rows: list[dict] = []
    state = "Подготовка"
    conditions = {
        "started_at": datetime.now().astimezone().isoformat(), "host": host_info(), "threads": args.threads,
        "device": "cuda:0", "selected_gpu": os.environ.get("BENCH_GPU", "0"),
        "input": "mono_16000_pcm_s16le", "concurrency": 1,
        "order": "Чередуется: Whisper → GigaAM, затем GigaAM → Whisper",
        "whisper": {"engine": "faster-whisper", "model": args.whisper_model, "compute_type": "float16",
                    "beam_size": args.beam_size, "vad_filter": True, "word_timestamps": True},
        "gigaam": {"asr_model": "v3_e2e_rnnt", "asr_backend": "официальный PyTorch CUDA",
                   "llm_options": {"num_thread": args.threads, "num_gpu": 999, "seed": 42},
                   "ollama_flash_attention": True, "ollama_parallel": 1},
        "isolation": "Отдельный процесс на систему и запись; другая система и её LLM выгружены",
        "preparations": [],
        "timers": {"clock": "perf_counter", "whisper": "Вызов распознавания и полное получение сегментов",
                   "gigaam_total": "Подача PCM, VAD, все реплики, голоса, эмоции и завершённый ответ LLM",
                   "gigaam_asr": "Сумма всех вызовов GigaAM-v3, включая пустые ответы",
                   "excluded": "Общая подготовка, загрузка и прогрев моделей, запись файлов отчёта"},
    }
    try:
        affinity = conditions["host"]["cpu_affinity"]
        if affinity and len(affinity) < args.threads:
            raise ValueError("Число потоков превышает доступные CPU; уменьшите BENCH_THREADS и CPUSET")
        log.info("Найдено записей: %d. NVIDIA GPU, %d вспомогательных потоков CPU, один файл за раз", len(files), args.threads)
        log.info("Общая подготовка аудио находится вне замеров систем")
        for index, source in enumerate(files):
            directory = f"{index + 1:04d}"
            (args.out / directory).mkdir()
            audio = prepare_audio(source, args.out / directory / "audio.wav")
            row = {"audio": audio, "directory": directory, "order": system_order(index)}
            rows.append(row)
            log.info("Подготовлена запись %s: %.3f с, SHA256 PCM %s; подготовка %.3f с",
                     source.name, audio.seconds, audio.sha256_pcm, audio.preparation_seconds)
        conditions["corpus"] = [asdict(row["audio"]) for row in rows]
        write_json(args.out / "условия.json", conditions)
        save_reports(args.out, rows, state)

        warm_source = (args.warmup_file or files[0]).resolve()
        warm = prepare_audio(warm_source, args.out / "прогрев.wav", max_seconds=args.warmup_seconds)
        conditions["warmup"] = {"audio": asdict(warm), "before_every_measurement": True,
                                "included_in_measurements": False}
        write_json(args.out / "условия.json", conditions)
        log.info("Для каждого замера — одинаковый прогрев на %.3f с из %s, вне таблицы", warm.seconds, warm_source.name)

        state = "Выполняется"
        for index, row in enumerate(rows):
            log.info("Запись %d/%d: %s; порядок %s", index + 1, len(rows), row["audio"].source, " → ".join(row["order"]))
            for system in row["order"]:
                payload = await run_worker({"system": system, "audio": asdict(row["audio"]),
                    "warmup": asdict(warm), "output": str(args.out / row["directory"]), "log_dir": str(args.out),
                    "threads": args.threads, "timeout": args.timeout, "whisper_model": args.whisper_model,
                    "whisper_cache": args.whisper_cache, "beam_size": args.beam_size})
                preparation = {"directory": row["directory"], **payload["preparation"]}
                conditions["preparations"].append(preparation)
                if preparation.get("gpu"):
                    if conditions.get("gpu") and conditions["gpu"] != preparation["gpu"]:
                        raise RuntimeError("GPU изменилась между системами; сравнение остановлено")
                    conditions["gpu"] = preparation["gpu"]
                write_json(args.out / "условия.json", conditions)
                result = payload["result"]
                row[system] = result
                save_result(args.out, row, system, result)
                save_reports(args.out, rows, state)
                if result["status"] in {"error", "timeout"}:
                    # Нативная модель может ещё работать после таймаута: другую не запускаем.
                    state = "Остановлен из-за ошибки"
                    log.error("Прогон остановлен; повторы и следующие системы не запускаются")
                    return 1
        state = "Завершён"
        log.info("Все записи обработаны. Результаты: %s", args.out)
        return 0
    except asyncio.CancelledError:
        state = "Прерван пользователем"
        log.warning("Прогон прерван; рабочий процесс остановлен, следующие системы не запускаются")
        raise
    except Exception as exc:
        state = "Остановлен из-за ошибки"
        log.exception("Не удалось завершить прогон (%s): %s", type(exc).__name__, exc)
        return 1
    finally:
        conditions["state"] = state
        conditions["finished_at"] = datetime.now().astimezone().isoformat()
        write_json(args.out / "условия.json", conditions)
        save_reports(args.out, rows, state)


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    args.audio_dir, args.out = args.audio_dir.resolve(), args.out.resolve()
    if args.phase != "local":
        from benchmark.server_run import main as server_main

        return server_main(args, cli)
    if not args.audio_dir.is_dir():
        cli.error("--audio-dir должен указывать на существующую папку")
    if args.out.exists():
        cli.error("Папка --out уже существует; выберите новую, чтобы сохранить предыдущий прогон")
    if args.audio_dir == args.out or args.audio_dir in args.out.parents:
        cli.error("Папка отчёта должна находиться вне папки исходных записей")
    if args.warmup_file and not args.warmup_file.is_file():
        cli.error("Файл --warmup-file не найден")
    args.out.mkdir(parents=True)
    setup_logging(args.out)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        log.warning("Остановлено пользователем; сохранённые результаты находятся в %s", args.out)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
