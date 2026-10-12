"""Одна запись GigaAM TensorRT; загрузка клиента и прогрев отдельно от замера."""
from __future__ import annotations

import asyncio
from datetime import datetime
import faulthandler
import json
import logging
import os
from pathlib import Path
import resource
import sys
import time

from benchmark.compare import Audio, read_audio, setup_logging
from benchmark.trt_common import save_json
from benchmark.tone_trt_prepare import gpu_snapshot

log = logging.getLogger("тестовый-gigaam-trt")


def statistics_delta(before: dict, after: dict, requests: int, model: str) -> dict:
    def find(stats):
        return next((row.get("inference_stats", {}) for row in stats.get("model_stats", [])
            if row.get("name") == model and row.get("version") == "1"), {})
    old, new = find(before), find(after)
    if not old or not new:
        return {"available": False, "note": "Сервер не вернул счётчики; исходная статистика сохранена"}
    delta = {"available": True}
    for key in ("success", "fail", "queue", "compute_input", "compute_infer", "compute_output"):
        count = int(new.get(key, {}).get("count", 0)) - int(old.get(key, {}).get("count", 0))
        nanoseconds = int(new.get(key, {}).get("ns", 0)) - int(old.get(key, {}).get("ns", 0))
        if count < 0 or nanoseconds < 0:
            raise RuntimeError("Счётчики Triton сбросились во время записи; сервер перезапускался")
        delta[key] = {"calls": count, "seconds": nanoseconds / 1e9}
    delta["matches_client_requests"] = delta["success"]["calls"] == requests
    if not delta["matches_client_requests"]:
        raise RuntimeError("Число успешных запросов Triton отличается от клиента; замер требует проверки")
    return delta


async def execute(request: dict) -> dict:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(request["threads"])
    from app.config import Settings
    from benchmark.gigaam_trt import MODEL, GigaTrtPipeline

    directory, system = Path(request["output"]), request["system"]
    preparation = {"system": system, "included_in_measurements": False, "pid": os.getpid(),
        "started_at": datetime.now().astimezone().isoformat(), "settings": request["protocol"],
        "enabled_stages": ["vad", "asr"]}
    result = {"status": "error", "error": None, "text": "", "elapsed_seconds": None}
    runner = None
    try:
        preparation["gpu"] = gpu_snapshot()
        samples, warm = read_audio(Audio(**request["audio"])), read_audio(Audio(**request["warmup"]))
        save_json(directory / f"{system}-подготовка.json", preparation)
        threads = request["threads"]
        # Настройки берутся у исходного монитора; его голоса, эмоции и LLM здесь выключены.
        cfg = Settings(asr_threads=threads, emo_enabled=False, llm_enabled=False, split_turns=False,
                       **request["vad_settings"])
        runner = GigaTrtPipeline(cfg, threads, request["timeout"], Path(request["model_dir"]))
        log.info("%s: загрузка клиента, декодера и поиска речи — вне замеров; GPU: %s", system, preparation["gpu"])
        started = time.perf_counter()
        await asyncio.wait_for(runner.load(), request["timeout"])
        engines = runner.engines
        preparation.update(load_seconds=time.perf_counter() - started, versions=engines.versions,
            vad_settings=request["vad_settings"], placement=engines.placement(),
            server_metadata=engines.server_metadata, model_metadata=engines.model_metadata,
            model_config=engines.config, engine_sha256=engines.engine_metadata["engine_sha256"])
        save_json(directory / f"{system}-подготовка.json", preparation)
        log.info("%s: начало прогрева — вне замеров", system)
        warmed = await runner.run(warm)
        save_json(directory / f"{system}-прогрев.json", warmed)
        if warmed["status"] != "ok":
            raise RuntimeError(f"Прогрев не завершён: {warmed.get('error') or warmed['status']}; нужна запись с речью")
        preparation.update(warmup_seconds=warmed["elapsed_seconds"], gpu_after_warmup=gpu_snapshot())
        preparation["server_statistics_before"] = runner.statistics()
        preparation["measurement_started_at"] = datetime.now().astimezone().isoformat()
        log.info("%s — начало измеряемой обработки записи", system)
        result = await runner.run(samples)
        preparation["measurement_finished_at"] = datetime.now().astimezone().isoformat()
        preparation["server_statistics_after"] = runner.statistics()
        result["triton_statistics_delta"] = statistics_delta(preparation["server_statistics_before"],
            preparation["server_statistics_after"], result["triton_requests"], MODEL)
        preparation["gpu_after_measurement"] = gpu_snapshot()
        log.info("%s: статус %s; обработка %.3f с; распознавание %.3f с; запросов Triton %d; части %s", system,
                 result["status"], result["elapsed_seconds"], result["asr_seconds"], result["triton_requests"], result["trt_stages"])
        for utterance in result["utterances"]:
            log.info("%s — реплика %.2f–%.2f с: %s", system, utterance["start"], utterance["end"], utterance["text"])
        if result["status"] not in {"ok", "no_speech"}:
            raise RuntimeError(result["error"])
    except Exception as exc:
        log.exception("Ошибка %s", system)
        result.update(status="timeout" if isinstance(exc, TimeoutError) else "error", error=str(exc),
            preparation_failed="measurement_started_at" not in preparation)
    finally:
        if runner:
            runner.close()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        preparation.update(finished_at=datetime.now().astimezone().isoformat(),
            process_usage={"peak_rss_mib": usage.ru_maxrss / 1024,
                "cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime})
        save_json(directory / f"{system}-подготовка.json", preparation)
    return {"result": result, "preparation": preparation}


def main() -> int:
    faulthandler.enable(all_threads=True)
    request = json.loads(Path(sys.argv[1]).read_text())
    setup_logging(Path(request["log_dir"]))
    payload = asyncio.run(execute(request))
    save_json(Path(request["output"]) / f"{request['system']}-ответ.json", payload)
    return 0 if payload["result"]["status"] in {"ok", "no_speech"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
