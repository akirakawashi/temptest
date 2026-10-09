"""Одна запись официального T-one; загрузка и прогрев отдельно от замера."""
from __future__ import annotations

from datetime import datetime
import faulthandler
import logging
import os
from pathlib import Path
import resource
import sys
import time
import json

from benchmark.compare import Audio, read_audio, setup_logging
from benchmark.tone_run import save_json
from benchmark.tone_trt_prepare import gpu_snapshot

log = logging.getLogger("тестовый-t-one-trt")


def statistics_delta(before, after, requests):
    def find(stats):
        return next((row.get("inference_stats", {}) for row in stats.get("model_stats", [])
            if row.get("name") == "streaming_acoustic" and row.get("version") == "1"), {})
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


def execute(request):
    from benchmark.tone_trt import TensorRTPipeline

    directory, system = Path(request["output"]), request["system"]
    preparation = {"system": system, "included_in_measurements": False, "pid": os.getpid(),
        "started_at": datetime.now().astimezone().isoformat(), "settings": request["protocol"]}
    result = {"status": "error", "error": None, "text": "", "elapsed_seconds": None}
    runner = None
    try:
        preparation["gpu"] = gpu_snapshot()
        samples, warm = read_audio(Audio(**request["audio"])), read_audio(Audio(**request["warmup"]))
        save_json(directory / f"{system}-подготовка.json", preparation)
        runner = TensorRTPipeline(system, Path(request["model_dir"]), request["timeout"])
        log.info("%s: загрузка клиента и декодера — вне замеров; GPU: %s", system, preparation["gpu"])
        started = time.perf_counter()
        runner.load()
        preparation.update(load_seconds=time.perf_counter() - started, versions=runner.versions,
            placement={"acoustic": "TensorRT, Triton GPU:0", "decoder": "CPU", "splitter": "CPU"},
            server_metadata=runner.server_metadata, model_metadata=runner.model_metadata,
            model_config=runner.config, engine_sha256=runner.engine_metadata["engine_sha256"])
        save_json(directory / f"{system}-подготовка.json", preparation)
        log.info("%s: начало прогрева — вне замеров", system)
        warmed = runner.run(warm)
        save_json(directory / f"{system}-прогрев.json", warmed)
        if warmed["status"] != "ok":
            raise RuntimeError(f"Прогрев не завершён: {warmed.get('error') or warmed['status']}; нужна запись с речью")
        preparation.update(warmup_seconds=warmed["elapsed_seconds"], gpu_after_warmup=gpu_snapshot())
        preparation["server_statistics_before"] = runner.statistics()
        preparation["measurement_started_at"] = datetime.now().astimezone().isoformat()
        log.info("%s — начало измеряемой обработки всей записи, контекст сохраняется", system)
        result = runner.run(samples)
        preparation["measurement_finished_at"] = datetime.now().astimezone().isoformat()
        preparation["server_statistics_after"] = runner.statistics()
        result["triton_statistics_delta"] = statistics_delta(preparation["server_statistics_before"],
            preparation["server_statistics_after"], result["triton_requests"])
        preparation["gpu_after_measurement"] = gpu_snapshot()
        log.info("%s: статус %s; обработка %.3f с; запросов Triton %d; этапы %s", system,
                 result["status"], result["elapsed_seconds"], result["triton_requests"], result["trt_stages"])
        for phrase in result["raw_phrases"]:
            log.info("%s — фраза: %s", system, phrase)
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


def main():
    faulthandler.enable(all_threads=True)
    request = json.loads(Path(sys.argv[1]).read_text())
    setup_logging(Path(request["log_dir"]))
    payload = execute(request)
    save_json(Path(request["output"]) / f"{request['system']}-ответ.json", payload)
    return 0 if payload["result"]["status"] in {"ok", "no_speech"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
