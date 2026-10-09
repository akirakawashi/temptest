"""Одна запись T-one в отдельном процессе; загрузка и прогрев вне замера."""
from __future__ import annotations

import asyncio
import faulthandler
import json
import logging
import os
import resource
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from benchmark.compare import Audio, read_audio, setup_logging
from benchmark.tone_run import save_json

log = logging.getLogger("тестовый-t-one")


async def execute(request: dict) -> dict:
    from app.config import Settings
    from benchmark.gpu import gpu_info, log_gpu_memory
    from benchmark.pipeline import versions
    from benchmark.tone import TonePipeline

    output = Path(request["output"])
    preparation = {"system": "tone", "included_in_measurements": False,
        "started_at": datetime.now().astimezone().isoformat(), "pid": os.getpid()}
    runner = None
    result = {"status": "error", "error": None, "text": "", "elapsed_seconds": None}
    try:
        preparation["gpu"] = gpu_info()
        preparation["versions"] = versions()
        preparation["phase"] = "Перед загрузкой моделей"
        save_json(output / "tone-подготовка.json", preparation)
        log.info("T-one: GPU %s; версии %s", preparation["gpu"], preparation["versions"])
        samples = read_audio(Audio(**request["audio"]))
        warm = read_audio(Audio(**request["warmup"]))
        cfg = Settings(asr_threads=request["threads"], emo_enabled=False, llm_enabled=False,
                       split_turns=False, **request["vad_settings"])
        preparation["settings"] = asdict(cfg)
        runner = TonePipeline(cfg, request["threads"], request["timeout"], Path(request["model_dir"]))
        log.info("T-one: загрузка моделей и прогрев — вне замеров")
        started = time.perf_counter()
        await asyncio.wait_for(runner.load(), request["timeout"])
        preparation.update(load_seconds=time.perf_counter() - started, phase="Перед прогревом",
            placement=runner.engines.placement(), model_sha256=runner.engines.model_hashes,
            versions={**preparation["versions"], **runner.engines.versions})
        save_json(output / "tone-подготовка.json", preparation)
        log.info("T-one: начало прогрева")
        warmed = await runner.run(warm)
        save_json(output / "tone-прогрев.json", warmed)
        if warmed["status"] != "ok":
            raise RuntimeError(f"Прогрев T-one не завершён: {warmed.get('error') or warmed['status']}. "
                               "Для записи с речью укажите --warmup-file внутри /recordings")
        preparation.update(warmup_seconds=warmed["elapsed_seconds"], phase="Перед замером",
            gpu_after_warmup=log_gpu_memory("прогрева T-one"))
        save_json(output / "tone-подготовка.json", preparation)
        log.info("T-one — начало измеряемой обработки")
        preparation["measurement_started_at"] = datetime.now().astimezone().isoformat()
        result = await runner.run(samples)
        preparation["measurement_finished_at"] = datetime.now().astimezone().isoformat()
        preparation["gpu_after_measurement"] = log_gpu_memory("замера T-one")
        log.info("T-one: статус %s; весь цикл %.3f с; распознавание %.3f с; текст %s",
                 result["status"], result["elapsed_seconds"], result["asr_seconds"], result["text"])
    except Exception as exc:
        log.exception("Ошибка T-one")
        result.update(status="timeout" if isinstance(exc, TimeoutError) else "error", error=str(exc),
                      preparation_failed="measurement_started_at" not in preparation)
    finally:
        if runner:
            runner.shutdown()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        preparation.update(finished_at=datetime.now().astimezone().isoformat(),
            process_usage={"peak_rss_mib": usage.ru_maxrss / 1024,
                           "cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime})
        save_json(output / "tone-подготовка.json", preparation)
    return {"result": result, "preparation": preparation}


def main():
    faulthandler.enable(all_threads=True)
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    setup_logging(Path(request["log_dir"]))
    payload = asyncio.run(execute(request))
    save_json(Path(request["output"]) / "tone-ответ.json", payload)
    return 0 if payload["result"]["status"] in {"ok", "no_speech"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
