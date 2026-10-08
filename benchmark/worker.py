"""Одна система и одна запись в отдельном процессе: VRAM освобождается при выходе."""
from __future__ import annotations

import asyncio
import faulthandler
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import sys
import time

from benchmark.compare import Audio, STATUS, read_audio, setup_logging, write_json

log = logging.getLogger("сравнение")


async def execute(request: dict) -> dict:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(request["threads"])
    from app.config import Settings
    from benchmark.gpu import gpu_info, require_idle_ollama
    from benchmark.pipeline import GigaPipeline, WhisperPipeline, versions

    system = request["system"]
    output = Path(request["output"])
    runner = None
    llm_prepared = False
    result = None
    measurement_started = None
    preparation = {"system": system, "included_in_measurements": False}
    try:
        log.info("%s: проверка GPU, PID %d", system, os.getpid())
        preparation["gpu"] = gpu_info()
        preparation["versions"] = versions()
        preparation["phase"] = "Перед загрузкой моделей"
        write_json(output / f"{system}-подготовка.json", preparation)
        log.info("%s: GPU %s; версии библиотек %s", system, preparation["gpu"], preparation["versions"])
        warm_samples = read_audio(Audio(**request["warmup"]))
        samples = read_audio(Audio(**request["audio"]))
        log.info("%s: загрузка моделей и прогрев — вне замеров; другая система выгружена", system)
        started = time.perf_counter()
        if system == "whisper":
            await require_idle_ollama(Settings().ollama_url, request["timeout"])
            runner = WhisperPipeline(request["whisper_model"], request["whisper_cache"],
                                     request["threads"], request["beam_size"])
            await asyncio.wait_for(asyncio.to_thread(runner.load), request["timeout"])
            preparation["placement"] = {"asr": "CTranslate2 CUDA:0, float16", "vad": runner.vad_providers}
        else:
            cfg = Settings(asr_threads=request["threads"], spk_threads=request["threads"],
                           emo_threads=request["threads"], emo_enabled=True, llm_enabled=True, llm_keep_alive=-1)
            cfg.llm_autopull = False
            runner = GigaPipeline(cfg, request["threads"], request["timeout"])
            preparation["settings"] = asdict(cfg)
            preparation["llm_runtime"] = runner.llm_runtime
            await require_idle_ollama(cfg.ollama_url, request["timeout"])
            await asyncio.wait_for(runner.load(), request["timeout"])
            llm_prepared = True
            preparation["placement"] = runner.engines.placement()
        preparation["load_seconds"] = time.perf_counter() - started
        preparation["phase"] = "Перед прогревом"
        write_json(output / f"{system}-подготовка.json", preparation)
        log.info("%s: начало прогрева", system)
        if system == "whisper":
            warm_result = await asyncio.wait_for(asyncio.to_thread(runner.run, warm_samples), request["timeout"])
        else:
            warm_result = await runner.run(warm_samples)
        write_json(output / f"{system}-прогрев.json", warm_result)
        if warm_result["status"] != "ok":
            advice = " Для прогрева выберите файл с речью через --warmup-file." if warm_result["status"] == "no_speech" else ""
            raise RuntimeError(f"Прогрев {system} не завершён: {warm_result.get('error') or STATUS[warm_result['status']]}.{advice}")
        preparation["warmup_seconds"] = warm_result["elapsed_seconds"]
        log.info("%s: прогрев завершён за %.3f с", system, preparation["warmup_seconds"])
        if system == "gigaam":
            from benchmark.gpu import log_gpu_memory

            log_gpu_memory("прогрева полного GigaAM")
            preparation["ollama_loaded_models"] = await runner.gpu_model_info()
            log.info("GigaAM: ASR, голоса, эмоции и VAD — CUDA; LLM — 100% GPU")
        else:
            log.info("Whisper: распознавание и VAD — CUDA")
        preparation["versions"] = {**versions(), **(runner.engines.versions if system == "gigaam" else {})}
        preparation["phase"] = "Перед замером"
        write_json(output / f"{system}-подготовка.json", preparation)
        log.info("%s — начало измеряемой обработки", system)
        measurement_started = time.perf_counter()
        if system == "whisper":
            result = await asyncio.wait_for(asyncio.to_thread(runner.run, samples), request["timeout"])
        else:
            result = await runner.run(samples)
            if result["status"] == "ok":
                result["ollama_loaded_models"] = await runner.gpu_model_info()
    except Exception as exc:
        log.exception("Ошибка %s: %s", system, exc)
        if result is None:
            result = {"elapsed_seconds": None if measurement_started is None else time.perf_counter() - measurement_started,
                      "text": "", "preparation_failed": measurement_started is None}
        result.update(status="timeout" if isinstance(exc, TimeoutError) else "error", error=str(exc))
    finally:
        if runner:
            log.info("%s: начало освобождения моделей", system)
            runner.shutdown()
            log.info("%s: модели освобождены", system)
            if system == "gigaam" and llm_prepared:
                try:
                    await runner.unload_llm()
                    log.info("LLM выгружена из GPU перед следующей системой")
                except Exception as exc:
                    log.exception("Не удалось освободить GPU от LLM: %s", exc)
                    if result is not None:
                        result.update(status="error", error=f"Не удалось выгрузить LLM: {exc}")
        write_json(output / f"{system}-подготовка.json", preparation)
    return {"result": result, "preparation": preparation}


def main() -> int:
    # Пишем в stderr: родитель сохраняет весь поток, включая SIGSEGV и ошибки
    # нативных библиотек, которые не проходят через logging/except.
    faulthandler.enable(all_threads=True)
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    setup_logging(Path(request["log_dir"]))
    payload = asyncio.run(execute(request))
    write_json(Path(request["output"]) / f"{request['system']}-ответ.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
