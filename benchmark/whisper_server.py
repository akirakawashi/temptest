"""Тестовая копия API из готового образа Whisper; запуск выполняет владелец."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark.compare import RussianFormatter

log = logging.getLogger("тестовый-whisper")
EXPECTED = {"faster-whisper": "1.2.1", "ctranslate2": "4.8.1"}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--download", action="store_true", help="Только скачать веса; без GPU и записей")
    args = cli.parse_args()
    handler = logging.StreamHandler()
    handler.setFormatter(RussianFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise RuntimeError(f"Версии тестового Whisper {versions} отличаются от ожидаемых {EXPECTED}")
    model, cache = os.environ.get("WHISPER_MODEL", "large-v3"), "/var/lib/whisper"
    log.info("Модель %s; версии %s; собственный кеш %s", model, versions, cache)
    if args.download:
        from faster_whisper.utils import download_model

        download_model(model, cache_dir=cache)
        log.info("Веса Whisper подготовлены; GPU и папка записей не подключены")
        return

    import ctranslate2
    import uvicorn

    if ctranslate2.get_cuda_device_count() != 1:
        raise RuntimeError("Тестовый Whisper должен видеть ровно одну GPU; переход на CPU запрещён")
    if os.environ.get("WHISPER_DEVICE") != "cuda" or os.environ.get("WHISPER_COMPUTE_TYPE") != "float16":
        raise RuntimeError("Тестовому Whisper нужны WHISPER_DEVICE=cuda и WHISPER_COMPUTE_TYPE=float16")
    if os.environ.get("WHISPER_LOCAL_ONLY") != "1":
        raise RuntimeError("При обработке записей разрешены только заранее скачанные локальные веса")

    # Запускаем API напрямую, обходя загрузки/внешние запросы vendor run.sh.
    sys.path.insert(0, "/opt/src")
    import api_server

    metadata = {"stand": "speech-comparison", "model": model, "versions": versions,
                "image": os.environ.get("BENCH_WHISPER_IMAGE"), "model_loaded": False,
                "language": os.environ.get("WHISPER_LANGUAGE"),
                "beam_size": int(os.environ.get("WHISPER_BEAM", "5")),
                "cpu_threads": int(os.environ.get("WHISPER_THREADS", "2")),
                "vad_filter": True, "vad_provider": "Штатный faster-whisper; провайдер не изменён",
                "diarization": False, "external_network": False,
                "api_source_sha256": hashlib.sha256(Path(api_server.__file__).read_bytes()).hexdigest()}
    original_load = api_server._load_model

    def checked_load():
        started = time.perf_counter()
        original_load()
        actual = api_server._model.model
        if actual.device != "cuda" or actual.compute_type != "float16":
            raise RuntimeError(f"Whisper загрузился на {actual.device}/{actual.compute_type}; прогон запрещён")
        # Контейнер видит только выбранную GPU; её физический номер может быть не 0.
        free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                                           "--format=csv,noheader,nounits"], text=True, timeout=10).strip())
        reserve = int(os.environ.get("BENCH_GPU_RESERVE_MIB", "4096"))
        if free < reserve:
            raise RuntimeError(f"После загрузки Whisper свободно {free} МиБ GPU; нужен резерв {reserve} МиБ")
        metadata.update(model_loaded=True, device=actual.device, compute_type=actual.compute_type,
                        load_seconds=time.perf_counter() - started, gpu_free_mib_after_load=free)
        log.info("Whisper загружен: CUDA/float16, %.3f с вне замеров, свободно %d МиБ GPU", metadata["load_seconds"], free)

    api_server._load_model = checked_load

    @api_server.app.get("/benchmark/metadata", include_in_schema=False)
    async def benchmark_metadata():
        return metadata

    # Собственная внутренняя сеть, без опубликованных портов и ключей прода.
    uvicorn.run(api_server.app, host="0.0.0.0", port=9000, workers=1, log_level="info")


if __name__ == "__main__":
    main()
