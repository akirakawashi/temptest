"""Настройки LLM файлового стенда; без загрузки моделей и обращения к API."""
from __future__ import annotations

import os

from app.config import Settings


def llm_runtime(threads: int, cfg: Settings | None = None, seed: int = 42) -> dict:
    cfg = cfg or Settings()
    value = os.environ.get("BENCH_LLM_NUM_BATCH", "64")
    try:
        batch = int(value)
    except ValueError:
        raise ValueError("BENCH_LLM_NUM_BATCH должен быть целым числом от 32 до 1024") from None
    if not 32 <= batch <= 1024:
        raise ValueError("BENCH_LLM_NUM_BATCH должен быть целым числом от 32 до 1024")
    cache = os.environ.get("BENCH_LLM_KV_CACHE_TYPE", "f16")
    if cache not in {"f16", "q8_0", "q4_0"}:
        raise ValueError("BENCH_LLM_KV_CACHE_TYPE должен быть f16, q8_0 или q4_0")
    return {"model": cfg.llm_model,
            "options": {"num_ctx": cfg.llm_num_ctx, "num_batch": batch,
                        "num_thread": threads, "num_gpu": 999, "seed": seed},
            # Фактический тип кеша и размер батча также видны в логи/ollama.log.
            "kv_cache_type_requested": cache, "flash_attention": True, "parallel": 1}
