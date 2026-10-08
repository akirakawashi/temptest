"""Настройки LLM файлового стенда; без загрузки моделей и обращения к API."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings


def llm_runtime(threads: int, cfg: Settings | None = None, seed: int = 42) -> dict:
    # В CPU-образе лежит только benchmark: app доступен лишь GPU-процессу.
    if cfg is None:
        model = os.environ.get("LLM_MODEL", "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:Q4_K_M").strip()
        try:
            context = int(os.environ.get("LLM_NUM_CTX", "8192"))
        except ValueError:
            context = 8192
    else:
        model, context = cfg.llm_model, cfg.llm_num_ctx
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
    return {"model": model,
            "options": {"num_ctx": context, "num_batch": batch,
                        "num_thread": threads, "num_gpu": 999, "seed": seed},
            # Фактический тип кеша и размер батча также видны в логи/ollama.log.
            "kv_cache_type_requested": cache, "flash_attention": True, "parallel": 1}
