"""Разбор разговора текстовой LLM через Ollama (работает в соседнем контейнере)."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Awaitable, Callable, Optional

import httpx

from .config import Settings
from .engines import Engines

log = logging.getLogger("llm")

SYSTEM_PROMPT = """Ты — аналитик качества разговоров. Тебе дают расшифровку разговора с пометками \
времени, имён собеседников и эмоций, определённых по голосу. Расшифровка сделана автоматически \
и может содержать ошибки распознавания — не придирайся к опечаткам.

Верни ТОЛЬКО один JSON-объект без пояснений со следующими полями:
{
  "summary": "суть разговора в 2–4 предложениях",
  "topic": "тема разговора одной фразой",
  "outcome": "чем закончился разговор: решён ли вопрос, о чём договорились",
  "speakers": [{"name": "имя как в расшифровке", "role": "предполагаемая роль", "tone": "как держался в разговоре"}],
  "agreements": ["договорённости и обещания"],
  "next_steps": ["что нужно сделать после разговора и кому"],
  "risks": ["конфликтные моменты, недовольство, риски"],
  "quality": {"score": число от 1 до 5, "comment": "оценка вежливости и результативности разговора"}
}
Пиши по-русски, кратко и по делу. Если данных для поля нет — верни пустой список или пустую строку. \
Не выдумывай факты, которых нет в расшифровке."""


def clip_transcript(text: str, max_chars: int) -> str:
    """Если разговор не помещается в контекст модели — оставить начало и конец."""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 3]
    tail = text[-(max_chars - len(head)):]
    head = head[: head.rfind("\n")] if "\n" in head else head
    tail = tail[tail.find("\n") + 1:] if "\n" in tail else tail
    return head + "\n[... середина разговора пропущена: не помещается в контекст модели ...]\n" + tail


class LLM:
    def __init__(self, cfg: Settings, engines: Engines):
        self.cfg = cfg
        self.eng = engines
        self.lock = asyncio.Lock()        # один разбор одновременно
        self.progress: Optional[float] = None

    def _set(self, state: str, detail: str = "") -> None:
        self.eng._set("llm", state, detail)

    @property
    def ready(self) -> bool:
        return self.eng.components["llm"].state == "ready"

    # ------------------------------------------------------------ подготовка
    async def prepare(self) -> None:
        """Дождаться Ollama и при необходимости скачать модель. Запускается при старте.

        Если сервер или модель недоступны, попытки продолжаются в фоне: когда Ollama
        поднимется позже, разбор заработает без перезапуска приложения.
        """
        cfg = self.cfg
        if not cfg.llm_enabled:
            self._set("off", "отключено в настройках (LLM_ENABLED=false)")
            return
        self._set("loading", "ожидание сервера Ollama")
        waited = 0.0
        while True:
            pause = 2.0 if waited < 60 else 15.0
            try:
                if await self._prepare_once():
                    return
                pause = 30.0                    # сервер отвечает, но модель не готова — не частим
            except asyncio.CancelledError:
                raise
            except httpx.TransportError:
                # сервер ещё не поднялся или соединение оборвалось
                if waited >= 60:
                    self._set("error", f"сервер Ollama недоступен по адресу {cfg.ollama_url}; пробую снова каждые 15 с")
            except Exception as exc:
                log.exception("LLM prepare failed")
                self._set("error", f"{type(exc).__name__}: {exc}"[:300])
                pause = 30.0
            waited += pause
            await asyncio.sleep(pause)

    async def _prepare_once(self) -> bool:
        """Одна попытка. True — модель готова; False — не готова, попробовать позже."""
        cfg = self.cfg
        async with httpx.AsyncClient(base_url=cfg.ollama_url, timeout=httpx.Timeout(30.0, read=None)) as cli:
            r = await cli.get("/api/version", timeout=5.0)
            r.raise_for_status()
            self.eng.versions["ollama"] = r.json().get("version", "?")
            r = await cli.post("/api/show", json={"model": cfg.llm_model, "name": cfg.llm_model}, timeout=30.0)
            if r.status_code == 200:
                self._set("ready", "модель загружена")
                return True
            if not cfg.llm_autopull:
                self._set("error", f"модель не скачана; выполните: ollama pull {cfg.llm_model}")
                return False
            return await self._pull(cli)

    async def _pull(self, cli: httpx.AsyncClient) -> bool:
        self._set("loading", "скачивание модели: 0%")
        last_pct = -1
        async with cli.stream("POST", "/api/pull", json={"model": self.cfg.llm_model, "name": self.cfg.llm_model, "stream": True}) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:200]
                self._set("error", f"Ollama отказала в скачивании: {resp.status_code} {body}")
                return False
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("error"):
                    self._set("error", f"скачивание не удалось: {str(msg['error'])[:260]}")
                    return False
                total, done = msg.get("total"), msg.get("completed")
                if total and total > 50_000_000 and done is not None:      # мелкие служебные слои не показываем
                    pct = int(done * 100 / total)
                    if pct != last_pct:
                        last_pct = pct
                        self.progress = pct / 100
                        self._set("loading", f"скачивание модели: {pct}% из {total / 1e9:.1f} ГБ")
                if msg.get("status") == "success":
                    self.progress = None
                    self._set("ready", "модель загружена")
                    return True
        self._set("error", "скачивание модели прервалось; будет продолжено")
        return False

    # ---------------------------------------------------------------- разбор
    async def analyze(
        self, transcript: str, emit: Callable[[dict], Awaitable[None]],
        *, options: Optional[dict] = None,
    ) -> None:
        """Запросить разбор и передавать ход генерации в интерфейс через emit."""
        cfg = self.cfg
        if not self.ready:
            c = self.eng.components["llm"]
            await emit({"type": "llm", "state": "error", "message": f"LLM не готова: {c.detail or c.state}"})
            return
        if not transcript.strip():
            await emit({"type": "llm", "state": "error", "message": "Пока нет ни одной реплики для разбора."})
            return
        if self.lock.locked():
            await emit({"type": "llm", "state": "error", "message": "Разбор уже выполняется, дождитесь результата."})
            return

        # русский текст — примерно 3 символа на токен; оставляем место под инструкцию и ответ
        budget_chars = max(2000, int((cfg.llm_num_ctx - 2300) * 2.6))
        clipped = clip_transcript(transcript, budget_chars)
        body = {
            "model": cfg.llm_model,
            "stream": True,
            "format": "json",
            # В JSON число означает секунды, строка — длительность с единицей.
            # Переменные окружения всегда строки: "-1" нужно отправлять как -1.
            "keep_alive": (int(cfg.llm_keep_alive) if str(cfg.llm_keep_alive).lstrip("+-").isdigit()
                           else cfg.llm_keep_alive),
            "options": {"temperature": 0.2, "num_ctx": cfg.llm_num_ctx, "num_predict": 1400},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Расшифровка разговора:\n\n" + clipped},
            ],
        }
        # Файловый стенд задаёт устройство, бюджет потоков и фиксированный seed.
        # При обычном запуске интерфейса остаются исходные параметры.
        body["options"].update(options or {})
        async with self.lock:
            started = time.time()
            await emit({"type": "llm", "state": "running", "chars": 0, "truncated": clipped != transcript})
            text, last_emit, stats = "", 0.0, {}
            try:
                async with httpx.AsyncClient(base_url=cfg.ollama_url, timeout=httpx.Timeout(30.0, read=600.0)) as cli:
                    async with cli.stream("POST", "/api/chat", json=body) as resp:
                        if resp.status_code != 200:
                            err = (await resp.aread()).decode("utf-8", "replace")[:300]
                            raise RuntimeError(f"Ollama ответила {resp.status_code}: {err}")
                        async for line in _lines(resp):
                            msg = json.loads(line)
                            if msg.get("error"):
                                raise RuntimeError(str(msg["error"]))
                            text += (msg.get("message") or {}).get("content", "")
                            if msg.get("done"):
                                stats = msg
                                break
                            if time.time() - last_emit > 0.4:
                                last_emit = time.time()
                                await emit({"type": "llm", "state": "running", "chars": len(text),
                                            "elapsed": round(time.time() - started, 1)})
            except Exception as exc:
                log.exception("LLM analyze failed")
                await emit({"type": "llm", "state": "error", "message": f"{type(exc).__name__}: {exc}"[:400]})
                return

        result = None
        try:
            result = json.loads(text)
            if not isinstance(result, dict):
                result = None
        except json.JSONDecodeError:
            result = None
        eval_count = stats.get("eval_count") or 0
        eval_ns = stats.get("eval_duration") or 0
        await emit(
            {
                "type": "llm",
                "state": "done",
                "result": result,
                "raw": None if result is not None else text,
                "elapsed": round(time.time() - started, 1),
                "tokens_out": eval_count,
                "tokens_in": stats.get("prompt_eval_count") or 0,
                "tokens_per_sec": round(eval_count / (eval_ns / 1e9), 1) if eval_ns else None,
                "load_sec": round((stats.get("load_duration") or 0) / 1e9, 1),
                "truncated": clipped != transcript,
                "completed": bool(stats.get("done")),
                "done_reason": stats.get("done_reason"),
            }
        )


async def _lines(resp: httpx.Response) -> AsyncIterator[str]:
    async for line in resp.aiter_lines():
        if line.strip():
            yield line
