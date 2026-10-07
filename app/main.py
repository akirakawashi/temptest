"""Веб-сервер: статика интерфейса, WebSocket для звука и событий, служебные API."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional, Set
from urllib.parse import urlparse

import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import SAMPLE_RATE, settings
from .engines import EMOTIONS, Engines
from .llm import LLM
from .session import Session

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
APP_VERSION = "1.1.0"

engines = Engines(settings)
llm: Optional[LLM] = None
clients: Set["Client"] = set()
proc = psutil.Process()
started_at = time.time()


class Client:
    """Одно подключение браузера."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.lock = asyncio.Lock()
        self.session: Optional[Session] = None
        self.alive = True

    async def send(self, msg: dict) -> None:
        if not self.alive:
            return
        try:
            async with self.lock:
                await self.ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            self.alive = False


def hello_payload() -> dict:
    return {
        "type": "hello",
        "version": APP_VERSION,
        "sample_rate": SAMPLE_RATE,
        "emotions": list(EMOTIONS),
        "components": engines.status(),
        "versions": engines.versions,
        "config": {
            "vad_threshold": settings.vad_threshold,
            "vad_min_silence": settings.vad_min_silence,
            "vad_max_speech": settings.vad_max_speech,
            "speaker_threshold": settings.speaker_threshold,
            "max_speakers": settings.max_speakers,
            "split_turns": settings.split_turns,
            "asr_threads": settings.asr_threads,
            "emo_threads": settings.emo_threads,
            "llm_model": settings.llm_model,
            "llm_num_ctx": settings.llm_num_ctx,
            "cpu_count": psutil.cpu_count() or 1,
            "mem_total_mb": int(psutil.virtual_memory().total / 2**20),
        },
    }


async def broadcast_status() -> None:
    msg = {"type": "status", "components": engines.status(), "versions": engines.versions}
    for c in list(clients):
        await c.send(msg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    loop = asyncio.get_running_loop()
    engines.on_change = lambda: loop.call_soon_threadsafe(lambda: asyncio.ensure_future(broadcast_status()))

    def load() -> None:
        engines.load_core()
        engines.load_emotions()

    threading.Thread(target=load, name="model-loader", daemon=True).start()
    llm = LLM(settings, engines)
    prepare = asyncio.create_task(llm.prepare())
    proc.cpu_percent(None)
    psutil.cpu_percent(None)
    yield
    prepare.cancel()


app = FastAPI(title="Речевой монитор", version=APP_VERSION, lifespan=lifespan)


@app.get("/api/status")
async def api_status() -> JSONResponse:
    return JSONResponse(
        {
            "ok": engines.core_ready,
            "version": APP_VERSION,
            "uptime_sec": int(time.time() - started_at),
            "components": engines.status(),
            "clients": len(clients),
        }
    )


def system_metrics() -> dict:
    vm = psutil.virtual_memory()
    cores = psutil.cpu_count() or 1
    return {
        "cpu_app": round(proc.cpu_percent(None) / cores, 1),     # доля всех ядер, занятая приложением
        "cpu_sys": round(psutil.cpu_percent(None), 1),
        "rss_mb": int(proc.memory_info().rss / 2**20),
        "mem_used_pct": round(vm.percent, 1),
        "mem_avail_mb": int(vm.available / 2**20),
        "threads": proc.num_threads(),
        "uptime_sec": int(time.time() - started_at),
    }


async def metrics_loop(client: Client) -> None:
    while client.alive:
        await asyncio.sleep(1.0)
        payload = {"type": "metrics", **system_metrics()}
        if client.session:
            payload.update(client.session.counters())
        await client.send(payload)


def same_origin(ws: WebSocket) -> bool:
    """Принимать соединения только со страниц самого приложения.

    Иначе любой сайт, открытый в том же браузере, мог бы подключиться к локальному сервису.
    """
    if os.environ.get("WS_ALLOW_ANY_ORIGIN", "").lower() in ("1", "true", "yes"):
        return True
    origin = ws.headers.get("origin")
    if not origin:
        return True                      # не браузер (скрипт, проверка)
    host = ws.headers.get("x-forwarded-host") or ws.headers.get("host") or ""
    return urlparse(origin).netloc.lower() == host.lower()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    if not same_origin(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    client = Client(ws)
    clients.add(client)
    session = Session(engines, settings, client.send)
    client.session = session
    session.start()
    ticker = asyncio.create_task(metrics_loop(client))
    await client.send(hello_payload())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if engines.core_ready:
                    await session.feed(msg["bytes"])
                continue
            text = msg.get("text")
            if not text:
                continue
            try:
                cmd = json.loads(text)
            except json.JSONDecodeError:
                continue
            kind = cmd.get("type")
            if kind == "stop":
                await session.flush()
            elif kind == "reset":
                await session.reset()
            elif kind == "rename":
                try:
                    sid = int(cmd.get("speaker"))
                except (TypeError, ValueError):
                    continue
                name = str(cmd.get("name") or "").strip()[:40]
                if name:
                    session.names[sid] = name
                else:
                    session.names.pop(sid, None)
            elif kind == "config":
                if "speaker_threshold" in cmd:
                    try:
                        thr = min(0.8, max(0.15, float(cmd["speaker_threshold"])))
                    except (TypeError, ValueError):
                        continue
                    session.speaker_threshold = thr
                    session.clusterer.threshold = thr
                if "split_turns" in cmd:
                    session.split_turns = bool(cmd["split_turns"])
                await client.send({"type": "config", "speaker_threshold": session.speaker_threshold,
                                   "split_turns": session.split_turns})
            elif kind == "analyze" and llm is not None:
                asyncio.create_task(llm.analyze(session.transcript(), client.send))
            elif kind == "ping":
                await client.send({"type": "pong", "t": cmd.get("t")})
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket failed")
    finally:
        client.alive = False
        clients.discard(client)
        ticker.cancel()
        await session.close()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), headers={"Cache-Control": "no-cache"})


class RevalidatedStatic(StaticFiles):
    """Файлы интерфейса с заголовком Cache-Control: no-cache.

    Без него браузер сам решает, сколько держать скрипты и стили в кэше, и после
    обновления образа может собрать страницу из новой разметки и старых скриптов —
    такая страница не запускается. С no-cache браузер каждый раз сверяет ETag
    (на localhost это мгновенно) и берёт новый файл, как только он изменился.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Адрес /assets, а не прежний /static: у тех, кто открывал первую версию, под /static
# в кэше браузера могли остаться её файлы без no-cache.
app.mount("/assets", RevalidatedStatic(directory=STATIC_DIR), name="assets")
