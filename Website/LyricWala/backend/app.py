"""Run from backend/: uvicorn app:app --host 127.0.0.1 --port 8000

The frontend is served independently on port 5173. This service loads the model
once at startup and sends real generation progress as newline-delimited JSON.
Use one Uvicorn worker per model/GPU with this in-process concurrency guard.
"""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import threading

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from engine import DemoEngine, DiffusionEngine, GenerationCancelled
from schemas import GenerateRequest

load_dotenv(Path(__file__).with_name(".env"))
logger = logging.getLogger("lyricwala")


@asynccontextmanager
async def lifespan(application: FastAPI):
    mode = os.environ.get("LYRICWALA_MODE", "demo").lower()
    if mode not in {"demo", "model"}:
        raise ValueError("LYRICWALA_MODE must be demo or model.")
    # A bad checkpoint fails startup visibly; it never silently returns demo lyrics.
    application.state.engine = await asyncio.to_thread(DiffusionEngine if mode == "model" else DemoEngine)
    application.state.generation_lock = threading.Lock()
    yield


app = FastAPI(title="LyricWala", version="1.0.0", lifespan=lifespan)
allowed_origins = [s.strip() for s in os.environ.get("LYRICWALA_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if s.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Accept"])


@app.get("/api/health")
async def health():
    return {"status": "ready", "mode": app.state.engine.mode, "busy": app.state.generation_lock.locked()}


@app.post("/api/generate")
async def generate(payload: GenerateRequest):
    lock = app.state.generation_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(429, "The studio is finishing another take. Please try again in a moment.")
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    cancelled = threading.Event()

    def publish(event):
        if not cancelled.is_set():
            loop.call_soon_threadsafe(events.put_nowait, event)

    def run_generation():
        try:
            result = app.state.engine.generate(payload, publish, cancelled)
            publish({"type": "result", **result})
        except GenerationCancelled:
            pass
        except ValueError as error:
            publish({"type": "error", "message": str(error)})
        except Exception:
            logger.exception("Lyric generation failed")
            publish({"type": "error", "message": "The model could not finish this take. Try again, or check the service logs."})
        finally:
            # The lock belongs to the actual worker, not the browser connection.
            # Disconnecting cannot let a second job overlap a still-running GPU pass.
            lock.release()
            if not loop.is_closed():
                loop.call_soon_threadsafe(events.put_nowait, None)

    # Schedule immediately so even a client disconnecting before the first chunk
    # cannot strand an acquired lock without an associated worker.
    worker = loop.run_in_executor(None, run_generation)

    async def stream():
        try:
            while True:
                event = await events.get()
                if event is None:
                    break
                yield json.dumps(event, ensure_ascii=False) + "\n"
        finally:
            cancelled.set()
            # Thread work cooperatively stops between denoising steps. Do not
            # cancel the future or release the model lock ahead of that work.
            _ = worker

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})
