"""Run from backend/: pytest -q. No GPU or model download required."""

import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import pytest

import app as service
from engine import DemoEngine, GenerationCancelled
from schemas import GenerateRequest


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LYRICWALA_MODE", "demo")
    with TestClient(service.app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ready", "mode": "demo", "busy": False}


@pytest.mark.parametrize("payload", [{"prompt": " "}, {"prompt": "x" * 601}, {"prompt": "Hi", "creativity": 101}, {"prompt": "Hi", "genre": "Invalid"}, {"prompt": "Hi", "extra": True}])
def test_invalid_requests(client, payload):
    assert client.post("/api/generate", json=payload).status_code == 422
    assert not service.app.state.generation_lock.locked()


def test_stream(client):
    response = client.post("/api/generate", json={"prompt": "A midnight drive"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    progress = [e["progress"] for e in events if e["type"] == "progress"]
    assert progress == sorted(progress)
    assert progress[-1] == 100
    assert events[-1]["type"] == "result"
    assert events[-1]["demo"] is True
    assert "[Chorus]" in events[-1]["lyrics"]
    assert not client.get("/api/health").json()["busy"]


def test_busy(client):
    service.app.state.generation_lock.acquire()
    try:
        assert client.post("/api/generate", json={"prompt": "Hello"}).status_code == 429
    finally:
        service.app.state.generation_lock.release()


def test_engine_failure_is_streamed_and_unlocks(client):
    class BrokenEngine:
        mode = "model"

        def generate(self, request, report, cancelled):
            raise ValueError("Shorten the prompt.")

    service.app.state.engine = BrokenEngine()
    response = client.post("/api/generate", json={"prompt": "Hello"})
    assert response.json() == {"type": "error", "message": "Shorten the prompt."}
    assert not service.app.state.generation_lock.locked()


def test_cors(client):
    response = client.options("/api/generate", headers={"Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    rejected = client.options("/api/generate", headers={"Origin": "https://unrelated.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in rejected.headers


def test_demo_cancellation():
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(GenerationCancelled):
        DemoEngine().generate(GenerateRequest(prompt="Hello"), lambda event: None, cancelled)


def test_conditioning():
    text = GenerateRequest(prompt="  A new start  ", genre="Rock", mood="Bold").model_prompt()
    assert text.endswith("A new start")
    assert "bold Rock" in text
