import pytest
from fastapi.testclient import TestClient

import app as app_module
from src.config import ConfigurationError
from src.graph import GenerationError, RetrievalError

RESULT = {"final_answer": "Answer (page 1).",
          "retrieved_context": [{"content": "text", "page": 1, "relevance_score": 0.8,
                                 "source": "Ebook-Agentic-AI.pdf", "chunk_id": "a"}],
          "confidence_score": 0.8, "citations": [{"source": "Ebook-Agentic-AI.pdf", "page": 1}]}


def client(monkeypatch, runner=None, graph_error=None):
    def fake_get_default_graph():
        if graph_error:
            raise graph_error
        return object()
    monkeypatch.setattr(app_module, "get_default_graph", fake_get_default_graph)
    monkeypatch.setattr(app_module, "run_graph", runner or (lambda q, g: RESULT))
    return TestClient(app_module.app)


def test_health(monkeypatch):
    with client(monkeypatch) as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_chat_ok(monkeypatch):
    with client(monkeypatch) as c:
        r = c.post("/chat", json={"query": "What is Agentic AI?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"final_answer", "retrieved_context", "confidence_score", "citations"}
    assert body["citations"][0]["page"] == 1


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "   "}, {}, {"query": "x" * 2001}, {"query": 5}])
def test_validation(monkeypatch, payload):
    with client(monkeypatch) as c:
        assert c.post("/chat", json=payload).status_code == 422


def test_missing_credentials_503_without_details(monkeypatch):
    with client(monkeypatch, graph_error=ConfigurationError("Missing OPENAI_API_KEY")) as c:
        r = c.post("/chat", json={"query": "hi there"})
    assert r.status_code == 503
    assert "OPENAI_API_KEY" not in r.text


@pytest.mark.parametrize("exc,status", [(RetrievalError("pinecone secret"), 503),
                                        (GenerationError("openai secret"), 502),
                                        (RuntimeError("Traceback secret"), 500)])
def test_pipeline_errors(monkeypatch, exc, status):
    def boom(q, g):
        raise exc
    with client(monkeypatch, runner=boom) as c:
        r = c.post("/chat", json={"query": "What is Agentic AI?"})
    assert r.status_code == status
    assert "secret" not in r.text and "Traceback" not in r.text
