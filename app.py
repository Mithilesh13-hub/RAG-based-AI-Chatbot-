"""FastAPI backend.  Run with:  uvicorn app:app --reload"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src.config import ConfigurationError
from src.graph import GenerationError, RetrievalError, get_default_graph, run_graph
from src.schemas import ChatRequest, ChatResponse

logger = logging.getLogger("rag.api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Build the (expensive) graph once at startup; tolerate missing credentials."""
    application.state.graph = None
    try:
        application.state.graph = get_default_graph()
    except ConfigurationError as exc:
        logger.warning("Graph not initialised: %s", exc)
    except Exception:  # noqa: BLE001
        logger.exception("Graph initialisation failed")
    yield


app = FastAPI(title="Agentic AI eBook RAG Chatbot", version="1.0.0", lifespan=lifespan)


def _get_graph(request: Request):
    if getattr(request.app.state, "graph", None) is None:
        try:
            request.app.state.graph = get_default_graph()
        except ConfigurationError as exc:
            logger.error("Configuration error: %s", exc)
            raise HTTPException(status_code=503, detail="The service is not configured correctly.") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Graph initialisation failed")
            raise HTTPException(status_code=503, detail="The knowledge base is unavailable.") from exc
    return request.app.state.graph


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    graph = _get_graph(request)
    try:
        result = run_graph(payload.query, graph)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RetrievalError as exc:
        raise HTTPException(status_code=503, detail="The knowledge base is temporarily unavailable.") from exc
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail="The language model is temporarily unavailable.") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc
    return ChatResponse(**result)
