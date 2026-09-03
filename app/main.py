"""FastAPI app: serves the UI and answers questions over the ingested documents.

Endpoints:
  GET  /            -> the minimal ask UI (static/index.html)
  GET  /health      -> liveness + how many chunks are indexed
  POST /ask         -> {question} -> {answer, grounded, sources[]}

Every expected failure (missing key, retrieval failure, API failure) is caught
and returned as a clean JSON message with a sensible status code -- never a raw
stack trace, never a silent hang.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .ingest import ingest
from .rag import (
    ConfigError,
    GenerationError,
    RagError,
    RetrievalError,
    answer,
)
from .store import get_collection

STATIC_DIR = config.ROOT_DIR / "static"

app = FastAPI(title="RAG Document Q&A Demo")


def _warm_embedder() -> None:
    """Load the ONNX embedding model into memory once, so the first real question
    isn't stuck ~15-20s waiting for it. Runs on a background thread so it never
    delays the server binding its port."""
    try:
        get_collection().query(query_texts=["warmup"], n_results=1)
        print("[startup] Embedder warmed.")
    except Exception as exc:
        print(f"[startup] Warm-up skipped: {exc}")


@app.on_event("startup")
def ensure_ingested() -> None:
    """Auto-ingest docs/ on first boot so a fresh deploy just works."""
    try:
        collection = get_collection()
        if collection.count() == 0:
            n = ingest()
            print(f"[startup] Indexed {n} chunks from {config.DOCS_DIR}.")
        else:
            print(f"[startup] Store already has {collection.count()} chunks.")
    except Exception as exc:  # never crash the server on startup ingest trouble
        print(f"[startup] Ingest skipped: {exc}")

    # Pre-load the embedder in the background so the first question is fast.
    threading.Thread(target=_warm_embedder, daemon=True).start()


class AskRequest(BaseModel):
    question: str


# GET and HEAD: uptime monitors ping with HEAD by default; answering it keeps
# the free instance warm without false "down" alerts.
@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict:
    try:
        count = get_collection().count()
    except Exception:
        count = -1
    return {"status": "ok", "indexed_chunks": count, "model": config.ANTHROPIC_MODEL}


@app.post("/ask")
def ask(req: AskRequest) -> JSONResponse:
    try:
        result = answer(req.question)
    except ConfigError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    except RetrievalError as exc:
        return JSONResponse(status_code=502, content={"error": str(exc)})
    except GenerationError as exc:
        return JSONResponse(status_code=502, content={"error": str(exc)})
    except RagError as exc:  # e.g. empty question
        return JSONResponse(status_code=400, content={"error": str(exc)})

    return JSONResponse(
        content={
            "answer": result.answer,
            "grounded": result.grounded,
            "sources": [
                {
                    "source": p.source,
                    "chunk_index": p.chunk_index,
                    "score": round(p.score, 3),
                    "text": p.text,
                }
                for p in result.sources
            ],
        }
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# Serve any other static assets (none required, but keeps the door open).
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
