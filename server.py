"""FastAPI backend for the web UI: chat with the assistant + view Phase 1/2/3
benchmark results, all in one place.

Run with the SYSTEM python3 (needs assistant.py's langchain/chroma stack,
same reason as assistant.py itself -- see its module docstring):

    pip install fastapi "uvicorn[standard]"   # one-time, into system python
    python3 server.py
    # open http://127.0.0.1:8000

Conversation history is kept in memory only (dict keyed by session_id),
lost on server restart -- fine for a local single-user demo, not meant to
be a persistence layer.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_data
from assistant import RAG_PROJECT_ROOT, answer_question
from benchmark import check_ollama_available
from models import MODELS
from src.ingestion.store import get_embeddings, get_vector_store
from src.retrieval.hybrid import HybridRetriever, RetrievalConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_ollama_available()
    yield


app = FastAPI(title="local-slm-benchmark", lifespan=lifespan)

DEFAULT_MODEL = "llama3.2-3b"
DEFAULT_K = 6
DEFAULT_HISTORY_TURNS = 3

# One retriever per (corpus_dir, k) pair, built lazily on first use -- avoids
# paying the embedding-model/reranker load cost at server startup if the UI
# is only opened to look at the dashboard tab.
_retrievers: dict[tuple[str, int], HybridRetriever] = {}
_embeddings = None

# session_id -> list[(question, answer)], capped per session at connect time.
_chat_history: dict[str, list[tuple[str, str]]] = {}


def get_retriever(corpus_dir: str, k: int) -> HybridRetriever:
    global _embeddings
    key = (corpus_dir, k)
    if key not in _retrievers:
        if _embeddings is None:
            _embeddings = get_embeddings()
        db = get_vector_store(persist_dir=corpus_dir, embeddings=_embeddings)
        _retrievers[key] = HybridRetriever(db, RetrievalConfig(final_k=k))
    return _retrievers[key]


class ChatRequest(BaseModel):
    session_id: str
    question: str
    model: str = DEFAULT_MODEL
    k: int = DEFAULT_K
    history_turns: int = DEFAULT_HISTORY_TURNS


@app.get("/api/models")
def get_models():
    return {"models": list(MODELS.keys()), "default": DEFAULT_MODEL}


@app.get("/api/benchmark/phase1")
def get_phase1():
    return benchmark_data.get_phase1_summary()


@app.get("/api/benchmark/phase2")
def get_phase2():
    return benchmark_data.get_phase2_summary()


@app.get("/api/benchmark/phase3")
def get_phase3():
    return benchmark_data.get_phase3_summary()


@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model {req.model!r}. Known: {list(MODELS.keys())}")
    tag = MODELS[req.model]

    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    history = _chat_history.get(req.session_id, [])
    retriever = get_retriever(str(RAG_PROJECT_ROOT / "chroma_index"), req.k)

    result = answer_question(retriever, tag, req.question, history=history or None)

    if result["success"] and req.history_turns > 0:
        history = history + [(req.question, result["answer"])]
        _chat_history[req.session_id] = history[-req.history_turns:]

    return result


@app.post("/api/chat/reset")
def reset_chat(session_id: str):
    _chat_history.pop(session_id, None)
    return {"ok": True}


# Static frontend last -- routes above take precedence over the catch-all.
app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
