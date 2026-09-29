"""FastAPI service exposing the RAG graph.

Run:  uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from src.graph import ask, build_rag_graph

graph = None


def get_graph():
    """Build the graph once; raises RuntimeError if API keys are missing."""
    global graph
    if graph is None:
        graph = build_rag_graph()
    return graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_graph()
    except RuntimeError as exc:
        # Start anyway so /health and /docs work; /chat reports the problem.
        print(f"WARNING: {exc}")
    yield


app = FastAPI(title="Agentic AI RAG Chatbot", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, examples=["What is Agentic AI?"])


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest) -> QueryResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")
    try:
        rag_graph = get_graph()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    # graph.invoke is blocking (network calls), so keep it off the event loop.
    try:
        return QueryResponse(**await run_in_threadpool(ask, rag_graph, query))
    except Exception as exc:
        # Upstream failures (model quota, network) are reported cleanly instead of a bare 500.
        raise HTTPException(status_code=503, detail=f"Model service unavailable: {exc}"[:500])
