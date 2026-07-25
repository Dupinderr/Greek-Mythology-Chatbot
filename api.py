"""
FastAPI wrapper around the mythology RAG agent.

Run locally:
    uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

import rag_core

app = FastAPI(
    title="Greek Mythology RAG Agent",
    description="Ask questions answered strictly from public-domain mythology texts.",
    version="1.0.0",
)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")
    source: Optional[str] = Field(
        None,
        description="Restrict retrieval to this source. Omit to search all sources.",
    )
    history: List[Turn] = Field(
        default_factory=list,
        description="Earlier turns, oldest first.",
    )


class ChatResponse(BaseModel):
    answer: str
    source_used: str
    tool_called: bool


@app.get("/health")
def health():
    """
    Liveness probe. Also reports what the vector store actually holds, so a
    container that started without its data is obvious immediately.
    """

    try:
        sources = rag_core.list_sources()
        chunks = rag_core.chunk_count()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"vector store unavailable: {exc}")

    return {
        "status": "ok" if chunks else "empty",
        "model": rag_core.MODEL,
        "embedding_model": rag_core.EMBEDDING_MODEL,
        "chunks": chunks,
        "sources": sources,
    }


@app.get("/sources")
def sources():
    return {
        "sources": rag_core.list_sources(),
        "max_sources": rag_core.MAX_SOURCES,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    available = rag_core.list_sources()

    if not available:
        raise HTTPException(status_code=503, detail="No sources loaded.")

    if request.source and request.source not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source '{request.source}'. Available: {available}",
        )

    # The graph may call the retriever more than once; we only need to know
    # whether it reached for it at all.
    called = []

    try:
        answer = rag_core.ask(
            question=request.question,
            source=request.source,
            history=[t.model_dump() for t in request.history],
            on_tool_call=lambda name, scope: called.append(name),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"agent failed: {exc}")

    return ChatResponse(
        answer=answer,
        source_used=request.source or "all sources",
        tool_called=bool(called),
    )
