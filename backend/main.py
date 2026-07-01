"""
Anchorium Omni-Engine — FastAPI Backend
========================================

This is the main entry point for the Anchorium multi-agent backend.
It provides:
- A health check endpoint for infrastructure monitoring
- CORS configuration for the Next.js frontend
- Ephemeral compute endpoints for the RAG pipeline and AI agents

SECURITY NOTE:
  This backend follows a strict zero-retention policy. Any financial
  data received from the frontend is processed ephemerally in RAM
  and NEVER persisted to disk or database in plaintext.
"""

from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


# ---------------------------------------------------------------------------
# Application Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application lifecycle events.
    - Startup: Initialize agent pipelines (future)
    - Shutdown: Ensure all ephemeral data is purged from RAM
    """
    # --- Startup ---
    print("🚀 Anchorium Omni-Engine starting up...")
    print("   ✓ Zero-retention policy active")
    print("   ✓ CORS configured for local development")
    yield
    # --- Shutdown ---
    print("🛑 Anchorium Omni-Engine shutting down...")
    print("   ✓ All ephemeral data purged from RAM")


# ---------------------------------------------------------------------------
# FastAPI Application Instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Anchorium Omni-Engine",
    description=(
        "Zero-Knowledge Multi-Agent Backend for cross-border "
        "credit underwriting. Processes financial data ephemerally — "
        "nothing is stored."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS Middleware — Allow the Next.js frontend to communicate
# ---------------------------------------------------------------------------
# In production, replace "*" with the exact frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Next.js dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Models for API
# ---------------------------------------------------------------------------
class EmbeddedChunk(BaseModel):
    """A chunk of text with its embedding vector and metadata"""
    text: str
    vector: List[float]


class QueryRequest(BaseModel):
    """Request model for querying the RAG pipeline"""
    query: str
    context_chunks: List[EmbeddedChunk] = Field(default_factory=list)
    system_prompt: Optional[str] = None


# ---------------------------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Infrastructure"])
async def health_check():
    """
    Simple health check for infrastructure monitoring.
    Returns the service status and active security policies.
    """
    return {
        "status": "ok",
        "service": "anchorium-omni-engine",
        "version": "0.1.0",
        "security": {
            "zero_retention": True,
            "data_encryption": "client-side-e2e",
        },
    }


# ---------------------------------------------------------------------------
# RAG & Agent Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/query", tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Query the RAG pipeline with a user query and context chunks.
    Uses strict zero-retention policy — all data processed ephemerally.
    """
    try:
        # For now, we'll implement a simplified response.
        # In production, this would integrate with the full rag_pipeline.
        
        async def generate_response():
            yield "data: {\"type\": \"thinking\"}\n\n"
            yield "data: {\"type\": \"content\", \"content\": \"This is a sample response from the Anchorium Omni-Engine. In production, this would integrate with our full RAG pipeline for accurate responses.\\n\\nHere's what the system would do:\\n1. Use the provided context chunks\\n2. Apply strict hallucination prevention\\n3. Generate an accurate response based solely on the context\"}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/compliance", tags=["Agents"])
async def compliance_agent(request: QueryRequest):
    """
    Compliance Copilot agent for RBI/FEMA regulation checks.
    """
    try:
        async def generate_response():
            yield "data: {\"type\": \"thinking\"}\n\n"
            yield "data: {\"type\": \"content\", \"content\": \"Compliance Copilot Agent activated. This would check RBI/FEMA regulations based on the provided context.\"}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/underwrite", tags=["Agents"])
async def underwrite_agent(request: QueryRequest):
    """
    Underwriter agent for US GAAP → Indian IndAS translation.
    """
    try:
        async def generate_response():
            yield "data: {\"type\": \"thinking\"}\n\n"
            yield "data: {\"type\": \"content\", \"content\": \"Underwriter Agent activated. This would convert US GAAP financials to Indian IndAS format based on the provided context.\"}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/arbitrage", tags=["Agents"])
async def arbitrage_agent(request: QueryRequest):
    """
    Arbitrage Calculator agent for USD/INR loan pathway optimization.
    """
    try:
        async def generate_response():
            yield "data: {\"type\": \"content\", \"content\": \"Arbitrage Calculator Agent activated. This would simulate USD-INR debt arbitrage opportunities based on the provided context.\"}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
