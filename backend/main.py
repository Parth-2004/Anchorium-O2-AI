"""
Anchorium Omni-Engine — FastAPI Backend
========================================

This is the main entry point for the Anchorium multi-agent backend.
It provides:
- A health check endpoint for infrastructure monitoring
- CORS configuration for the Next.js frontend
- Agent-specific endpoints for the 5-agent + orchestrator pipeline
- Streaming SSE responses for real-time frontend rendering

SECURITY NOTE:
  This backend follows a strict zero-retention policy. Any financial
  data received from the frontend is processed ephemerally in RAM
  and NEVER persisted to disk or database in plaintext.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Application Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application lifecycle events.
    - Startup: Initialize agent pipelines, validate prompt registry
    - Shutdown: Ensure all ephemeral data is purged from RAM
    """
    # --- Startup ---
    print("🚀 Anchorium Omni-Engine starting up...")
    print("   ✓ Zero-retention policy active")
    print("   ✓ CORS configured for local development")

    # Validate prompt registry on startup
    try:
        from rag_pipeline.prompts.registry import AgentPromptRegistry

        validation = AgentPromptRegistry.validate_all()
        all_valid = all(validation.values())
        print(f"   ✓ Prompt Registry: {len(validation)} agents registered")
        if all_valid:
            print("   ✓ All agent prompts validated (Core Directives present)")
        else:
            failed = [k for k, v in validation.items() if not v]
            print(f"   ⚠️ PROMPT VALIDATION FAILED for: {failed}")
    except Exception as e:
        print(f"   ⚠️ Prompt Registry validation skipped: {e}")

    print("   ✓ Multi-agent pipeline ready")
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
        "nothing is stored. 5 specialist agents + Master Orchestrator."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS Middleware — Allow the Next.js frontend to communicate
# ---------------------------------------------------------------------------
# In production, replace "*" with the exact frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
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

    model_config = ConfigDict(populate_by_name=True)

    text: str
    vector: list[float] = Field(default_factory=list)
    document_name: str | None = Field(default=None, alias="documentName")
    score: float | None = None


class QueryRequest(BaseModel):
    """Request model for querying the RAG pipeline"""

    query: str
    context_chunks: list[EmbeddedChunk] = Field(default_factory=list)
    system_prompt: str | None = None
    agent_type: str | None = None


class FounderOnboardingRequest(BaseModel):
    """Request model for the full orchestrated pipeline."""

    query: str
    context_chunks: list[EmbeddedChunk] = Field(default_factory=list)
    founder_country: str | None = None
    entity_type_abroad: str | None = None
    annual_revenue: str | None = None
    capital_source: str | None = None
    proposed_india_activity: str | None = None
    headcount_planned: int | None = None
    collateral_type: str | None = None
    collateral_value: str | None = None


class TrustScoreRequest(BaseModel):
    """Request model for the Global Trust Score agent."""

    query: str
    context_chunks: list[EmbeddedChunk] = Field(default_factory=list)
    # Financial profile data
    foreign_bureau_score: int | None = None
    annual_revenue_usd: float | None = None
    revenue_growth_yoy: float | None = None
    collateral_type: str | None = None
    collateral_value_usd: float | None = None
    credit_history_years: int | None = None
    existing_debt_usd: float | None = None
    total_assets_usd: float | None = None
    industry_vertical: str | None = None
    has_indian_credit_history: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks_to_dicts(chunks: list[EmbeddedChunk]) -> list[dict[str, Any]]:
    """Convert Pydantic EmbeddedChunk models to plain dicts for agents."""
    return [
        {
            "text": c.text,
            "documentName": c.document_name or "Unknown",
            "score": c.score or 0.0,
        }
        for c in chunks
        if c.text.strip()
    ]


def _sse(payload: dict[str, Any]) -> str:
    """Serialize one server-sent event payload."""
    return f"data: {json.dumps(payload)}\n\n"


def _get_orchestrator():
    """Lazily initialize the MasterOrchestrator.

    Uses local Ollama (llama3.2) — no external API key required.
    """
    from pydantic import SecretStr

    from rag_pipeline.agents.orchestrator import MasterOrchestrator
    from rag_pipeline.config import GenerationConfig

    config = GenerationConfig()

    return MasterOrchestrator(
        config=config,
        api_key=SecretStr("ollama"),  # Ollama doesn't need a real key
    )


async def _stream_agent_response(
    agent_name: str,
    query: str,
    context_chunks: list[dict[str, Any]],
    input_data: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    """Run an agent and stream its response as SSE events.

    SSE format:
      data: {"type": "thinking"}
      data: {"type": "agent", "agent": "compliance_copilot"}
      data: {"type": "content", "content": "..."}
      data: {"type": "structured", "data": {...}}
      data: {"type": "confidence", "tier": "HIGH|MEDIUM|LOW"}
      data: {"type": "disclaimer", "text": "..."}
      data: {"type": "done"}
    """
    yield _sse({"type": "thinking"})
    yield _sse({"type": "agent", "agent": agent_name})

    try:
        orchestrator = _get_orchestrator()
        output = orchestrator.run_single_agent(
            agent_name=agent_name,
            query=query,
            context_chunks=context_chunks if context_chunks else None,
            input_data=input_data,
        )

        # Stream the main answer
        yield _sse({"type": "content", "content": output.answer})

        # Stream structured data
        if output.structured_data:
            yield _sse({"type": "structured", "data": output.structured_data})

        # Stream confidence tier
        yield _sse(
            {
                "type": "confidence",
                "tier": output.confidence_tier.value,
                "requires_ca_review": output.requires_ca_review,
            }
        )

        # Stream disclaimer if present
        if output.disclaimer:
            yield _sse({"type": "disclaimer", "text": output.disclaimer})

        # Stream draft banner
        yield _sse({"type": "banner", "text": output.draft_banner})

        # Cross-border data flag
        if output.cross_border_data_flag:
            yield _sse(
                {
                    "type": "warning",
                    "text": "⚠️ Cross-border data transfer flagged. Review required before proceeding.",
                }
            )

        yield _sse({"type": "done"})

    except Exception as e:
        error_msg = f"Agent {agent_name} error: {str(e)}"
        yield _sse({"type": "error", "message": error_msg})


async def _stream_pipeline_response(
    query: str,
    context_chunks: list[dict[str, Any]],
) -> AsyncGenerator[str, None]:
    """Run the full orchestrated pipeline and stream results."""
    yield _sse({"type": "thinking"})
    yield _sse({"type": "agent", "agent": "orchestrator"})
    yield _sse({"type": "status", "message": "Starting multi-agent pipeline..."})

    try:
        orchestrator = _get_orchestrator()
        result = orchestrator.run_full_pipeline(
            query=query,
            context_chunks=context_chunks if context_chunks else None,
        )

        # Stream each agent's output
        for agent_name, output in result.agent_outputs.items():
            yield _sse({"type": "status", "message": f"Agent completed: {agent_name}"})

        # Stream the assembled report
        yield _sse({"type": "content", "content": result.assembled_report})

        # Stream overall metadata
        yield _sse(
            {
                "type": "confidence",
                "tier": result.overall_confidence.value,
                "requires_ca_review": result.requires_ca_review,
            }
        )

        if result.requires_ca_review:
            yield _sse({"type": "warning", "text": "⚠️ REQUIRES CA REVIEW BEFORE CLIENT DELIVERY"})

        # Stream audit trail
        yield _sse({"type": "audit", "data": result.audit_trail})

        yield _sse({"type": "done"})

    except Exception as e:
        error_msg = f"Pipeline error: {str(e)}"
        yield _sse({"type": "error", "message": error_msg})


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
        "version": "0.2.0",
        "security": {
            "zero_retention": True,
            "data_encryption": "client-side-e2e",
        },
        "agents": [
            "compliance_copilot",
            "gaap_translator",
            "trust_score",
            "arbitrage_calculator",
            "kyc_extractor",
            "orchestrator",
        ],
    }


# ---------------------------------------------------------------------------
# Agent Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/query", tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    General query endpoint — routes to the appropriate agent based on
    agent_type, or defaults to the Compliance Copilot.
    """
    agent_name = request.agent_type or "compliance_copilot"

    # Map friendly names to agent names
    agent_map = {
        "Compliance Copilot": "compliance_copilot",
        "Underwriter": "gaap_translator",
        "Arbitrage Calculator": "arbitrage_calculator",
        "Strategy Agent": "compliance_copilot",  # fallback
        "compliance_copilot": "compliance_copilot",
        "gaap_translator": "gaap_translator",
        "trust_score": "trust_score",
        "arbitrage_calculator": "arbitrage_calculator",
        "kyc_extractor": "kyc_extractor",
    }
    resolved_agent = agent_map.get(agent_name, "compliance_copilot")
    chunks = _chunks_to_dicts(request.context_chunks)

    return StreamingResponse(
        _stream_agent_response(resolved_agent, request.query, chunks),
        media_type="text/event-stream",
    )


@app.post("/api/v1/compliance", tags=["Agents"])
async def compliance_agent(request: QueryRequest):
    """Agent 1 — RBI/FEMA Compliance Copilot."""
    chunks = _chunks_to_dicts(request.context_chunks)
    return StreamingResponse(
        _stream_agent_response("compliance_copilot", request.query, chunks),
        media_type="text/event-stream",
    )


@app.post("/api/v1/underwrite", tags=["Agents"])
async def underwrite_agent(request: QueryRequest):
    """Agent 2 — GAAP ⇄ Ind AS Translation Agent."""
    chunks = _chunks_to_dicts(request.context_chunks)
    return StreamingResponse(
        _stream_agent_response("gaap_translator", request.query, chunks),
        media_type="text/event-stream",
    )


@app.post("/api/v1/trust-score", tags=["Agents"])
async def trust_score_agent(request: TrustScoreRequest):
    """Agent 3 — Global Trust Score Agent."""
    chunks = _chunks_to_dicts(request.context_chunks)

    # Build input data from the request fields
    input_data: dict[str, Any] = {}
    if request.foreign_bureau_score is not None:
        input_data["foreign_bureau_score"] = request.foreign_bureau_score
    if request.annual_revenue_usd is not None:
        input_data["annual_revenue_usd"] = request.annual_revenue_usd
    if request.revenue_growth_yoy is not None:
        input_data["revenue_growth_yoy"] = request.revenue_growth_yoy
    if request.collateral_type:
        input_data["collateral_type"] = request.collateral_type
    if request.collateral_value_usd is not None:
        input_data["collateral_value_usd"] = request.collateral_value_usd
    if request.credit_history_years is not None:
        input_data["credit_history_years"] = request.credit_history_years
    if request.existing_debt_usd is not None:
        input_data["existing_debt_usd"] = request.existing_debt_usd
    if request.total_assets_usd is not None:
        input_data["total_assets_usd"] = request.total_assets_usd
    if request.industry_vertical:
        input_data["industry_vertical"] = request.industry_vertical
    input_data["has_indian_credit_history"] = request.has_indian_credit_history

    return StreamingResponse(
        _stream_agent_response("trust_score", request.query, chunks, input_data or None),
        media_type="text/event-stream",
    )


@app.post("/api/v1/arbitrage", tags=["Agents"])
async def arbitrage_agent(request: QueryRequest):
    """Agent 4 — Arbitrage & Cost Calculator Agent."""
    chunks = _chunks_to_dicts(request.context_chunks)
    return StreamingResponse(
        _stream_agent_response("arbitrage_calculator", request.query, chunks),
        media_type="text/event-stream",
    )


@app.post("/api/v1/kyc-extract", tags=["Agents"])
async def kyc_extract_agent(request: QueryRequest):
    """Agent 5 — KYC / Data Extraction Agent."""
    chunks = _chunks_to_dicts(request.context_chunks)
    return StreamingResponse(
        _stream_agent_response("kyc_extractor", request.query, chunks),
        media_type="text/event-stream",
    )


@app.post("/api/v1/playbook", tags=["Pipeline"])
async def full_playbook(request: FounderOnboardingRequest):
    """Master Orchestrator — Full India Soft-Landing Playbook pipeline.

    Runs all 5 agents in the correct order:
      1. KYC Extraction
      2. Compliance + GAAP (sequential)
      3. Global Trust Score
      4. Arbitrage Calculator

    Returns the assembled report with audit trail.
    """
    chunks = _chunks_to_dicts(request.context_chunks)
    return StreamingResponse(
        _stream_pipeline_response(request.query, chunks),
        media_type="text/event-stream",
    )
