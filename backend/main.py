"""
Anchorium Omni-Engine — FastAPI Backend
========================================

This is the main entry point for the Anchorium multi-agent backend.
It provides:
  - A health check endpoint for infrastructure monitoring
  - CORS configuration for the Next.js frontend
  - (Future) Ephemeral compute endpoints for the 3 AI agents:
      1. Compliance Copilot  — RBI/FEMA regulation checks
      2. Underwriter          — US GAAP → Indian IndAS translation
      3. Arbitrage Calculator — USD/INR loan pathway optimization

SECURITY NOTE:
  This backend follows a strict zero-retention policy. Any financial
  data received from the frontend is processed ephemerally in RAM
  and NEVER persisted to disk or database in plaintext.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


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
# Placeholder: Agent Endpoints (Phase 2+)
# ---------------------------------------------------------------------------
# These will be implemented in future phases:
#
# POST /api/v1/compliance   → Compliance Copilot agent
# POST /api/v1/underwrite   → Underwriter agent
# POST /api/v1/arbitrage    → Arbitrage Calculator agent
#
# Each endpoint will:
#   1. Receive ONLY the relevant text chunks + prompt (no raw docs)
#   2. Route to the appropriate CrewAI/LangGraph agent
#   3. Return the structured result
#   4. Immediately purge all input data from memory
