"""Configuration management for the Anchorium RAG Pipeline.

Uses Pydantic Settings V2 for type-safe, environment-variable-driven
configuration with nested models for each subsystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIKeysConfig(BaseSettings):
    """API key configuration for all external services.

    All keys are stored as SecretStr to prevent accidental logging.
    NOTE: When using Ollama locally, these keys are not required.
    """

    model_config = SettingsConfigDict(env_prefix="ANCHORIUM_")

    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key (optional — not needed when using Ollama).",
    )
    pinecone_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Pinecone API key for vector database operations (optional).",
    )
    cohere_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Cohere API key for reranking search results (optional).",
    )
    langsmith_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="LangSmith API key for observability tracing (optional).",
    )


class PineconeConfig(BaseSettings):
    """Pinecone vector database configuration."""

    model_config = SettingsConfigDict(env_prefix="ANCHORIUM_PINECONE_")

    index_name: str = Field(
        default="anchorium-legal-rag",
        description="Name of the Pinecone index.",
    )
    cloud: str = Field(
        default="aws",
        description="Cloud provider for the Pinecone serverless index.",
    )
    region: str = Field(
        default="us-east-1",
        description="Region for the Pinecone serverless index.",
    )
    dimension: int = Field(
        default=1536,
        description="Embedding vector dimension. 1536 for text-embedding-3-small, 1024 for voyage-law-2.",
    )
    metric: Literal["cosine", "euclidean", "dotproduct"] = Field(
        default="cosine",
        description="Distance metric for vector similarity.",
    )
    batch_size: int = Field(
        default=100,
        description="Number of vectors to upsert in a single batch.",
    )


class ChunkingConfig(BaseSettings):
    """Configuration for legal-structural document chunking."""

    model_config = SettingsConfigDict(env_prefix="ANCHORIUM_CHUNKING_")

    max_chunk_tokens: int = Field(
        default=512,
        description="Maximum number of tokens per chunk.",
    )
    overlap_tokens: int = Field(
        default=64,
        description="Number of overlapping tokens between adjacent chunks.",
    )
    min_chunk_tokens: int = Field(
        default=50,
        description="Minimum token count — smaller chunks are merged with neighbors.",
    )
    tokenizer_model: str = Field(
        default="cl100k_base",
        description="Tiktoken encoding model for token counting.",
    )


class RetrievalConfig(BaseSettings):
    """Configuration for hybrid search and reranking."""

    model_config = SettingsConfigDict(env_prefix="ANCHORIUM_RETRIEVAL_")

    alpha: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Alpha blend weight: Score = α·Sparse + (1-α)·Dense. Higher alpha emphasizes keyword (BM25) matching."
        ),
    )
    top_k_raw: int = Field(
        default=20,
        description="Number of raw results from each search modality before blending.",
    )
    top_k_reranked: int = Field(
        default=5,
        description="Number of results returned after Cohere reranking.",
    )
    temporal_boost_factor: float = Field(
        default=0.1,
        description=("Recency boost multiplier per year. Score *= (1 + factor * years_newer_than_oldest)."),
    )
    cohere_rerank_model: str = Field(
        default="rerank-v3.5",
        description="Cohere reranking model identifier.",
    )


class EmbeddingConfig(BaseSettings):
    """Configuration for the embedding model."""

    model_config = SettingsConfigDict(env_prefix="ANCHORIUM_EMBEDDING_")

    provider: Literal["openai", "voyage"] = Field(
        default="openai",
        description="Embedding provider: 'openai' for text-embedding-3-small, 'voyage' for voyage-law-2.",
    )
    openai_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name.",
    )
    voyage_model: str = Field(
        default="voyage-law-2",
        description="VoyageAI embedding model name.",
    )
    batch_size: int = Field(
        default=64,
        description="Number of texts to embed in a single API call.",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retries for embedding API calls.",
    )


class GenerationConfig(BaseSettings):
    """Configuration for LLM generation and hallucination validation.

    Defaults are tuned for Hugging Face Transformers.
    """

    model_config = SettingsConfigDict(env_prefix="ANCHORIUM_GENERATION_")

    hf_model_name: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        description="Hugging Face model for generation.",
    )
    hf_device: str = Field(
        default="cpu",
        description="Device to run Hugging Face model on ('cpu', 'cuda', 'mps').",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Low for deterministic compliance outputs.",
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens in the generated response.",
    )
    validation_max_retries: int = Field(
        default=2,
        description="Maximum regeneration attempts if hallucination is detected.",
    )
    validation_model: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        description="Model used for the self-reflection hallucination check.",
    )
    context_window_limit: int = Field(
        default=120000,
        description="Maximum context window tokens for the generation model.",
    )


class AgentConfig(BaseSettings):
    """Configuration for the multi-agent Omni-Engine pipeline.

    Per-agent overrides allow tuning model, temperature, and token limits
    independently for each specialist agent.
    """

    model_config = SettingsConfigDict(env_prefix="ANCHORIUM_AGENT_")

    # Per-agent model overrides (None = inherit from GenerationConfig)
    compliance_model: str | None = Field(
        default=None,
        description="Model override for Agent 1 (Compliance Copilot).",
    )
    gaap_model: str | None = Field(
        default=None,
        description="Model override for Agent 2 (GAAP Translator).",
    )
    trust_score_model: str | None = Field(
        default=None,
        description="Model override for Agent 3 (Global Trust Score).",
    )
    arbitrage_model: str | None = Field(
        default=None,
        description="Model override for Agent 4 (Arbitrage Calculator).",
    )
    kyc_model: str | None = Field(
        default=None,
        description="Model override for Agent 5 (KYC Extractor).",
    )

    # Shared agent settings
    stale_source_months: int = Field(
        default=12,
        ge=1,
        description=(
            "Number of months after which a regulatory source is considered "
            "stale and triggers MEDIUM confidence (Core Directives Rule 3)."
        ),
    )
    enable_arbitrage_code_execution: bool = Field(
        default=False,
        description=(
            "Safety toggle: if True, the Arbitrage Calculator agent may "
            "execute structured calculations via a sandboxed Python runtime. "
            "Default False for safety."
        ),
    )
    max_orchestration_agents: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Maximum number of agents the orchestrator may invoke in a single pipeline run.",
    )


class AppSettings(BaseSettings):
    """Root application settings composing all subsystem configurations.

    Loads from a `.env` file in the project root. All nested configs
    are initialized from their respective environment variable prefixes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configurations
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    pinecone: PineconeConfig = Field(default_factory=PineconeConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)

    # Filesystem paths
    data_dir: Path = Field(
        default=Path("data"),
        description="Root directory for ingested documents and indices.",
    )
    bm25_index_path: Path = Field(
        default=Path("data/bm25_index.pkl"),
        description="File path for the serialized BM25 index.",
    )
    bm25_chunks_path: Path = Field(
        default=Path("data/bm25_chunks.pkl"),
        description="File path for the serialized BM25 chunk metadata.",
    )

    # Observability
    langsmith_project: str = Field(
        default="anchorium-rag-pipeline",
        description="LangSmith project name for tracing.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Structured logging level.",
    )


def load_settings() -> AppSettings:
    """Load and validate application settings from environment variables.

    Returns:
        Fully validated AppSettings instance.

    Raises:
        pydantic.ValidationError: If required environment variables are missing
            or values fail validation constraints.
    """
    return AppSettings()
