"""CLI entry point for the Anchorium Cross-Border Legal & Regulatory RAG Pipeline.

Exposes three commands via `typer`:

* **ingest** — Parse a regulatory PDF, chunk it, and index it into both
  Pinecone (dense) and BM25 (sparse) stores.
* **query** — Run a compliance question through the full hybrid retrieval →
  Cohere reranking → LLM generation → hallucination validation pipeline.
* **supersede** — Mark an older circular as inactive when a newer one replaces it.

Usage examples::

    # Ingest a new RBI circular
    python -m rag_pipeline.main ingest \
        --file docs/rbi_circular_2025_42.pdf \
        --issuing-body RBI \
        --circular-number "RBI/2024-25/42" \
        --effective-date 2025-01-15 \
        --title "Master Direction on External Commercial Borrowings"

    # Query the system
    python -m rag_pipeline.main query \
        "What are the FEMA regulations for SBLC-backed lending via GIFT City OBUs?"

    # Supersede an old circular with a new one
    python -m rag_pipeline.main supersede \
        --old-circular "RBI/2023-24/18" \
        --new-circular "RBI/2024-25/42"
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_pipeline.config import AppSettings, load_settings
from rag_pipeline.generation import (
    ComplianceResponse,
    generate_compliance_response,
)
from rag_pipeline.indexing import (
    BM25IndexManager,
    EmbeddingService,
    IndexingResult,
    PineconeIndexer,
    index_document,
)
from rag_pipeline.ingestion import (
    DocumentChunk,
    DocumentMetadata,
    IssuingBody,
    ingest_document,
)
from rag_pipeline.retrieval import (
    MetadataFilter,
    RetrievalResult,
    retrieve,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

_LOG_CONFIGURED = False


def _configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog with human-readable console output.

    Called once at application start. Subsequent calls are no-ops.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR.
    """
    global _LOG_CONFIGURED  # noqa: PLW0603
    if _LOG_CONFIGURED:
        return

    import logging

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _LOG_CONFIGURED = True


logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="anchorium-rag",
    help=(
        "Anchorium Cross-Border Legal & Regulatory RAG Pipeline — "
        "Automated Compliance Copilot for GIFT City OBU Lending."
    ),
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_settings_or_exit() -> AppSettings:
    """Load application settings, printing a helpful error on failure.

    Returns:
        Validated ``AppSettings`` instance.
    """
    try:
        settings = load_settings()
        return settings
    except Exception as exc:
        console.print(
            Panel(
                f"[bold red]Configuration Error[/bold red]\n\n{exc}\n\n"
                "Ensure all required environment variables are set or "
                "provide a `.env` file in the project root.\n\n"
                "Required variables:\n"
                "  • ANCHORIUM_OPENAI_API_KEY\n"
                "  • ANCHORIUM_PINECONE_API_KEY\n"
                "  • ANCHORIUM_COHERE_API_KEY",
                title="⚠ Startup Failed",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _parse_issuing_body(value: str) -> IssuingBody:
    """Parse a string into an ``IssuingBody`` enum member.

    Args:
        value: Case-insensitive issuing body name.

    Returns:
        The matching ``IssuingBody`` variant.

    Raises:
        typer.BadParameter: If the value does not match any variant.
    """
    normalised = value.upper().replace(" ", "_").replace("-", "_")
    try:
        return IssuingBody(normalised)
    except ValueError:
        valid = ", ".join(member.value for member in IssuingBody)
        raise typer.BadParameter(f"Invalid issuing body '{value}'. Must be one of: {valid}")


def _display_indexing_result(result: IndexingResult) -> None:
    """Render an ``IndexingResult`` as a rich table.

    Args:
        result: The indexing result to display.
    """
    table = Table(title="📦 Indexing Result", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Total Chunks", str(result.total_chunks))
    table.add_row("Successfully Indexed", f"[green]{result.successfully_indexed}[/green]")
    table.add_row("Failed Chunks", f"[red]{result.failed_chunks}[/red]" if result.failed_chunks > 0 else "0")
    table.add_row("Index Name", result.index_name)
    table.add_row("Duration", f"{result.duration_seconds:.2f}s")

    console.print(table)


def _display_compliance_response(response: ComplianceResponse) -> None:
    """Render a ``ComplianceResponse`` in the console with rich formatting.

    Args:
        response: The compliance response to display.
    """
    # Verification status
    if response.is_verified:
        status = "[bold green]✅ VERIFIED[/bold green]"
    else:
        status = "[bold red]⚠ UNVERIFIED[/bold red]"

    console.print()
    console.print(
        Panel(
            f"[bold]Query:[/bold] {response.query}\n"
            f"[bold]Status:[/bold] {status}\n"
            f"[bold]Model:[/bold] {response.model_used}\n"
            f"[bold]Attempts:[/bold] {response.generation_attempts}\n"
            f"[bold]Context Chunks:[/bold] {response.context_chunks_used}\n"
            f"[bold]Validation Confidence:[/bold] "
            f"{response.validation_result.confidence_score:.1%}",
            title="📋 Compliance Response",
            border_style="green" if response.is_verified else "red",
        )
    )

    # Answer
    console.print()
    console.print(
        Panel(
            response.answer,
            title="📝 Answer",
            border_style="blue",
        )
    )

    # Citations
    if response.citations:
        citation_table = Table(
            title="📚 Citations",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
        )
        citation_table.add_column("#", style="dim", width=3)
        citation_table.add_column("Circular", width=20)
        citation_table.add_column("Body", width=8)
        citation_table.add_column("Date", width=12)
        citation_table.add_column("Path", width=35)
        citation_table.add_column("Excerpt", width=50)

        for idx, cit in enumerate(response.citations, start=1):
            citation_table.add_row(
                str(idx),
                cit.circular_number,
                cit.issuing_body,
                cit.effective_date,
                cit.hierarchical_path,
                cit.relevant_excerpt[:100] + "…" if len(cit.relevant_excerpt) > 100 else cit.relevant_excerpt,
            )

        console.print()
        console.print(citation_table)

    # Refusal reason
    if response.refusal_reason:
        console.print()
        console.print(
            Panel(
                response.refusal_reason,
                title="🚫 Refusal Reason",
                border_style="red",
            )
        )

    # Validation details
    if response.validation_result.failed_claims:
        console.print()
        failed_table = Table(
            title="⚠ Failed Claims",
            show_header=True,
            header_style="bold red",
        )
        failed_table.add_column("#", style="dim", width=3)
        failed_table.add_column("Unsupported Claim")

        for idx, claim in enumerate(response.validation_result.failed_claims, start=1):
            failed_table.add_row(str(idx), claim)

        console.print(failed_table)


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    file: Path = typer.Option(
        ...,
        "--file",
        "-f",
        help="Path to the regulatory PDF file to ingest.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    issuing_body: str = typer.Option(
        ...,
        "--issuing-body",
        "-b",
        help="Issuing regulatory body (RBI, SEBI, IT_DEPT, FEMA, OTHER).",
    ),
    circular_number: str = typer.Option(
        ...,
        "--circular-number",
        "-c",
        help="Official circular reference (e.g. 'RBI/2024-25/42').",
    ),
    effective_date: str = typer.Option(
        ...,
        "--effective-date",
        "-d",
        help="Effective date in ISO format (YYYY-MM-DD).",
    ),
    title: str = typer.Option(
        "",
        "--title",
        "-t",
        help="Document title. Defaults to the filename if not provided.",
    ),
    version: int = typer.Option(
        1,
        "--version",
        "-v",
        help="Version number for this circular (used for supersession tracking).",
    ),
    supersede_old: str | None = typer.Option(
        None,
        "--supersede",
        "-s",
        help="If this circular supersedes an older one, provide the old circular number here.",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Output the indexing result as JSON instead of a rich table.",
    ),
) -> None:
    """Ingest a regulatory PDF into the RAG pipeline.

    Parses the document with layout-aware extraction, chunks it by legal
    clause boundaries, generates embeddings, and upserts into both Pinecone
    (dense) and BM25 (sparse) indices.
    """
    _configure_logging()
    settings = _load_settings_or_exit()

    # Parse inputs
    parsed_body = _parse_issuing_body(issuing_body)

    try:
        parsed_date = date.fromisoformat(effective_date)
    except ValueError as exc:
        console.print(f"[red]Invalid date format:[/red] {effective_date}. Use YYYY-MM-DD.")
        raise typer.Exit(code=1) from exc

    doc_title = title if title else file.stem.replace("_", " ").title()

    console.print(
        Panel(
            f"[bold]File:[/bold] {file}\n"
            f"[bold]Issuing Body:[/bold] {parsed_body.value}\n"
            f"[bold]Circular:[/bold] {circular_number}\n"
            f"[bold]Effective Date:[/bold] {parsed_date.isoformat()}\n"
            f"[bold]Title:[/bold] {doc_title}\n"
            f"[bold]Version:[/bold] {version}"
            + (f"\n[bold]Supersedes:[/bold] {supersede_old}" if supersede_old else ""),
            title="📄 Ingestion Parameters",
            border_style="cyan",
        )
    )

    # Step 1: Handle supersession of old circular
    if supersede_old:
        console.print(f"\n[yellow]Superseding old circular:[/yellow] {supersede_old}")
        try:
            pinecone_indexer = PineconeIndexer(
                config=settings.pinecone,
                api_key=settings.api_keys.pinecone_api_key,
            )
            updated = pinecone_indexer.supersede_document(supersede_old)
            console.print(f"[green]Marked {updated} chunks as inactive for {supersede_old}[/green]")
        except Exception as exc:
            console.print(f"[red]Warning: Could not supersede {supersede_old}: {exc}[/red]")
            logger.warning(
                "supersession_failed_continuing",
                old_circular=supersede_old,
                error=str(exc),
            )

    # Step 2: Parse and chunk the document
    console.print("\n[cyan]Step 1/3:[/cyan] Parsing PDF and chunking…")
    t_start = time.perf_counter()

    metadata = DocumentMetadata(
        document_title=doc_title,
        issuing_body=parsed_body,
        circular_number=circular_number,
        effective_date=parsed_date,
        is_active=True,
        version=version,
        source_file=str(file),
        total_pages=0,  # Will be updated by ingest_document
    )

    try:
        chunks: list[DocumentChunk] = ingest_document(
            file_path=file,
            metadata=metadata,
            chunking_config=settings.chunking,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]File not found:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        console.print(f"[red]PDF parsing failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    parse_duration = time.perf_counter() - t_start
    console.print(f"  [green]✓[/green] Parsed {len(chunks)} chunks in {parse_duration:.1f}s")

    if not chunks:
        console.print("[yellow]No chunks produced — the document may be empty or unparseable.[/yellow]")
        raise typer.Exit(code=0)

    # Step 3: Embed and index
    console.print("[cyan]Step 2/3:[/cyan] Generating embeddings…")
    console.print("[cyan]Step 3/3:[/cyan] Upserting to Pinecone + updating BM25 index…")

    try:
        result: IndexingResult = index_document(
            chunks=chunks,
            settings=settings,
        )
    except ValueError as exc:
        console.print(f"[red]Indexing error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[red]Indexing failed:[/red] {exc}")
        logger.exception("indexing_failed")
        raise typer.Exit(code=1) from exc

    # Display result
    if output_json:
        console.print(result.model_dump_json(indent=2))
    else:
        _display_indexing_result(result)

    console.print("\n[bold green]✅ Ingestion complete.[/bold green]")


@app.command()
def query(
    question: str = typer.Argument(
        ...,
        help="The compliance question to answer.",
    ),
    issuing_body: str | None = typer.Option(
        None,
        "--issuing-body",
        "-b",
        help="Filter results to a specific issuing body (RBI, SEBI, etc.).",
    ),
    circular_number: str | None = typer.Option(
        None,
        "--circular",
        "-c",
        help="Filter results to a specific circular number.",
    ),
    min_date: str | None = typer.Option(
        None,
        "--min-date",
        help="Only retrieve circulars effective on or after this date (YYYY-MM-DD).",
    ),
    max_date: str | None = typer.Option(
        None,
        "--max-date",
        help="Only retrieve circulars effective on or before this date (YYYY-MM-DD).",
    ),
    include_inactive: bool = typer.Option(
        False,
        "--include-inactive",
        help="Include superseded (inactive) circulars in retrieval.",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Output the full response as JSON instead of rich formatting.",
    ),
) -> None:
    """Query the compliance RAG system.

    Runs the full pipeline: hybrid search (BM25 + dense) → alpha-blended
    fusion → temporal boost → Cohere reranking → LLM generation →
    hallucination validation.
    """
    _configure_logging()
    settings = _load_settings_or_exit()

    # Build metadata filter
    filter_body: IssuingBody | None = None
    if issuing_body:
        filter_body = _parse_issuing_body(issuing_body)

    parsed_min_date: date | None = None
    if min_date:
        try:
            parsed_min_date = date.fromisoformat(min_date)
        except ValueError as exc:
            console.print(f"[red]Invalid min-date format:[/red] {min_date}. Use YYYY-MM-DD.")
            raise typer.Exit(code=1) from exc

    parsed_max_date: date | None = None
    if max_date:
        try:
            parsed_max_date = date.fromisoformat(max_date)
        except ValueError as exc:
            console.print(f"[red]Invalid max-date format:[/red] {max_date}. Use YYYY-MM-DD.")
            raise typer.Exit(code=1) from exc

    filters = MetadataFilter(
        issuing_body=filter_body,
        min_effective_date=parsed_min_date,
        max_effective_date=parsed_max_date,
        circular_number=circular_number,
        active_only=not include_inactive,
    )

    console.print(
        Panel(
            f"[bold]Query:[/bold] {question}\n"
            f"[bold]Filters:[/bold] {filters.model_dump(exclude_none=True, exclude_defaults=True) or 'None'}",
            title="🔍 Compliance Query",
            border_style="cyan",
        )
    )

    # Initialize services
    console.print("\n[cyan]Initializing services…[/cyan]")

    try:
        embedding_svc = EmbeddingService(
            config=settings.embedding,
            openai_api_key=settings.api_keys.openai_api_key,
        )

        pinecone_idx = PineconeIndexer(
            config=settings.pinecone,
            api_key=settings.api_keys.pinecone_api_key,
        )

        bm25_mgr = BM25IndexManager()
        try:
            bm25_mgr.load_index(
                settings.bm25_index_path,
                settings.bm25_chunks_path,
            )
        except FileNotFoundError:
            console.print(
                "[yellow]⚠ No BM25 index found. Sparse search will be unavailable. "
                "Run 'ingest' first to build the index.[/yellow]"
            )
            bm25_mgr.build_index([])

    except Exception as exc:
        console.print(f"[red]Service initialization failed:[/red] {exc}")
        logger.exception("service_init_failed")
        raise typer.Exit(code=1) from exc

    # Step 1: Retrieval
    console.print("[cyan]Step 1/2:[/cyan] Running hybrid retrieval + reranking…")
    t_start = time.perf_counter()

    try:
        retrieval_result: RetrievalResult = retrieve(
            query=question,
            settings=settings,
            embedding_service=embedding_svc,
            pinecone_indexer=pinecone_idx,
            bm25_manager=bm25_mgr,
            filters=filters,
        )
    except Exception as exc:
        console.print(f"[red]Retrieval failed:[/red] {exc}")
        logger.exception("retrieval_failed")
        raise typer.Exit(code=1) from exc

    retrieval_duration = time.perf_counter() - t_start
    console.print(
        f"  [green]✓[/green] Retrieved {len(retrieval_result.scored_chunks)} chunks "
        f"(sparse: {retrieval_result.total_sparse_hits}, "
        f"dense: {retrieval_result.total_dense_hits}, "
        f"reranked: {retrieval_result.reranking_applied}) "
        f"in {retrieval_duration:.1f}s"
    )

    if not retrieval_result.scored_chunks:
        console.print(
            "\n[yellow]No relevant regulatory documents found for this query. "
            "Ensure documents have been ingested first.[/yellow]"
        )
        raise typer.Exit(code=0)

    # Step 2: Generation + Validation
    console.print("[cyan]Step 2/2:[/cyan] Generating compliance response + validation…")

    try:
        response: ComplianceResponse = generate_compliance_response(
            query=question,
            context_chunks=retrieval_result.scored_chunks,
            settings=settings,
        )
    except Exception as exc:
        console.print(f"[red]Generation failed:[/red] {exc}")
        logger.exception("generation_failed")
        raise typer.Exit(code=1) from exc

    # Display result
    if output_json:
        console.print(response.model_dump_json(indent=2))
    else:
        _display_compliance_response(response)

    console.print()


@app.command()
def supersede(
    old_circular: str = typer.Option(
        ...,
        "--old-circular",
        help="The circular number to mark as superseded (inactive).",
    ),
    new_circular: str | None = typer.Option(
        None,
        "--new-circular",
        help="The replacing circular number (for audit logging only).",
    ),
    delete: bool = typer.Option(
        False,
        "--delete",
        help="Permanently delete the old circular's vectors instead of marking inactive.",
    ),
) -> None:
    """Mark an older circular as superseded by a newer one.

    By default, sets ``is_active=False`` on all vectors belonging to the
    old circular. With ``--delete``, permanently removes them instead.
    """
    _configure_logging()
    settings = _load_settings_or_exit()

    console.print(
        Panel(
            f"[bold]Old Circular:[/bold] {old_circular}\n"
            f"[bold]New Circular:[/bold] {new_circular or 'N/A'}\n"
            f"[bold]Action:[/bold] {'DELETE' if delete else 'Mark Inactive'}",
            title="🔄 Supersession",
            border_style="yellow",
        )
    )

    try:
        pinecone_indexer = PineconeIndexer(
            config=settings.pinecone,
            api_key=settings.api_keys.pinecone_api_key,
        )
    except Exception as exc:
        console.print(f"[red]Failed to connect to Pinecone:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if delete:
        console.print(f"\n[red]Permanently deleting vectors for {old_circular}…[/red]")
        try:
            count = pinecone_indexer.delete_document(old_circular)
            console.print(f"[green]✓ Deleted {count} vectors for {old_circular}.[/green]")
        except Exception as exc:
            console.print(f"[red]Deletion failed:[/red] {exc}")
            logger.exception("deletion_failed", circular=old_circular)
            raise typer.Exit(code=1) from exc
    else:
        console.print(f"\n[yellow]Marking {old_circular} as inactive…[/yellow]")
        try:
            count = pinecone_indexer.supersede_document(old_circular)
            console.print(f"[green]✓ Marked {count} vectors as inactive for {old_circular}.[/green]")
        except Exception as exc:
            console.print(f"[red]Supersession failed:[/red] {exc}")
            logger.exception("supersession_failed", circular=old_circular)
            raise typer.Exit(code=1) from exc

    if new_circular:
        console.print(f"\n[dim]Audit log: {old_circular} superseded by {new_circular}[/dim]")

    console.print("\n[bold green]✅ Supersession complete.[/bold green]")


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------


def run_ingestion(
    file_path: Path,
    issuing_body: IssuingBody,
    circular_number: str,
    effective_date: date,
    document_title: str = "",
    version: int = 1,
    supersede_old: str | None = None,
    settings: AppSettings | None = None,
) -> IndexingResult:
    """Programmatic API for document ingestion.

    Provides the same functionality as the ``ingest`` CLI command but
    returns a typed ``IndexingResult`` for use in scripts and services.

    Args:
        file_path: Path to the regulatory PDF.
        issuing_body: Regulatory body that issued the document.
        circular_number: Official circular reference.
        effective_date: Date the circular comes into force.
        document_title: Human-readable document title.
        version: Version number for supersession tracking.
        supersede_old: If set, marks the specified old circular as inactive.
        settings: Application settings. Loaded from env if not provided.

    Returns:
        ``IndexingResult`` with statistics about the indexing operation.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError: If PDF parsing fails.
        ValueError: If no chunks are produced.
    """
    _configure_logging()

    if settings is None:
        settings = load_settings()

    log = logger.bind(operation="run_ingestion", circular=circular_number)
    log.info("programmatic_ingestion_started", file=str(file_path))

    # Handle supersession
    if supersede_old:
        pinecone_indexer = PineconeIndexer(
            config=settings.pinecone,
            api_key=settings.api_keys.pinecone_api_key,
        )
        updated = pinecone_indexer.supersede_document(supersede_old)
        log.info("supersession_applied", old_circular=supersede_old, updated_count=updated)

    # Build metadata
    doc_title = document_title if document_title else file_path.stem.replace("_", " ").title()
    metadata = DocumentMetadata(
        document_title=doc_title,
        issuing_body=issuing_body,
        circular_number=circular_number,
        effective_date=effective_date,
        is_active=True,
        version=version,
        source_file=str(file_path),
        total_pages=0,
    )

    # Parse and chunk
    chunks = ingest_document(
        file_path=file_path,
        metadata=metadata,
        chunking_config=settings.chunking,
    )

    if not chunks:
        raise ValueError(f"No chunks produced from {file_path}")

    # Index
    result = index_document(chunks=chunks, settings=settings)

    log.info(
        "programmatic_ingestion_complete",
        chunks=result.total_chunks,
        indexed=result.successfully_indexed,
        duration=result.duration_seconds,
    )
    return result


def run_query(
    question: str,
    filters: MetadataFilter | None = None,
    settings: AppSettings | None = None,
) -> ComplianceResponse:
    """Programmatic API for compliance queries.

    Provides the same functionality as the ``query`` CLI command but
    returns a typed ``ComplianceResponse`` for use in scripts and services.

    Args:
        question: Natural-language compliance question.
        filters: Optional metadata constraints for retrieval.
        settings: Application settings. Loaded from env if not provided.

    Returns:
        ``ComplianceResponse`` with the verified (or flagged) answer.
    """
    _configure_logging()

    if settings is None:
        settings = load_settings()

    log = logger.bind(operation="run_query")
    log.info("programmatic_query_started", query_preview=question[:120])

    # Initialize services
    embedding_svc = EmbeddingService(
        config=settings.embedding,
        openai_api_key=settings.api_keys.openai_api_key,
    )
    pinecone_idx = PineconeIndexer(
        config=settings.pinecone,
        api_key=settings.api_keys.pinecone_api_key,
    )
    bm25_mgr = BM25IndexManager()
    try:
        bm25_mgr.load_index(settings.bm25_index_path, settings.bm25_chunks_path)
    except FileNotFoundError:
        log.warning("bm25_index_not_found_building_empty")
        bm25_mgr.build_index([])

    # Retrieve
    retrieval_result = retrieve(
        query=question,
        settings=settings,
        embedding_service=embedding_svc,
        pinecone_indexer=pinecone_idx,
        bm25_manager=bm25_mgr,
        filters=filters,
    )

    # Generate
    response = generate_compliance_response(
        query=question,
        context_chunks=retrieval_result.scored_chunks,
        settings=settings,
    )

    log.info(
        "programmatic_query_complete",
        is_verified=response.is_verified,
        citations=len(response.citations),
    )
    return response


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
