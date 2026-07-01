"""Layout-aware PDF parsing, metadata extraction, and legal-structural chunking.

Handles RBI / FEMA regulatory documents by:
1. Extracting text with layout information via *pdfplumber*.
2. Falling back to *unstructured* for complex scanned layouts.
3. Splitting content into token-bounded chunks that never break
   mid-clause and carry a deterministic ``chunk_id`` plus a full
   hierarchical path (e.g. ``Chapter III > Section 6 > Clause 6.2(a)``).
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal

import structlog
import tiktoken
from pydantic import BaseModel, Field

from rag_pipeline.config import ChunkingConfig

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns compiled once at module level for heading detection
# ---------------------------------------------------------------------------
_CHAPTER_RE = re.compile(r"^(?:CHAPTER|Chapter)\s+[IVXLCDM0-9]+", re.MULTILINE)
_SECTION_RE = re.compile(r"^(?:SECTION|Section)\s+\d+", re.MULTILINE)
_SUBSECTION_RE = re.compile(r"^\d+\.\d+", re.MULTILINE)
_CLAUSE_RE = re.compile(r"^(?:\(\w+\)|Clause\s+)", re.MULTILINE)


# ===================================================================
# Pydantic Models
# ===================================================================

class IssuingBody(str, Enum):
    """Enumeration of recognised regulatory / issuing bodies."""

    RBI = "RBI"
    SEBI = "SEBI"
    IT_DEPT = "IT_DEPT"
    FEMA = "FEMA"
    OTHER = "OTHER"


class DocumentMetadata(BaseModel):
    """Metadata for a single regulatory document.

    Tracks provenance (issuing body, circular number), temporal validity
    (effective date, active flag), and versioning.
    """

    document_title: str = Field(
        ..., description="Full title of the regulatory document."
    )
    issuing_body: IssuingBody = Field(
        ..., description="Regulatory body that issued the document."
    )
    circular_number: str = Field(
        ...,
        description="Official circular reference, e.g. 'RBI/2024-25/42'.",
    )
    effective_date: date = Field(
        ..., description="Date from which the circular is in force."
    )
    is_active: bool = Field(
        default=True,
        description="Whether the document is currently in force.",
    )
    version: int = Field(
        default=1,
        description="Internal version counter for superseded circulars.",
    )
    source_file: str = Field(
        ..., description="Original filename or path of the ingested PDF."
    )
    total_pages: int = Field(
        ..., ge=0, description="Total number of pages in the source PDF."
    )


class HierarchicalElement(BaseModel):
    """A single layout element within a parsed page.

    Each element carries its nesting level (0=document … 4=clause),
    the extracted text, a page reference, and an element-type tag.
    """

    level: int = Field(
        ...,
        ge=0,
        le=4,
        description="Nesting depth: 0=document, 1=chapter, 2=section, 3=subsection, 4=clause.",
    )
    heading: str = Field(
        default="",
        description="Heading text if the element represents a heading.",
    )
    content: str = Field(
        ..., description="Full textual content of the element."
    )
    page_number: int = Field(..., ge=1, description="1-indexed source page.")
    element_type: Literal["heading", "paragraph", "table", "list_item"] = Field(
        ..., description="Semantic type of the element."
    )


class ParsedPage(BaseModel):
    """All extracted information for a single PDF page."""

    page_number: int = Field(..., ge=1, description="1-indexed page number.")
    raw_text: str = Field(
        default="", description="Concatenated raw text of the page."
    )
    tables: list[str] = Field(
        default_factory=list,
        description="Markdown-formatted table strings extracted from the page.",
    )
    elements: list[HierarchicalElement] = Field(
        default_factory=list,
        description="Structured hierarchical elements detected on the page.",
    )


class DocumentChunk(BaseModel):
    """A token-bounded chunk of document text ready for embedding.

    Carries a deterministic ``chunk_id``, a hierarchical path string,
    and all metadata required by the retrieval layer.
    """

    chunk_id: str = Field(
        ..., description="Deterministic SHA-256 hash identifying this chunk."
    )
    text: str = Field(..., description="Chunk text content.")
    token_count: int = Field(
        ..., ge=0, description="Number of tokens in the chunk text."
    )
    metadata: DocumentMetadata = Field(
        ..., description="Parent document metadata."
    )
    hierarchical_path: str = Field(
        default="",
        description="Path such as 'Chapter III > Section 6 > Clause 6.2(a)'.",
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="Pages spanned by this chunk.",
    )
    contains_table: bool = Field(
        default=False,
        description="Whether the chunk includes tabular data.",
    )
    chunk_index: int = Field(
        ..., ge=0, description="0-indexed position within the document."
    )


# ===================================================================
# LayoutAwareParser
# ===================================================================

class LayoutAwareParser:
    """Layout-aware PDF parser with *pdfplumber* primary and *unstructured* fallback.

    Extracts raw text, markdown-formatted tables, and hierarchical
    structural elements from each page of a regulatory PDF.
    """

    def __init__(self) -> None:
        self._log: structlog.stdlib.BoundLogger = logger.bind(component="LayoutAwareParser")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_pdf(self, file_path: Path) -> list[ParsedPage]:
        """Parse a PDF file into a list of ``ParsedPage`` objects.

        Attempts *pdfplumber* first. If it raises an exception the parser
        falls back to the *unstructured* library.

        Args:
            file_path: Filesystem path to the source PDF.

        Returns:
            Ordered list of parsed pages.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            RuntimeError: If both primary and fallback parsing fail.
        """
        resolved_path = file_path.resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"PDF not found: {resolved_path}")

        self._log.info("parse_pdf.start", file_path=str(resolved_path))

        try:
            import pdfplumber  # type: ignore[import-untyped]

            pages: list[ParsedPage] = []
            with pdfplumber.open(str(resolved_path)) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    try:
                        raw_text = page.extract_text() or ""
                    except Exception as text_err:
                        self._log.warning(
                            "parse_pdf.text_extraction_failed",
                            page=page_idx,
                            error=str(text_err),
                        )
                        raw_text = ""

                    tables = self._extract_tables(page)
                    elements = self._build_elements(raw_text, tables, page_idx)

                    pages.append(
                        ParsedPage(
                            page_number=page_idx,
                            raw_text=raw_text,
                            tables=tables,
                            elements=elements,
                        )
                    )

            self._log.info(
                "parse_pdf.pdfplumber_success",
                pages_parsed=len(pages),
            )
            return pages

        except ImportError:
            self._log.error(
                "parse_pdf.pdfplumber_import_error",
                hint="Install pdfplumber: pip install pdfplumber",
            )
            raise RuntimeError("pdfplumber is not installed.")

        except Exception as pdfplumber_err:
            self._log.warning(
                "parse_pdf.pdfplumber_failed_falling_back",
                error=str(pdfplumber_err),
            )
            try:
                return self._fallback_parse_with_unstructured(resolved_path)
            except Exception as fallback_err:
                self._log.error(
                    "parse_pdf.fallback_also_failed",
                    primary_error=str(pdfplumber_err),
                    fallback_error=str(fallback_err),
                )
                raise RuntimeError(
                    f"Both pdfplumber and unstructured failed for {resolved_path}. "
                    f"Primary: {pdfplumber_err!r} — Fallback: {fallback_err!r}"
                ) from fallback_err

    # ------------------------------------------------------------------
    # Table extraction
    # ------------------------------------------------------------------

    def _extract_tables(self, page: object) -> list[str]:
        """Extract tables from a *pdfplumber* page and convert to Markdown.

        Args:
            page: A ``pdfplumber.page.Page`` instance.

        Returns:
            List of Markdown-formatted table strings.
        """
        markdown_tables: list[str] = []

        try:
            raw_tables: list[list[list[str | None]]] | None = page.extract_tables()  # type: ignore[union-attr]
            if not raw_tables:
                return markdown_tables

            for table in raw_tables:
                md = self._table_to_markdown(table)
                if md:
                    markdown_tables.append(md)

        except Exception as table_err:
            self._log.warning(
                "extract_tables.failed",
                error=str(table_err),
            )

        return markdown_tables

    @staticmethod
    def _table_to_markdown(table: list[list[str | None]]) -> str:
        """Convert a 2-D list (pdfplumber table) to a Markdown string.

        Args:
            table: Rows × Columns matrix; cells may be ``None``.

        Returns:
            Markdown table string or empty string if the table is empty.
        """
        if not table:
            return ""

        cleaned_rows: list[list[str]] = []
        for row in table:
            cleaned_rows.append(
                [
                    (cell or "").replace("\n", " ").strip()
                    for cell in row
                ]
            )

        if not cleaned_rows:
            return ""

        col_count = max(len(row) for row in cleaned_rows)
        # Pad short rows
        for row in cleaned_rows:
            while len(row) < col_count:
                row.append("")

        lines: list[str] = []
        header = "| " + " | ".join(cleaned_rows[0]) + " |"
        separator = "| " + " | ".join("---" for _ in range(col_count)) + " |"
        lines.append(header)
        lines.append(separator)
        for row in cleaned_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Heading / element detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_heading_level(text: str) -> int | None:
        """Detect a hierarchical heading level from the text.

        Returns:
            Integer level (1-4) or ``None`` if no heading pattern matches.
        """
        stripped = text.strip()
        if _CHAPTER_RE.match(stripped):
            return 1
        if _SECTION_RE.match(stripped):
            return 2
        if _SUBSECTION_RE.match(stripped):
            return 3
        if _CLAUSE_RE.match(stripped):
            return 4
        return None

    def _build_elements(
        self,
        raw_text: str,
        tables: list[str],
        page_number: int,
    ) -> list[HierarchicalElement]:
        """Construct ``HierarchicalElement`` objects from raw text and tables.

        Text is split on blank lines. Each paragraph is classified as a
        heading (with its level) or a paragraph / list-item.

        Args:
            raw_text: Full page text.
            tables: Markdown tables already extracted for this page.
            page_number: 1-indexed page number.

        Returns:
            Ordered list of hierarchical elements for the page.
        """
        elements: list[HierarchicalElement] = []

        # Split text into non-empty paragraphs.
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw_text) if p.strip()]

        for para in paragraphs:
            level = self._detect_heading_level(para)
            if level is not None:
                elements.append(
                    HierarchicalElement(
                        level=level,
                        heading=para,
                        content=para,
                        page_number=page_number,
                        element_type="heading",
                    )
                )
            elif re.match(r"^\s*[\-\u2022\u25cf\u2013•]\s+", para):
                elements.append(
                    HierarchicalElement(
                        level=4,
                        heading="",
                        content=para,
                        page_number=page_number,
                        element_type="list_item",
                    )
                )
            else:
                elements.append(
                    HierarchicalElement(
                        level=4,
                        heading="",
                        content=para,
                        page_number=page_number,
                        element_type="paragraph",
                    )
                )

        # Append tables as dedicated elements.
        for tbl in tables:
            elements.append(
                HierarchicalElement(
                    level=4,
                    heading="",
                    content=tbl,
                    page_number=page_number,
                    element_type="table",
                )
            )

        return elements

    # ------------------------------------------------------------------
    # Fallback parser
    # ------------------------------------------------------------------

    def _fallback_parse_with_unstructured(
        self, file_path: Path
    ) -> list[ParsedPage]:
        """Parse a PDF using the *unstructured* library as a fallback.

        Args:
            file_path: Resolved path to the PDF.

        Returns:
            List of ``ParsedPage`` objects constructed from *unstructured*
            elements.

        Raises:
            ImportError: If *unstructured* is not installed.
            RuntimeError: If partition_pdf itself fails.
        """
        self._log.info(
            "fallback_parse_with_unstructured.start",
            file_path=str(file_path),
        )

        try:
            from unstructured.partition.pdf import partition_pdf  # type: ignore[import-untyped]
        except ImportError as imp_err:
            self._log.error(
                "fallback_parse.unstructured_not_installed",
                hint="pip install 'unstructured[pdf]'",
            )
            raise ImportError(
                "unstructured is not installed — cannot fall back."
            ) from imp_err

        try:
            elements = partition_pdf(
                filename=str(file_path),
                strategy="hi_res",
                infer_table_structure=True,
            )
        except Exception as part_err:
            self._log.error(
                "fallback_parse.partition_pdf_failed",
                error=str(part_err),
            )
            raise RuntimeError(
                f"unstructured partition_pdf failed: {part_err!r}"
            ) from part_err

        page_map: dict[int, ParsedPage] = {}

        for elem in elements:
            page_num: int = int(getattr(elem.metadata, "page_number", 1) or 1)
            if page_num not in page_map:
                page_map[page_num] = ParsedPage(
                    page_number=page_num,
                    raw_text="",
                    tables=[],
                    elements=[],
                )

            parsed_page = page_map[page_num]
            text_content: str = str(elem)

            # Accumulate raw text
            parsed_page.raw_text = (
                (parsed_page.raw_text + "\n" + text_content).strip()
            )

            category: str = getattr(elem, "category", "NarrativeText") or "NarrativeText"

            if category == "Table":
                parsed_page.tables.append(text_content)
                parsed_page.elements.append(
                    HierarchicalElement(
                        level=4,
                        heading="",
                        content=text_content,
                        page_number=page_num,
                        element_type="table",
                    )
                )
            elif category in {"Title", "Header"}:
                level = self._detect_heading_level(text_content) or 1
                parsed_page.elements.append(
                    HierarchicalElement(
                        level=level,
                        heading=text_content,
                        content=text_content,
                        page_number=page_num,
                        element_type="heading",
                    )
                )
            elif category == "ListItem":
                parsed_page.elements.append(
                    HierarchicalElement(
                        level=4,
                        heading="",
                        content=text_content,
                        page_number=page_num,
                        element_type="list_item",
                    )
                )
            else:
                level = self._detect_heading_level(text_content)
                if level is not None:
                    parsed_page.elements.append(
                        HierarchicalElement(
                            level=level,
                            heading=text_content,
                            content=text_content,
                            page_number=page_num,
                            element_type="heading",
                        )
                    )
                else:
                    parsed_page.elements.append(
                        HierarchicalElement(
                            level=4,
                            heading="",
                            content=text_content,
                            page_number=page_num,
                            element_type="paragraph",
                        )
                    )

        sorted_pages = sorted(page_map.values(), key=lambda p: p.page_number)
        self._log.info(
            "fallback_parse_with_unstructured.success",
            pages_parsed=len(sorted_pages),
        )
        return sorted_pages


# ===================================================================
# LegalStructuralChunker
# ===================================================================

class LegalStructuralChunker:
    """Token-bounded, clause-boundary-respecting chunker for legal documents.

    Guarantees that:
    * No chunk exceeds ``max_chunk_tokens``.
    * Clauses are never split mid-sentence.
    * Under-sized chunks are merged with neighbours (but never across
      section boundaries).
    * Every chunk receives a deterministic ``chunk_id`` derived from the
      document's circular number, chunk position, and leading text.
    """

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config
        self._log: structlog.stdlib.BoundLogger = logger.bind(
            component="LegalStructuralChunker"
        )

        try:
            self._encoder: tiktoken.Encoding = tiktoken.get_encoding(
                config.tokenizer_model
            )
        except Exception as enc_err:
            self._log.error(
                "chunker_init.tokenizer_load_failed",
                tokenizer_model=config.tokenizer_model,
                error=str(enc_err),
            )
            raise ValueError(
                f"Failed to load tiktoken encoding '{config.tokenizer_model}': {enc_err!r}"
            ) from enc_err

        self._log.info(
            "chunker_init.success",
            max_chunk_tokens=config.max_chunk_tokens,
            overlap_tokens=config.overlap_tokens,
            min_chunk_tokens=config.min_chunk_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(
        self,
        parsed_pages: list[ParsedPage],
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:
        """Chunk a parsed document into embedding-ready ``DocumentChunk`` objects.

        Processing steps:
        1. Flatten all elements across pages.
        2. Identify clause / section boundaries.
        3. Group elements into boundary-respecting segments.
        4. Split or join segments to satisfy ``max_chunk_tokens`` /
           ``min_chunk_tokens``.
        5. Assign deterministic ``chunk_id`` values.

        Args:
            parsed_pages: Output of ``LayoutAwareParser.parse_pdf``.
            metadata: Document-level metadata to attach to every chunk.

        Returns:
            Ordered list of ``DocumentChunk`` instances.
        """
        self._log.info(
            "chunk_document.start",
            circular_number=metadata.circular_number,
            pages=len(parsed_pages),
        )

        # 1. Flatten elements.
        all_elements: list[HierarchicalElement] = []
        for page in parsed_pages:
            all_elements.extend(page.elements)

        if not all_elements:
            self._log.warning("chunk_document.no_elements")
            return []

        # 2. Detect clause boundaries.
        boundaries = self._detect_clause_boundaries(all_elements)

        # 3. Group elements into boundary-respecting segments.
        segments = self._group_by_boundaries(all_elements, boundaries)

        # 4. Build raw chunks respecting max_chunk_tokens.
        raw_chunks = self._segments_to_chunks(segments, metadata)

        # 5. Merge small chunks.
        merged_chunks = self._merge_small_chunks(raw_chunks)

        # 6. Re-index and assign deterministic IDs.
        final_chunks: list[DocumentChunk] = []
        for idx, chunk in enumerate(merged_chunks):
            chunk_id = self._generate_chunk_id(
                metadata.circular_number, idx, chunk.text
            )
            final_chunks.append(
                chunk.model_copy(
                    update={
                        "chunk_id": chunk_id,
                        "chunk_index": idx,
                        "token_count": self._count_tokens(chunk.text),
                    }
                )
            )

        self._log.info(
            "chunk_document.complete",
            total_chunks=len(final_chunks),
        )
        return final_chunks

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken.

        Args:
            text: Input string.

        Returns:
            Token count.
        """
        return len(self._encoder.encode(text))

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------

    def _detect_clause_boundaries(
        self, elements: list[HierarchicalElement]
    ) -> list[int]:
        """Return indices at which a new clause or section begins.

        An element is considered a boundary when:
        * It is a heading (any level), **or**
        * Its level is ≤ 3 (chapter / section / subsection).

        The first element is always a boundary.

        Args:
            elements: Flattened element list.

        Returns:
            Sorted list of boundary indices.
        """
        boundaries: list[int] = [0]
        for idx in range(1, len(elements)):
            elem = elements[idx]
            if elem.element_type == "heading" or elem.level <= 3:
                boundaries.append(idx)
        return boundaries

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_boundaries(
        elements: list[HierarchicalElement],
        boundaries: list[int],
    ) -> list[list[HierarchicalElement]]:
        """Group elements into segments delimited by *boundaries*.

        Args:
            elements: Full flattened element list.
            boundaries: Sorted start-indices of each segment.

        Returns:
            List of element groups.
        """
        segments: list[list[HierarchicalElement]] = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(elements)
            segment = elements[start:end]
            if segment:
                segments.append(segment)
        return segments

    # ------------------------------------------------------------------
    # Segment → Chunk conversion
    # ------------------------------------------------------------------

    def _segments_to_chunks(
        self,
        segments: list[list[HierarchicalElement]],
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:
        """Convert boundary-respecting segments into ``DocumentChunk`` objects.

        If a single segment exceeds ``max_chunk_tokens`` its content is
        split on sentence / line boundaries within the segment.

        Args:
            segments: Element groups from ``_group_by_boundaries``.
            metadata: Document metadata.

        Returns:
            Raw (pre-merge) chunk list.
        """
        chunks: list[DocumentChunk] = []
        # Maintain a heading stack to build hierarchical paths.
        heading_stack: list[str] = []

        flat_idx = 0  # running element index across segments

        for segment in segments:
            # Update heading stack with any headings in this segment.
            for elem in segment:
                if elem.element_type == "heading" and elem.heading:
                    level = elem.level
                    # Trim stack to one-below current level, then push.
                    heading_stack = heading_stack[: max(level - 1, 0)]
                    heading_stack.append(elem.heading.strip())

            hierarchical_path = " > ".join(heading_stack) if heading_stack else ""

            segment_text = "\n\n".join(e.content for e in segment)
            page_numbers = sorted({e.page_number for e in segment})
            contains_table = any(e.element_type == "table" for e in segment)
            token_count = self._count_tokens(segment_text)

            if token_count <= self._config.max_chunk_tokens:
                chunks.append(
                    DocumentChunk(
                        chunk_id="",  # placeholder — assigned later
                        text=segment_text,
                        token_count=token_count,
                        metadata=metadata,
                        hierarchical_path=hierarchical_path,
                        page_numbers=page_numbers,
                        contains_table=contains_table,
                        chunk_index=0,  # re-indexed later
                    )
                )
            else:
                # Split oversized segment on sentence boundaries.
                sub_chunks = self._split_oversized_segment(
                    segment_text,
                    page_numbers=page_numbers,
                    hierarchical_path=hierarchical_path,
                    contains_table=contains_table,
                    metadata=metadata,
                )
                chunks.extend(sub_chunks)

            flat_idx += len(segment)

        return chunks

    def _split_oversized_segment(
        self,
        text: str,
        *,
        page_numbers: list[int],
        hierarchical_path: str,
        contains_table: bool,
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:
        """Split text that exceeds *max_chunk_tokens* on sentence boundaries.

        Uses a greedy approach: accumulate sentences until the token limit
        is about to be exceeded, then start a new chunk with an overlap
        of ``overlap_tokens`` worth of trailing text from the previous chunk.

        Args:
            text: Oversized segment text.
            page_numbers: Pages covered by the segment.
            hierarchical_path: Path string for all resulting chunks.
            contains_table: Whether any table is present.
            metadata: Document metadata.

        Returns:
            List of sub-chunks.
        """
        # Split on sentence-ending punctuation or double newline.
        sentences = re.split(r"(?<=[.;:!?\n])\s+", text)
        sentences = [s for s in sentences if s.strip()]

        sub_chunks: list[DocumentChunk] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if (
                current_tokens + sentence_tokens > self._config.max_chunk_tokens
                and current_sentences
            ):
                chunk_text = " ".join(current_sentences)
                sub_chunks.append(
                    DocumentChunk(
                        chunk_id="",
                        text=chunk_text,
                        token_count=self._count_tokens(chunk_text),
                        metadata=metadata,
                        hierarchical_path=hierarchical_path,
                        page_numbers=page_numbers,
                        contains_table=contains_table,
                        chunk_index=0,
                    )
                )

                # Carry overlap from the end of the previous chunk.
                overlap_text = chunk_text
                overlap_tokens = self._count_tokens(overlap_text)
                overlap_sentences: list[str] = []
                for s in reversed(current_sentences):
                    s_tok = self._count_tokens(s)
                    if overlap_tokens <= self._config.overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens -= s_tok

                current_sentences = overlap_sentences
                current_tokens = self._count_tokens(" ".join(current_sentences))

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Flush remaining.
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            sub_chunks.append(
                DocumentChunk(
                    chunk_id="",
                    text=chunk_text,
                    token_count=self._count_tokens(chunk_text),
                    metadata=metadata,
                    hierarchical_path=hierarchical_path,
                    page_numbers=page_numbers,
                    contains_table=contains_table,
                    chunk_index=0,
                )
            )

        return sub_chunks

    # ------------------------------------------------------------------
    # Small-chunk merging
    # ------------------------------------------------------------------

    def _merge_small_chunks(
        self, chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """Merge chunks smaller than *min_chunk_tokens* with a neighbour.

        Merge direction:
        * Prefer merging with the **predecessor** chunk.
        * If no predecessor exists (first chunk), merge with the successor.
        * Never merge across different ``hierarchical_path`` values
          (i.e. section boundaries are respected).

        Args:
            chunks: Pre-merge chunk list.

        Returns:
            Chunk list with small chunks absorbed.
        """
        if not chunks:
            return chunks

        merged: list[DocumentChunk] = [chunks[0]]

        for chunk in chunks[1:]:
            prev = merged[-1]
            tok = self._count_tokens(chunk.text)

            # Can merge with predecessor?
            same_section = prev.hierarchical_path == chunk.hierarchical_path
            combined_tokens = self._count_tokens(prev.text) + tok

            if (
                tok < self._config.min_chunk_tokens
                and same_section
                and combined_tokens <= self._config.max_chunk_tokens
            ):
                merged_text = prev.text + "\n\n" + chunk.text
                merged_pages = sorted(set(prev.page_numbers + chunk.page_numbers))
                merged[-1] = prev.model_copy(
                    update={
                        "text": merged_text,
                        "token_count": self._count_tokens(merged_text),
                        "page_numbers": merged_pages,
                        "contains_table": prev.contains_table or chunk.contains_table,
                    }
                )
            else:
                merged.append(chunk)

        # Second pass: handle first-chunk undersize by merging forward.
        if (
            len(merged) >= 2
            and self._count_tokens(merged[0].text) < self._config.min_chunk_tokens
            and merged[0].hierarchical_path == merged[1].hierarchical_path
        ):
            combined_tokens = (
                self._count_tokens(merged[0].text)
                + self._count_tokens(merged[1].text)
            )
            if combined_tokens <= self._config.max_chunk_tokens:
                merged_text = merged[0].text + "\n\n" + merged[1].text
                merged_pages = sorted(
                    set(merged[0].page_numbers + merged[1].page_numbers)
                )
                merged[1] = merged[1].model_copy(
                    update={
                        "text": merged_text,
                        "token_count": self._count_tokens(merged_text),
                        "page_numbers": merged_pages,
                        "contains_table": merged[0].contains_table
                        or merged[1].contains_table,
                    }
                )
                merged.pop(0)

        return merged

    # ------------------------------------------------------------------
    # Hierarchical path building
    # ------------------------------------------------------------------

    def _build_hierarchical_path(
        self,
        elements: list[HierarchicalElement],
        current_idx: int,
    ) -> str:
        """Build a hierarchical path by scanning headings up to *current_idx*.

        Walks backward from the beginning of the element list, maintaining
        a stack of headings keyed by their level.

        Args:
            elements: Full flattened element list.
            current_idx: Index of the target element.

        Returns:
            Path string like ``'Chapter III > Section 6 > Clause 6.2(a)'``.
        """
        stack: dict[int, str] = {}

        for idx in range(current_idx + 1):
            elem = elements[idx]
            if elem.element_type == "heading" and elem.heading:
                level = elem.level
                # Remove deeper levels from the stack.
                keys_to_remove = [k for k in stack if k >= level]
                for k in keys_to_remove:
                    del stack[k]
                stack[level] = elem.heading.strip()

        if not stack:
            return ""

        return " > ".join(
            stack[k] for k in sorted(stack)
        )

    # ------------------------------------------------------------------
    # Deterministic chunk ID
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_chunk_id(
        circular_number: str,
        chunk_index: int,
        text: str,
    ) -> str:
        """Generate a deterministic SHA-256 chunk ID.

        The hash is derived from ``circular_number + chunk_index +
        text[:100]`` to guarantee reproducibility.

        Args:
            circular_number: Document circular number.
            chunk_index: 0-indexed position of the chunk.
            text: Chunk text (only the first 100 chars are used).

        Returns:
            Hex-encoded SHA-256 digest.
        """
        payload = f"{circular_number}|{chunk_index}|{text[:100]}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ===================================================================
# Convenience function
# ===================================================================

def ingest_document(
    file_path: Path,
    metadata: DocumentMetadata,
    chunking_config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Parse and chunk a regulatory PDF in a single call.

    This is the primary entry-point for the ingestion sub-system. It
    creates a ``LayoutAwareParser``, parses the PDF, then feeds the
    result into a ``LegalStructuralChunker``.

    Args:
        file_path: Path to the source PDF file.
        metadata: Pre-populated document metadata. The ``total_pages``
            field will be updated to match the actual page count if it
            differs.
        chunking_config: Optional chunking configuration; defaults to
            ``ChunkingConfig()`` which reads from environment variables
            or uses built-in defaults.

    Returns:
        List of ``DocumentChunk`` objects ready for embedding and
        indexing.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        RuntimeError: If PDF parsing fails.
    """
    log: structlog.stdlib.BoundLogger = logger.bind(
        function="ingest_document",
        file_path=str(file_path),
        circular_number=metadata.circular_number,
    )
    log.info("ingest_document.start")

    config = chunking_config or ChunkingConfig()

    parser = LayoutAwareParser()
    parsed_pages = parser.parse_pdf(file_path)

    # Reconcile total_pages in metadata with actual parsed pages.
    actual_pages = len(parsed_pages)
    if metadata.total_pages != actual_pages:
        log.info(
            "ingest_document.page_count_reconciled",
            declared=metadata.total_pages,
            actual=actual_pages,
        )
        metadata = metadata.model_copy(update={"total_pages": actual_pages})

    chunker = LegalStructuralChunker(config)
    chunks = chunker.chunk_document(parsed_pages, metadata)

    log.info(
        "ingest_document.complete",
        total_chunks=len(chunks),
        total_pages=actual_pages,
    )
    return chunks
