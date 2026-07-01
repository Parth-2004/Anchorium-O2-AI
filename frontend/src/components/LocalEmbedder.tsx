/**
 * ==========================================================================
 * LocalEmbedder — In-Browser Document Embedding Component
 * ==========================================================================
 */

"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A single embedded chunk with its source text and 384-dim vector */
interface EmbeddingResult {
  text: string;
  vector: number[];
}

/** Status of the embedding pipeline */
type PipelineStatus =
  | "idle"
  | "reading-file"
  | "loading-model"
  | "embedding"
  | "complete"
  | "error";

/** Model download progress from the Web Worker */
interface DownloadProgress {
  loaded: number;
  total: number;
  file: string;
}

// ---------------------------------------------------------------------------
// Text Chunking — splits document text into overlapping windows
// ---------------------------------------------------------------------------

/**
 * Splits a large text into overlapping chunks suitable for embedding.
 */
function chunkText(text: string, chunkSize = 500, overlap = 50): string[] {
  const chunks: string[] = [];
  const cleanedText = text.replace(/\s+/g, " ").trim();

  if (cleanedText.length <= chunkSize) {
    return [cleanedText];
  }

  let start = 0;
  while (start < cleanedText.length) {
    const end = Math.min(start + chunkSize, cleanedText.length);
    const chunk = cleanedText.slice(start, end).trim();

    if (chunk.length > 0) {
      chunks.push(chunk);
    }

    start += chunkSize - overlap;
  }

  return chunks;
}

// ---------------------------------------------------------------------------
// PDF Text Extraction — Uses pdf.js loaded from CDN
// ---------------------------------------------------------------------------

async function extractPdfText(file: File): Promise<string> {
  const cdnUrl = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.9.155/+esm";
  const pdfjsLib = await (new Function("url", "return import(url)"))(cdnUrl);

  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.9.155/build/pdf.worker.min.mjs";

  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

  const pageTexts: string[] = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const textContent = await page.getTextContent();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const pageText = textContent.items.map((item: any) => item.str).join(" ");
    pageTexts.push(pageText);
  }

  return pageTexts.join("\n\n");
}

// ---------------------------------------------------------------------------
// LocalEmbedder Component
// ---------------------------------------------------------------------------

export default function LocalEmbedder({
  onEmbeddingsComplete,
}: {
  onEmbeddingsComplete?: (embeddings: EmbeddingResult[], fileName: string) => void;
}) {
  const [status, setStatus] = useState<PipelineStatus>("idle");
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [fileName, setFileName] = useState<string>("");
  const [chunks, setChunks] = useState<string[]>([]);
  const [embeddings, setEmbeddings] = useState<EmbeddingResult[]>([]);
  const [downloadProgress, setDownloadProgress] =
    useState<DownloadProgress | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [expandedChunk, setExpandedChunk] = useState<number | null>(null);

  const workerRef = useRef<Worker | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
    };
  }, []);

  // ---------------------------------------------------------------------------
  // Worker Communication
  // ---------------------------------------------------------------------------

  const embedChunks = useCallback((textChunks: string[]) => {
    workerRef.current?.terminate();

    const worker = new Worker(
      new URL("../workers/embedding.worker.ts", import.meta.url),
      { type: "module" }
    );

    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent) => {
      const data = event.data;

      switch (data.type) {
        case "status":
          setStatusMessage(data.message);
          if (data.message.includes("Loading embedding model")) {
            setStatus("loading-model");
          }
          break;

        case "progress":
          setDownloadProgress({
            loaded: data.loaded,
            total: data.total,
            file: data.file,
          });
          break;

        case "result":
          setEmbeddings(data.embeddings);
          setStatus("complete");
          setStatusMessage(
            `✓ Generated ${data.embeddings.length} embeddings (384 dimensions each)`
          );
          setDownloadProgress(null);
          if (onEmbeddingsComplete && fileName) {
            onEmbeddingsComplete(data.embeddings, fileName);
          }
          console.group("🔒 Anchorium Local Embeddings");
          console.log("Model: Xenova/all-MiniLM-L6-v2 (384-dim)");
          console.log("Processing: 100% in-browser — zero data exfiltration");
          console.table(
            data.embeddings.map((e: EmbeddingResult, i: number) => ({
              chunk: i + 1,
              textPreview: e.text.slice(0, 80) + "...",
              vectorDims: e.vector.length,
              vectorSample: `[${e.vector.slice(0, 4).map((v: number) => v.toFixed(4)).join(", ")}, ...]`,
            }))
          );
          console.groupEnd();
          break;

        case "error":
          setStatus("error");
          setStatusMessage(`Error: ${data.message}`);
          break;
      }
    };

    worker.onerror = (error) => {
      setStatus("error");
      setStatusMessage(`Worker error: ${error.message}`);
    };

    setStatus("embedding");
    setStatusMessage("Preparing embedding pipeline...");
    worker.postMessage({ type: "embed", chunks: textChunks });
  }, [fileName, onEmbeddingsComplete]);

  // ---------------------------------------------------------------------------
  // File Processing Pipeline
  // ---------------------------------------------------------------------------

  const processFile = useCallback(
    async (file: File) => {
      const isText = file.name.endsWith(".txt");
      const isPdf = file.name.endsWith(".pdf");

      if (!isText && !isPdf) {
        setStatus("error");
        setStatusMessage("Please upload a .txt or .pdf file.");
        return;
      }

      setFileName(file.name);
      setEmbeddings([]);
      setExpandedChunk(null);
      setStatus("reading-file");
      setStatusMessage(`Reading ${file.name}...`);

      try {
        let text: string;

        if (isPdf) {
          setStatusMessage(`Parsing PDF: ${file.name}...`);
          text = await extractPdfText(file);
        } else {
          text = await file.text();
        }

        if (!text.trim()) {
          setStatus("error");
          setStatusMessage("The file appears to be empty or has no extractable text.");
          return;
        }

        const textChunks = chunkText(text);
        setChunks(textChunks);
        setStatusMessage(`Split into ${textChunks.length} chunk(s). Starting embedding...`);

        embedChunks(textChunks);
      } catch (error) {
        setStatus("error");
        setStatusMessage(
          `Failed to process file: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    },
    [embedChunks]
  );

  // ---------------------------------------------------------------------------
  // Drag & Drop Handlers
  // ---------------------------------------------------------------------------

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);

      const file = e.dataTransfer.files[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  // ---------------------------------------------------------------------------
  // Helper: Format bytes to human-readable string
  // ---------------------------------------------------------------------------

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6 animate-fade-in">
      {/* ------------------------------------------------------------------ */}
      {/* Upload Zone                                                        */}
      {/* ------------------------------------------------------------------ */}
      <div
        id="upload-zone"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`
          relative overflow-hidden cursor-pointer
          rounded-[var(--radius-xl)] p-10 text-center
          border-2 border-dashed transition-all duration-300
          ${
            isDragOver
              ? "border-[var(--anchorium-gold)] bg-[var(--surface-highlight)] scale-[1.01]"
              : "border-[var(--border-default)] hover:border-[var(--border-accent)] hover:bg-[var(--surface-highlight)]"
          }
        `}
        role="button"
        tabIndex={0}
        aria-label="Upload a document for local embedding"
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
        }}
      >
        <div
          className="absolute inset-0 opacity-0 transition-opacity duration-500 pointer-events-none"
          style={{
            background: "var(--gradient-glow)",
            opacity: isDragOver ? 0.6 : 0,
          }}
        />

        <div className="relative z-10">
          <div className="mx-auto w-16 h-16 rounded-2xl bg-[var(--surface-overlay)] flex items-center justify-center mb-4 border border-[var(--border-subtle)]">
            <svg
              className="w-8 h-8"
              style={{ color: "var(--anchorium-gold)" }}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
              />
            </svg>
          </div>

          <h3
            className="text-lg font-semibold mb-1"
            style={{ color: "var(--text-primary)" }}
          >
            Drop your document here
          </h3>
          <p style={{ color: "var(--text-secondary)" }} className="text-sm">
            Supports <span className="font-mono text-xs px-1.5 py-0.5 rounded" style={{ background: "var(--surface-overlay)" }}>.txt</span> and{" "}
            <span className="font-mono text-xs px-1.5 py-0.5 rounded" style={{ background: "var(--surface-overlay)" }}>.pdf</span> files
            — all processing happens locally in your browser
          </p>

          <div className="inline-flex items-center gap-1.5 mt-4 px-3 py-1.5 rounded-full text-xs font-medium"
            style={{
              background: "rgba(212, 175, 55, 0.1)",
              color: "var(--anchorium-gold)",
              border: "1px solid rgba(212, 175, 55, 0.2)",
            }}
          >
            <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z"
                clipRule="evenodd"
              />
            </svg>
            Zero-Knowledge — Your data never leaves this device
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.pdf"
          onChange={handleFileInput}
          className="hidden"
          id="file-upload-input"
          aria-hidden="true"
        />
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Status / Progress Bar                                              */}
      {/* ------------------------------------------------------------------ */}
      {status !== "idle" && (
        <div
          className="glass-panel rounded-[var(--radius-lg)] p-5 animate-fade-in"
          id="status-panel"
        >
          <div className="flex items-center gap-3 mb-3">
            {(status === "reading-file" ||
              status === "loading-model" ||
              status === "embedding") && (
              <div
                className="w-5 h-5 rounded-full border-2 border-t-transparent animate-spin"
                style={{ borderColor: "var(--anchorium-gold)", borderTopColor: "transparent" }}
              />
            )}

            {status === "complete" && (
              <div
                className="w-5 h-5 rounded-full flex items-center justify-center"
                style={{ background: "rgba(212, 175, 55, 0.2)" }}
              >
                <svg
                  className="w-3 h-3"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="var(--anchorium-gold)"
                  strokeWidth={3}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              </div>
            )}

            {status === "error" && (
              <div className="w-5 h-5 rounded-full flex items-center justify-center bg-red-500/20">
                <svg className="w-3 h-3 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
            )}

            <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
              {statusMessage}
            </span>
          </div>

          {downloadProgress && downloadProgress.total > 0 && (
            <div>
              <div className="flex justify-between text-xs mb-1.5" style={{ color: "var(--text-muted)" }}>
                <span>Downloading: {downloadProgress.file.split("/").pop()}</span>
                <span>
                  {formatBytes(downloadProgress.loaded)} / {formatBytes(downloadProgress.total)}
                </span>
              </div>
              <div
                className="h-1.5 rounded-full overflow-hidden"
                style={{ background: "var(--surface-overlay)" }}
              >
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.round((downloadProgress.loaded / downloadProgress.total) * 100)}%`,
                    background: "var(--gradient-primary)",
                  }}
                />
              </div>
            </div>
          )}

          {fileName && (
            <div className="mt-3 flex items-center gap-2">
              <span
                className="text-xs font-mono px-2 py-1 rounded"
                style={{ background: "var(--surface-overlay)", color: "var(--text-muted)" }}
              >
                📄 {fileName}
              </span>
              {chunks.length > 0 && (
                <span
                  className="text-xs px-2 py-1 rounded"
                  style={{ background: "var(--surface-overlay)", color: "var(--text-muted)" }}
                >
                  {chunks.length} chunk{chunks.length !== 1 ? "s" : ""}
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Embedding Results                                                  */}
      {/* ------------------------------------------------------------------ */}
      {embeddings.length > 0 && (
        <div className="space-y-3 animate-fade-in" id="results-panel">
          <div className="flex items-center justify-between">
            <h3
              className="text-sm font-semibold uppercase tracking-wider"
              style={{ color: "var(--text-muted)" }}
            >
              Embedding Results
            </h3>
            <span
              className="text-xs px-2.5 py-1 rounded-full font-mono"
              style={{
                background: "rgba(212, 175, 55, 0.1)",
                color: "var(--anchorium-gold)",
                border: "1px solid rgba(212, 175, 55, 0.2)",
              }}
            >
              384-dim vectors
            </span>
          </div>

          {embeddings.map((embedding, index) => (
            <div
              key={index}
              className="glass-panel glass-panel-hover rounded-[var(--radius-md)] overflow-hidden transition-all duration-200"
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <button
                onClick={() =>
                  setExpandedChunk(expandedChunk === index ? null : index)
                }
                className="w-full flex items-center justify-between p-4 text-left hover:bg-white/[0.02] transition-colors"
                id={`chunk-${index}`}
              >
                <div className="flex items-center gap-3">
                  <span
                    className="w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold"
                    style={{
                      background: "var(--surface-overlay)",
                      color: "var(--anchorium-gold)",
                    }}
                  >
                    {index + 1}
                  </span>
                  <span
                    className="text-sm truncate max-w-md"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {embedding.text.slice(0, 100)}
                    {embedding.text.length > 100 ? "..." : ""}
                  </span>
                </div>

                <svg
                  className={`w-4 h-4 transition-transform duration-200 ${
                    expandedChunk === index ? "rotate-180" : ""
                  }`}
                  style={{ color: "var(--text-muted)" }}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {expandedChunk === index && (
                <div
                  className="px-4 pb-4 space-y-3 animate-fade-in border-t"
                  style={{ borderColor: "var(--border-subtle)" }}
                >
                  <div className="mt-3">
                    <div
                      className="text-xs font-semibold uppercase tracking-wider mb-1.5"
                      style={{ color: "var(--text-muted)" }}
                    >
                      Source Text
                    </div>
                    <p
                      className="text-sm leading-relaxed p-3 rounded-lg"
                      style={{
                        background: "var(--surface-overlay)",
                        color: "var(--text-secondary)",
                      }}
                    >
                      {embedding.text}
                    </p>
                  </div>

                  <div>
                    <div
                      className="text-xs font-semibold uppercase tracking-wider mb-1.5"
                      style={{ color: "var(--text-muted)" }}
                    >
                      Vector ({embedding.vector.length} dimensions)
                    </div>
                    <div
                      className="font-mono text-xs p-3 rounded-lg overflow-x-auto"
                      style={{
                        background: "var(--surface-overlay)",
                        color: "var(--anchorium-gold)",
                      }}
                    >
                      [{embedding.vector.slice(0, 8).map((v) => v.toFixed(6)).join(", ")}
                      , ... , {embedding.vector.slice(-2).map((v) => v.toFixed(6)).join(", ")}]
                    </div>
                  </div>

                  <div className="flex gap-4 text-xs" style={{ color: "var(--text-muted)" }}>
                    <span>
                      Magnitude:{" "}
                      <span style={{ color: "var(--text-secondary)" }}>
                        {Math.sqrt(
                          embedding.vector.reduce((sum, v) => sum + v * v, 0)
                        ).toFixed(6)}
                      </span>
                    </span>
                    <span>
                      Chars:{" "}
                      <span style={{ color: "var(--text-secondary)" }}>
                        {embedding.text.length}
                      </span>
                    </span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
