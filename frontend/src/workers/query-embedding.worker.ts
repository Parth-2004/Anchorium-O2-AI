/**
 * ==========================================================================
 * Query Embedding Worker — Anchorium Omni-Engine
 * ==========================================================================
 *
 * This Web Worker handles QUERY embedding for the RAG pipeline.
 * It reuses the same Xenova/all-MiniLM-L6-v2 model as the document
 * embedding worker, but is dedicated to single-query embedding
 * for real-time semantic search.
 *
 * COMMUNICATION PROTOCOL:
 *   Main → Worker:
 *     { type: "embed-query", query: string }
 *
 *   Worker → Main:
 *     { type: "query-result", vector: number[] }
 *     { type: "status",       message: string }
 *     { type: "error",        message: string }
 */

import { pipeline, FeatureExtractionPipeline, ProgressInfo } from "@huggingface/transformers";

// ---------------------------------------------------------------------------
// Singleton pipeline (cached after first load)
// ---------------------------------------------------------------------------
let embeddingPipeline: FeatureExtractionPipeline | null = null;

async function getEmbeddingPipeline(): Promise<FeatureExtractionPipeline> {
  if (embeddingPipeline) return embeddingPipeline;

  self.postMessage({
    type: "status",
    message: "Loading query embedding model...",
  });

  embeddingPipeline = (await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2",
    {
      progress_callback: (progress: ProgressInfo) => {
        if (progress.status === "done") {
          self.postMessage({
            type: "status",
            message: `Model ready: ${progress.file ?? ""}`,
          });
        }
      },
    },
  )) as FeatureExtractionPipeline;

  self.postMessage({ type: "status", message: "✓ Query embedding model ready." });
  return embeddingPipeline;
}

// ---------------------------------------------------------------------------
// Message handler — embed a single query string
// ---------------------------------------------------------------------------
self.addEventListener("message", async (event: MessageEvent) => {
  const { type, query } = event.data;

  if (type !== "embed-query" || typeof query !== "string") {
    self.postMessage({ type: "error", message: "Invalid message. Expected embed-query." });
    return;
  }

  try {
    const pipe = await getEmbeddingPipeline();

    const output = await pipe(query, {
      pooling: "mean",
      normalize: true,
    });

    const vector = Array.from(output.data as Float32Array);
    self.postMessage({ type: "query-result", vector });
  } catch (error) {
    self.postMessage({
      type: "error",
      message: `Query embedding failed: ${error instanceof Error ? error.message : String(error)}`,
    });
  }
});
