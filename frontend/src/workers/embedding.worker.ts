/**
 * ==========================================================================
 * Embedding Web Worker — Anchorium Omni-Engine
 * ==========================================================================
 *
 * This Web Worker runs the HuggingFace Transformers.js embedding pipeline
 * OFF the main thread, ensuring zero UI jank during model loading and
 * vector generation.
 *
 * PRIVACY ARCHITECTURE:
 *   - The embedding model runs 100% inside the user's browser.
 *   - No text data is ever sent to any external server.
 *   - The model weights are cached in the browser's Cache Storage after
 *     the first download (~23MB for all-MiniLM-L6-v2).
 *
 * COMMUNICATION PROTOCOL (postMessage):
 *   Main → Worker:
 *     { type: "embed", chunks: string[] }
 *
 *   Worker → Main:
 *     { type: "status",   message: string }
 *     { type: "progress", loaded: number, total: number, file: string }
 *     { type: "result",   embeddings: { text: string, vector: number[] }[] }
 *     { type: "error",    message: string }
 */

import { pipeline, FeatureExtractionPipeline, ProgressInfo } from "@huggingface/transformers";

// ---------------------------------------------------------------------------
// Singleton: Lazily-loaded embedding pipeline
// ---------------------------------------------------------------------------
let embeddingPipeline: FeatureExtractionPipeline | null = null;

/**
 * Loads the embedding model (Xenova/all-MiniLM-L6-v2) if not already cached.
 * Sends progress updates back to the main thread during download.
 */
async function getEmbeddingPipeline(): Promise<FeatureExtractionPipeline> {
  if (embeddingPipeline) return embeddingPipeline;

  self.postMessage({
    type: "status",
    message: "Loading embedding model (first time may take 10-20s)...",
  });

  embeddingPipeline = (await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2",
    {
      // Send download progress back to main thread
      progress_callback: (progress: ProgressInfo) => {
        if (progress.status === "progress") {
          self.postMessage({
            type: "progress",
            loaded: progress.loaded ?? 0,
            total: progress.total ?? 0,
            file: progress.file ?? "",
          });
        } else if (progress.status === "done") {
          self.postMessage({
            type: "status",
            message: `Model file loaded: ${progress.file ?? "unknown"}`,
          });
        }
      },
    },
  )) as FeatureExtractionPipeline;

  self.postMessage({
    type: "status",
    message: "✓ Embedding model ready — all processing is local.",
  });

  return embeddingPipeline;
}

// ---------------------------------------------------------------------------
// Message Handler — Process embedding requests
// ---------------------------------------------------------------------------
self.addEventListener("message", async (event: MessageEvent) => {
  const { type, chunks } = event.data;

  if (type !== "embed" || !Array.isArray(chunks)) {
    self.postMessage({
      type: "error",
      message: `Invalid message type: ${type}. Expected "embed" with chunks array.`,
    });
    return;
  }

  try {
    // Step 1: Ensure the model is loaded
    const pipe = await getEmbeddingPipeline();

    // Step 2: Generate embeddings for each chunk
    self.postMessage({
      type: "status",
      message: `Embedding ${chunks.length} chunk(s)...`,
    });

    const embeddings: { text: string; vector: number[] }[] = [];

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];

      // Run the embedding pipeline on this chunk
      // Output shape: [1, sequence_length, 384] → we mean-pool to [384]
      const output = await pipe(chunk, {
        pooling: "mean",
        normalize: true,
      });

      // Extract the 384-dimensional vector
      const vector = Array.from(output.data as Float32Array);

      embeddings.push({ text: chunk, vector });

      // Progress update per chunk
      self.postMessage({
        type: "status",
        message: `Embedded chunk ${i + 1} of ${chunks.length}`,
      });
    }

    // Step 3: Send all results back
    self.postMessage({ type: "result", embeddings });
  } catch (error) {
    self.postMessage({
      type: "error",
      message: `Embedding failed: ${error instanceof Error ? error.message : String(error)}`,
    });
  }
});
