/**
 * ==========================================================================
 * Chat Component — Anchorium Omni-Engine "Omni-Chat"
 * ==========================================================================
 *
 * The command-center chat interface for the zero-knowledge RAG pipeline.
 *
 * FEATURES:
 *   1. Semantic search: Embeds user queries locally, runs cosine similarity
 *      against document vectors, extracts top-K context (score > threshold).
 *   2. Quick Action buttons: Pre-engineered prompts for multi-agent triggers.
 *   3. Streaming responses: SSE from the backend with real-time rendering.
 *   4. Zero data retention: Chat history lives ONLY in React state.
 *
 * PRIVACY:
 *   - Query embedding runs 100% in-browser via Web Worker.
 *   - Only the top 3-5 relevant text chunks + prompt are sent to backend.
 *   - Full conversation history is NEVER sent to the server.
 *   - Backend processes ephemerally and immediately purges data.
 */

"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface EmbeddingResult {
  text: string;
  vector: number[];
  documentName?: string;
  isRegulatory?: boolean;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  contextChunks?: { text: string; score: number; documentName?: string }[];
  agentType?: string;
}

interface ChatProps {
  contextChunks: EmbeddingResult[];
  onBack?: () => void;
}

// ---------------------------------------------------------------------------
// Quick Action Button Definitions — Multi-Agent Triggers
// ---------------------------------------------------------------------------

const QUICK_ACTIONS = [
  {
    id: "gaap-convert",
    icon: "⚡",
    label: "US GAAP → IndAS",
    endpoint: "/api/v1/underwrite",
    agentType: "Underwriter",
    prompt:
      "Convert all US GAAP financial figures found in my uploaded documents into their Indian IndAS equivalents. " +
      "Show a detailed line-by-line mapping table with: (1) the original US GAAP line item, " +
      "(2) the corresponding IndAS standard reference, (3) any adjustment amounts, " +
      "(4) the converted IndAS value. Flag any items where GAAP and IndAS treatment differs materially.",
  },
  {
    id: "compliance",
    icon: "⚖️",
    label: "RBI & FEMA Check",
    endpoint: "/api/v1/compliance",
    agentType: "Compliance Copilot",
    prompt:
      "Perform a comprehensive RBI (Reserve Bank of India) and FEMA (Foreign Exchange Management Act) " +
      "compliance check on the financial data in my uploaded documents. Specifically check: " +
      "(1) FDI sectoral caps and entry routes, (2) ECB (External Commercial Borrowing) limits, " +
      "(3) Pricing guidelines for share transfers, (4) Reporting requirements (FC-GPR, FC-TRS, FEMA Annual Return). " +
      "Flag any compliance risks or required regulatory filings.",
  },
  {
    id: "arbitrage",
    icon: "🧮",
    label: "USD-INR Arbitrage",
    endpoint: "/api/v1/arbitrage",
    agentType: "Arbitrage Calculator",
    prompt:
      "Using the financial data from my uploaded documents, simulate USD-INR debt arbitrage pathways. " +
      "Compare: (1) Raising debt in USD and converting to INR, (2) Direct INR borrowing from Indian banks, " +
      "(3) ECB route with RBI-mandated spread caps. Factor in current USD/INR depreciation trends, " +
      "hedging costs (6-month forward premium), and withholding tax differentials. " +
      "Output a ranked table of the 3 cheapest all-in cost pathways.",
  },
  {
    id: "playbook",
    icon: "📄",
    label: "Soft-Landing Playbook",
    endpoint: "/api/v1/query",
    agentType: "Strategy Agent",
    prompt:
      "Generate a comprehensive India Soft-Landing Playbook based on my uploaded financial documents. " +
      "Include: (1) Recommended entity structure (subsidiary vs. branch vs. LLP), " +
      "(2) Capital infusion strategy and FDI compliance timeline, " +
      "(3) Banking relationship setup — which banks are best for foreign-owned entities, " +
      "(4) Tax registration checklist (PAN, TAN, GST, Professional Tax), " +
      "(5) Key milestones with estimated timelines and costs in both USD and INR.",
  },
];

// ---------------------------------------------------------------------------
// RAG System Prompt — Strict zero-hallucination
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are an elite cross-border corporate finance AI powering the Anchorium Omni-Engine.

STRICT RULES:
1. Answer the user's question using ONLY the provided context chunks below.
2. If the math, data, or information needed is NOT in the provided context, you MUST say: "⚠️ Insufficient data provided in the uploaded documents to answer this question accurately."
3. NEVER hallucinate, fabricate, or infer numbers that are not explicitly stated in the context.
4. When performing calculations, show your work step-by-step.
5. Always cite the document name of the chunk you are using for each claim (e.g., "According to sales_report.pdf, ...").
6. Format financial figures with proper currency symbols and thousand separators.
7. If you identify data quality issues in the context (e.g., conflicting numbers), flag them explicitly.

You are operating under a zero-knowledge architecture. The user's raw documents are processed locally — you only receive the most relevant text excerpts.`;

// ---------------------------------------------------------------------------
// Cosine Similarity — Local vector matching
// ---------------------------------------------------------------------------

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Performs local semantic search against the embedded document chunks.
 * Returns the top-K chunks with similarity score > threshold.
 */
function semanticSearch(
  queryVector: number[],
  chunks: EmbeddingResult[],
  topK = 5,
  threshold = 0.3
): { text: string; score: number; documentName?: string }[] {
  return chunks
    .map((c) => ({
      text: c.text,
      score: cosineSimilarity(queryVector, c.vector),
      documentName: c.documentName,
    }))
    .filter((item) => item.score >= threshold)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// ---------------------------------------------------------------------------
// SVG Icon Components (finance-themed, matching Anchorium aesthetic)
// ---------------------------------------------------------------------------

const IconChart = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M4 19V5" strokeLinecap="round" />
    <path d="M4 19h16" strokeLinecap="round" />
    <path d="M8 15l4-4 3 3 5-5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const IconShield = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M12 3l8 4v6c0 5-3.5 9-8 10-4.5-1-8-5-8-10V7l8-4z" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const IconArrows = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M7 17l-5-5 5-5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M2 12h12" strokeLinecap="round" />
    <path d="M17 7l5 5-5 5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M22 12H10" strokeLinecap="round" />
  </svg>
);

const IconBook = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M4 19V5a2 2 0 012-2h11a2 2 0 012 2v14" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M6 17h14" strokeLinecap="round" />
    <path d="M6 21h14" strokeLinecap="round" />
  </svg>
);

const ACTION_ICONS: Record<string, React.ReactNode> = {
  "gaap-convert": <IconChart />,
  "compliance": <IconShield />,
  "arbitrage": <IconArrows />,
  "playbook": <IconBook />,
};

// ---------------------------------------------------------------------------
// Chat Component
// ---------------------------------------------------------------------------

export default function Chat({ contextChunks, onBack }: ChatProps) {
  // --- State ---
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isEmbeddingQuery, setIsEmbeddingQuery] = useState(false);

  const queryWorkerRef = useRef<Worker | null>(null);
  const chatWorkerRef = useRef<Worker | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  // --- Auto-scroll ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // --- Cleanup worker on unmount ---
  useEffect(() => {
    return () => {
      queryWorkerRef.current?.terminate();
      chatWorkerRef.current?.terminate();
    };
  }, []);

  // --- Auto-resize textarea ---
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
  };

  // ---------------------------------------------------------------------------
  // Embed Query Locally via Web Worker
  // ---------------------------------------------------------------------------

  function embedQueryLocally(query: string): Promise<number[]> {
    return new Promise((resolve, reject) => {
      if (!queryWorkerRef.current) {
        queryWorkerRef.current = new Worker(
          new URL("../workers/query-embedding.worker.ts", import.meta.url),
          { type: "module" }
        );
      }

      const worker = queryWorkerRef.current;

      const handler = (event: MessageEvent) => {
        const data = event.data;
        if (data.type === "query-result") {
          worker.removeEventListener("message", handler);
          resolve(data.vector);
        } else if (data.type === "error") {
          worker.removeEventListener("message", handler);
          reject(new Error(data.message));
        }
      };

      worker.addEventListener("message", handler);
      worker.postMessage({ type: "embed-query", query });
    });
  }

  // ---------------------------------------------------------------------------
  // Process Query: Embed → Search → Stream
  // ---------------------------------------------------------------------------

  const processQuery = useCallback(
    async (query: string, endpoint = "/api/v1/query", agentType = "RAG") => {
      if (!query.trim() || isProcessing) return;

      // Add user message
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content: query,
        timestamp: new Date(),
        agentType,
      };
      setMessages((prev) => [...prev, userMsg]);
      setInputValue("");
      setIsProcessing(true);
      setIsEmbeddingQuery(true);

      if (inputRef.current) inputRef.current.style.height = "auto";

      try {
        // Step 1: Embed the query locally
        const queryVector = await embedQueryLocally(query);
        setIsEmbeddingQuery(false);

        // Step 2: Cosine similarity search
        const relevantChunks = semanticSearch(queryVector, contextChunks, 5, 0.3);

        // Step 3: Assemble strict RAG prompt
        const contextText = relevantChunks.length > 0
          ? relevantChunks
              .map((c, i) => `--- Chunk ${i + 1} from Document: "${c.documentName || "Unknown"}" (${(c.score * 100).toFixed(1)}% match) ---\n${c.text}`)
              .join("\n\n")
          : "No relevant context found in the uploaded documents.";

        const assembledPrompt = `${SYSTEM_PROMPT}\n\n=== RETRIEVED CONTEXT ===\n${contextText}\n\n=== USER QUERY ===\n${query}`;

        // Step 4: Add placeholder assistant message
        const assistantId = `assistant-${Date.now()}`;
        setMessages((prev) => [
          ...prev,
          {
            id: assistantId,
            role: "assistant",
            content: "",
            timestamp: new Date(),
            contextChunks: relevantChunks,
            agentType,
          },
        ]);

        // Step 5: Stream from local LLM
        await streamFromLocalWorker(query, contextText, relevantChunks, assistantId);
      } catch (error) {
        setIsEmbeddingQuery(false);
        setMessages((prev) => [
          ...prev,
          {
            id: `error-${Date.now()}`,
            role: "assistant",
            content: `⚠️ Error: ${error instanceof Error ? error.message : String(error)}`,
            timestamp: new Date(),
          },
        ]);
      } finally {
        setIsProcessing(false);
      }
    },
    [contextChunks, isProcessing]
  );

  // ---------------------------------------------------------------------------
  // Stream Response from Backend (SSE)
  // ---------------------------------------------------------------------------

  async function streamFromLocalWorker(
    query: string,
    contextText: string,
    relevantChunks: { text: string; score: number; documentName?: string }[],
    messageId: string
  ) {
    if (relevantChunks.length === 0) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === messageId ? { ...m, content: "⚠️ Insufficient data provided in the uploaded documents to answer this question accurately. Please upload documents containing relevant financial information." } : m
        )
      );
      return;
    }

    return new Promise<void>((resolve, reject) => {
      if (!chatWorkerRef.current) {
        chatWorkerRef.current = new Worker(
          new URL("../workers/chat-generation.worker.ts", import.meta.url),
          { type: "module" }
        );
      }

      const worker = chatWorkerRef.current;
      let currentContent = "";
      let hasStartedStreaming = false;

      const handler = (event: MessageEvent) => {
        const { type, messageId: eventMessageId, text, message } = event.data;

        if (type === "status") {
          console.log("[Local LLM Status]:", message);
          if (!hasStartedStreaming) {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === messageId ? { ...m, content: `*[System]: ${message}*` } : m
              )
            );
          }
        } else if (type === "stream-chunk" && eventMessageId === messageId) {
          hasStartedStreaming = true;
          currentContent += text;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === messageId ? { ...m, content: currentContent } : m
            )
          );
        } else if (type === "stream-complete" && eventMessageId === messageId) {
          worker.removeEventListener("message", handler);
          resolve();
        } else if (type === "error" && eventMessageId === messageId) {
          worker.removeEventListener("message", handler);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === messageId ? { ...m, content: `⚠️ Error communicating with Local AI model: ${message}` } : m
            )
          );
          reject(new Error(message));
        }
      };

      worker.addEventListener("message", handler);
      
      worker.postMessage({
        type: "generate",
        messageId,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: `=== RETRIEVED CONTEXT ===\n${contextText}\n\n=== USER QUERY ===\n${query}` }
        ]
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Send handlers
  // ---------------------------------------------------------------------------

  const handleSend = () => processQuery(inputValue);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex flex-col h-full" style={{ background: "var(--surface-base)" }}>
      {/* ================================================================ */}
      {/* Chat Header                                                     */}
      {/* ================================================================ */}
      <div
        className="flex items-center justify-between px-6 py-4 border-b glass-panel"
        style={{ borderColor: "var(--border-default)" }}
      >
        <div className="flex items-center gap-3">
          {onBack && (
            <button
              onClick={onBack}
              className="p-2 rounded-lg transition-all duration-200"
              style={{ color: "var(--anchorium-gold)" }}
              onMouseEnter={(e) => { e.currentTarget.style.background = "var(--surface-overlay)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
          )}
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{
              background: "rgba(212, 175, 55, 0.12)",
              border: "1px solid rgba(212, 175, 55, 0.25)",
            }}
          >
            <svg
              className="w-4.5 h-4.5"
              style={{ color: "var(--anchorium-gold)" }}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z"
              />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold tracking-wide uppercase" style={{ color: "var(--text-primary)", letterSpacing: "0.12em" }}>
              Omni-Chat
            </h3>
            <p className="text-[11px]" style={{ color: "var(--text-muted)" }}>
              {contextChunks.length} chunks • Zero-knowledge RAG
            </p>
          </div>
        </div>

        {/* Processing indicator */}
        {isProcessing && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full animate-fade-in"
            style={{
              background: "rgba(212, 175, 55, 0.08)",
              border: "1px solid rgba(212, 175, 55, 0.15)",
            }}
          >
            <div
              className="w-3 h-3 rounded-full border-2 border-t-transparent animate-spin"
              style={{ borderColor: "var(--anchorium-gold)", borderTopColor: "transparent" }}
            />
            <span className="text-[11px]" style={{ color: "var(--anchorium-gold)" }}>
              {isEmbeddingQuery ? "Embedding locally..." : "Streaming..."}
            </span>
          </div>
        )}
      </div>

      {/* ================================================================ */}
      {/* Message History                                                  */}
      {/* ================================================================ */}
      <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5" id="chat-messages">
        {/* Welcome state */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-5 animate-fade-in">
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center animate-pulse-gold"
              style={{
                background: "rgba(212, 175, 55, 0.1)",
                border: "1px solid rgba(212, 175, 55, 0.25)",
              }}
            >
              <svg className="w-7 h-7" style={{ color: "var(--anchorium-gold)" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                />
              </svg>
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
                Welcome to <span className="text-gradient">Omni-Chat</span>
              </h3>
              <p className="text-sm max-w-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                Your document has been embedded locally. Ask any question or use the quick action buttons below to trigger specialized agents.
              </p>
            </div>
            {/* Dotted grid hint — mimicking the Anchorium Works dot pattern */}
            <div className="flex gap-2 mt-4 opacity-30">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="flex flex-col gap-2">
                  {Array.from({ length: 3 }).map((_, j) => (
                    <div
                      key={j}
                      className="w-1 h-1 rounded-full"
                      style={{ background: "var(--anchorium-gold)" }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Messages */}
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === "user" ? "justify-end" : "justify-start"} animate-fade-in`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-5 py-4 ${
                message.role === "user" ? "rounded-br-sm" : "rounded-bl-sm"
              }`}
              style={
                message.role === "user"
                  ? {
                      background: "rgba(212, 175, 55, 0.12)",
                      border: "1px solid rgba(212, 175, 55, 0.25)",
                      color: "var(--text-primary)",
                    }
                  : message.role === "system"
                  ? {
                      background: "var(--surface-overlay)",
                      border: "1px solid var(--border-subtle)",
                      color: "var(--text-secondary)",
                    }
                  : {
                      background: "var(--surface-raised)",
                      border: "1px solid var(--border-subtle)",
                      color: "var(--text-primary)",
                    }
              }
            >
              {/* Agent badge */}
              {message.agentType && message.role === "assistant" && (
                <div className="flex items-center gap-1.5 mb-2.5">
                  <span
                    className="text-[10px] font-semibold uppercase tracking-[0.15em] px-2 py-0.5 rounded"
                    style={{
                      background: "rgba(212, 175, 55, 0.12)",
                      color: "var(--anchorium-gold)",
                      border: "1px solid rgba(212, 175, 55, 0.2)",
                    }}
                  >
                    {message.agentType}
                  </span>
                </div>
              )}

              {/* Content or loading dots */}
              <div className="text-sm leading-relaxed whitespace-pre-wrap">
                {message.content || (
                  <span className="flex items-center gap-2" style={{ color: "var(--text-muted)" }}>
                    <span className="flex gap-1">
                      <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: "var(--anchorium-gold)", animationDelay: "0ms" }} />
                      <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: "var(--anchorium-gold)", animationDelay: "150ms" }} />
                      <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: "var(--anchorium-gold)", animationDelay: "300ms" }} />
                    </span>
                    Processing...
                  </span>
                )}
              </div>

              {/* Context chunks used — collapsible */}
              {message.contextChunks && message.contextChunks.length > 0 && message.content && (
                <details className="mt-3 group">
                  <summary
                    className="text-xs cursor-pointer select-none flex items-center gap-1"
                    style={{ color: "var(--text-muted)" }}
                  >
                    <svg className="w-3 h-3 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                    </svg>
                    {message.contextChunks.length} context chunk{message.contextChunks.length !== 1 ? "s" : ""} retrieved
                  </summary>
                  <div className="mt-2 space-y-1.5">
                    {message.contextChunks.map((chunk, i) => (
                      <div
                        key={i}
                        className="text-xs p-2.5 rounded-lg"
                        style={{ background: "var(--surface-overlay)", color: "var(--text-muted)" }}
                      >
                        <div className="flex justify-between font-semibold text-[10px] mb-1 opacity-80" style={{ color: "var(--anchorium-gold)" }}>
                          <span>📄 {chunk.documentName || "Unknown Source"}</span>
                          <span className="font-mono">{(chunk.score * 100).toFixed(0)}% match</span>
                        </div>
                        <p className="leading-relaxed">
                          {chunk.text.slice(0, 150)}
                          {chunk.text.length > 150 ? "..." : ""}
                        </p>
                      </div>
                    ))}
                  </div>
                </details>
              )}

              {/* Timestamp */}
              <div className="text-[10px] mt-2.5 opacity-40" style={{ color: "var(--text-muted)" }}>
                {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </div>
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {isProcessing && messages.length > 0 && !messages[messages.length - 1]?.content && null}

        <div ref={messagesEndRef} />
      </div>

      {/* ================================================================ */}
      {/* Quick Action Buttons — Premium gold-bordered style               */}
      {/* ================================================================ */}
      <div className="px-6 py-3 border-t" style={{ borderColor: "var(--border-subtle)" }}>
        <div className="flex gap-2 overflow-x-auto pb-1">
          {QUICK_ACTIONS.map((action) => (
            <button
              key={action.id}
              onClick={() => processQuery(action.prompt, action.endpoint, action.agentType)}
              disabled={isProcessing}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-medium whitespace-nowrap
                         transition-all duration-200 flex-shrink-0 disabled:opacity-40"
              style={{
                background: "var(--surface-overlay)",
                color: "var(--text-secondary)",
                border: "1px solid var(--border-default)",
              }}
              onMouseEnter={(e) => {
                if (!isProcessing) {
                  e.currentTarget.style.borderColor = "rgba(212, 175, 55, 0.5)";
                  e.currentTarget.style.background = "rgba(212, 175, 55, 0.08)";
                  e.currentTarget.style.color = "var(--anchorium-gold)";
                  e.currentTarget.style.boxShadow = "0 0 16px rgba(212, 175, 55, 0.1)";
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "var(--border-default)";
                e.currentTarget.style.background = "var(--surface-overlay)";
                e.currentTarget.style.color = "var(--text-secondary)";
                e.currentTarget.style.boxShadow = "none";
              }}
              id={`action-${action.id}`}
            >
              <span style={{ color: "var(--anchorium-gold)" }}>
                {ACTION_ICONS[action.id] || action.icon}
              </span>
              <span>{action.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ================================================================ */}
      {/* Chat Input — Anchored to bottom                                  */}
      {/* ================================================================ */}
      <div
        className="px-6 py-4 border-t"
        style={{ borderColor: "var(--border-default)", background: "var(--surface-raised)" }}
      >
        <div
          className="flex items-end gap-3 rounded-2xl px-4 py-3 transition-all duration-200"
          style={{
            background: "var(--surface-overlay)",
            border: "1px solid var(--border-default)",
          }}
        >
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onFocus={(e) => {
              const parent = e.target.parentElement;
              if (parent) {
                parent.style.borderColor = "rgba(212, 175, 55, 0.4)";
                parent.style.boxShadow = "0 0 16px rgba(212, 175, 55, 0.08)";
              }
            }}
            onBlur={(e) => {
              const parent = e.target.parentElement;
              if (parent) {
                parent.style.borderColor = "var(--border-default)";
                parent.style.boxShadow = "none";
              }
            }}
            placeholder="Ask about your financial documents..."
            disabled={isProcessing}
            rows={1}
            className="flex-1 bg-transparent resize-none outline-none text-sm leading-relaxed placeholder-[var(--text-muted)]"
            style={{ color: "var(--text-primary)", maxHeight: "140px" }}
            id="chat-input"
          />

          <button
            onClick={handleSend}
            disabled={isProcessing || !inputValue.trim()}
            className="flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center transition-all duration-200"
            style={{
              background: inputValue.trim() && !isProcessing ? "var(--gradient-primary)" : "var(--surface-overlay)",
              opacity: inputValue.trim() && !isProcessing ? 1 : 0.35,
              cursor: inputValue.trim() && !isProcessing ? "pointer" : "not-allowed",
            }}
            onMouseEnter={(e) => {
              if (inputValue.trim() && !isProcessing) {
                e.currentTarget.style.boxShadow = "0 0 20px rgba(212, 175, 55, 0.3)";
              }
            }}
            onMouseLeave={(e) => { e.currentTarget.style.boxShadow = "none"; }}
            id="send-button"
            aria-label="Send message"
          >
            <svg
              className="w-4 h-4"
              style={{ color: inputValue.trim() && !isProcessing ? "#000" : "var(--text-muted)" }}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>

        <p className="text-center text-[10px] mt-2.5 tracking-wide" style={{ color: "var(--text-muted)" }}>
          🔒 Query embedded locally • Only relevant chunks sent to backend • Zero data retention
        </p>
      </div>
    </div>
  );
}
