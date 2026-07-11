/**
 * ==========================================================================
 * O2Engine.tsx — Anchorium Omni Engine (Single-File Component)
 * ==========================================================================
 *
 * A cinematic, two-state single-page application:
 *
 *   STATE 1 — HERO ANIMATION:
 *     Full-viewport landing with staggered gold text animation, permanent
 *     background video, and a premium CTA button.
 *
 *   STATE 2 — O2 DASHBOARD:
 *     Split-screen glassmorphism layout:
 *       Left  → The Vault (drag-and-drop PDF upload)
 *       Right → Omni-Chat (Ollama llama3.2 powered chatbot)
 *
 * PERMANENT BACKGROUND:
 *   A fixed <video> element at z-index: -1 that NEVER unmounts.
 *   A dark overlay at z-index: 0 provides contrast.
 *   All content sits at z-index: 10+.
 *
 * CHAT ENGINE:
 *   Connects to local Ollama instance (http://localhost:11434)
 *   running the llama3.2 model for streaming responses.
 *
 * Theme: Deep black (#0B0B0C), Gold (#D4AF37), Cream (#F1E5AC)
 * Font:  Helvetica Now Var → sans-serif fallback
 */

"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Upload, Send, CheckCircle, Shield, FileText, X, Sparkles } from "lucide-react";
import { FadeUp } from "./FadeUp";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type AppState = "hero" | "dashboard";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  status: "uploading" | "embedded" | "error";
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const OLLAMA_URL = "http://localhost:11434/api/chat";
const OLLAMA_MODEL = "llama3.2";

const HEADING_WORDS = "OMNI ENGINE: CROSS-BORDER CREDIT ARBITRAGE, SOLVED.".split(" ");

const QUICK_ACTIONS = [
  { label: "Convert US GAAP to IndAS", icon: "Zap" },
  { label: "Run FEMA Check", icon: "Scale" },
  { label: "Analyze SBLC Terms", icon: "Building" },
  { label: "Draft Compliance Report", icon: "Clipboard" },
];

const SYSTEM_PROMPT = `You are Omni Engine — an elite AI compliance and financial structuring copilot built by Anchorium Works.

Your specialties:
- Cross-border credit arbitrage between US and India
- SBLC-backed INR liquidity structuring
- FEMA (Foreign Exchange Management Act) compliance
- US GAAP to IndAS conversion
- Zero-knowledge underwriting for international founders expanding to India
- RBI regulatory frameworks, ECB norms, FDI sectoral caps
- Corporate entity structuring (WOS, JV, LLP, Branch Office)

Rules:
- Be precise, professional, and structured in your responses.
- Use clear headings, bullet points, and tables when helpful.
- Always flag when something requires CA/legal review.
- Cite relevant regulatory sections (FEMA, RBI circulars, Companies Act).
- If uncertain, clearly state your confidence level.
- Never fabricate regulatory citations.`;

// ---------------------------------------------------------------------------
// Utility: Generate unique ID
// ---------------------------------------------------------------------------
const uid = () => Math.random().toString(36).slice(2, 11);

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function O2Engine() {
  // -- App State --
  const [appState, setAppState] = useState<AppState>("hero");

  // -- Chat State --
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // -- Upload State --
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // -- Auto-scroll chat --
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // -- Welcome message on dashboard entry --
  useEffect(() => {
    if (appState === "dashboard" && messages.length === 0) {
      setMessages([
        {
          id: uid(),
          role: "assistant",
          content:
            "Omni Engine initialized. Secure enclave active.\n\nHow can I assist you with cross-border structuring, FEMA compliance, or credit underwriting today?",
          timestamp: new Date(),
        },
      ]);
    }
  }, [appState, messages.length]);

  // =========================================================================
  // OLLAMA CHAT — Streaming via llama3.2
  // =========================================================================

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isGenerating) return;

      const userMsg: ChatMessage = {
        id: uid(),
        role: "user",
        content: text.trim(),
        timestamp: new Date(),
      };

      const assistantId = uid();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setInputValue("");
      setIsGenerating(true);

      // Build Ollama message history (last 10 messages for context)
      const historyForOllama = [
        { role: "system" as const, content: SYSTEM_PROMPT },
        ...messages.slice(-10).map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        })),
        { role: "user" as const, content: text.trim() },
      ];

      try {
        const controller = new AbortController();
        abortRef.current = controller;

        const response = await fetch(OLLAMA_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: OLLAMA_MODEL,
            messages: historyForOllama,
            stream: true,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Ollama returned ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        let accumulated = "";

        if (reader) {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split("\n").filter(Boolean);

            for (const line of lines) {
              try {
                const parsed = JSON.parse(line);
                if (parsed.message?.content) {
                  accumulated += parsed.message.content;
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantId
                        ? { ...m, content: accumulated, isStreaming: true }
                        : m
                    )
                  );
                }
              } catch {
                // Skip malformed JSON lines
              }
            }
          }
        }

        // Finalize
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: accumulated || "I couldn't generate a response. Please try again.", isStreaming: false }
              : m
          )
        );
      } catch (err: unknown) {
        const errorMessage =
          err instanceof Error && err.name === "AbortError"
            ? "Response cancelled."
            : "Could not connect to Ollama. Make sure `ollama serve` is running and the `llama3.2` model is pulled.\n\n```\nollama pull llama3.2\nollama serve\n```";

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: errorMessage, isStreaming: false }
              : m
          )
        );
      } finally {
        setIsGenerating(false);
        abortRef.current = null;
      }
    },
    [isGenerating, messages]
  );

  // =========================================================================
  // FILE UPLOAD HANDLERS
  // =========================================================================

  const handleFiles = useCallback((fileList: FileList | null) => {
    if (!fileList) return;
    const newFiles: UploadedFile[] = Array.from(fileList).map((f) => ({
      id: uid(),
      name: f.name,
      size: f.size,
      status: "uploading" as const,
    }));
    setFiles((prev) => [...prev, ...newFiles]);

    // Simulate embedding delay per file
    newFiles.forEach((file) => {
      setTimeout(
        () => {
          setFiles((prev) =>
            prev.map((f) => (f.id === file.id ? { ...f, status: "embedded" } : f))
          );
        },
        1200 + Math.random() * 800
      );
    });
  }, []);

  const removeFile = useCallback((fileId: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId));
  }, []);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // =========================================================================
  // RENDER
  // =========================================================================

  return (
    <div className="min-h-screen relative overflow-hidden" style={{ fontFamily: "'Helvetica Now Var', sans-serif" }}>
      {/* ================================================================ */}
      {/* INLINE FONT IMPORT                                               */}
      {/* ================================================================ */}
      <style>{`
        @import url('https://db.onlinewebfonts.com/c/e66905e07608167a84e6ad52f638c3c6?family=Helvetica+Now+Var');

        /* Chat scrollbar */
        .o2-scrollbar::-webkit-scrollbar { width: 5px; }
        .o2-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .o2-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(212, 175, 55, 0.15);
          border-radius: 10px;
        }
        .o2-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(212, 175, 55, 0.3);
        }

        /* Hide scrollbar for quick actions row */
        .o2-hide-scrollbar::-webkit-scrollbar { display: none; }
        .o2-hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }

        /* Pulsing dot */
        @keyframes o2-pulse {
          0%, 100% { opacity: 1; box-shadow: 0 0 4px rgba(34, 197, 94, 0.4); }
          50% { opacity: 0.5; box-shadow: 0 0 12px rgba(34, 197, 94, 0.8); }
        }
        .o2-pulse { animation: o2-pulse 2s ease-in-out infinite; }

        /* Streaming cursor blink */
        @keyframes o2-blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        .o2-cursor::after {
          content: '▊';
          color: #D4AF37;
          animation: o2-blink 0.8s step-end infinite;
          margin-left: 2px;
        }

        /* Drag-over glow */
        @keyframes o2-border-glow {
          0%, 100% { border-color: rgba(212, 175, 55, 0.3); }
          50% { border-color: rgba(212, 175, 55, 0.7); }
        }
        .o2-drag-active {
          animation: o2-border-glow 1s ease-in-out infinite;
          background: rgba(212, 175, 55, 0.04) !important;
        }
      `}</style>

      {/* ================================================================ */}
      {/* PERMANENT BACKGROUND VIDEO — NEVER UNMOUNTS                      */}
      {/* ================================================================ */}
      <video
        autoPlay
        muted
        loop
        playsInline
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100%",
          height: "100vh",
          objectFit: "cover",
          zIndex: -1,
        }}
      >
        <source
          src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260514_135830_bb6491d1-9b66-4aec-9722-13b4dfe3fb46.mp4"
          type="video/mp4"
        />
      </video>

      {/* Dark overlay for contrast */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(11, 11, 12, 0.75)",
          zIndex: 0,
          pointerEvents: "none",
        }}
      />

      {/* ================================================================ */}
      {/* STATE MANAGER — AnimatePresence wraps both states                 */}
      {/* ================================================================ */}
      <div className="relative w-full min-h-screen" style={{ zIndex: 10 }}>
        <AnimatePresence mode="wait">
          {appState === "hero" ? (
            /* ============================================================ */
            /* STATE 1: THE HERO ANIMATION                                  */
            /* ============================================================ */
            <motion.div
              key="hero-state"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, filter: "blur(12px)", scale: 0.97 }}
              transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
              className="h-screen flex flex-col justify-center"
              style={{ padding: "70px 32px 32px 32px" }}
            >
              {/* Mobile padding override */}
              <div
                className="max-w-[760px] flex flex-col items-start w-full"
                style={{ marginLeft: "max(32px, 8vw)" }}
              >
                {/* ---- HEADING ---- */}
                <h2
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "0.25em",
                    margin: 0,
                    padding: 0,
                    lineHeight: 1.1,
                  }}
                >
                  {HEADING_WORDS.map((word, i) => (
                    <FadeUp
                      key={`word-${i}`}
                      delay={0.15 + i * 0.08}
                      duration={0.7}
                      y={32}
                      as="span"
                    >
                      <span
                        style={{
                          display: "inline-block",
                          fontSize: "clamp(26px, 3.5vw, 48px)",
                          fontWeight: 700,
                          textTransform: "uppercase",
                          color: "#D4AF37",
                          textShadow: "0 4px 24px rgba(212, 175, 55, 0.2)",
                        }}
                      >
                        {word}
                      </span>
                    </FadeUp>
                  ))}
                </h2>

                {/* ---- SUBTEXT ---- */}
                <FadeUp delay={0.9} y={24} duration={0.8}>
                  <p
                    style={{
                      color: "#F1E5AC",
                      opacity: 0.9,
                      maxWidth: "480px",
                      marginTop: "24px",
                      fontSize: "clamp(14px, 1.4vw, 18px)",
                      lineHeight: 1.65,
                      letterSpacing: "0.01em",
                    }}
                  >
                    Automating SBLC-backed INR liquidity, real-time FEMA
                    compliance, and zero-knowledge underwriting for
                    international founders expanding to India.
                  </p>
                </FadeUp>

                {/* ---- CTA BUTTON ---- */}
                <FadeUp delay={1.1} duration={0.8} y={24}>
                  <motion.button
                    onClick={() => setAppState("dashboard")}
                    whileHover={{
                      scale: 1.05,
                      boxShadow: "0 0 30px rgba(212, 175, 55, 0.4)",
                    }}
                    whileTap={{ scale: 0.97 }}
                    style={{
                      marginTop: "40px",
                      background: "#D4AF37",
                      color: "#0B0B0C",
                      textTransform: "uppercase",
                      fontWeight: 700,
                      fontSize: "14px",
                      letterSpacing: "0.08em",
                      padding: "16px 32px",
                      border: "1px solid #F1E5AC",
                      boxShadow: "0 0 20px rgba(212, 175, 55, 0.15)",
                      cursor: "pointer",
                      borderRadius: "2px",
                    }}
                  >
                    GET STARTED: INITIALIZE OMNI ENGINE
                  </motion.button>
                </FadeUp>
              </div>
            </motion.div>
          ) : (
            /* ============================================================ */
            /* STATE 2: THE O2 DASHBOARD                                    */
            /* ============================================================ */
            <motion.div
              key="dashboard-state"
              initial={{ opacity: 0, filter: "blur(12px)", y: 16 }}
              animate={{ opacity: 1, filter: "blur(0px)", y: 0 }}
              exit={{ opacity: 0, filter: "blur(12px)", y: -16 }}
              transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
              className="h-screen w-full flex flex-col md:flex-row gap-5 p-4 md:p-6 box-border"
            >
              {/* -------------------------------------------------------- */}
              {/* LEFT PANEL — THE VAULT                                    */}
              {/* -------------------------------------------------------- */}
              <div
                className="w-full md:w-[380px] flex flex-col rounded-2xl overflow-hidden border"
                style={{
                  background: "rgba(11, 11, 12, 0.4)",
                  backdropFilter: "blur(16px)",
                  WebkitBackdropFilter: "blur(16px)",
                  borderColor: "rgba(212, 175, 55, 0.2)",
                  boxShadow: "0 8px 40px rgba(0, 0, 0, 0.6)",
                }}
              >
                {/* Vault Header */}
                <div
                  className="flex items-center gap-3 px-6 py-4 border-b"
                  style={{
                    borderColor: "rgba(212, 175, 55, 0.12)",
                    background: "rgba(0, 0, 0, 0.3)",
                  }}
                >
                  <Shield size={18} color="#D4AF37" />
                  <h3
                    style={{
                      color: "#D4AF37",
                      textTransform: "uppercase",
                      letterSpacing: "0.15em",
                      fontWeight: 600,
                      fontSize: "13px",
                      margin: 0,
                    }}
                  >
                    The Vault
                  </h3>
                  <span
                    style={{
                      marginLeft: "auto",
                      fontSize: "10px",
                      color: "rgba(241, 229, 172, 0.4)",
                      textTransform: "uppercase",
                      letterSpacing: "0.1em",
                    }}
                  >
                    100% Local
                  </span>
                </div>

                {/* Upload Zone */}
                <div className="flex-1 flex flex-col p-5 gap-4">
                  {/* Drag & Drop Area */}
                  <div
                    className={`flex-1 border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-6 text-center transition-all duration-300 cursor-pointer group ${
                      isDragOver ? "o2-drag-active" : ""
                    }`}
                    style={{
                      borderColor: isDragOver
                        ? "rgba(212, 175, 55, 0.6)"
                        : "rgba(212, 175, 55, 0.2)",
                      minHeight: files.length > 0 ? "120px" : "200px",
                    }}
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setIsDragOver(true);
                    }}
                    onDragLeave={() => setIsDragOver(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setIsDragOver(false);
                      handleFiles(e.dataTransfer.files);
                    }}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf,.xlsx,.csv,.doc,.docx"
                      multiple
                      className="hidden"
                      onChange={(e) => handleFiles(e.target.files)}
                    />
                    <div
                      className="w-14 h-14 rounded-full flex items-center justify-center mb-4 transition-transform duration-500 group-hover:scale-110"
                      style={{
                        background: "rgba(11, 11, 12, 0.8)",
                        border: "1px solid rgba(212, 175, 55, 0.15)",
                      }}
                    >
                      <Upload
                        size={22}
                        className="transition-colors"
                        style={{ color: isDragOver ? "#F1E5AC" : "#D4AF37" }}
                      />
                    </div>
                    <p
                      className="transition-colors group-hover:text-[#D4AF37]"
                      style={{
                        color: "#F1E5AC",
                        fontWeight: 500,
                        fontSize: "15px",
                        margin: "0 0 4px 0",
                      }}
                    >
                      {isDragOver ? "Drop files here" : "Drag & Drop Documents"}
                    </p>
                    <p style={{ color: "rgba(241, 229, 172, 0.45)", fontSize: "12px", margin: 0 }}>
                      PDF, XLSX, CSV, DOCX supported
                    </p>
                  </div>

                  {/* Uploaded Files List */}
                  {files.length > 0 && (
                    <div className="flex flex-col gap-2 o2-scrollbar" style={{ maxHeight: "240px", overflowY: "auto" }}>
                      {files.map((file) => (
                        <motion.div
                          key={file.id}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, x: -20 }}
                          className="flex items-center gap-3 px-4 py-3 rounded-lg border"
                          style={{
                            background:
                              file.status === "embedded"
                                ? "rgba(34, 197, 94, 0.06)"
                                : "rgba(11, 11, 12, 0.6)",
                            borderColor:
                              file.status === "embedded"
                                ? "rgba(34, 197, 94, 0.2)"
                                : "rgba(212, 175, 55, 0.1)",
                          }}
                        >
                          {file.status === "embedded" ? (
                            <CheckCircle size={16} color="#22c55e" />
                          ) : (
                            <FileText size={16} color="#D4AF37" className="animate-pulse" />
                          )}
                          <div className="flex-1 min-w-0">
                            <p
                              className="truncate"
                              style={{
                                color: file.status === "embedded" ? "#22c55e" : "#F1E5AC",
                                fontSize: "12px",
                                fontWeight: 600,
                                margin: 0,
                              }}
                            >
                              {file.name}
                            </p>
                            <p style={{ color: "rgba(241, 229, 172, 0.4)", fontSize: "10px", margin: 0 }}>
                              {file.status === "embedded"
                                ? "Embedded Locally — 100% Privacy"
                                : `Embedding... ${formatFileSize(file.size)}`}
                            </p>
                          </div>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              removeFile(file.id);
                            }}
                            className="opacity-40 hover:opacity-100 transition-opacity"
                          >
                            <X size={14} color="#F1E5AC" />
                          </button>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* -------------------------------------------------------- */}
              {/* RIGHT PANEL — OMNI-CHAT                                   */}
              {/* -------------------------------------------------------- */}
              <div
                className="flex-1 flex flex-col rounded-2xl overflow-hidden border min-w-0"
                style={{
                  background: "rgba(11, 11, 12, 0.4)",
                  backdropFilter: "blur(16px)",
                  WebkitBackdropFilter: "blur(16px)",
                  borderColor: "rgba(212, 175, 55, 0.2)",
                  boxShadow: "0 8px 40px rgba(0, 0, 0, 0.6)",
                }}
              >
                {/* Chat Header */}
                <div
                  className="flex items-center justify-between px-6 py-4 border-b"
                  style={{
                    borderColor: "rgba(212, 175, 55, 0.12)",
                    background: "rgba(0, 0, 0, 0.3)",
                  }}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="w-2 h-2 rounded-full o2-pulse"
                      style={{ background: "#22c55e" }}
                    />
                    <h3
                      style={{
                        color: "#D4AF37",
                        textTransform: "uppercase",
                        letterSpacing: "0.15em",
                        fontWeight: 600,
                        fontSize: "13px",
                        margin: 0,
                      }}
                    >
                      Omni-Chat
                    </h3>
                  </div>
                  <div className="flex items-center gap-3">
                    <span
                      className="hidden sm:inline-block px-3 py-1.5 rounded border"
                      style={{
                        fontSize: "10px",
                        color: "rgba(241, 229, 172, 0.5)",
                        borderColor: "rgba(241, 229, 172, 0.12)",
                        textTransform: "uppercase",
                        letterSpacing: "0.12em",
                        background: "rgba(0, 0, 0, 0.2)",
                      }}
                    >
                      Model: llama3.2
                    </span>
                    <Sparkles size={14} color="#D4AF37" style={{ opacity: 0.5 }} />
                  </div>
                </div>

                {/* Chat Messages */}
                <div
                  className="flex-1 overflow-y-auto p-5 md:p-6 flex flex-col gap-5 o2-scrollbar"
                >
                  {messages.map((msg) => (
                    <motion.div
                      key={msg.id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                      className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={`max-w-[85%] md:max-w-[75%] px-5 py-4 ${
                          msg.role === "user"
                            ? "rounded-2xl rounded-tr-sm"
                            : "rounded-2xl rounded-tl-sm"
                        }`}
                        style={
                          msg.role === "user"
                            ? {
                                background: "#D4AF37",
                                color: "#0B0B0C",
                                boxShadow: "0 4px 16px rgba(212, 175, 55, 0.15)",
                              }
                            : {
                                background: "rgba(11, 11, 12, 0.7)",
                                border: "1px solid rgba(212, 175, 55, 0.25)",
                                color: "#F1E5AC",
                                boxShadow: "0 4px 16px rgba(0, 0, 0, 0.3)",
                              }
                        }
                      >
                        <p
                          className={msg.isStreaming ? "o2-cursor" : ""}
                          style={{
                            fontSize: "14px",
                            lineHeight: 1.65,
                            margin: 0,
                            fontWeight: msg.role === "user" ? 600 : 400,
                            whiteSpace: "pre-wrap",
                            wordBreak: "break-word",
                          }}
                        >
                          {msg.content}
                        </p>
                      </div>
                    </motion.div>
                  ))}
                  <div ref={chatEndRef} />
                </div>

                {/* Quick Action Buttons */}
                <div
                  className="px-5 md:px-6 py-3 flex gap-2 overflow-x-auto o2-hide-scrollbar border-t"
                  style={{
                    borderColor: "rgba(212, 175, 55, 0.08)",
                    background: "rgba(0, 0, 0, 0.15)",
                  }}
                >
                  {QUICK_ACTIONS.map((action, i) => (
                    <motion.button
                      key={i}
                      whileHover={{ scale: 1.03 }}
                      whileTap={{ scale: 0.97 }}
                      onClick={() => sendMessage(action.label)}
                      disabled={isGenerating}
                      className="whitespace-nowrap rounded-full border transition-all duration-200 disabled:opacity-40"
                      style={{
                        padding: "8px 16px",
                        borderColor: "rgba(212, 175, 55, 0.25)",
                        background: "rgba(11, 11, 12, 0.7)",
                        color: "#D4AF37",
                        fontSize: "11px",
                        textTransform: "uppercase",
                        letterSpacing: "0.08em",
                        fontWeight: 500,
                        cursor: isGenerating ? "not-allowed" : "pointer",
                      }}
                      onMouseEnter={(e) => {
                        if (!isGenerating) {
                          e.currentTarget.style.background = "#D4AF37";
                          e.currentTarget.style.color = "#0B0B0C";
                          e.currentTarget.style.borderColor = "#D4AF37";
                        }
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = "rgba(11, 11, 12, 0.7)";
                        e.currentTarget.style.color = "#D4AF37";
                        e.currentTarget.style.borderColor = "rgba(212, 175, 55, 0.25)";
                      }}
                    >
                      {action.icon} {action.label}
                    </motion.button>
                  ))}
                </div>

                {/* Chat Input */}
                <div
                  className="px-5 md:px-6 py-4"
                  style={{ background: "rgba(0, 0, 0, 0.25)" }}
                >
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      sendMessage(inputValue);
                    }}
                    className="relative flex items-center rounded-xl border overflow-hidden transition-all duration-300"
                    style={{
                      background: "rgba(11, 11, 12, 0.8)",
                      borderColor: "rgba(212, 175, 55, 0.25)",
                    }}
                    onFocus={(e) => {
                      (e.currentTarget as HTMLElement).style.borderColor = "#D4AF37";
                      (e.currentTarget as HTMLElement).style.boxShadow = "0 0 20px rgba(212, 175, 55, 0.12)";
                    }}
                    onBlur={(e) => {
                      (e.currentTarget as HTMLElement).style.borderColor = "rgba(212, 175, 55, 0.25)";
                      (e.currentTarget as HTMLElement).style.boxShadow = "none";
                    }}
                  >
                    <input
                      type="text"
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      placeholder={
                        isGenerating
                          ? "Omni Engine is thinking..."
                          : "Query the Omni Engine..."
                      }
                      disabled={isGenerating}
                      className="flex-1 bg-transparent outline-none disabled:opacity-50"
                      style={{
                        color: "#F1E5AC",
                        padding: "16px 20px",
                        fontSize: "14px",
                        fontWeight: 300,
                        letterSpacing: "0.01em",
                      }}
                    />
                    <button
                      type="submit"
                      disabled={isGenerating || !inputValue.trim()}
                      className="p-3 mr-2 rounded-lg transition-all duration-200 disabled:opacity-30"
                      style={{
                        background: inputValue.trim()
                          ? "rgba(212, 175, 55, 0.15)"
                          : "transparent",
                        color: "#D4AF37",
                        cursor:
                          isGenerating || !inputValue.trim()
                            ? "not-allowed"
                            : "pointer",
                      }}
                      onMouseEnter={(e) => {
                        if (inputValue.trim() && !isGenerating) {
                          e.currentTarget.style.background = "#D4AF37";
                          e.currentTarget.style.color = "#0B0B0C";
                        }
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = inputValue.trim()
                          ? "rgba(212, 175, 55, 0.15)"
                          : "transparent";
                        e.currentTarget.style.color = "#D4AF37";
                      }}
                    >
                      <Send size={18} />
                    </button>
                  </form>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
