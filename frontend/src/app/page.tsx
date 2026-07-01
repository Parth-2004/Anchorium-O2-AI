/**
 * ==========================================================================
 * Home Page — Anchorium Omni-Engine
 * ==========================================================================
 *
 * Two-state UI:
 *   1. UPLOAD STATE: Premium landing page (Anchorium Works aesthetic) with
 *      document upload and local embedding.
 *   2. DASHBOARD STATE: Split-screen — The Vault (left) + Omni-Chat (right).
 *
 * Design: Inspired by the Anchorium Works brand — black + gold, serif headings,
 * dot grid patterns, institutional finance aesthetic.
 */

"use client";

import React, { useState, useEffect, useRef } from "react";
import LocalEmbedder from "@/components/LocalEmbedder";
import Chat from "@/components/Chat";

interface EmbeddingResult {
  text: string;
  vector: number[];
  documentName?: string;
}

// ---------------------------------------------------------------------------
// Dot Grid Background Component — Signature Anchorium visual
// ---------------------------------------------------------------------------

function DotGrid() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
      <svg width="100%" height="100%" className="opacity-[0.12]">
        <defs>
          <pattern id="dot-grid" x="0" y="0" width="28" height="28" patternUnits="userSpaceOnUse">
            <circle cx="1.5" cy="1.5" r="1" fill="#d4af37" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#dot-grid)" />
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Navigation Bar
// ---------------------------------------------------------------------------

function NavBar({ showDashboard }: { showDashboard: boolean }) {
  return (
    <nav
      className="sticky top-0 z-50 border-b px-6 py-3"
      style={{
        borderColor: showDashboard ? "var(--border-default)" : "rgba(212, 175, 55, 0.1)",
        background: showDashboard ? "var(--surface-raised)" : "rgba(0, 0, 0, 0.85)",
        backdropFilter: "blur(24px)",
      }}
    >
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{
              background: "var(--gradient-primary)",
              boxShadow: "0 0 16px rgba(212, 175, 55, 0.3)",
            }}
          >
            <svg className="w-5 h-5 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 2a2 2 0 100 4a2 2 0 000-4zM12 6v4m0 0l-4 8m4-8l4 8M6 18h12" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-[0.15em] uppercase" style={{ color: "var(--text-primary)" }}>
              Anchorium
              <span className="text-gradient ml-1 font-normal tracking-[0.08em]">Works</span>
            </h1>
            <p className="text-[9px] uppercase tracking-[0.25em]" style={{ color: "var(--text-muted)" }}>
              Omni-Engine
            </p>
          </div>
        </div>

        {/* Nav links (visible on upload page) */}
        {!showDashboard && (
          <div className="hidden md:flex items-center gap-6">
            {["Vision", "Capabilities", "Metrics", "Connect"].map((item) => (
              <span
                key={item}
                className="text-[11px] uppercase tracking-[0.2em] cursor-default transition-colors duration-200"
                style={{ color: "var(--text-muted)" }}
                onMouseEnter={(e) => { e.currentTarget.style.color = "var(--anchorium-gold)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text-muted)"; }}
              >
                {item}
              </span>
            ))}
          </div>
        )}

        {/* Status badge */}
        <div className="flex items-center gap-3">
          {showDashboard && (
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded text-[10px] uppercase tracking-[0.15em]"
              style={{
                color: "var(--anchorium-gold)",
                border: "1px solid rgba(212, 175, 55, 0.2)",
              }}
            >
              <span className="relative flex h-1.5 w-1.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ background: "var(--anchorium-gold)" }} />
                <span className="relative inline-flex rounded-full h-1.5 w-1.5" style={{ background: "var(--anchorium-gold)" }} />
              </span>
              Local Mode
            </div>
          )}
          <div
            className="px-4 py-1.5 rounded text-[10px] uppercase tracking-[0.2em] font-semibold transition-all duration-200"
            style={{
              border: "1px solid rgba(212, 175, 55, 0.4)",
              color: "var(--anchorium-gold)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(212, 175, 55, 0.1)";
              e.currentTarget.style.boxShadow = "0 0 12px rgba(212, 175, 55, 0.15)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "transparent";
              e.currentTarget.style.boxShadow = "none";
            }}
          >
            Dashboard
          </div>
        </div>
      </div>
    </nav>
  );
}

// ---------------------------------------------------------------------------
// Home Page Component
// ---------------------------------------------------------------------------

interface ProcessedDocument {
  id: string;
  fileName: string;
  chunksCount: number;
}

export default function Home() {
  const [embeddings, setEmbeddings] = useState<EmbeddingResult[]>([]);
  const [documents, setDocuments] = useState<ProcessedDocument[]>([]);
  const [showDashboard, setShowDashboard] = useState(false);
  const [transitionPhase, setTransitionPhase] = useState<"none" | "fading" | "entering">("none");
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  const handleEmbeddingsComplete = (newEmbeddings: EmbeddingResult[], newFileName: string, isLast: boolean) => {
    setEmbeddings((prev) => {
      const filtered = prev.filter((chunk) => chunk.documentName !== newFileName);
      return [...filtered, ...newEmbeddings];
    });

    setDocuments((prev) => {
      const filtered = prev.filter((doc) => doc.fileName !== newFileName);
      const newDoc: ProcessedDocument = {
        id: Math.random().toString(36).substring(2, 9),
        fileName: newFileName,
        chunksCount: newEmbeddings.length,
      };
      return [...filtered, newDoc];
    });

    if (isLast && !showDashboard) {
      // Smooth transition to dashboard
      setTransitionPhase("fading");
      setTimeout(() => {
        setShowDashboard(true);
        setTransitionPhase("entering");
        setTimeout(() => setTransitionPhase("none"), 500);
      }, 400);
    }
  };

  const handleRemoveDocument = (docId: string) => {
    const docToRemove = documents.find((d) => d.id === docId);
    if (!docToRemove) return;

    setDocuments((prev) => prev.filter((d) => d.id !== docId));
    setEmbeddings((prev) => prev.filter((chunk) => chunk.documentName !== docToRemove.fileName));
  };

  // Automatically exit dashboard if all documents are removed
  useEffect(() => {
    if (showDashboard && documents.length === 0) {
      setShowDashboard(false);
    }
  }, [documents, showDashboard]);

  return (
    <div
      className="flex flex-col flex-1 min-h-screen"
      style={{
        background: "var(--surface-base)",
        transition: "opacity 400ms ease",
        opacity: transitionPhase === "fading" ? 0 : 1,
      }}
    >
      <NavBar showDashboard={showDashboard} />

      <main className="flex-1 flex flex-col">
        {!showDashboard ? (
          /* ============================================================ */
          /* UPLOAD STATE — Premium Anchorium Works landing               */
          /* ============================================================ */
          <>
            <section className="relative flex-1 flex flex-col items-center justify-center min-h-[85vh] overflow-hidden py-12">
              {/* Dot grid background */}
              <DotGrid />

              {/* Radial gold glow */}
              <div
                className="absolute inset-0 pointer-events-none"
                style={{ background: "radial-gradient(ellipse at 50% 30%, rgba(212, 175, 55, 0.06) 0%, transparent 60%)" }}
              />

              {/* EST. badge */}
              <div
                className="absolute top-8 left-8 text-[10px] tracking-[0.3em] uppercase hidden md:flex items-center gap-2"
                style={{ color: "var(--text-muted)" }}
              >
                <span className="w-2 h-2 border" style={{ borderColor: "var(--text-muted)" }} />
                Est. 2026
              </div>

              {/* Location badge */}
              <div
                className="absolute top-8 right-8 text-[10px] tracking-[0.3em] uppercase hidden md:flex items-center gap-2"
                style={{ color: "var(--text-muted)" }}
              >
                Gurugram, India
                <span className="w-2 h-2 border" style={{ borderColor: "var(--text-muted)" }} />
              </div>

              {/* Hero text */}
              <div className="relative z-10 text-center px-6 max-w-4xl animate-fade-in mb-10">
                <h2
                  className="text-4xl sm:text-5xl lg:text-6xl font-light leading-[1.15] tracking-tight mb-4"
                  style={{
                    color: "var(--text-primary)",
                    fontFamily: "Georgia, 'Times New Roman', serif",
                  }}
                >
                  Unlock your
                  <em
                    className="not-italic mx-3"
                    style={{
                      color: "var(--anchorium-gold)",
                      fontStyle: "italic",
                      fontFamily: "Georgia, 'Times New Roman', serif",
                    }}
                  >
                    financial
                  </em>
                  growth
                </h2>

                <p
                  className="text-sm sm:text-base max-w-xl mx-auto leading-relaxed animate-fade-in"
                  style={{
                    color: "var(--text-secondary)",
                    animationDelay: "200ms",
                  }}
                >
                  Process your documents privately. Our AI runs entirely inside your browser — your data never touches our servers.
                </p>
              </div>

              {/* Upload Box integrated immediately below hero */}
              <div className="relative z-10 w-full max-w-3xl px-6 animate-fade-in" style={{ animationDelay: "400ms" }}>
                <LocalEmbedder onEmbeddingsComplete={handleEmbeddingsComplete} />
              </div>
            </section>
          </>
        ) : (
          /* ============================================================ */
          /* DASHBOARD STATE — Split-screen Vault + Omni-Chat             */
          /* ============================================================ */
          <div
            className={`flex flex-1 max-h-[calc(100vh-52px)] ${
              transitionPhase === "entering" ? "animate-fade-in" : ""
            }`}
          >
            {/* ------ Left Panel: The Vault ------ */}
            <div
              className="w-[30%] min-w-[280px] border-r flex flex-col overflow-hidden"
              style={{ borderColor: "var(--border-default)" }}
            >
              <div className="p-5 flex-1 flex flex-col">
                <div className="flex items-center gap-3 mb-5">
                  <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center"
                    style={{
                      background: "rgba(212, 175, 55, 0.1)",
                      border: "1px solid rgba(212, 175, 55, 0.2)",
                    }}
                  >
                    <svg className="w-5 h-5" style={{ color: "var(--anchorium-gold)" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                    </svg>
                  </div>
                  <div>
                    <h3
                      className="text-sm font-semibold tracking-[0.12em] uppercase"
                      style={{ color: "var(--text-primary)" }}
                    >
                      The Vault
                    </h3>
                    <p className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
                      Secure Local Storage
                    </p>
                  </div>
                </div>

                {/* Document List */}
                <div className="flex-1 flex flex-col min-h-0 mb-4 overflow-y-auto scrollbar-none space-y-2">
                  <div className="text-[10px] uppercase tracking-wider font-semibold opacity-60 mb-1" style={{ color: "var(--text-muted)" }}>
                    Active Documents ({documents.length}/6)
                  </div>
                  {documents.map((doc) => (
                    <div
                      key={doc.id}
                      className="p-3 rounded-xl flex items-center justify-between transition-all duration-200"
                      style={{
                        background: "var(--surface-raised)",
                        border: "1px solid var(--border-subtle)",
                      }}
                    >
                      <div className="truncate flex-1 pr-2">
                        <div className="text-xs font-semibold truncate" style={{ color: "var(--text-primary)" }}>
                          📄 {doc.fileName}
                        </div>
                        <div className="text-[10px] opacity-75" style={{ color: "var(--text-muted)" }}>
                          {doc.chunksCount} chunks
                        </div>
                      </div>
                      <button
                        onClick={() => handleRemoveDocument(doc.id)}
                        className="text-xs p-1 opacity-60 hover:opacity-100 hover:text-red-400 transition-all duration-150"
                        title="Remove document"
                      >
                        🗑️
                      </button>
                    </div>
                  ))}
                </div>

                {/* Stats bar */}
                <div className="flex gap-2 mb-4">
                  {[
                    { label: "Vector Dims", value: "384" },
                    { label: "Total Chunks", value: embeddings.length.toString() },
                    { label: "Privacy", value: "100%" },
                  ].map((stat) => (
                    <div
                      key={stat.label}
                      className="flex-1 px-1.5 py-2 rounded-lg text-center"
                      style={{ background: "var(--surface-overlay)", border: "1px solid var(--border-subtle)" }}
                    >
                      <div className="text-xs font-bold" style={{ color: "var(--anchorium-gold)" }}>
                        {stat.value}
                      </div>
                      <div className="text-[8px] uppercase tracking-[0.05em]" style={{ color: "var(--text-muted)" }}>
                        {stat.label}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Upload More Button */}
                <button
                  onClick={() => setIsUploadModalOpen(true)}
                  disabled={documents.length >= 6}
                  className="w-full text-[11px] uppercase tracking-[0.15em] px-4 py-3 rounded-lg transition-all duration-200 font-semibold mb-2"
                  style={{
                    background: "var(--gradient-primary)",
                    color: "#000",
                    opacity: documents.length >= 6 ? 0.5 : 1,
                    cursor: documents.length >= 6 ? "not-allowed" : "pointer",
                  }}
                  onMouseEnter={(e) => {
                    if (documents.length < 6) {
                      e.currentTarget.style.boxShadow = "0 0 16px rgba(212, 175, 55, 0.35)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.boxShadow = "none";
                  }}
                >
                  + Upload More
                </button>

                {/* Clear Vault / Reset button */}
                <button
                  onClick={() => {
                    setDocuments([]);
                    setEmbeddings([]);
                    setShowDashboard(false);
                  }}
                  className="w-full text-[11px] uppercase tracking-[0.15em] px-4 py-3 rounded-lg transition-all duration-200"
                  style={{
                    background: "var(--surface-overlay)",
                    color: "var(--text-muted)",
                    border: "1px solid var(--border-subtle)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "rgba(212, 175, 55, 0.3)";
                    e.currentTarget.style.color = "var(--text-primary)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--border-subtle)";
                    e.currentTarget.style.color = "var(--text-muted)";
                  }}
                >
                  Reset Vault
                </button>
              </div>
            </div>

            {/* ------ Right Panel: Omni-Chat ------ */}
            <div className="flex-1 flex flex-col min-w-0">
              <Chat
                contextChunks={embeddings}
                onBack={() => setShowDashboard(false)}
              />
            </div>
          </div>
        )}
      </main>

      {/* Upload More Modal */}
      {isUploadModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in">
          <div
            className="w-full max-w-2xl rounded-2xl p-6 border shadow-2xl space-y-4 relative"
            style={{
              background: "var(--surface-raised)",
              borderColor: "rgba(212, 175, 55, 0.3)",
              boxShadow: "0 0 40px rgba(212, 175, 55, 0.15)",
            }}
          >
            <div className="flex items-center justify-between pb-2 border-b" style={{ borderColor: "var(--border-subtle)" }}>
              <div>
                <h3 className="text-base font-semibold uppercase tracking-wider text-gradient">
                  Upload Additional Document
                </h3>
                <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                  Maximum 6 active documents in Vault
                </p>
              </div>
              <button
                onClick={() => setIsUploadModalOpen(false)}
                className="text-[10px] font-semibold uppercase tracking-[0.15em] px-3 py-1.5 rounded-lg transition-all duration-200 border"
                style={{
                  background: "var(--surface-overlay)",
                  color: "var(--text-muted)",
                  borderColor: "var(--border-subtle)"
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = "white";
                  e.currentTarget.style.borderColor = "white";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = "var(--text-muted)";
                  e.currentTarget.style.borderColor = "var(--border-subtle)";
                }}
              >
                Close
              </button>
            </div>
            
            <div className="max-h-[70vh] overflow-y-auto pr-1">
              <LocalEmbedder
                multiple={true}
                onEmbeddingsComplete={(newEmbs, name, isLast) => {
                  handleEmbeddingsComplete(newEmbs, name, isLast);
                  if (isLast) {
                    setIsUploadModalOpen(false);
                  }
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Footer — only on upload page */}
      {!showDashboard && (
        <footer
          className="border-t px-6 py-4 text-center text-[10px] uppercase tracking-[0.2em]"
          style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
        >
          Anchorium Omni-Engine v0.1.0 — Your data stays on your device. We never store, share, or train on your financial documents.
        </footer>
      )}
    </div>
  );
}
