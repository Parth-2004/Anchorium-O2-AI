"use client";

import React, { useState } from "react";
import LocalEmbedder from "@/components/LocalEmbedder";
import Chat from "@/components/Chat";

interface EmbeddingResult {
  text: string;
  vector: number[];
}

export default function Home() {
  const [embeddings, setEmbeddings] = useState<EmbeddingResult[]>([]);
  const [fileName, setFileName] = useState<string>("");
  const [showDashboard, setShowDashboard] = useState(false);

  const handleEmbeddingsComplete = (newEmbeddings: EmbeddingResult[], newFileName: string) => {
    setEmbeddings(newEmbeddings);
    setFileName(newFileName);
    setShowDashboard(true);
  };

  return (
    <div className="flex flex-col flex-1 min-h-screen" style={{ background: "var(--surface-base)" }}>
      {/* Navigation */}
      <nav
        className="sticky top-0 z-50 glass-panel border-b px-6 py-3"
        style={{ borderColor: "var(--border-default)" }}
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{
                background: "var(--gradient-primary)",
                boxShadow: "0 0 16px rgba(212, 175, 55, 0.3)",
              }}
            >
              <svg
                className="w-5 h-5 text-black"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 2a2 2 0 100 4a2 2 0 000-4zM12 6v4m0 0l-4 8m4-8l4 8M6 18h12"
                />
              </svg>
            </div>

            <div>
              <h1
                className="text-base font-bold tracking-tight"
                style={{ color: "var(--text-primary)" }}
              >
                Anchorium
                <span className="text-gradient ml-1.5 font-light">Omni-Engine</span>
              </h1>
              <p className="text-[10px] uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
                Zero-Knowledge Financial Intelligence
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs"
              style={{
                background: "rgba(212, 175, 55, 0.1)",
                color: "var(--anchorium-gold)",
                border: "1px solid rgba(212, 175, 55, 0.2)",
              }}
            >
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                  style={{ background: "var(--anchorium-gold)" }}
                />
                <span className="relative inline-flex rounded-full h-2 w-2"
                  style={{ background: "var(--anchorium-gold)" }}
                />
              </span>
              Local Mode
            </div>
          </div>
        </div>
      </nav>

      <main className="flex-1 flex flex-col">
        {!showDashboard ? (
          <>
            <section className="relative overflow-hidden">
              <div
                className="absolute inset-0 pointer-events-none"
                style={{ background: "var(--gradient-glow)" }}
              />

              <div className="relative max-w-5xl mx-auto px-6 pt-16 pb-8 text-center">
                <h2
                  className="text-3xl sm:text-4xl font-bold tracking-tight mb-3 animate-fade-in"
                  style={{ color: "var(--text-primary)" }}
                >
                  Process Documents
                  <span className="text-gradient"> Privately</span>
                </h2>
                <p
                  className="text-base max-w-2xl mx-auto leading-relaxed animate-fade-in"
                  style={{ color: "var(--text-secondary)", animationDelay: "100ms" }}
                >
                  Upload your financial documents below. Our AI runs
                  <strong style={{ color: "var(--text-primary)" }}> entirely inside your browser</strong>
                  — your data never touches our servers. Every vector is generated
                  locally using the MiniLM embedding model.
                </p>

                <div className="flex flex-wrap justify-center gap-3 mt-6 animate-fade-in" style={{ animationDelay: "200ms" }}>
                  {[
                    { label: "End-to-End Encrypted" },
                    { label: "In-Browser AI" },
                    { label: "Local Vector Search" },
                    { label: "Ephemeral Compute" },
                  ].map((badge, idx) => (
                    <div
                      key={badge.label}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm"
                      style={{
                        background: "var(--surface-overlay)",
                        color: "var(--text-secondary)",
                        border: "1px solid var(--border-default)",
                      }}
                    >
                      {badge.label}
                    </div>
                  ))}
                </div>
              </div>
            </section>

            <section className="px-6 pb-16">
              <LocalEmbedder onEmbeddingsComplete={handleEmbeddingsComplete} />
            </section>
          </>
        ) : (
          <div className="flex flex-1 max-h-[calc(100vh-80px)]">
            {/* Left Panel */}
            <div className="w-1/2 border-r border-[var(--border-default)] flex flex-col overflow-hidden">
              <div className="p-6 border-b border-[var(--border-default)] glass-panel">
                <div className="flex items-center gap-3 mb-4">
                  <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center"
                    style={{ background: "var(--surface-overlay)" }}
                  >
                    <svg
                      className="w-5 h-5"
                      style={{ color: "var(--anchorium-gold)" }}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4"
                      />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
                      The Vault
                    </h3>
                    <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                      {fileName} • {embeddings.length} chunks embedded locally
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setShowDashboard(false)}
                  className="text-sm px-4 py-2 rounded-lg transition-all duration-200"
                  style={{
                    background: "var(--surface-overlay)",
                    color: "var(--text-secondary)",
                    border: "1px solid var(--border-default)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--border-accent)";
                    e.currentTarget.style.color = "var(--text-primary)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--border-default)";
                    e.currentTarget.style.color = "var(--text-secondary)";
                  }}
                >
                  ← Upload a new document
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {embeddings.map((chunk, idx) => (
                  <div
                    key={idx}
                    className="glass-panel rounded-xl p-4 transition-all animate-fade-in"
                    style={{ animationDelay: `${idx * 30}ms` }}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <span
                        className="text-xs font-mono px-2 py-1 rounded"
                        style={{
                          background: "var(--surface-overlay)",
                          color: "var(--anchorium-gold)",
                        }}
                      >
                        Chunk {idx + 1}
                      </span>
                    </div>
                    <p
                      className="text-sm leading-relaxed"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {chunk.text.slice(0, 300)}
                      {chunk.text.length > 300 ? "…" : ""}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Right Panel */}
            <div className="w-1/2 flex flex-col">
              <Chat
                contextChunks={embeddings}
                onBack={() => setShowDashboard(false)}
              />
            </div>
          </div>
        )}
      </main>

      {!showDashboard && (
        <footer
          className="border-t px-6 py-4 text-center text-xs"
          style={{
            borderColor: "var(--border-default)",
            color: "var(--text-muted)",
          }}
        >
          <p>
            Anchorium Omni-Engine v0.1.0 — Your data stays on your device.
            We never store, share, or train on your financial documents.
          </p>
        </footer>
      )}
    </div>
  );
}
