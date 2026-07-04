/**
 * ==========================================================================
 * Home Page — Anchorium Omni-Engine
 * ==========================================================================
 *
 * Two-state UI:
 *   1. UPLOAD STATE: Premium landing page (Anchorium Works aesthetic) with
 *      document upload and local embedding — cinematic video background.
 *   2. DASHBOARD STATE: Split-screen — The Vault (left) + Omni-Chat (right).
 *
 * Design: Inspired by the Anchorium Works brand — black + gold, serif headings,
 * dot grid patterns, institutional finance aesthetic.
 *
 * IMPORTANT: The background video is ALWAYS rendered and fixed behind
 * all content. It never unmounts during state transitions.
 */

"use client";

import React, { useState } from "react";
import LocalEmbedder from "@/components/LocalEmbedder";
import Chat from "@/components/Chat";
import { FadeUp } from "@/components/FadeUp";

interface EmbeddingResult {
  text: string;
  vector: number[];
  documentName?: string;
  isRegulatory?: boolean;
}

// ---------------------------------------------------------------------------
// Navigation Bar
// ---------------------------------------------------------------------------

function NavBar({ showDashboard }: { showDashboard: boolean }) {
  return (
    <nav
      className="sticky top-0 z-50 border-b px-6 py-3"
      style={{
        borderColor: showDashboard ? "var(--border-default)" : "rgba(212, 175, 55, 0.08)",
        background: showDashboard ? "rgba(10, 10, 10, 0.92)" : "rgba(0, 0, 0, 0.4)",
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
            <h1 className="text-sm font-semibold tracking-[0.15em] uppercase" style={{ color: "#F1E5AC" }}>
              Anchorium
              <span className="text-gradient ml-1 font-normal tracking-[0.08em]">Works</span>
            </h1>
            <p className="text-[9px] uppercase tracking-[0.25em]" style={{ color: "rgba(241, 229, 172, 0.5)" }}>
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
                style={{ color: "rgba(241, 229, 172, 0.45)" }}
                onMouseEnter={(e) => { e.currentTarget.style.color = "#D4AF37"; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = "rgba(241, 229, 172, 0.45)"; }}
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
              color: "#D4AF37",
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
  const [isRegulatoryLoaded, setIsRegulatoryLoaded] = useState(false);
  const [isRegulatoryLoading, setIsRegulatoryLoading] = useState(false);

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

  const handleToggleRegulatory = async () => {
    if (isRegulatoryLoaded) {
      const remainingDocs = documents.filter((d) => d.id !== "regulatory-db");
      setDocuments(remainingDocs);
      setEmbeddings((prev) => prev.filter((chunk) => !chunk.isRegulatory));
      setIsRegulatoryLoaded(false);
      if (remainingDocs.length === 0) {
        setShowDashboard(false);
      }
    } else {
      setIsRegulatoryLoading(true);
      try {
        const res = await fetch("/regulatory_embeddings.json");
        if (!res.ok) throw new Error("Failed to load regulatory data.");
        const data: EmbeddingResult[] = await res.json();
        const dataWithFlag = data.map(d => ({ ...d, isRegulatory: true }));
        
        setEmbeddings((prev) => [...prev, ...dataWithFlag]);
        setDocuments((prev) => [
          ...prev,
          {
            id: "regulatory-db",
            fileName: "Verified Indian Regulatory Database (20 PDFs)",
            chunksCount: data.length,
          }
        ]);
        setIsRegulatoryLoaded(true);
      } catch (err) {
        console.error(err);
        alert("Failed to load Regulatory Knowledge Base. Ensure it is generated.");
      } finally {
        setIsRegulatoryLoading(false);
      }
    }
  };

  const handleRemoveDocument = (docId: string) => {
    if (docId === "regulatory-db") {
      handleToggleRegulatory();
      return;
    }
    const docToRemove = documents.find((d) => d.id === docId);
    if (!docToRemove) return;

    const remainingDocs = documents.filter((d) => d.id !== docId);
    setDocuments(remainingDocs);
    setEmbeddings((prev) => prev.filter((chunk) => chunk.documentName !== docToRemove.fileName));
    if (remainingDocs.length === 0) {
      setShowDashboard(false);
    }
  };

  return (
    <div
      className="flex flex-col flex-1 min-h-screen"
      style={{
        /* Transparent background so video shines through */
        background: "transparent",
        transition: "opacity 400ms ease",
        opacity: transitionPhase === "fading" ? 0 : 1,
      }}
    >
      {/* ================================================================== */}
      {/* PERMANENT BACKGROUND VIDEO — Always rendered, never unmounted      */}
      {/* ================================================================== */}
      <video
        autoPlay
        muted
        loop
        playsInline
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100vh',
          objectFit: 'cover',
          zIndex: -2,
        }}
      >
        <source
          src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260514_135830_bb6491d1-9b66-4aec-9722-13b4dfe3fb46.mp4"
          type="video/mp4"
        />
      </video>

      {/* Video Overlay — tints the video dark so text is readable */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(11, 11, 12, 0.72)',
          zIndex: -1,
          pointerEvents: 'none',
        }}
      />

      {/* ================================================================== */}
      {/* GOLD DOT GRID PATTERN — Subtle animated overlay for premium feel   */}
      {/* ================================================================== */}
      {!showDashboard && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 0,
            pointerEvents: 'none',
            backgroundImage: 'radial-gradient(circle, rgba(212, 175, 55, 0.12) 1px, transparent 1px)',
            backgroundSize: '28px 28px',
            maskImage: 'radial-gradient(ellipse 80% 70% at 50% 50%, black 30%, transparent 75%)',
            WebkitMaskImage: 'radial-gradient(ellipse 80% 70% at 50% 50%, black 30%, transparent 75%)',
          }}
        />
      )}

      <NavBar showDashboard={showDashboard} />

      <main className="flex-1 flex flex-col relative" style={{ zIndex: 1 }}>
        {!showDashboard ? (
          /* ============================================================ */
          /* UPLOAD STATE — Premium Anchorium Works landing               */
          /* ============================================================ */

          <section
            style={{
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              height: 'calc(100vh - 56px)',
              padding: '0 32px',
              textAlign: 'center',
              position: 'relative',
            }}
          >
            {/* Corner accents — like the reference image */}
            <div className="hidden md:block" style={{ position: 'absolute', top: '32px', left: '40px' }}>
              <FadeUp delay={0.1} duration={0.6} y={10}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <div style={{ width: '6px', height: '6px', border: '1px solid #D4AF37', transform: 'rotate(45deg)' }} />
                  <span style={{ fontSize: '10px', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'rgba(241, 229, 172, 0.5)' }}>
                    EST. 2026
                  </span>
                </div>
              </FadeUp>
            </div>
            <div className="hidden md:block" style={{ position: 'absolute', top: '32px', right: '40px' }}>
              <FadeUp delay={0.15} duration={0.6} y={10}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '10px', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'rgba(241, 229, 172, 0.5)' }}>
                    Gurugram, India
                  </span>
                  <div style={{ width: '6px', height: '6px', border: '1px solid #D4AF37', transform: 'rotate(45deg)' }} />
                </div>
              </FadeUp>
            </div>

            {/* Main content — centered like reference */}
            <div style={{ maxWidth: '860px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>

              {/* Heading — Large, elegant, centered */}
              <FadeUp delay={0.2} duration={1} y={40}>
                <h2
                  style={{
                    fontSize: 'clamp(32px, 5.5vw, 72px)',
                    fontWeight: 300,
                    lineHeight: 1.08,
                    letterSpacing: '-0.02em',
                    color: '#F1E5AC',
                    margin: 0,
                    fontFamily: 'Georgia, "Times New Roman", serif',
                  }}
                >
                  The Ultimate
                </h2>
              </FadeUp>

              <FadeUp delay={0.4} duration={1} y={40}>
                <h2
                  style={{
                    fontSize: 'clamp(34px, 5.8vw, 78px)',
                    fontWeight: 700,
                    fontStyle: 'italic',
                    lineHeight: 1.08,
                    letterSpacing: '-0.01em',
                    color: '#D4AF37',
                    margin: '4px 0',
                    fontFamily: 'Georgia, "Times New Roman", serif',
                    textShadow: '0 4px 30px rgba(212, 175, 55, 0.25)',
                  }}
                >
                  Cross-Border
                </h2>
              </FadeUp>

              <FadeUp delay={0.6} duration={1} y={40}>
                <h2
                  style={{
                    fontSize: 'clamp(32px, 5.5vw, 72px)',
                    fontWeight: 300,
                    lineHeight: 1.08,
                    letterSpacing: '-0.02em',
                    color: '#F1E5AC',
                    margin: 0,
                    fontFamily: 'Georgia, "Times New Roman", serif',
                  }}
                >
                  Credit Engine.
                </h2>
              </FadeUp>

              {/* Subtext */}
              <FadeUp delay={0.9} duration={0.8} y={20}>
                <p
                  style={{
                    marginTop: '32px',
                    fontSize: 'clamp(13px, 1.4vw, 17px)',
                    lineHeight: 1.7,
                    color: 'rgba(241, 229, 172, 0.7)',
                    maxWidth: '520px',
                    letterSpacing: '0.01em',
                  }}
                >
                  Seamlessly move your foreign startup to India with
                  AI‑powered compliance and smart banking.
                </p>
              </FadeUp>

              {/* CTA Button — Bracketed style like the reference */}
              <FadeUp delay={1.15} duration={0.8} y={20}>
                <button
                  onClick={() => setIsUploadModalOpen(true)}
                  style={{
                    marginTop: '48px',
                    fontSize: '12px',
                    fontWeight: 600,
                    letterSpacing: '0.18em',
                    color: '#D4AF37',
                    textTransform: 'uppercase',
                    backgroundColor: 'transparent',
                    padding: '18px 40px',
                    border: '1px solid rgba(212, 175, 55, 0.5)',
                    cursor: 'pointer',
                    transition: 'all 0.4s cubic-bezier(0.22, 1, 0.36, 1)',
                    position: 'relative',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#D4AF37';
                    e.currentTarget.style.color = '#0B0B0C';
                    e.currentTarget.style.boxShadow = '0 0 40px rgba(212, 175, 55, 0.35)';
                    e.currentTarget.style.borderColor = '#D4AF37';
                    e.currentTarget.style.transform = 'scale(1.03)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.color = '#D4AF37';
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.borderColor = 'rgba(212, 175, 55, 0.5)';
                    e.currentTarget.style.transform = 'scale(1)';
                  }}
                >
                  [ &nbsp; ACCESS PLATFORM &nbsp; ]
                </button>
              </FadeUp>
            </div>

            {/* Bottom decorative line */}
            <FadeUp delay={1.4} duration={1} y={0}>
              <div
                style={{
                  position: 'absolute',
                  bottom: '40px',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  width: '60px',
                  height: '1px',
                  background: 'linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.4), transparent)',
                }}
              />
            </FadeUp>
          </section>
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
              style={{
                borderColor: "var(--border-default)",
                background: "rgba(10, 10, 10, 0.88)",
                backdropFilter: "blur(20px)",
              }}
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
                        background: "rgba(26, 26, 26, 0.8)",
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
                      style={{ background: "rgba(26, 26, 26, 0.6)", border: "1px solid var(--border-subtle)" }}
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

                {/* Regulatory Database Toggle */}
                <button
                  onClick={handleToggleRegulatory}
                  disabled={isRegulatoryLoading}
                  className="w-full flex items-center justify-between px-4 py-3 rounded-lg transition-all duration-200 mb-3"
                  style={{
                    background: isRegulatoryLoaded ? "var(--gradient-primary)" : "rgba(26, 26, 26, 0.6)",
                    color: isRegulatoryLoaded ? "#000" : "var(--text-primary)",
                    border: isRegulatoryLoaded ? "none" : "1px solid var(--border-subtle)",
                    boxShadow: isRegulatoryLoaded ? "0 0 16px rgba(212, 175, 55, 0.2)" : "none",
                  }}
                >
                  <div className="flex flex-col items-start text-left">
                    <span className="text-[11px] uppercase tracking-[0.1em] font-bold">
                      {isRegulatoryLoaded ? "✓ Regulatory DB Active" : "Regulatory DB (20 PDFs)"}
                    </span>
                    <span className="text-[9px] opacity-70" style={{ color: isRegulatoryLoaded ? "#333" : "var(--text-muted)" }}>
                      Verified Indian Regulatory Framework
                    </span>
                  </div>
                  {isRegulatoryLoading ? (
                    <div className="w-4 h-4 rounded-full border-2 border-t-transparent animate-spin" style={{ borderColor: isRegulatoryLoaded ? "#000" : "var(--anchorium-gold)", borderTopColor: "transparent" }} />
                  ) : (
                    <div className="w-8 h-4 rounded-full flex items-center p-0.5 transition-all" style={{ background: isRegulatoryLoaded ? "rgba(0,0,0,0.5)" : "var(--surface-highlight)" }}>
                      <div className={`w-3 h-3 rounded-full bg-white transition-all ${isRegulatoryLoaded ? "translate-x-4" : "translate-x-0"}`} />
                    </div>
                  )}
                </button>

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
                    background: "rgba(26, 26, 26, 0.6)",
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
            <div
              className="flex-1 flex flex-col min-w-0"
              style={{
                background: "rgba(10, 10, 10, 0.85)",
                backdropFilter: "blur(16px)",
              }}
            >
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
              background: "rgba(10, 10, 10, 0.95)",
              borderColor: "rgba(212, 175, 55, 0.3)",
              boxShadow: "0 0 40px rgba(212, 175, 55, 0.15)",
            }}
          >
            <div className="flex items-center justify-between pb-2 border-b" style={{ borderColor: "var(--border-subtle)" }}>
              <div>
                <h3 className="text-base font-semibold uppercase tracking-wider text-gradient">
                  {showDashboard ? "Upload Additional Document" : "Initialize Omni-Engine"}
                </h3>
                <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                  Maximum 6 active documents in Vault
                </p>
              </div>
              <button
                onClick={() => setIsUploadModalOpen(false)}
                className="text-[10px] font-semibold uppercase tracking-[0.15em] px-3 py-1.5 rounded-lg transition-all duration-200 border"
                style={{
                  background: "rgba(26, 26, 26, 0.8)",
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
          className="relative border-t px-6 py-4 text-center text-[10px] uppercase tracking-[0.2em]"
          style={{ borderColor: "rgba(212, 175, 55, 0.08)", color: "rgba(241, 229, 172, 0.35)", zIndex: 1 }}
        >
          Anchorium Omni-Engine v0.1.0 — Your data stays on your device. We never store, share, or train on your financial documents.
        </footer>
      )}
    </div>
  );
}
