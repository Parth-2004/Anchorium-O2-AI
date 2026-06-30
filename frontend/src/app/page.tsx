/**
 * ==========================================================================
 * Home Page — Anchorium Omni-Engine
 * ==========================================================================
 *
 * The main entry point for the application. Renders the branding header
 * and the LocalEmbedder component for in-browser document processing.
 */

import LocalEmbedder from "@/components/LocalEmbedder";

export default function Home() {
  return (
    <div className="flex flex-col flex-1 min-h-screen" style={{ background: "var(--surface-base)" }}>
      {/* ================================================================ */}
      {/* Top Navigation Bar                                               */}
      {/* ================================================================ */}
      <nav
        className="sticky top-0 z-50 glass-panel border-b px-6 py-3"
        style={{ borderColor: "var(--border-subtle)" }}
      >
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          {/* Logo / Brand */}
          <div className="flex items-center gap-3">
            {/* Logo mark — stylized anchor */}
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{
                background: "var(--gradient-primary)",
                boxShadow: "0 0 16px rgba(0, 212, 255, 0.25)",
              }}
            >
              <svg
                className="w-5 h-5 text-white"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 2a2 2 0 100 4 2 2 0 000-4zM12 6v4m0 0l-4 8m4-8l4 8M6 18h12"
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

          {/* Status indicator */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs"
              style={{
                background: "rgba(16, 185, 129, 0.1)",
                color: "var(--anchorium-emerald)",
                border: "1px solid rgba(16, 185, 129, 0.15)",
              }}
            >
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                  style={{ background: "var(--anchorium-emerald)" }}
                />
                <span className="relative inline-flex rounded-full h-2 w-2"
                  style={{ background: "var(--anchorium-emerald)" }}
                />
              </span>
              Local Mode
            </div>
          </div>
        </div>
      </nav>

      {/* ================================================================ */}
      {/* Main Content                                                     */}
      {/* ================================================================ */}
      <main className="flex-1 flex flex-col">
        {/* Hero section */}
        <section className="relative overflow-hidden">
          {/* Background gradient glow */}
          <div
            className="absolute inset-0 pointer-events-none"
            style={{ background: "var(--gradient-glow)" }}
          />

          <div className="relative max-w-5xl mx-auto px-6 pt-16 pb-8 text-center">
            <h2
              className="text-3xl sm:text-4xl font-bold tracking-tight mb-3"
              style={{ color: "var(--text-primary)" }}
            >
              Process Documents{" "}
              <span className="text-gradient">Privately</span>
            </h2>
            <p
              className="text-base max-w-2xl mx-auto leading-relaxed"
              style={{ color: "var(--text-secondary)" }}
            >
              Upload your financial documents below. Our AI runs{" "}
              <strong style={{ color: "var(--text-primary)" }}>
                entirely inside your browser
              </strong>{" "}
              — your data never touches our servers. Every vector is generated
              locally using the MiniLM embedding model.
            </p>

            {/* Architecture badges */}
            <div className="flex flex-wrap justify-center gap-3 mt-6">
              {[
                { icon: "🔒", label: "End-to-End Encrypted" },
                { icon: "🧠", label: "In-Browser AI" },
                { icon: "⚡", label: "Local Vector Search" },
                { icon: "💨", label: "Ephemeral Compute" },
              ].map((badge) => (
                <div
                  key={badge.label}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs"
                  style={{
                    background: "var(--surface-raised)",
                    color: "var(--text-secondary)",
                    border: "1px solid var(--border-subtle)",
                  }}
                >
                  <span>{badge.icon}</span>
                  {badge.label}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Embedder component */}
        <section className="px-6 pb-16">
          <LocalEmbedder />
        </section>
      </main>

      {/* ================================================================ */}
      {/* Footer                                                           */}
      {/* ================================================================ */}
      <footer
        className="border-t px-6 py-4 text-center text-xs"
        style={{
          borderColor: "var(--border-subtle)",
          color: "var(--text-muted)",
        }}
      >
        <p>
          Anchorium Omni-Engine v0.1.0 — Your data stays on your device.
          We never store, share, or train on your financial documents.
        </p>
      </footer>
    </div>
  );
}
