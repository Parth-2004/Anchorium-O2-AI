"use client";

import React, { useState, useRef, useEffect } from "react";

interface EmbeddingResult {
  text: string;
  vector: number[];
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface ChatProps {
  contextChunks: EmbeddingResult[];
  onBack?: () => void;
}

const cosineSimilarity = (vecA: number[], vecB: number[]): number => {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (normA * normB);
};

// Finance Icons
const FinanceIconChart = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M4 19V5" strokeLinecap="round"/>
    <path d="M4 19h16" strokeLinecap="round"/>
    <path d="M8 15l4-4 3 3 5-5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const FinanceIconShield = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M12 3l8 4v6c0 5-3.5 9-8 10-4.5-1-8-5-8-10V7l8-4z" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const FinanceIconArrows = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M7 17l-5-5 5-5" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M2 12h12" strokeLinecap="round"/>
    <path d="M17 7l5 5-5 5" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M22 12H10" strokeLinecap="round"/>
  </svg>
);

const FinanceIconBook = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M4 19V5a2 2 0 0 1 2-2h11a2 2 0 0 1 2 2v14" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M6 17h14" strokeLinecap="round"/>
    <path d="M6 21h14" strokeLinecap="round"/>
  </svg>
);

export default function Chat({ contextChunks, onBack }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const retrieveRelevantChunks = (query: string, topK: number = 5): EmbeddingResult[] => {
    const dummyQueryEmbedding = new Array(384).fill(0).map(() => Math.random());
    return [...contextChunks]
      .map((chunk) => ({
        ...chunk,
        score: cosineSimilarity(dummyQueryEmbedding, chunk.vector),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .filter((c) => c.score > 0.75);
  };

  const sendMessage = async (content: string, endpoint: string = "/api/v1/query") => {
    if (!content.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const relevantChunks = retrieveRelevantChunks(content);
      const systemPrompt =
        "You are an elite cross-border corporate finance AI. Answer the user's question using ONLY the provided context. If the math or data is not in the context, say 'Insufficient data provided.' Never hallucinate.";

      const response = await fetch("http://localhost:8000" + endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: content,
          context_chunks: relevantChunks,
          system_prompt: systemPrompt,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);

      if (reader) {
        let done = false;
        while (!done) {
          const { value, done: doneReading } = await reader.read();
          done = doneReading;
          if (value) {
            const chunk = decoder.decode(value);
            const lines = chunk.split("\n");
            for (const line of lines) {
              if (line.startsWith("data: ")) {
                try {
                  const data = JSON.parse(line.slice(6));
                  if (data.type === "content") {
                    assistantMessage.content += data.content;
                    setMessages((prev) =>
                      prev.map((m) => (m.id === assistantMessage.id ? { ...assistantMessage } : m))
                    );
                  }
                } catch (e) {
                  // Ignore invalid JSON
                }
              }
            }
          }
        }
      }
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, an error occurred. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const quickActions = [
    {
      label: "Convert US GAAP to Indian Banking Format",
      endpoint: "/api/v1/underwrite",
      icon: <FinanceIconChart />,
    },
    {
      label: "Run RBI & FEMA Compliance Check",
      endpoint: "/api/v1/compliance",
      icon: <FinanceIconShield />,
    },
    {
      label: "Simulate USD-INR Debt Arbitrage",
      endpoint: "/api/v1/arbitrage",
      icon: <FinanceIconArrows />,
    },
    {
      label: "Generate India Soft-Landing Playbook",
      endpoint: "/api/v1/query",
      icon: <FinanceIconBook />,
      defaultQuery:
        "Generate a comprehensive India soft-landing playbook for expanding foreign startups, covering regulatory, financial, and operational considerations.",
    },
  ];

  return (
    <div className="flex flex-col h-full bg-[var(--surface-base)]">
      {/* Chat Header */}
      <div className="border-b border-[var(--border-default)] px-6 py-4 flex items-center gap-4 glass-panel">
        {onBack && (
          <button
            onClick={onBack}
            className="p-2 rounded-lg hover:bg-[var(--surface-overlay)] transition-all duration-200"
            style={{ color: "var(--text-accent)" }}
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
          </button>
        )}
        <div className="flex-1">
          <h2 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
            Omni-Chat
          </h2>
          <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
            Ask questions about your document or use quick actions
          </p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-4 animate-fade-in">
            <div
              className="w-20 h-20 rounded-2xl flex items-center justify-center animate-pulse-gold"
              style={{ background: "var(--gradient-primary)" }}
            >
              <svg
                className="w-10 h-10 text-black"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                />
              </svg>
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-semibold" style={{ color: "var(--text-primary)" }}>
                Welcome to Omni-Chat
              </h3>
              <p style={{ color: "var(--text-secondary)" }}>
                Your document has been processed locally. Ask any question or use the quick action
                buttons below.
              </p>
            </div>
          </div>
        )}

        {messages.map((message, idx) => (
          <div
            key={message.id}
            className={`flex gap-4 ${
              message.role === "user" ? "justify-end" : "justify-start"
            } animate-fade-in`}
            style={{ animationDelay: `${idx * 50}ms` }}
          >
            <div
              className={`max-w-3xl p-4 rounded-2xl ${
                message.role === "user"
                  ? "rounded-tr-sm"
                  : "rounded-tl-sm glass-panel"
              }`}
              style={
                message.role === "user"
                  ? {
                      background: "var(--gradient-primary)",
                      color: "black",
                    }
                  : {}
              }
            >
              <p className="whitespace-pre-wrap leading-relaxed">
                {message.content}
              </p>
              <p
                className="text-xs mt-2 opacity-70"
                style={
                  message.role === "user"
                    ? {}
                    : { color: "var(--text-muted)" }
                }
              >
                {message.timestamp.toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-4 justify-start animate-fade-in">
            <div className="p-4 rounded-2xl rounded-tl-sm glass-panel flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full animate-pulse"
                style={{ background: "var(--anchorium-gold)" }}
              />
              <div
                className="w-2 h-2 rounded-full animate-pulse"
                style={{
                  background: "var(--anchorium-gold)",
                  animationDelay: "0.2s",
                }}
              />
              <div
                className="w-2 h-2 rounded-full animate-pulse"
                style={{
                  background: "var(--anchorium-gold)",
                  animationDelay: "0.4s",
                }}
              />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Quick Actions */}
      <div className="border-t border-[var(--border-default)] p-4 glass-panel">
        <div className="flex flex-wrap gap-2 mb-4">
          {quickActions.map((action, idx) => (
            <button
              key={idx}
              onClick={() =>
                sendMessage(
                  action.defaultQuery || action.label,
                  action.endpoint
                )
              }
              disabled={isLoading}
              className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2 disabled:opacity-50 glass-panel-hover"
              style={{
                color: "var(--text-primary)",
              }}
            >
              <span style={{ color: "var(--text-accent)" }}>{action.icon}</span>
              <span className="truncate max-w-xs">{action.label}</span>
            </button>
          ))}
        </div>

        {/* Input */}
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage(input);
              }
            }}
            placeholder="Ask a question about your document..."
            disabled={isLoading}
            className="flex-1 px-4 py-3 rounded-xl border bg-[var(--surface-overlay)] text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none transition-all duration-200"
            style={{
              borderColor: "var(--border-subtle)",
            }}
            onFocus={(e) => {
              e.target.style.borderColor = "var(--border-accent)";
              e.target.style.boxShadow = "var(--shadow-glow)";
            }}
            onBlur={(e) => {
              e.target.style.borderColor = "var(--border-subtle)";
              e.target.style.boxShadow = "none";
            }}
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={isLoading || !input.trim()}
            className="px-6 py-3 rounded-xl font-medium transition-all duration-200 disabled:opacity-50"
            style={{
              background: "var(--gradient-primary)",
              color: "black",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.boxShadow = "var(--shadow-glow-strong)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.boxShadow = "none";
            }}
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
