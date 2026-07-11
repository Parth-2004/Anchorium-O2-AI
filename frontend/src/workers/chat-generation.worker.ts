/**
 * ==========================================================================
 * Anchorium Omni-Brain — Multi-Agent Regulatory Advisor Engine
 * ==========================================================================
 *
 * This worker handles the chat generation pipeline. It supports two modes:
 *
 * 1. BACKEND MODE (default): Sends queries + context chunks to the FastAPI
 *    backend, which routes to the appropriate specialist agent via the
 *    Omni-Engine's 5-agent architecture. Responses stream back via SSE.
 *
 * 2. OFFLINE MODE (fallback): Uses the built-in knowledge base for basic
 *    regulatory Q&A when the backend is unreachable. This mode does NOT
 *    use LLM generation — it's a pattern-matched, template-based fallback.
 *
 * Architecture (Backend Mode):
 *   1. Query Understanding → Intent classification + agent routing
 *   2. Backend Call → SSE stream to /api/v1/{agent} endpoint
 *   3. Response Assembly → Parses SSE events and streams to UI
 *
 * Architecture (Offline Mode):
 *   1. Query Understanding → Deep intent classification + entity extraction
 *   2. Knowledge Brain → Pre-built expert knowledge on Indian regulations
 *   3. Context Integration → Weaves in specific facts from uploaded PDFs
 *   4. Conversational Generation → Produces natural, advisor-like responses
 *
 * SECURITY: Only the top 3-5 relevant text chunks + prompt are sent to
 * the backend. Full conversation history is NEVER sent.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface RelevantChunk {
  text: string;
  score: number;
  documentName?: string;
}

interface GenerateMessage {
  type: "generate";
  messageId: string;
  query: string;
  chunks: RelevantChunk[];
  agentType: string;
  endpoint?: string;
}

interface QueryAnalysis {
  intent: string;
  entities: string[];
  subTopics: string[];
  isYesNoQuestion: boolean;
  isHowQuestion: boolean;
  isWhatQuestion: boolean;
  isCanQuestion: boolean;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const BACKEND_URL = "http://localhost:8000";
const BACKEND_TIMEOUT_MS = 300000; // 5 minute timeout for local LLM agent responses

// Agent routing map — maps intent → backend agent name
const INTENT_TO_AGENT: Record<string, string> = {
  loan_eligibility: "compliance_copilot",
  fdi_rules: "compliance_copilot",
  fema_compliance: "compliance_copilot",
  ecb_rules: "compliance_copilot",
  compliance_process: "compliance_copilot",
  entity_structure: "compliance_copilot",
  investment_rules: "compliance_copilot",
  foreign_person_rules: "compliance_copilot",
  gaap_conversion: "gaap_translator",
  tax_guidance: "compliance_copilot",
  remittance_rules: "compliance_copilot",
  currency_arbitrage: "arbitrage_calculator",
  company_setup: "compliance_copilot",
  india_expansion: "compliance_copilot",
  general_query: "compliance_copilot",
};

// Agent type from Chat.tsx → backend agent name
const AGENT_TYPE_MAP: Record<string, string> = {
  "Compliance Copilot": "compliance_copilot",
  Underwriter: "gaap_translator",
  "Arbitrage Calculator": "arbitrage_calculator",
  "Strategy Agent": "compliance_copilot",
  "Trust Score": "trust_score",
  "KYC Extractor": "kyc_extractor",
};

// Agent → endpoint map
const AGENT_ENDPOINT: Record<string, string> = {
  compliance_copilot: "/api/v1/compliance",
  gaap_translator: "/api/v1/underwrite",
  arbitrage_calculator: "/api/v1/arbitrage",
  trust_score: "/api/v1/trust-score",
  kyc_extractor: "/api/v1/kyc-extract",
};

// ---------------------------------------------------------------------------
// BRAIN MODULE 1: Deep Query Understanding
// ---------------------------------------------------------------------------

function analyzeQuery(query: string): QueryAnalysis {
  const q = query.toLowerCase().trim();

  // Detect question type
  const isYesNoQuestion =
    /^(can|will|is|are|do|does|should|could|would|may|am)\b/i.test(q) ||
    /\b(possible|allowed|permitted|eligible|able)\b/i.test(q);
  const isHowQuestion =
    /^how\b/i.test(q) || /\b(process|procedure|steps|way to)\b/i.test(q);
  const isWhatQuestion =
    /^what\b/i.test(q) ||
    /\b(define|meaning|explain|tell me about)\b/i.test(q);
  const isCanQuestion =
    /^can\b/i.test(q) || /\b(allowed|permitted|eligible)\b/i.test(q);

  // Extract entities
  const entities: string[] = [];
  const entityPatterns: [RegExp, string][] = [
    [/\b(loan|loans|lending|borrow|borrowing|credit)\b/i, "loans"],
    [/\b(fdi|foreign direct investment)\b/i, "FDI"],
    [/\b(fema|foreign exchange)\b/i, "FEMA"],
    [/\b(rbi|reserve bank)\b/i, "RBI"],
    [/\b(tax|taxes|gst|income tax|withholding)\b/i, "taxation"],
    [/\b(gaap|ind[\s-]?as|accounting standard|ifrs)\b/i, "accounting"],
    [
      /\b(company|incorporate|registration|setup|establish)\b/i,
      "company_setup",
    ],
    [/\b(subsidiary|branch|liaison|project office)\b/i, "entity_type"],
    [/\b(remittance|transfer|send money|repatriate)\b/i, "remittance"],
    [
      /\b(nri|non[\s-]?resident|foreign national|foreigner|overseas)\b/i,
      "foreign_person",
    ],
    [/\b(investment|invest|portfolio|shares|equity)\b/i, "investment"],
    [/\b(ecb|external commercial borrowing)\b/i, "ECB"],
    [/\b(currency|usd|inr|dollar|rupee|exchange rate)\b/i, "currency"],
    [/\b(arbitrage|hedging|forward)\b/i, "arbitrage"],
    [/\b(compliance|audit|filing|report|return)\b/i, "compliance"],
    [/\b(india|indian|domestic)\b/i, "india"],
    [/\b(business|startup|entrepreneur|founder)\b/i, "business"],
    [/\b(profit|revenue|income|earnings|financial)\b/i, "financials"],
    [/\b(capital|equity|debt|funding|raise)\b/i, "capital"],
    [/\b(expand|expansion|grow|growth|scale)\b/i, "expansion"],
    [/\b(pan|tan|gstin|din|cin)\b/i, "identifiers"],
    [/\b(bank account|current account|savings)\b/i, "banking"],
    [/\b(sector|sector cap|sectoral)\b/i, "sectors"],
    [
      /\b(approval|permission|clearance|automatic route|government route)\b/i,
      "approval_route",
    ],
    [/\b(trust score|gts|credit score|cibil|fico)\b/i, "trust_score"],
    [/\b(kyc|know your customer|aml|identity)\b/i, "kyc"],
    [/\b(playbook|soft.?landing|onboard)\b/i, "playbook"],
  ];

  for (const [pattern, entity] of entityPatterns) {
    if (pattern.test(q)) entities.push(entity);
  }

  // Classify primary intent
  let intent = "general_query";
  if (entities.includes("loans") || /loan|borrow|credit/i.test(q))
    intent = "loan_eligibility";
  else if (entities.includes("FDI") || /fdi|foreign.*invest.*india/i.test(q))
    intent = "fdi_rules";
  else if (
    entities.includes("accounting") ||
    /gaap|ind.?as|convert.*account/i.test(q)
  )
    intent = "gaap_conversion";
  else if (
    entities.includes("company_setup") ||
    /setup|start.*business|incorporate|register.*company/i.test(q)
  )
    intent = "company_setup";
  else if (entities.includes("FEMA") || /fema|foreign exchange/i.test(q))
    intent = "fema_compliance";
  else if (entities.includes("taxation")) intent = "tax_guidance";
  else if (entities.includes("remittance")) intent = "remittance_rules";
  else if (entities.includes("ECB")) intent = "ecb_rules";
  else if (entities.includes("arbitrage") || entities.includes("currency"))
    intent = "currency_arbitrage";
  else if (entities.includes("compliance")) intent = "compliance_process";
  else if (entities.includes("entity_type")) intent = "entity_structure";
  else if (entities.includes("investment")) intent = "investment_rules";
  else if (entities.includes("expansion") && entities.includes("india"))
    intent = "india_expansion";
  else if (entities.includes("business") && entities.includes("india"))
    intent = "india_expansion";
  else if (entities.includes("foreign_person"))
    intent = "foreign_person_rules";

  // Extract sub-topics
  const subTopics: string[] = [];
  if (/expand|grow|scale/i.test(q)) subTopics.push("business_expansion");
  if (/india/i.test(q)) subTopics.push("india_context");
  if (/foreign|overseas|abroad/i.test(q)) subTopics.push("cross_border");
  if (/startup|founder/i.test(q)) subTopics.push("startup_specific");

  return {
    intent,
    entities,
    subTopics,
    isYesNoQuestion,
    isHowQuestion,
    isWhatQuestion,
    isCanQuestion,
  };
}

// ---------------------------------------------------------------------------
// BACKEND MODE: Stream from FastAPI agents
// ---------------------------------------------------------------------------

async function streamFromBackend(
  messageId: string,
  query: string,
  chunks: RelevantChunk[],
  agentName: string,
  endpoint: string
): Promise<boolean> {
  try {
    self.postMessage({
      type: "status",
      message: `Connecting to ${agentName} agent...`,
    });

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS);

    const response = await fetch(`${BACKEND_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        context_chunks: chunks.map((c) => ({
          text: c.text,
          documentName: c.documentName || "Unknown",
          score: c.score || 0,
        })),
        agent_type: agentName,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response stream");

    const decoder = new TextDecoder();
    let buffer = "";
    let gotContent = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);

          switch (event.type) {
            case "thinking":
              self.postMessage({
                type: "status",
                message: `${agentName} is analyzing...`,
              });
              break;

            case "agent":
              self.postMessage({
                type: "status",
                message: `Agent: ${event.agent}`,
              });
              break;

            case "status":
              self.postMessage({
                type: "status",
                message: event.message,
              });
              break;

            case "content":
              gotContent = true;
              // Stream word-by-word for typing effect
              const words = event.content.split(" ");
              let wordBuffer = "";
              for (let i = 0; i < words.length; i++) {
                wordBuffer += words[i] + " ";
                if (wordBuffer.length > 12 || i === words.length - 1) {
                  self.postMessage({
                    type: "stream-chunk",
                    messageId,
                    text: wordBuffer,
                  });
                  wordBuffer = "";
                  await new Promise((r) => setTimeout(r, 8));
                }
              }
              break;

            case "structured":
              if (event.data && event.data.chart_data) {
                self.postMessage({
                  type: "stream-chart",
                  messageId,
                  chartData: event.data.chart_data,
                });
              }
              break;

            case "confidence":
              // Send confidence tier as a styled badge
              const tierColors: Record<string, string> = {
                HIGH: "[HIGH]",
                MEDIUM: "[MEDIUM]",
                LOW: "[LOW]",
              };
              const badge = tierColors[event.tier] || "[-]";
              self.postMessage({
                type: "stream-chunk",
                messageId,
                text: `\n\n---\n\n${badge} **Confidence:** ${event.tier}`,
              });
              if (event.requires_ca_review) {
                self.postMessage({
                  type: "stream-chunk",
                  messageId,
                  text: ` | **Requires CA Review**`,
                });
              }
              self.postMessage({
                type: "stream-chunk",
                messageId,
                text: `\n`,
              });
              break;

            case "disclaimer":
              self.postMessage({
                type: "stream-chunk",
                messageId,
                text: `\n\n> **Disclaimer:** ${event.text}\n`,
              });
              break;

            case "banner":
              // Already prepended by the agent, but reinforce in UI
              break;

            case "warning":
              self.postMessage({
                type: "stream-chunk",
                messageId,
                text: `\n\n> ${event.text}\n`,
              });
              break;

            case "audit":
              // Optionally show audit trail
              self.postMessage({
                type: "stream-chunk",
                messageId,
                text: `\n\n<details><summary>Audit Trail</summary>\n\n\`\`\`json\n${JSON.stringify(event.data, null, 2)}\n\`\`\`\n</details>\n`,
              });
              break;

            case "done":
              break;

            case "error":
              self.postMessage({
                type: "stream-chunk",
                messageId,
                text: `\n\n**Error:** ${event.message}\n`,
              });
              break;
          }
        } catch {
          // Skip unparseable SSE lines
        }
      }
    }

    if (gotContent) {
      self.postMessage({ type: "stream-complete", messageId });
      return true;
    }

    return false;
  } catch (error) {
    const msg =
      error instanceof Error ? error.message : "Backend unreachable";
    self.postMessage({
      type: "status",
      message: `Backend unavailable (${msg}). Falling back to offline mode...`,
    });
    return false;
  }
}

// ---------------------------------------------------------------------------
// OFFLINE MODE: Local knowledge base (fallback)
// ---------------------------------------------------------------------------

interface KnowledgeEntry {
  directAnswer: string;
  explanation: string;
  keyPoints: string[];
  importantNote?: string;
}

function getExpertKnowledge(
  intent: string
): KnowledgeEntry {
  const knowledgeBase: Record<string, KnowledgeEntry> = {
    loan_eligibility: {
      directAnswer:
        "Yes, foreign nationals and companies can obtain loans in India, but there are specific rules set by the RBI under FEMA regulations.",
      explanation:
        'The Reserve Bank of India (RBI) allows foreign entities to borrow money in India through several channels. The most common route for foreign companies is through **External Commercial Borrowings (ECB)** or by taking loans from Indian banks after setting up a local entity (like a subsidiary or branch office). The rules depend on your residency status, the purpose of the loan, and the amount you need.',
      keyPoints: [
        "You need to first set up a legal entity in India (subsidiary, branch office, or LLP) before applying for loans from Indian banks.",
        "Foreign companies can borrow through the **ECB route** — this lets you raise funds from overseas sources in foreign currency.",
        "Indian banks can lend to your Indian subsidiary once it's registered, just like any other Indian company.",
        "There are limits on interest rates, loan amounts, and end-use restrictions set by the RBI.",
        "For amounts under a certain threshold, you can go through the **automatic route** (no prior RBI approval needed). Larger amounts may require government approval.",
      ],
      importantNote:
        "The exact loan limits and interest rate caps change periodically. Always check the latest RBI Master Direction on ECB for current numbers.",
    },

    gaap_conversion: {
      directAnswer:
        "Converting from US GAAP to Indian Accounting Standards (Ind AS) involves mapping your financial statements to the Indian framework, which is largely converged with IFRS.",
      explanation:
        "Ind AS (Indian Accounting Standards) is India's version of IFRS. If your company currently follows US GAAP, you'll need to convert your books to Ind AS once you set up operations in India. Key differences include revenue recognition timing, lease accounting treatment, and financial instrument classification.",
      keyPoints: [
        "Ind AS is largely based on IFRS, so it's closer to international standards than US GAAP.",
        "**Revenue recognition** under Ind AS 115 follows a 5-step model similar to ASC 606 in US GAAP.",
        "**Lease accounting** requires most leases to be recognized on the balance sheet (similar to ASC 842).",
        "You'll need to prepare an **opening Ind AS balance sheet** at the transition date.",
        "A qualified Indian Chartered Accountant (CA) should oversee the conversion process.",
      ],
      importantNote:
        "Companies with a net worth of ₹250 crore or more are mandatorily required to follow Ind AS.",
    },

    general_query: {
      directAnswer:
        "I can help you understand Indian regulations for setting up and running a foreign business in India.",
      explanation:
        "As your Anchorium regulatory advisor, I have deep knowledge of Indian business regulations including FEMA compliance, FDI rules, RBI guidelines, company incorporation, taxation, and accounting standards. For the most accurate and grounded analysis, please upload your regulatory documents so I can cite specific sources.",
      keyPoints: [
        "Ask me about **FDI rules** — which sectors are open, what approvals you need.",
        "Ask me about **company setup** — how to register, what entity type to choose.",
        "Ask me about **loans and financing** — ECBs, local bank loans, and capital requirements.",
        "Ask me about **tax and compliance** — corporate tax rates, GST, TDS, and filing deadlines.",
        "Ask me about **FEMA and RBI rules** — foreign exchange regulations, remittances, and reporting.",
      ],
    },
  };

  return knowledgeBase[intent] || knowledgeBase["general_query"];
}

function extractSupportingFacts(
  chunks: RelevantChunk[],
  analysis: QueryAnalysis
): string[] {
  const facts: string[] = [];
  const seen = new Set<string>();

  for (const chunk of chunks.slice(0, 5)) {
    const text = chunk.text
      .replace(/[\r\n]+/g, " ")
      .replace(/\s{2,}/g, " ")
      .replace(/Updated as on [A-Za-z]+ \d{1,2}, \d{4}/gi, "")
      .replace(/www\.rbi\.org\.in/gi, "")
      .replace(/RBI\/FED\/\d{4}-\d{2}\/\d+/gi, "")
      .replace(/FED Master Direction No\.\d+\/\d{4}-\d{2}/gi, "")
      .trim();

    const sentences = text
      .split(/(?<=[.!?])\s+/)
      .filter((s) => s.length > 50 && s.length < 400);

    for (const sent of sentences) {
      const lower = sent.toLowerCase();
      const isRelevant = analysis.entities.some((entity) => {
        const entityPatterns: Record<string, RegExp> = {
          loans: /loan|borrow|lend|credit|interest/i,
          FDI: /fdi|foreign.*direct.*invest|invest.*india/i,
          FEMA: /fema|foreign.*exchange|authorised.*dealer/i,
          taxation: /tax|gst|withhold|deduct/i,
          accounting: /gaap|ind.?as|accounting|revenue.*recogni/i,
          company_setup: /company|incorporat|registr|mca/i,
          remittance: /remit|transfer|repatri/i,
          ECB: /ecb|external.*commercial|borrow.*overseas/i,
          investment: /invest|equity|share|portfolio/i,
          foreign_person:
            /non.?resident|nri|foreign.*national|person.*outside/i,
          compliance: /compliance|filing|return|report/i,
          banking: /bank.*account|current.*account|nre|nro/i,
          expansion: /business|expand|operat/i,
          india: /india|domestic|rupee/i,
          business: /business|company|entity/i,
        };
        return entityPatterns[entity]?.test(lower);
      });

      if (isRelevant) {
        const key = lower.substring(0, 60);
        if (!seen.has(key)) {
          seen.add(key);
          facts.push(sent.trim());
          if (facts.length >= 3) return facts;
        }
      }
    }
  }

  return facts;
}

function generateOfflineResponse(
  query: string,
  analysis: QueryAnalysis,
  knowledge: KnowledgeEntry,
  supportingFacts: string[]
): string {
  let response = "";

  // Draft banner — mandatory per Core Directives
  response +=
    "> **DRAFT — PENDING HUMAN REVIEW. NOT LEGAL, TAX, OR CREDIT ADVICE.**\n\n";
  response += "> *Offline mode — for grounded, source-cited analysis, ensure the backend is running.*\n\n";

  // Opening
  if (analysis.isYesNoQuestion || analysis.isCanQuestion) {
    response += `**${knowledge.directAnswer}**\n\n`;
  } else if (analysis.isHowQuestion) {
    response += `Here's how it works:\n\n`;
  } else {
    response += `**${knowledge.directAnswer}**\n\n`;
  }

  // Body
  response += `${knowledge.explanation}\n\n`;

  // Key Points
  response += `**Here's what you need to know:**\n\n`;
  for (const point of knowledge.keyPoints) {
    response += `• ${point}\n\n`;
  }

  // Supporting evidence
  if (supportingFacts.length > 0) {
    response += `---\n\n`;
    response += `**From your uploaded documents:**\n\n`;
    for (const fact of supportingFacts) {
      const simplified = fact
        .replace(/Authorised Dealer Category[- ]I banks?/gi, "authorized banks")
        .replace(/person resident in India/gi, "Indian resident")
        .replace(/person resident outside India/gi, "foreign resident")
        .replace(/Reserve Bank of India/gi, "the RBI")
        .replace(/Reserve Bank/gi, "the RBI");
      response += `> *"${simplified.substring(0, 250)}${simplified.length > 250 ? "..." : ""}"*\n\n`;
    }
  }

  // Important note
  if (knowledge.importantNote) {
    response += `---\n\n`;
    response += `**Pro Tip:** ${knowledge.importantNote}\n`;
  }

  // Confidence badge (offline = always MEDIUM)
  response += `\n\n---\n\n**Confidence:** MEDIUM (offline mode — no RAG grounding)\n`;

  return response;
}

// ---------------------------------------------------------------------------
// Message Handler
// ---------------------------------------------------------------------------

self.addEventListener("message", async (event: MessageEvent) => {
  const data = event.data as GenerateMessage;
  if (data.type !== "generate") return;

  const { messageId, query, chunks, agentType, endpoint: requestedEndpoint } = data;

  try {
    self.postMessage({
      type: "status",
      message: "Analyzing your question...",
    });

    await new Promise((r) => setTimeout(r, 200));

    // Step 1: Analyze the query to determine the best agent
    const analysis = analyzeQuery(query);

    // Step 2: Determine which agent to route to
    const resolvedAgent =
      AGENT_TYPE_MAP[agentType] ||
      INTENT_TO_AGENT[analysis.intent] ||
      "compliance_copilot";

    const endpoint = requestedEndpoint || AGENT_ENDPOINT[resolvedAgent] || "/api/v1/query";

    // Step 3: Try backend first
    const backendSuccess = await streamFromBackend(
      messageId,
      query,
      chunks,
      resolvedAgent,
      endpoint
    );

    if (backendSuccess) return; // Done — backend handled it

    // Step 4: Fallback to offline mode
    self.postMessage({
      type: "status",
      message: "Using offline knowledge base...",
    });

    await new Promise((r) => setTimeout(r, 200));

    const knowledge = getExpertKnowledge(analysis.intent);
    const supportingFacts = extractSupportingFacts(chunks, analysis);
    const answer = generateOfflineResponse(
      query,
      analysis,
      knowledge,
      supportingFacts
    );

    // Stream the response with typing effect
    const words = answer.split(" ");
    let buffer = "";

    for (let i = 0; i < words.length; i++) {
      buffer += words[i] + " ";
      if (buffer.length > 12 || i === words.length - 1) {
        self.postMessage({ type: "stream-chunk", messageId, text: buffer });
        buffer = "";
        await new Promise((r) => setTimeout(r, 12));
      }
    }

    self.postMessage({ type: "stream-complete", messageId });
  } catch (error) {
    self.postMessage({
      type: "error",
      messageId,
      message: `Analysis failed: ${error instanceof Error ? error.message : String(error)}`,
    });
  }
});
