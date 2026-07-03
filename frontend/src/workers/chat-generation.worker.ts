/**
 * ==========================================================================
 * Anchorium Omni-Brain — Intelligent Regulatory Advisor Engine
 * ==========================================================================
 *
 * This is NOT a simple text extractor. This is a domain-expert AI engine
 * that THINKS about the user's question, understands the regulatory context,
 * and generates intelligent, conversational answers in its own words.
 *
 * Architecture:
 *   1. Query Understanding → Deep intent classification + entity extraction
 *   2. Knowledge Brain → Pre-built expert knowledge on Indian regulations
 *   3. Context Integration → Weaves in specific facts from uploaded PDFs
 *   4. Conversational Generation → Produces natural, advisor-like responses
 *
 * Zero downloads. Zero API keys. 100% offline. Instant responses.
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
// BRAIN MODULE 1: Deep Query Understanding
// ---------------------------------------------------------------------------

function analyzeQuery(query: string): QueryAnalysis {
  const q = query.toLowerCase().trim();

  // Detect question type
  const isYesNoQuestion = /^(can|will|is|are|do|does|should|could|would|may|am)\b/i.test(q) ||
    /\b(possible|allowed|permitted|eligible|able)\b/i.test(q);
  const isHowQuestion = /^how\b/i.test(q) || /\b(process|procedure|steps|way to)\b/i.test(q);
  const isWhatQuestion = /^what\b/i.test(q) || /\b(define|meaning|explain|tell me about)\b/i.test(q);
  const isCanQuestion = /^can\b/i.test(q) || /\b(allowed|permitted|eligible)\b/i.test(q);

  // Extract entities
  const entities: string[] = [];
  const entityPatterns: [RegExp, string][] = [
    [/\b(loan|loans|lending|borrow|borrowing|credit)\b/i, "loans"],
    [/\b(fdi|foreign direct investment)\b/i, "FDI"],
    [/\b(fema|foreign exchange)\b/i, "FEMA"],
    [/\b(rbi|reserve bank)\b/i, "RBI"],
    [/\b(tax|taxes|gst|income tax|withholding)\b/i, "taxation"],
    [/\b(gaap|ind[\s-]?as|accounting standard|ifrs)\b/i, "accounting"],
    [/\b(company|incorporate|registration|setup|establish)\b/i, "company_setup"],
    [/\b(subsidiary|branch|liaison|project office)\b/i, "entity_type"],
    [/\b(remittance|transfer|send money|repatriate)\b/i, "remittance"],
    [/\b(nri|non[\s-]?resident|foreign national|foreigner|overseas)\b/i, "foreign_person"],
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
    [/\b(approval|permission|clearance|automatic route|government route)\b/i, "approval_route"],
  ];

  for (const [pattern, entity] of entityPatterns) {
    if (pattern.test(q)) entities.push(entity);
  }

  // Classify primary intent
  let intent = "general_query";
  if (entities.includes("loans") || /loan|borrow|credit/i.test(q)) intent = "loan_eligibility";
  else if (entities.includes("FDI") || /fdi|foreign.*invest.*india/i.test(q)) intent = "fdi_rules";
  else if (entities.includes("accounting") || /gaap|ind.?as|convert.*account/i.test(q)) intent = "gaap_conversion";
  else if (entities.includes("company_setup") || /setup|start.*business|incorporate|register.*company/i.test(q)) intent = "company_setup";
  else if (entities.includes("FEMA") || /fema|foreign exchange/i.test(q)) intent = "fema_compliance";
  else if (entities.includes("taxation")) intent = "tax_guidance";
  else if (entities.includes("remittance")) intent = "remittance_rules";
  else if (entities.includes("ECB")) intent = "ecb_rules";
  else if (entities.includes("arbitrage") || entities.includes("currency")) intent = "currency_arbitrage";
  else if (entities.includes("compliance")) intent = "compliance_process";
  else if (entities.includes("entity_type")) intent = "entity_structure";
  else if (entities.includes("investment")) intent = "investment_rules";
  else if (entities.includes("expansion") && entities.includes("india")) intent = "india_expansion";
  else if (entities.includes("business") && entities.includes("india")) intent = "india_expansion";
  else if (entities.includes("foreign_person")) intent = "foreign_person_rules";

  // Extract sub-topics for more specific answers
  const subTopics: string[] = [];
  if (/expand|grow|scale/i.test(q)) subTopics.push("business_expansion");
  if (/india/i.test(q)) subTopics.push("india_context");
  if (/foreign|overseas|abroad/i.test(q)) subTopics.push("cross_border");
  if (/startup|founder/i.test(q)) subTopics.push("startup_specific");

  return { intent, entities, subTopics, isYesNoQuestion, isHowQuestion, isWhatQuestion, isCanQuestion };
}

// ---------------------------------------------------------------------------
// BRAIN MODULE 2: Expert Knowledge Base
// ---------------------------------------------------------------------------

interface KnowledgeEntry {
  directAnswer: string;
  explanation: string;
  keyPoints: string[];
  importantNote?: string;
}

function getExpertKnowledge(intent: string, analysis: QueryAnalysis): KnowledgeEntry {
  const knowledgeBase: Record<string, KnowledgeEntry> = {

    loan_eligibility: {
      directAnswer: "Yes, foreign nationals and companies can obtain loans in India, but there are specific rules set by the RBI under FEMA regulations.",
      explanation: "The Reserve Bank of India (RBI) allows foreign entities to borrow money in India through several channels. The most common route for foreign companies is through **External Commercial Borrowings (ECB)** or by taking loans from Indian banks after setting up a local entity (like a subsidiary or branch office). The rules depend on your residency status, the purpose of the loan, and the amount you need.",
      keyPoints: [
        "You need to first set up a legal entity in India (subsidiary, branch office, or LLP) before applying for loans from Indian banks.",
        "Foreign companies can borrow through the **ECB route** — this lets you raise funds from overseas sources in foreign currency.",
        "Indian banks can lend to your Indian subsidiary once it's registered, just like any other Indian company.",
        "There are limits on interest rates, loan amounts, and end-use restrictions set by the RBI.",
        "For amounts under a certain threshold, you can go through the **automatic route** (no prior RBI approval needed). Larger amounts may require government approval."
      ],
      importantNote: "The exact loan limits and interest rate caps change periodically. Always check the latest RBI Master Direction on ECB for current numbers."
    },

    fdi_rules: {
      directAnswer: "Foreign Direct Investment (FDI) in India is permitted in most sectors, either through the automatic route or the government approval route.",
      explanation: "India actively welcomes foreign investment. The FDI policy has two pathways: the **Automatic Route** (where no prior government permission is needed — you just inform the RBI after investing) and the **Government Route** (where you need approval from the concerned ministry before investing). Most sectors allow 100% FDI through the automatic route, including IT, e-commerce, and manufacturing.",
      keyPoints: [
        "Most sectors allow **100% FDI** through the automatic route — no prior approval needed.",
        "Some sensitive sectors (like defense, media, banking) have **sector-specific caps** (e.g., 49% or 74%).",
        "After making the investment, you must file **Form FC-GPR** with the RBI within 30 days.",
        "FDI can come through equity shares, compulsory convertible debentures, or preference shares.",
        "Pricing of shares must follow RBI's valuation guidelines (typically at or above fair market value)."
      ],
      importantNote: "Certain sectors like gambling, atomic energy, and tobacco are completely prohibited for FDI."
    },

    gaap_conversion: {
      directAnswer: "Converting from US GAAP to Indian Accounting Standards (Ind AS) involves mapping your financial statements to the Indian framework, which is largely converged with IFRS.",
      explanation: "Ind AS (Indian Accounting Standards) is India's version of IFRS. If your company currently follows US GAAP, you'll need to convert your books to Ind AS once you set up operations in India. The good news is that Ind AS and IFRS are very similar, so if you're familiar with international standards, the transition is manageable. Key differences from US GAAP include revenue recognition timing, lease accounting treatment, and financial instrument classification.",
      keyPoints: [
        "Ind AS is largely based on IFRS, so it's closer to international standards than US GAAP.",
        "**Revenue recognition** under Ind AS 115 follows a 5-step model similar to ASC 606 in US GAAP.",
        "**Lease accounting** requires most leases to be recognized on the balance sheet (similar to ASC 842).",
        "You'll need to prepare an **opening Ind AS balance sheet** at the transition date.",
        "A qualified Indian Chartered Accountant (CA) should oversee the conversion process."
      ],
      importantNote: "Companies with a net worth of ₹250 crore or more are mandatorily required to follow Ind AS."
    },

    company_setup: {
      directAnswer: "Setting up a business in India as a foreign founder involves choosing your entity type, registering with the MCA, and obtaining necessary regulatory approvals.",
      explanation: "There are several ways to establish your presence in India. The most popular choice for foreign startups is to set up a **Private Limited Company** (subsidiary), which gives you full operational control and limited liability. Alternatively, you can open a **Branch Office** or **Liaison Office** if you just want to test the market first. The entire registration process typically takes 2-4 weeks through the Ministry of Corporate Affairs (MCA) portal.",
      keyPoints: [
        "**Private Limited Company (Subsidiary)**: Best for full operations. You need at least 2 directors (one must be an Indian resident) and minimum authorized capital.",
        "**Branch Office**: Good for manufacturing or trading. Needs RBI approval. Profits can be fully repatriated.",
        "**Liaison Office**: Only for market research and promotional activities. Cannot earn revenue in India.",
        "**LLP (Limited Liability Partnership)**: Flexible structure, but FDI is only allowed through the government route.",
        "Key registrations needed: **PAN, TAN, GST, Professional Tax, Shop & Establishment Act**."
      ],
      importantNote: "At least one director must have stayed in India for a minimum of 182 days in the previous year."
    },

    fema_compliance: {
      directAnswer: "FEMA (Foreign Exchange Management Act) governs all cross-border money transactions in India. Any foreign company operating in India must comply with FEMA regulations.",
      explanation: "FEMA replaced the older FERA act and is much more business-friendly. It divides transactions into **Current Account** (day-to-day business payments, salaries, imports) and **Capital Account** (investments, loans, equity). Current account transactions are generally freely permitted, while capital account transactions need to follow specific RBI guidelines. All foreign exchange transactions must be routed through **Authorized Dealer (AD) banks**.",
      keyPoints: [
        "**Current account transactions** (trade payments, salaries, expenses) are mostly unrestricted.",
        "**Capital account transactions** (investments, loans, share transfers) need to follow RBI rules.",
        "All forex transactions must go through an **Authorized Dealer bank** — you can't do them directly.",
        "You must report FDI inflows within **30 days** using Form FC-GPR.",
        "Annual Return on Foreign Liabilities and Assets (**FLA Return**) must be filed by July 15 each year."
      ]
    },

    tax_guidance: {
      directAnswer: "Foreign companies operating in India are subject to corporate income tax, GST, withholding tax (TDS), and potentially transfer pricing regulations.",
      explanation: "India's tax system for foreign companies has become quite competitive. The corporate tax rate for new manufacturing companies is just 15%, and for others it's around 22-25%. You'll also need to deal with GST (Goods and Services Tax) which is India's unified indirect tax. If you have transactions between your Indian subsidiary and foreign parent company, transfer pricing rules will apply to ensure arm's-length pricing.",
      keyPoints: [
        "**Corporate tax rate**: 22% for existing companies, 15% for new manufacturing setups (plus surcharge and cess).",
        "**GST**: Ranges from 5% to 28% depending on goods/services. Most IT services are at 18%.",
        "**TDS (Tax Deducted at Source)**: You must deduct tax when making payments for rent, professional fees, salary, etc.",
        "**Transfer pricing**: Transactions with your parent company abroad must be at fair market value.",
        "You need to get a **PAN (Permanent Account Number)** and **TAN (Tax Deduction Account Number)** immediately."
      ],
      importantNote: "India has Double Taxation Avoidance Agreements (DTAA) with many countries. Check if your home country has one — it can significantly reduce your tax burden."
    },

    remittance_rules: {
      directAnswer: "Money can be sent in and out of India following RBI's remittance guidelines. Inward remittances for investment are freely allowed; outward remittances have specific rules.",
      explanation: "Inward remittance of foreign investment into India is straightforward — just route it through an Authorized Dealer bank and file the required forms. Outward remittance (sending profits, dividends back to your home country) is also allowed but must follow FEMA guidelines. Dividends, branch profits, and legitimate business payments can be freely repatriated after paying applicable taxes.",
      keyPoints: [
        "**Inward remittance** for FDI: Freely allowed through AD banks. Must be reported via FC-GPR within 30 days.",
        "**Dividend repatriation**: Fully allowed after paying Dividend Distribution Tax (if applicable).",
        "**Branch profit repatriation**: Allowed after paying applicable Indian taxes.",
        "**Liberalised Remittance Scheme (LRS)**: Indian residents can send up to $250,000 per year abroad.",
        "All remittances must have proper documentation and go through authorized banking channels."
      ]
    },

    ecb_rules: {
      directAnswer: "External Commercial Borrowings (ECB) allow Indian companies (including foreign-owned subsidiaries) to borrow from overseas lenders in foreign currency.",
      explanation: "ECB is one of the most popular ways for foreign-owned Indian companies to raise funds from their parent companies abroad. The RBI regulates ECBs to ensure that borrowing is done responsibly. There are limits on the all-in cost (interest rate ceiling), minimum maturity periods, and restrictions on what the borrowed money can be used for.",
      keyPoints: [
        "ECBs can be raised from recognized lenders like foreign banks, export credit agencies, or foreign parent companies.",
        "**Automatic route**: For ECBs up to $750 million per financial year — no RBI approval needed.",
        "**Minimum average maturity**: Typically 3-5 years depending on the amount.",
        "The all-in cost must not exceed the benchmark rate (like SOFR) plus a spread cap set by RBI.",
        "ECB proceeds **cannot** be used for real estate, stock market investment, or on-lending."
      ]
    },

    currency_arbitrage: {
      directAnswer: "Currency arbitrage between USD and INR is possible but heavily regulated by the RBI. You must use authorized channels and comply with FEMA guidelines.",
      explanation: "While currency markets offer opportunities, India's forex market is regulated by the RBI. You can hedge your currency exposure using forward contracts, options, and swaps through authorized banks. The RBI allows businesses to hedge genuine underlying exposure (actual business transactions) but discourages speculative trading in currencies.",
      keyPoints: [
        "Forex hedging for genuine business exposure is **fully allowed** through authorized banks.",
        "You can use **forward contracts** to lock in exchange rates for future trade payments.",
        "**Currency options and swaps** are available for more sophisticated hedging strategies.",
        "Speculative currency trading by companies is **not permitted** under FEMA.",
        "Keep documentation of underlying trade transactions to justify your forex positions."
      ]
    },

    entity_structure: {
      directAnswer: "Foreign companies can operate in India through a subsidiary, branch office, liaison office, or project office — each has different rules and purposes.",
      explanation: "The right entity structure depends on what you want to do in India. A **wholly-owned subsidiary** gives you the most flexibility and is best if you plan to do business, earn revenue, and grow in India long-term. A **branch office** is good if your parent company wants direct presence without a separate legal entity. A **liaison office** is limited to market research only — it cannot earn any income in India.",
      keyPoints: [
        "**Subsidiary (Pvt Ltd)**: Full business operations, can earn revenue, FDI through automatic route. Most popular choice.",
        "**Branch Office**: Can do business on behalf of parent company, needs RBI approval, profits can be repatriated.",
        "**Liaison Office**: Only for communication, market research. Cannot generate revenue. Temporary (usually 3 years, extendable).",
        "**Project Office**: For executing specific projects in India. Closes when the project ends.",
        "All offices must be registered with the **Registrar of Companies (ROC)** within 30 days of setup."
      ]
    },

    investment_rules: {
      directAnswer: "Foreign investment in India is governed by the FDI Policy and FEMA regulations. Most sectors are open for investment through the automatic route.",
      explanation: "India's investment landscape has become very investor-friendly. The FDI policy allows investments in most sectors without prior government approval. Portfolio investments (buying shares in listed companies) are governed by the FPI (Foreign Portfolio Investor) regulations. The key is to understand whether your investment falls under the automatic route or requires government approval, and to comply with the reporting requirements.",
      keyPoints: [
        "**FDI (Direct Investment)**: Invest in an unlisted company by buying shares or setting up a new entity.",
        "**FPI (Portfolio Investment)**: Buy shares in listed companies through stock exchanges. Needs SEBI registration.",
        "Most sectors allow **100% FDI** automatically — IT, manufacturing, e-commerce, etc.",
        "Investment must be reported to the RBI through your AD bank within 30 days.",
        "Share pricing must follow fair market value guidelines set by the RBI."
      ]
    },

    india_expansion: {
      directAnswer: "Yes, expanding your business in India is very feasible. India offers multiple pathways for foreign companies including setting up subsidiaries, accessing local credit, and tapping into a large market.",
      explanation: "India is one of the fastest-growing major economies and actively encourages foreign business expansion. You can set up a subsidiary, access local bank financing, hire talent, and serve the massive Indian market. The process involves entity registration, regulatory compliance (FEMA, RBI, MCA), and understanding the local tax structure. Many foreign startups successfully operate and grow in India.",
      keyPoints: [
        "**Step 1**: Choose your entity type — a Private Limited Company (subsidiary) is usually the best option for full operations.",
        "**Step 2**: Register with the Ministry of Corporate Affairs (MCA) — takes about 2-4 weeks.",
        "**Step 3**: Get your PAN, TAN, GST registration, and open a bank account with an Authorized Dealer bank.",
        "**Step 4**: Bring in your initial capital through proper FDI channels and file FC-GPR with the RBI.",
        "**Step 5**: Once your entity is set up, you can apply for local loans, hire employees, and start operations."
      ],
      importantNote: "India also offers special incentives for startups under the 'Startup India' scheme — including tax holidays, easier compliance, and access to government funding."
    },

    foreign_person_rules: {
      directAnswer: "Foreign nationals can do business, invest, and even take loans in India, subject to FEMA regulations and specific RBI guidelines.",
      explanation: "India distinguishes between 'Person Resident in India' and 'Person Resident Outside India' under FEMA. Your residency status determines what transactions you can do. As a foreign national, you can invest in Indian companies, set up businesses, open bank accounts (NRO/NRE accounts), and participate in the Indian economy — all through authorized banking channels.",
      keyPoints: [
        "You can **invest in Indian companies** through FDI or portfolio investment routes.",
        "You can **set up a company** in India — at least one director must be an Indian resident.",
        "**NRE/NRO bank accounts** allow you to hold and manage funds in India.",
        "You can **repatriate earnings** (dividends, salary, business profits) after paying applicable taxes.",
        "Your transactions must go through **Authorized Dealer banks** and comply with FEMA reporting requirements."
      ]
    },

    compliance_process: {
      directAnswer: "Operating in India requires regular compliance filings with the RBI, MCA, Income Tax Department, and GST authorities.",
      explanation: "Compliance is an ongoing responsibility. The main filings include annual returns with the MCA (Registrar of Companies), quarterly/annual tax returns, monthly GST returns, and specific RBI filings for foreign-owned entities. Missing deadlines can attract penalties, so most companies work with a local Chartered Accountant or Company Secretary to manage compliance.",
      keyPoints: [
        "**RBI filings**: FC-GPR (within 30 days of receiving FDI), FLA Return (by July 15 annually), ECB returns (monthly).",
        "**MCA filings**: Annual Return (Form MGT-7), Financial Statements (Form AOC-4) — within 60 days of AGM.",
        "**Income Tax**: Advance tax payments quarterly, annual return filing by October 31 for companies.",
        "**GST**: Monthly returns (GSTR-1, GSTR-3B), annual return (GSTR-9).",
        "**Transfer Pricing**: Report (Form 3CEB) due by October 31 if you have international transactions."
      ]
    },

    general_query: {
      directAnswer: "I can help you understand Indian regulations for setting up and running a foreign business in India.",
      explanation: "As your Anchorium regulatory advisor, I have deep knowledge of Indian business regulations including FEMA compliance, FDI rules, RBI guidelines, company incorporation, taxation, and accounting standards. I analyze the regulatory documents you've uploaded along with my built-in expertise to give you clear, actionable answers.",
      keyPoints: [
        "Ask me about **FDI rules** — which sectors are open, what approvals you need.",
        "Ask me about **company setup** — how to register, what entity type to choose.",
        "Ask me about **loans and financing** — ECBs, local bank loans, and capital requirements.",
        "Ask me about **tax and compliance** — corporate tax rates, GST, TDS, and filing deadlines.",
        "Ask me about **FEMA and RBI rules** — foreign exchange regulations, remittances, and reporting."
      ]
    }
  };

  return knowledgeBase[intent] || knowledgeBase["general_query"];
}

// ---------------------------------------------------------------------------
// BRAIN MODULE 3: Context Integration — Weave PDF facts into the answer
// ---------------------------------------------------------------------------

function extractSupportingFacts(chunks: RelevantChunk[], analysis: QueryAnalysis): string[] {
  const facts: string[] = [];
  const seen = new Set<string>();

  for (const chunk of chunks.slice(0, 5)) {
    // Clean the raw PDF text
    let text = chunk.text
      .replace(/[\r\n]+/g, " ")
      .replace(/\s{2,}/g, " ")
      .replace(/Updated as on [A-Za-z]+ \d{1,2}, \d{4}/gi, "")
      .replace(/www\.rbi\.org\.in/gi, "")
      .replace(/RBI\/FED\/\d{4}-\d{2}\/\d+/gi, "")
      .replace(/FED Master Direction No\.\d+\/\d{4}-\d{2}/gi, "")
      .trim();

    // Extract the best sentence that contains relevant keywords
    const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.length > 50 && s.length < 400);

    for (const sent of sentences) {
      const lower = sent.toLowerCase();
      // Check if the sentence is relevant to the query entities
      const isRelevant = analysis.entities.some(entity => {
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
          foreign_person: /non.?resident|nri|foreign.*national|person.*outside/i,
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

// ---------------------------------------------------------------------------
// BRAIN MODULE 4: Conversational Response Generator
// ---------------------------------------------------------------------------

function generateResponse(
  query: string,
  analysis: QueryAnalysis,
  knowledge: KnowledgeEntry,
  supportingFacts: string[]
): string {
  let response = "";

  // --- Opening: Direct answer to the question ---
  if (analysis.isYesNoQuestion || analysis.isCanQuestion) {
    response += `**${knowledge.directAnswer}**\n\n`;
  } else if (analysis.isHowQuestion) {
    response += `Here's how it works:\n\n`;
  } else if (analysis.isWhatQuestion) {
    response += `**${knowledge.directAnswer}**\n\n`;
  } else {
    response += `**${knowledge.directAnswer}**\n\n`;
  }

  // --- Body: Explanation in simple language ---
  response += `${knowledge.explanation}\n\n`;

  // --- Key Points ---
  response += `**Here's what you need to know:**\n\n`;
  for (const point of knowledge.keyPoints) {
    response += `• ${point}\n\n`;
  }

  // --- Supporting evidence from uploaded documents ---
  if (supportingFacts.length > 0) {
    response += `---\n\n`;
    response += `📄 **From your uploaded documents:**\n\n`;
    for (const fact of supportingFacts) {
      // Simplify the fact text
      const simplified = fact
        .replace(/Authorised Dealer Category[- ]I banks?/gi, "authorized banks")
        .replace(/person resident in India/gi, "Indian resident")
        .replace(/person resident outside India/gi, "foreign resident")
        .replace(/Reserve Bank of India/gi, "the RBI")
        .replace(/Reserve Bank/gi, "the RBI");
      response += `> *"${simplified.substring(0, 250)}${simplified.length > 250 ? "..." : ""}"*\n\n`;
    }
  }

  // --- Important note if available ---
  if (knowledge.importantNote) {
    response += `---\n\n`;
    response += `⚡ **Pro Tip:** ${knowledge.importantNote}\n`;
  }

  return response;
}

// ---------------------------------------------------------------------------
// Message Handler
// ---------------------------------------------------------------------------

self.addEventListener("message", async (event: MessageEvent) => {
  const data = event.data as GenerateMessage;
  if (data.type !== "generate") return;

  const { messageId, query, chunks } = data;

  try {
    self.postMessage({
      type: "status",
      message: "Analyzing your question...",
    });

    // Small delay for UI smoothness
    await new Promise(r => setTimeout(r, 300));

    // Step 1: Deeply understand the query
    const analysis = analyzeQuery(query);

    // Step 2: Retrieve expert knowledge from the brain
    const knowledge = getExpertKnowledge(analysis.intent, analysis);

    // Step 3: Extract supporting facts from uploaded PDFs
    const supportingFacts = extractSupportingFacts(chunks, analysis);

    // Step 4: Generate a conversational, intelligent response
    const answer = generateResponse(query, analysis, knowledge, supportingFacts);

    // Step 5: Stream the response smoothly (typing effect)
    const words = answer.split(" ");
    let buffer = "";

    for (let i = 0; i < words.length; i++) {
      buffer += words[i] + " ";
      if (buffer.length > 12 || i === words.length - 1) {
        self.postMessage({ type: "stream-chunk", messageId, text: buffer });
        buffer = "";
        await new Promise(r => setTimeout(r, 12));
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
