# Anchorium Omni-Engine: Scaling, Accuracy, and Feature Roadmap

This document outlines the strategic roadmap for the Anchorium Omni-Engine. It addresses three core pillars: drastically improving the accuracy and reliability of the AI output, scaling the architecture for B2B enterprise delivery, and introducing unique features to differentiate the product.

---

## 1. AI Accuracy & Grounding Improvements

Since the core value proposition is delivering highly accurate, regulatory-compliant answers, we must move beyond standard RAG patterns to a highly defensive, verifiable AI architecture.

### 1.1 Self-Correction and Verification Agent
- **The Problem:** Single-pass LLMs are prone to hallucination or ignoring specific constraints.
- **The Solution:** Introduce a new **"Fact-Checker / Verification Agent"** into the pipeline.
- **Implementation:** After the Orchestrator assembles the report, it passes the draft and the *original source chunks* to the Verification Agent. This agent's sole prompt is to attempt to falsify every claim made in the draft against the provided context. If a claim cannot be perfectly mapped to a source, it forcibly downgrades the confidence tier to `LOW` and adds an inline warning.

### 1.2 Advanced RAG: Semantic Chunking & Reranking
- **The Problem:** Standard fixed-size chunking splits critical context across boundaries, confusing the LLM.
- **The Solution:**
  - Implement **Semantic Chunking**: Group text by structural elements (e.g., keeping an entire FEMA sub-section or GAAP definition intact) rather than arbitrary token limits.
  - Integrate a **Cross-Encoder Reranker** (e.g., `bge-reranker-v2-m3`): After initial vector retrieval, pass the top 20 results through a reranker to re-order them based on exact semantic relevance to the query, providing only the top 5 highly relevant chunks to the LLM.

### 1.3 Strict "Quote-First" Prompt Engineering
- **The Problem:** LLMs naturally want to summarize, which dilutes regulatory exactness.
- **The Solution:** Update the Core Directives to mandate **Quote-First Generation**. The LLM must output the exact, verbatim text from the regulatory source in a blockquote *before* it is allowed to summarize or interpret it.

### 1.4 Dynamic Knowledge Graph (Entity Resolution)
- **The Problem:** RAG struggles with multi-hop reasoning (e.g., "How does Rule A from Document 1 interact with Policy B from Document 2?").
- **The Solution:** Build a lightweight Knowledge Graph connecting regulatory concepts (e.g., "Foreign Direct Investment" -> "Requires CA Signoff" -> "FEMA Section X"). The engine queries both the vector DB and the Knowledge Graph, feeding the LLM explicit relationship data.

---

## 2. B2B Enterprise Scaling Strategy

To transition this into a SaaS/On-Prem product sold to specific organizations (Chartered Accountant firms, Banks, Consultancies), the architecture must support multi-tenancy and enterprise security.

### 2.1 Multi-Tenant "Walled Garden" Architecture
- **Tenant Isolation:** Every enterprise client gets a logical Tenant ID.
- **Custom Knowledge Bases:** While all clients share the baseline "Indian Regulatory DB", each enterprise can upload their own proprietary documents (internal firm policies, past case studies, specific lending criteria). The vector DB must enforce row-level security based on `tenant_id`.

### 2.2 Role-Based Access Control (RBAC) & Approvals
- **The Problem:** The AI currently outputs "DRAFT" but lacks a human workflow.
- **The Solution:** Introduce user roles: `Analyst`, `Reviewer`, `Partner`.
  - Analysts can generate drafts.
  - The UI provides an interface for Reviewers/Partners to interactively edit, approve, or reject specific sections of the AI draft.
  - Once approved, the system mathematically locks the final report and removes the "DRAFT" banner.

### 2.3 Deployment Models: Cloud vs. On-Premise
- **SaaS Cloud:** Secure AWS/Azure deployment with VPC isolation per client. Use enterprise-grade APIs (e.g., Azure OpenAI or AWS Bedrock) for faster, scalable inference compared to local Ollama, while maintaining zero-retention compliance via BAA agreements.
- **On-Premise / Edge:** For ultra-strict banks, package the entire system (Next.js, FastAPI, local Ollama/vLLM) into a unified Docker/Kubernetes appliance that can be deployed entirely behind their firewall with no internet access required.

### 2.4 API-First Extensibility
- **The Solution:** Expose the Orchestrator as a REST/GraphQL API. This allows enterprise clients to plug the Omni-Engine directly into their existing CRM (Salesforce) or ERP (SAP) systems to automatically trigger reports when a new founder record is created.

---

## 3. Unique New Features & Capabilities

To differentiate the Omni-Engine from standard text-based chat applications, we must introduce multi-modal capabilities and data visualization.

### 3.1 Dynamic Financial Visualizations
- **The Feature:** Instead of returning text tables for the "GAAP Translation" or "Arbitrage Calculator", the AI generates structured JSON that the frontend renders as interactive charts.
- **Implementation:**
  - Use libraries like `Recharts` or `Chart.js` in the Next.js frontend.
  - The LLM outputs: `{"chart_type": "bar", "data": [{"label": "Borrow", "value": 100}, {"label": "Sell", "value": 150}]}`
  - Visualizing the margin call risk (downside scenario) on a line graph showing asset value vs. loan-to-value (LTV) trigger points.

### 3.2 Visual Document Processing (Vision LLM)
- **The Feature:** Financial documents often contain complex tables, scanned signatures, and diagrams that break standard text extractors.
- **Implementation:** Integrate a Vision model (e.g., Llama-3-Vision, GPT-4o, or Claude 3.5 Sonnet) specifically for the **KYC Extraction Agent**. Users can upload screenshots or scanned PDFs, and the Vision model will accurately parse tabular data into JSON.

### 3.3 Live Data Integrations
- **The Feature:** The Arbitrage Calculator currently relies on static/assumed interest rates and FX rates.
- **Implementation:** Integrate live APIs (e.g., Bloomberg, Reuters, or open FX APIs) to fetch real-time SOFR rates, EURIBOR, and INR/USD exchange rates at the exact moment the report is generated, citing the live timestamp.

### 3.4 Interactive Scenario Modeling ("What-If" Sliders)
- **The Feature:** Once the AI generates the initial arbitrage calculation, the user is presented with UI sliders.
- **Implementation:** The user can drag a slider to change the "Collateral Value Drop" from 10% to 30%. The frontend immediately recalculates the risk without needing to call the LLM again, providing an interactive financial modeling experience.

### 3.5 Automated Pre-Filled Forms
- **The Feature:** Going beyond just a report, the engine outputs actionable paperwork.
- **Implementation:** Using the data extracted and the compliance path chosen, the system automatically populates the exact PDF forms required (e.g., RBI Form FC-GPR, KYC Annexures) ready for the founder to sign.