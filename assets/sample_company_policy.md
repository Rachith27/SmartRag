# ACME Corp — Enterprise Artificial Intelligence & Data Privacy Policy
**Document ID:** ACME-SEC-2026-V4  
**Effective Date:** January 1, 2026  
**Applicability:** All Full-time Employees, Contractors, and Third-Party Vendors  

---

## 1. Executive Summary & Core Principles
At ACME Corp, we encourage the responsible adoption of Artificial Intelligence (AI) to enhance productivity, software engineering, and scientific research. However, safeguarding proprietary intellectual property (IP), customer Personally Identifiable Information (PII), and regulatory compliance is paramount.

### Core Mandates:
1. **Zero Public Training Data:** No proprietary ACME source code, internal financial projections, or unreleased product schematics may be entered into public AI models (e.g., consumer versions of ChatGPT, Claude, or Gemini) where data is retained for model retraining.
2. **Approved Enterprise Tools:** Employees must strictly use the **ACME AI Gateway** (`ai-gateway.internal.acme.com`), which is governed by enterprise zero-data-retention agreements with LLM providers.
3. **Mandatory Human-in-the-Loop Review:** AI-generated source code, contracts, or architectural blueprints must undergo peer code review by at least one Senior Engineer prior to deployment in production environments.

---

## 2. Data Classification & AI Usage Tiers

| Data Tier | Description | AI Usage Permission | Required Guardrails |
| :--- | :--- | :--- | :--- |
| **Tier 1: Public** | Open-source code, press releases, public documentation | Permitted on Public & Enterprise AI | Standard professional judgment |
| **Tier 2: Internal** | Internal wikis, architecture notes, non-sensitive Slack discussions | **Enterprise AI Gateway Only** | Must log request ID in project tracker |
| **Tier 3: Confidential** | Unreleased source code, M&A strategy, customer PII, financial ledgers | **Prohibited on ALL External Models** | Must use air-gapped local models (`Llama-3.3-70B-Local`) |

---

## 3. Incident Reporting & Penalties
Any accidental transmission of **Tier 3 Confidential Data** to an external or unapproved AI service must be reported to the Security Operations Center (SOC) at `security-incident@acme.com` within **2 hours** of discovery.

Failure to report data breaches or willful violation of the zero-data-retention mandate will result in immediate suspension of network privileges and formal disciplinary review by the Ethics & Compliance Committee.

---

## 4. Hardware & Local AI Allocation
Senior AI Engineers and Researchers working on Tier 3 datasets are eligible for specialized on-premise hardware workstations equipped with **NVIDIA H200 Tensor Core GPUs**. Requests must be submitted via Jira ticket (`#HW-AI-REQ`) and approved by the VP of Infrastructure.
