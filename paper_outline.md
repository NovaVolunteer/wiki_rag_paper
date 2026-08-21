# Outline: LLM-Compiled Wiki vs. Vector RAG for Classroom Question Answering

*Extension of: "Using Educational Data to Explore Multimodal (Audio, Visual, & Textual) LLM Retrieval Techniques" (Wright et al.)*

---

## 1. Introduction

**1.1 Motivation**
- LLMs are increasingly deployed as course-specific tutoring tools, but need grounding in vetted course materials to align with instructional content and level.
- Our prior work built a multimodal RAG system over DS3001 course data and showed retrieval improves contextual grounding — but also that *fixed retrieval strategies are suboptimal*: performance depended heavily on query specificity and context composition (5T+5I vs. 10T+10I trade-offs).
- Vector RAG retrieves isolated chunks at query time. An alternative: compile the corpus *at ingest time* into a structured, linked wiki (Karpathy-style LLM-compiled wiki) that the model browses at question time — trading chunk retrieval for curated synthesis, explicit concept links, and provenance.

**1.2 The gap**
- Head-to-head wiki-vs-RAG comparisons exist (Cochran, 2026 preregistered study) but not in an educational setting with real classroom data.
- Graph/structured retrieval literature suggests structure helps multi-hop and relational questions but can hurt simple fact lookup (Peng et al., 2024; Zhou et al., 2025; AboulEla et al., 2025) — untested on course corpora.

**1.3 Research questions**
- RQ1: Does an LLM-compiled wiki improve response quality over vector RAG on classroom QA, holding LLM, corpus, and questions constant?
- RQ2: How do gains vary by question type — fact lookup vs. multi-hop concept linking vs. synthesis/explanation? (Extends the generic/specific split from the prior paper.)
- RQ3: What are the trade-offs in citation support, robustness to curriculum updates, and ingest/query cost?

**1.4 Contributions**
- First wiki-vs-RAG comparison on an authentic multimodal course corpus (DS3001).
- A question taxonomy and evaluation protocol extending our prior 30-question generic/specific set.
- Evidence on when structure-at-ingest beats retrieval-at-query for educational deployment.

---

## 2. Background

**2.1 Retrieval-Augmented Generation**
- Original RAG framework (Lewis et al., 2020); RAG as a reliability layer — grounding, hallucination reduction, knowledge updating (Peng et al., 2024; Neha et al., 2025; Li et al., 2024).
- Strong domain results: clinical GPT-4 RAG at 96.4% accuracy with no hallucinations (Ke et al., 2025); MEGA-RAG in public health (Xu et al., 2025). But gains are contingent on retrieval relevance, task type, and evaluation design (Zhou et al., 2025; Brown et al., 2025).

**2.2 Structured and graph-based retrieval**
- Graph-enhanced RAG captures entity relations, supports multi-hop reasoning and explainability (Peng et al., 2024; Zhu et al., 2025).
- Mixed evidence: KG²RAG improves HotpotQA F1 (Zhu et al., 2025); KG-RAG improves multi-hop with single-hop trade-off (Linders & Tomczak, 2025); Graph RAG underperforms naïve RAG on SQuAD-V2/TriviaQA (AboulEla et al., 2025).
- LLM-compiled wikis as an alternative structuring approach: Cochran (2026) — vector RAG wins single-fact lookup, wiki wins cross-paper synthesis; no single architecture wins every dimension.

**2.3 RAG in education**
- Retrieval should align to vetted course materials and curriculum, not just general correctness (Li et al., 2025; Jain et al., 2025).
- Groundedness–helpfulness trade-off: humans preferred RAG in math tutoring, unless grounding was too rigid (Levonian et al., 2023).
- Educational KG + Agentic-RAG systems show high retrieval accuracy and learner satisfaction (Gao et al., 2025); RAG+ICL for question generation (Maity et al., 2024).

**2.4 Prior work: multimodal RAG on DS3001 (this paper's foundation)**
- Summary of our prior findings: multimodal retrieval improved context recall; selectively swapping text for top-ranked images helped specific questions (perfect recall at 5T+5I) but hurt faithfulness/factual correctness on generic ones; naive fixed strategies are suboptimal.
- Key takeaway motivating this study: *how context is structured and selected matters more than how much is retrieved* — which motivates comparing an ingest-time structured representation (wiki) against query-time chunk retrieval.

**2.5 Evaluating RAG systems**
- Limits of BLEU/ROUGE under LLM stochasticity; LLM-as-judge and embedding-based metrics; RAGAS framework (as used in prior paper); multi-endpoint evaluation to avoid winner-take-all conclusions (Lyu et al., 2024; Cochran, 2026).

---

## 3. Study Design

**3.1 Overview**
- Controlled head-to-head comparison: same corpus, same base LLM, same prompts, same question set; only the knowledge representation differs. Modeled on the preregistered Cochran (2026) design combined with the classroom evaluation approach of Jain et al. (2025).

**3.2 Corpus**
- DS3001/DS-3021 course materials (reused from prior paper): lecture slides (text + images), lecture audio transcripts, ML papers and textbooks.
- Same extraction pipeline as prior work for comparability.

**3.3 Conditions**
- **Arm A — Vector RAG (control):** replication of prior pipeline. Chunked text (1500 tokens, 100 overlap), all-mpnet-base-v2 embeddings, Pinecone; best-performing configuration from prior study as the baseline.
- **Arm B — LLM-compiled wiki:** corpus compiled at ingest into a linked wiki (raw/ + wiki/ structure): concept pages, cross-links, source citations back to raw materials. At query time the LLM navigates/reads relevant pages rather than receiving similarity-ranked chunks.
- Hybrid (wiki navigation + vector retrieval) deferred to future work; this study is a clean two-arm comparison.
- Held constant: generator LLM, system prompt, context token budget, question set, decoding parameters.

**3.4 Question set**
- Extend prior 30-question generic/specific set into a fuller taxonomy: (a) single-fact lookup, (b) multi-hop concept linking, (c) thematic explanation/synthesis, (d) contradiction/ambiguity handling, (e) curriculum-update questions.
- Human-generated (as before) with reference answers and word limits; balanced across course topics.

**3.5 Metrics**
- RAGAS core (continuity with prior paper): Context Recall, Faithfulness, Factual Correctness.
- New endpoints: claim–citation alignment, answer synthesis quality (blinded LLM-judge + human rubric), robustness under knowledge-shift (revised materials test, per Jain et al., 2025 / Hou et al., 2024), and cost (ingest compute, query tokens, latency).

**3.6 Procedure**
- Build both indexes from frozen corpus snapshot → run all questions through each arm → bootstrapped sampling for metric stability (10 rounds, as in prior paper) → blinded judging for preference/synthesis endpoints.
- Knowledge-shift phase: modify a subset of source materials, re-ingest both arms, re-ask affected questions, measure staleness.

**3.7 Hypotheses**
- H1: Vector RAG ≥ wiki on single-fact lookup (accuracy, cost).
- H2: Wiki > vector RAG on multi-hop and synthesis questions.
- H3: Wiki provides better claim–citation alignment and provenance.
- H4: Trade-offs, not a single winner, across endpoints (consistent with Cochran, 2026).
