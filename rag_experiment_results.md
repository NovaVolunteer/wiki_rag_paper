# RAG Pipeline Experiment — Results

Two runs of Arm A (vector RAG) against Arm B (LLM-compiled wiki,
full-context), same 59-question count both times, same answerer/judge
setup. The corpus and question scope grew between them — see below.

## Setup (both runs)

- **Answerer:** `claude-opus-5`, effort=medium (both arms).
- **Judge:** `gpt-5-mini` via OpenAI, graded against the full wiki in both
  arms (same instrument — only the answerer's context differs between arms).
- **Arm A config:** chunks of 1,500 tokens / 100-token overlap, embedded
  with `sentence-transformers/all-mpnet-base-v2` (768-d, cosine), indexed
  in Pinecone (serverless, aws/us-east-1, index `llm-wiki-rag`). Top-k = 5
  chunks retrieved per question. Replicates the prior study's best
  configuration (paper §3.3, Arm A).
- **Arm B config:** full wiki content cached in the system prompt
  (`wiki_qa_eval.py`).

## Run 2 (current) — 2026-09-01, 7-topic corpus

**Corpus:** `wiki/` — 12 Markdown pages, 7 of them DS-3021 course topics
(knn, decision-trees, ml-bias, ml-eval-metrics, dt-regression,
kmeans-clustering, ensemble-methods — the last 4 added 2026-08-31).
Chunked into **21 vectors** (up from 9 in run 1) — top-k=5 now covers
~24% of the index per query instead of >50%, so retrieval can actually
miss things.

**Questions:** `course_questions.json` regenerated to cover all 7 topics
instead of 3, holding the total at 59 (not expanded, per instruction): 45
topic-page questions (6–7 per topic, weighted slightly toward the
most-cross-referenced pages: decision-trees, ml-eval-metrics,
ensemble-methods) + 14 cross-page synthesis questions spanning the full
7-topic set. IDs were reassigned 0–58; the old question set is not
preserved separately, but run 1's results below stand on their own.

### Headline results

| | Arm A — Vector RAG | Arm B — Full wiki |
|---|---|---|
| Avg score (1–10) | **9.05** | **9.95** |
| Grounded rate | **81%** | **98%** |
| Graded | 59/59 | 59/59 |
| Errors | 0 | 0 |

Source-page retrieval hit rate (topic-page questions): **96%** (43/45).

### By question class — the H2 pattern shows up

| | Arm A avg | Arm A grounded | Arm B avg | Arm B grounded |
|---|---|---|---|---|
| topic-page (n=45) | 9.33 | 87% | 9.96 | 98% |
| cross-page (n=14) | **8.14** | **64%** | 9.93 | 100% |

This is the gap run 1 was too small to surface: cross-page synthesis
questions take a much larger RAG penalty than single-topic questions —
both in score (−1.79 vs −0.63 points relative to Arm B) and especially in
groundedness (64% vs 87%). Full-context answering barely notices the
difference between question types; retrieval clearly does. Directionally
consistent with the paper's H2 (wiki favored on multi-hop/synthesis).

**Retrieval misses (2 of 45 topic-page questions):**
- id 18 (ml-bias.md) — retrieved index/getting-started/dt-regression/
  pilot-plan/README instead; scored 3, not grounded.
- id 36 (kmeans-clustering.md) — retrieved pilot-plan/status/
  getting-started/README/index instead; scored 4, not grounded.

Both misses pulled in the same handful of short meta/navigation pages
(index.md, getting-started.md, status.md, course/README.md,
course/pilot-plan.md) instead of the actual target topic page — those
pages are short and generic, so their chunks may be scoring deceptively
close to on-topic chunks; worth a look if this recurs at larger scale.

**Grounding gap even on correct retrieval:** 9 of the 11 ungrounded Arm A
answers *did* retrieve the right page (or, for cross-page questions, drew
on relevant retrieved chunks) — the RAG answerer still added unsupported
detail that full-context answering didn't. This replicates run 1's
finding: retrieval failure isn't the whole story, chunked context itself
seems to invite more fabrication than the full wiki does.

**Lowest-scoring questions (Arm A), all cross-page except one:**
1. `[3]` Interventions described in the ml-bias companion Value_1.pptx deck
   — topic-page, not a retrieval miss, just thin/wrong detail.
2. `[3]` R→Python conversion pattern across 5 pages — cross-page.
3. `[3]` Version/currency warnings across all pages — cross-page (this
   question type structurally needs the *whole* corpus, worst case for
   chunked retrieval).
4. `[4]` Confusion-matrix R→Python calls — topic-page (ml-eval-metrics).
5. `[4]` Kmeans canonical slide-deck version — topic-page, retrieval miss
   (id 36 above).

## Run 1 (superseded) — 2026-08-31, 3-topic pilot corpus

8-page wiki (3 course topics only), 9 chunks total, top-k=5 retrieved
over half the index every query — under-tested retrieval failure by
construction. Kept here for the record, not as the current baseline.

| | Arm A — Vector RAG | Arm B — Full wiki |
|---|---|---|
| Avg score | 9.59 | 9.37 |
| Grounded rate | 86% | 100% |

By class: topic-page 9.56/87% (A) vs 9.44/100% (B); cross-page 9.71/86%
(A) vs 9.14/100% (B) — RAG did *not* show a cross-page weakness at this
scale, which run 2 confirms was an artifact of too-small an index rather
than evidence against H2.

## Artifacts

- `rag_course_eval_results.jsonl` / `.csv` — Arm A, run 2 (current).
- `course_eval_results.jsonl` / `.csv` — Arm B, run 2 (current).
- `rag_rerun.log` / `wiki_rerun.log` — stdout from the run 2 eval calls.
- `rag_index_rebuild.log` — the 21-vector Pinecone index rebuild.
- `course_questions.json` — the current 59-question, 7-topic set.
- Run 1's `.jsonl`/`.csv` files were overwritten by run 2 (same output
  basenames); run 1's numbers above are preserved from the earlier writeup
  and are not independently re-derivable from files in this repo anymore.

## Caveat for the paper

Both runs are still well below DS3001 scale. Run 2's 21-chunk index is a
meaningful step up from run 1's 9 and is now large enough to show the
directional H2 pattern (cross-page/synthesis favors the wiki), but the
absolute numbers (96% retrieval hit rate, top-k=5 over 21 chunks ≈ 24% of
the index) are still generous compared to what a full-course-scale corpus
would present. Treat this as continued pipeline validation plus a
stronger directional signal, not the paper's final result.
