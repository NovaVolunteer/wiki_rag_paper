# 2026-11-19: Expanding the study to the full course, rerunning, and updating the paper

A record of what was done on 2026-11-19, why, and what came out of it. The goal is that the experiment can be understood and reproduced later without the chat history.

**Scope of the request:**
- add all DS-3021 course materials, not just the 7 pilot topics, plus the instructor's online book (MLBook, https://novavolunteer.github.io/MLBook/);
- add them to the Pinecone index and the wiki;
- rerun the experiment;
- update the paper.

---

## 1. Decisions made (and why)

| Decision | Choice | Reason |
|---|---|---|
| Legacy `DS-3021/xx_*` folders (reinforcement learning, text mining, Git primer) | **Excluded** | Not part of the course as taught. |
| Question set | **Keep a random half of the 7-topic set, generate 50 new questions on the 9 new topics.** The instructor reviews them before any run. | Keeps continuity with the earlier question set while covering the new material. |
| Judge reference material | **Wiki + gold raw files + raw files the answer cites (+ A-ret's retrieved excerpts)**, not the whole raw corpus | The whole wiki plus the whole raw corpus (~306k tokens) does not fit reliably in `gpt-5-mini`'s ~272k-token input limit. |
| Retrieval depth / navigation budget | **Primary: k=10 chunks, 15,000-token B-nav budget. Sensitivity run: k=5 / 7,500.** | See §4. Fixed before any full-course results existed. |
| Page splitting | **Not done** | Splitting pages to fit the budget would tune the representation to the test. Page length reflects how much source material a topic has. |
| Question edits after generation | **By hand, no further OpenAI calls** (the user's instruction) | Fixes were small and verifiable directly from the raw files. |
| Old pilot (59 questions, RAG over wiki chunks) and the abandoned 7-topic run | **Not mentioned anywhere in the paper** (the user's instruction) | The five-condition, full-course design is far more sound. The old results are archived, not reported. |

---

## 2. Corpus expansion

**Script:** `ingest_course_materials.py` (new). It copies files into `raw/course/<source>/<topic>/<date>-<desc><ext>` and never overwrites. The script holds the include list (`DS3021`, `MLBOOK`) and an explicit `SKIPPED` dict that records the reason for every exclusion.

- **MLBook** was ingested from a clone of `NovaVolunteer/MLBook` at commit `40b4199e…` (2026-05-21).
  - Run with: `python ingest_course_materials.py --mlbook <clone path>`.
- **51 files were added.** The corpus went from 30 to **81 documents** (56 DS-3021 + 25 MLBook), about **235k tokens** (embedding tokenizer), over **16 topics**:
  - the 7 original pilot topics: knn, decision-trees, ml-eval-metrics, dt-regression, kmeans-clustering, ensemble-methods, ml-bias;
  - 9 new topics: course-overview, dev-environment, python-basics, ml-foundations, data-prep, linear-regression, logistic-regression, pca, final-project.
- **Deduplication.** Normalized extracted text was hashed to find duplicates.
  - Exact duplicates across the sources were ingested once.
  - Where both sources held different versions of a notebook, the current MLBook version was kept.
  - These were dropped: superseded slide decks, PDF exports of included decks, `.py` exports of included notebooks, the MLBook title page, data files and images, and one legacy binary `.ppt`.
- **Fix:** MLBook `ch4.ipynb` was mislabeled. It was renamed to `2026-05-21-acquiring-data-from-apis.ipynb`, and the manifest and `data-prep.md` were updated to match.

**`raw_corpus.py` (modified):**
- `RAW_COURSE = raw/course`, `SOURCES = ["DS-3021", "MLBook"]`, `TOPICS = PILOT_TOPICS + NEW_TOPICS`.
- `RAW_CITE_RE` now matches both sources. This regex derives gold raw files from wiki citations.
- Each doc now has a `source` field. Path component 3 is still the topic, which `eval_conditions.py` relies on.
- `raw_corpus.json` was rebuilt: 81 docs.

**Pinecone:**
- New namespace **`raw-v2`** in index `llm-wiki-rag`: 203 chunks of 1,500 tokens with 100-token overlap, all-mpnet-base-v2 embeddings.
- `raw-v1` (the old 7-topic index) was left in place.

## 3. Wiki update

- **New concept pages** in `wiki/course/`: course-overview, dev-environment, python-basics, ml-foundations, data-prep, linear-regression, logistic-regression, pca, final-project.
- **Updated pages** (cross-links to the new material): knn, decision-trees, dt-regression, kmeans-clustering.
- **`wiki/index.md`:** the course section is regrouped into Getting started / Foundations / Models / Ethics and capstone.
- **`wiki/course/README.md`:** an expansion note and the MLBook paths.
- **Result:** 21 wiki pages (16 concept pages plus navigation and meta pages), about 71k tokens. The index is 765 tokens. Concept pages have a median of about 3,900 tokens (range 1,300–7,500).

## 4. Budget choice (k and the B-nav budget)

- **What k means.** k counts retrieved *chunks*, not topics or files. The 10 nearest chunks can come from any files or topics, and nothing guarantees that every topic a cross-page question needs is represented.
- **Why k=10.** The prior study used k=5 over 80 chunks, about 6% of the corpus per question. k=10 over 203 chunks is about 5%, which keeps the retrievable share roughly constant as the corpus grew.
- **Matched budgets.** B-nav's budget equals A-ret's retrieved-context budget (k × 1,500 tokens). Both are raised together, so the comparison is between representations, not between amounts of context.
- **What k=10 allows.** At k=5 (7,500 tokens), B-nav could read only the index plus one or two pages. At k=10 (15,000 tokens), it can read the index plus three median pages, or the index plus any two pages. The only exception is the two longest pages, which would lose about 90 tokens.
- **Sensitivity run.** k=5 / 7,500 tokens for A-ret and B-nav, to show whether the result depends on the budget.

## 5. Question set (100 questions: 45 topic-page / 45 cross-page / 10 unanswerable)

**Script:** `expand_questions.py` (new).

**Retained half: 50 questions.** A random sample from `results/archive_7topic_100q/course_questions_raw.json` with `SEED=20261119`, stratified by class: 23 topic-page, 22 cross-page, 5 unanswerable. The retained unanswerable controls were re-checked against the expanded corpus.

**New half: 50 questions.** Generated by `gpt-5.5` from the **raw files only** (the generator never sees the wiki):
- 22 topic-page: 3 each for linear-regression, data-prep, ml-foundations and python-basics, and 2 for each other new topic;
- 23 cross-page: 21 random topic pairs and 2 random triples, each including at least one new topic;
- 5 unanswerable controls, each verified against the full corpus.

**Problems during generation:**
- **OpenAI credit ran out (429).** The user added credit.
- **OpenAI rate limit (1M tokens/min).** Six parallel full-corpus checks of about 225k tokens each lost generated questions. Fixed with:
  - checkpointing to `results/expand_questions_checkpoint.json`;
  - at most 2 concurrent full-corpus checks;
  - `OpenAI(max_retries=10)`.
- **Too few unanswerable controls.** Only 2 of 10 candidates passed verification. A top-up script brought the set to 100.

**Hand edits after review.** No API calls were used. Each edited question has a `review_note`, and replaced questions keep the original in `replaced`. The edited reference answers and replaced questions carry `gold_annotation: "manual:claude-2026-11-19 (no API; written from the raw files)"`. The pre-edit backup is `results/course_questions_raw.pre-edit.json`.

| ID | Change |
|---|---|
| 53 | Reference answer fixed. The source says "clone it to your GitHub account", which is really a fork. |
| 71 | Reference answer fixed. The 60/40 split is within the project grade; the syllabus weights the project at 25% of the course. **Flagged for the instructor to confirm.** |
| 75, 78, 87, 90, 92 | Replaced. Each was two unrelated questions joined by "and" (mostly dev-environment pairs). New pairs: linear-regression + dt-regression; python-basics + data-prep; knn + pca; logistic-regression + ml-eval-metrics; pca + kmeans. |
| 93 | Replaced the ml-bias + dev-environment + final-project triple with course-overview + ml-foundations + final-project. |
| 95 | Unanswerable status confirmed by search: the corpus has no Minkowski distance or distance-weighted voting. |
| 97 | Kept but **borderline**. The course uses `passthrough` inside a ColumnTransformer but never covers `remainder=`. |
| 98 | Replaced with a t-SNE/UMAP vs PCA question. The original, about `get_feature_names_out`, is answerable, because the linear-regression notebooks use it. |

The instructor approved all 100 questions (`review_status: "approved"`) before the run.

## 6. Evaluation code changes (`eval_conditions.py`, `analyze_conditions.py`)

- `PINECONE_NAMESPACE = "raw-v2"`, `TOP_K = 10`, `NAV_BUDGET = TOP_K * CHUNK_TOKENS`.
- New CLI flags:
  - `--top-k` (also sets the B-nav budget);
  - `--results-dir`;
  - `--judge-context {gold,full}`, default `gold`.
- The judge's reference is built by `judge_raw_material()`: the cached full-wiki prefix, then the gold raw files, the raw files the answer cites, and, for A-ret, the retrieved excerpts.
- A-ret rows log `retrieved_excerpts`. Every row records `judge_context`, `top_k` and `nav_budget`.
- `analyze_conditions.py` gained `--results-dir`. Its planned contrasts:
  - B-nav − A-ret;
  - B-full − A-full;
  - B-full − B-nav;
  - each grounded condition − C0.
- **Bug fixed:** a Python `global` declaration came after the variable's first use in `main()`. It was moved to the top.
- **Safety step:** old partial results were moved into the archive first, so `--resume` could not silently reuse 7-topic rows.

**Fixed across all conditions:**
- answerer `claude-opus-5`, effort medium, up to 16,000 output tokens, 1M-token context window;
- judge `gpt-5-mini`, from a different provider than the answerer;
- the same question set and shared instructions.

## 7. Runs

```
python eval_conditions.py --condition all                                         # primary, k=10 / 15k
python eval_conditions.py --condition A-ret --top-k 5 --results-dir results/budget7500
python eval_conditions.py --condition B-nav --top-k 5 --results-dir results/budget7500
python eval_conditions.py --condition A-full --resume                             # reran 15 errored rows
python eval_conditions.py --condition B-nav  --resume                             # reran 51 errored rows
python analyze_conditions.py
python analyze_conditions.py --results-dir results/budget7500
```

- The log is `results/run_full.log`.
- **Errors:** 66 rows (15 A-full, 51 B-nav) failed with transient `APIConnectionError`, most likely a network interruption. They were rerun unchanged with `--resume`.
- **Final counts:** 500/500 primary rows and 200/200 sensitivity rows, with no errors.
- **Outputs:**
  - primary: `results/<cond>.jsonl/.csv`, `results/RESULTS.md`, `results/summary.json`;
  - sensitivity: the same files under `results/budget7500/`.

## 8. Outcomes

**Primary run, k=10 / 15k, all 100 questions:**

| Condition | Score (refusal = 1) | Grounded | Fabrication | All gold sources in context | Mean input tokens |
|---|---|---|---|---|---|
| C0 | 8.65 | 32% | 58% | — | 131 |
| A-ret | 9.71 | 97% | 0% | 76% | 18,847 |
| A-full | 9.89 | 100% | 0% | 90% | 338,671 |
| B-nav | 9.84 | 96% | 0% | 83% | 13,937 |
| B-full | 9.85 | 97% | 0% | 90% | 96,031 |

Gold coverage tops out at 90% because the 10 unanswerable controls have no gold sources.

**B-nav − A-ret (paired bootstrap, B=20,000, 95% CI):**

| | Primary, k=10 | Sensitivity, k=5 |
|---|---|---|
| Cross-page score | **+0.47 [0.24, 0.73]** | +0.33 [−0.09, 0.76] |
| Cross-page grounded | +4 points [0, 11] | **+20 points [4, 36]** (93% vs 73%) |
| Topic-page score | −0.20 [−0.53, 0.02] | −0.44 [−1.11, 0.09] |
| Topic-page grounded | −7 points [−16, 0] | −7 points [−20, 7] |
| Overall score | +0.13 [−0.07, 0.32] | +0.06 [−0.33, 0.43] |

**Other findings:**
- **Long context.** B-full − A-full is −0.04 [−0.26, 0.10]: with everything in context, the two representations are indistinguishable. A-full scores 9.89 on cross-page questions against A-ret's 9.49, so the cross-page gap is a retrieval effect.
- **Navigation loses nothing measurable.** B-full − B-nav is +0.01 [−0.09, 0.10]. B-nav reads about 3 of 21 pages, about 14% of the wiki's tokens.
- **Why the wiki wins cross-page.** Every gold source was in context for 87% of B-nav's cross-page questions, against 71% for A-ret. A-ret's 13 cross-page questions with incomplete coverage scored 9.23, against 9.59 for its fully covered ones.
- **Why RAG wins topic-page.** Gold coverage is the same (98% each). The difference comes from detail the wiki summary omits. Example: Q6 asks about split-selection bias and the GUIDE algorithm. The wiki never mentions GUIDE, so B-nav scored 4 and B-full refused, while A-ret and A-full both scored 10.
- **Query cost.** B-nav averages 2.35 calls per question and 20,186 billed input tokens, against A-ret's 1 call and 18,847. Latency is about 25 s vs 28 s. Ingest cost was not measured.
- **Budget use.** B-nav hit the 15k cap on 10 of 100 questions and the 7.5k cap on 69 of 100.
- **Hypotheses:**
  - H1 (A-ret at least as good on topic-page questions, at lower cost): supported on quality, weak on cost.
  - H2 (B-nav better on cross-page questions): supported.
  - H3 (trade-offs rather than one winner): supported.
- **Main caveat: ceiling effect.** Every grounded condition scores between 9.7 and 9.9.

## 9. Paper changes (`draft/draft_sections_1-3.tex`)

- **Methods §3.1 Corpus:** 16 topics, MLBook footnote with the commit, 81 docs (56 + 25), ~235k tokens, curation rules, and the 16-page, ~71k-token wiki.
- **Methods §3.2 Conditions:**
  - A-ret: 203 chunks, what k counts, why k=10.
  - B-nav: the 15,000-token budget, plus a **Choice of budget** paragraph.
  - A new **Token budgets and context windows** paragraph, added at the user's request. It covers:
    - the single tokenizer used for all budgets;
    - what the budget counts (the index is charged first);
    - how truncation and forced answering work;
    - why B-nav's multi-call billing is reported but not counted against the budget;
    - the answerer's tokenizer counting 1.35–1.45 times as many tokens;
    - the 1M-token answerer context window;
    - prompt caching;
    - the judge's smaller context window.
- **Methods §3.3 Questions:** the two-halves construction.
- **Methods §3.5 Scoring:** the judge uses gold context; the ~272k input limit; the cached wiki prefix. The refusal rule is stated on its own terms.
- **Methods §3.6:** primary k=10, secondary k=5.
- **The "What Has Been Run" subsection was removed.** It described the pilot.
- **Results §4:** fully rewritten.
  - Tables: the headline table across all five conditions, and A-ret vs B-nav by class at both budgets.
  - Subsections: matched-budget comparison, budget sensitivity, long context and navigation cost, hypotheses, scope.
- **Discussion §5:** rewritten.
  - RQ1/RQ2: selection, not content, drives the cross-page gain; compilation loses detail.
  - Grounding is mostly a matter of having material.
  - Limitations, and future work (a harder question set; a hybrid that navigates the wiki and falls back to the raw files a page cites).
- **Abstract and contributions (1) and (3):** rewritten to match.
- **All references to the old pilot were removed.**
- The draft compiles cleanly: no errors, no undefined references, no overfull boxes. The PDF is `draft/_tmp_build.pdf`.
  - Build with `TEXINPUTS=".;../AuthorKit27/AuthorKit27;"`, running pdflatex, then bibtex, then pdflatex twice.

## 10. Open items

- [ ] Q71: confirm the intended reading of the 60/40 project-grade split.
- [ ] Q97 (borderline unanswerable): keep or drop.
- [ ] Verify the `gpt-5-mini` ~272k input limit cited in §3.5. It came from model knowledge, not from a doc or the code.
- [ ] Commit this session's results and paper changes (not yet committed when this note was written).
- [ ] Not yet done:
  - repeated runs to separate answerer and judge variance from question sampling;
  - human validation of the judge;
  - the RAGAS triad;
  - the knowledge-shift phase;
  - measuring ingest cost;
  - a harder question set to get off the ceiling.

## 11. Where things are

| What | Where |
|---|---|
| Raw corpus | `raw/course/{DS-3021,MLBook}/<topic>/`, extracted to `raw_corpus.json` |
| Ingest manifest (includes and skips with reasons) | `ingest_course_materials.py` |
| Wiki | `wiki/index.md`, `wiki/course/*.md` |
| Questions | `course_questions_raw.json`; pre-edit backup `results/course_questions_raw.pre-edit.json` |
| Primary results | `results/*.jsonl`, `results/RESULTS.md`, `results/summary.json`, `results/run_full.log` |
| Sensitivity results | `results/budget7500/` |
| Archived 7-topic run (not reported) | `results/archive_7topic_100q/` |
| Run log table and commands | `results/README.md` |
| Paper | `draft/draft_sections_1-3.tex` |
