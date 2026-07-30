# Pilot Plan: Wiki vs. Plain RAG for Course Delivery

The project goal (see the space's top-level instructions) is to test the wiki framework as a model for course delivery *compared to other LLM formats* — not just to check whether a course wiki can be built. This page is the evaluation design so the pilot actually produces that comparison instead of just three nice-looking pages.

## What's built so far

Three topic pages ([[course/knn]], [[course/decision-trees]], [[course/ml-bias]]) synthesizing DS-3021's week 6, 8, and 14 materials. Chosen for variety: KNN is procedural/code-heavy, Decision Trees is mathematical, ML Bias is discussion/case-study-driven.

## The comparison

**Condition A — Wiki.** Answer a question by reading the relevant `wiki/course/*.md` page(s) and following their links, the way [[getting-started]] describes for the rest of this space.

**Condition B — Plain RAG baseline.** Answer the same question with only the raw source files available (`raw/course/DS-3021/<topic>/`) and no synthesized page — i.e. retrieve-and-read the actual slides/notebooks/conversion docs fresh, the way a generic "upload your files and ask" tool would.

## Question set to run against both conditions

Deliberately mixed, because the hypothesis is that the wiki wins on some question types and loses on others:

1. **Conceptual/connective** (wiki should win): "Why does KNN require feature scaling but Decision Trees don't?" "How do the fairness metrics in week 14 relate to the confusion matrix concepts from week 6?"
2. **Exact/code-level** (RAG may win, or tie): "What's the exact scikit-learn call used to compute variable importance for a decision tree in this course?" "What k value did the in-class KNN example land on, and what specific code produced that answer?"
3. **Cross-topic synthesis** (wiki should win clearly — this is the case the framework is built for): "Which weeks' materials would I need to review to understand why Amazon's and Google's bias examples happened, using concepts from earlier weeks?"
4. **Currency/version questions** (tests whether curation actually mattered): "Which Decision Trees slide deck is the current one, and how do I know?" — the wiki page states this explicitly; a naive RAG pass over all three duplicate decks in the original folder would likely blend or contradict itself across versions.

## What to record per question

For each question, note: which condition answered correctly, which condition cited its source precisely enough to verify, and — for the conceptual questions — which condition actually drew the cross-topic connection versus just paraphrasing one document.

## Known confound to watch for

The RAG condition as run here reads from the *curated* `raw/course/` copies (junk and duplicate decks already stripped), not the original messy `DS-3021/` folder. That's a deliberate simplification to isolate "does synthesis help" from "does cleanup help" — but it means this pilot is not yet testing the realistic worst case, where a plain RAG tool is pointed at the full original folder with all its duplicate/legacy files and lock files still in it. Worth running that messier version too if the clean-RAG condition doesn't show a clear enough gap.

## Incremental ingest (class notes + discussion board)

The three topic pages so far came from a single bulk ingest — that only tests whether synthesis beats raw retrieval, not whether the wiki actually *compounds*. `raw/course/DS-3021/classnotes/` and `raw/course/DS-3021/discussion-board/` (see `WIKI.md`'s ingest steps) exist to test the compounding claim directly: as real notes and threads come in through the semester and get folded into the topic pages, later questions should get *better* answers than the first bulk-ingested version did, without needing to re-derive anything already captured. Worth comparing "wiki after N incremental ingests" against "wiki right after initial bulk ingest" on the same questions to see if that's actually true here, not just assumed.

## Next step

Not yet run — this page defines the test, it doesn't report results. Once a set of questions is answered under both conditions, log outcomes here or in a new `wiki/course/pilot-results.md` page rather than losing them to chat history, per the wiki framework's own "answers should get filed back in" principle.
