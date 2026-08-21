# Wiki arm results — 2026-08-10

Run of the wiki condition (Arm B) over the DS-3021 course question set.
**Headline: the wiki arm saturates the instrument. This is a measurement problem, not a result.**

## Configuration

| | |
|---|---|
| Answerer | `claude-opus-5`, effort `medium`, full wiki in a cached system block |
| Judge | `claude-haiku-4-5` |
| Corpus | `wiki/` — 8 pages, 32,435 chars |
| Questions | 59 (filtered from 145 — see below) |
| Errors | 0 |

**Judge deviation.** The script's design is cross-provider — Claude answers, GPT grades, so no
model grades its own homework. `api.openai.com` is unreachable from the sandbox this ran in, so
grading used a different Anthropic model instead. Same provider, different model: a weaker
independence guarantee. The OpenAI path is still the script default; re-run locally to restore it.

## Question set filtering

The original 145-question set was generated over all of `wiki/`, so 86 of the questions were
about the wiki rather than about the course — `index.md`, `status.md`, `getting-started.md`,
`course/README.md`, `course/pilot-plan.md`, plus 11 cross-page questions on pilot design and ATA
workstreams. Those test wiki-governance knowledge, not classroom QA. Filter logic and the
per-question rationale are in `filter_course_questions.py`.

Kept: 45 topic-page + 14 cross-page = **59**.
By type: 18 factual, 18 conceptual, 18 synthesis, 5 currency.

## Results

**n=59, mean 9.37/10, median 9, sd 0.69, range 6–10.**
**57 of 59 (97%) scored 9 or 10. Grounded rate 100% — zero ungrounded answers.**

Score distribution:

| Score | 1–5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| n | 0 | 1 | 0 | 1 | 31 | 26 |

By question type:

| Type | n | Mean | sd |
|---|---|---|---|
| factual | 18 | 9.78 | 0.42 |
| currency | 5 | 9.40 | 0.49 |
| conceptual | 18 | 9.28 | 0.99 |
| synthesis | 18 | 9.06 | 0.23 |

By source page:

| Page | n | Mean | sd |
|---|---|---|---|
| `course/ml-bias.md` | 15 | 9.53 | 0.50 |
| `course/knn.md` | 15 | 9.47 | 0.62 |
| `course/decision-trees.md` | 15 | 9.33 | 1.01 |
| (cross-page) | 14 | 9.14 | 0.35 |

## Bootstrap error distributions

20,000 resamples of the question set, BCa intervals. **Resampling estimates question-sampling
error only** — "if a different set of questions had been drawn from the same generator, how much
would the mean move?" It does *not* estimate judge stochasticity (same answer re-graded) or
answerer stochasticity (same question re-answered); both require repeated runs, not resampling.
Worth separating explicitly in the paper, since the prior work's "10 bootstrapped rounds"
conflates the two.

Overall: **mean 9.37, 95% BCa [9.15, 9.53], SE 0.090.** P(score ≥ 9) = 0.966 [0.881, 1.000].

| Type | n | Mean | 95% BCa | SE |
|---|---|---|---|---|
| factual | 18 | 9.78 | [9.56, 9.94] | 0.097 |
| currency | 5 | 9.40 | [9.00, 9.80] | 0.220 |
| conceptual | 18 | 9.28 | [8.61, 9.61] | 0.232 |
| synthesis | 18 | 9.06 | [9.00, 9.28] | 0.054 |

Pairwise differences (bootstrap of the contrast directly, rather than eyeballing CI overlap):

| Contrast | Diff | 95% CI | P(diff>0) | |
|---|---|---|---|---|
| factual − synthesis | 0.72 | [0.50, 0.94] | 1.000 | **excludes 0** |
| factual − conceptual | 0.50 | [0.06, 1.06] | 0.981 | **excludes 0** |
| factual − currency | 0.38 | [−0.13, 0.83] | 0.935 | |
| currency − synthesis | 0.34 | [−0.06, 0.80] | 0.918 | |
| conceptual − synthesis | 0.22 | [−0.28, 0.61] | 0.800 | |
| currency − conceptual | 0.12 | [−0.47, 0.77] | 0.639 | |
| topic-page − cross-page | 0.30 | [−0.00, 0.57] | 0.975 | |

Detectability floor for Arm A — power to detect a downward shift of *d* points, via simulation
(600 sims per cell; 95% bootstrap CI on the wiki-minus-RAG difference excluding zero):

| cell | n | d=0.25 | 0.5 | 0.75 | 1.0 | 1.5 |
|---|---|---|---|---|---|---|
| overall | 59 | 0.54 | **0.96** | 1.00 | 1.00 | 1.00 |
| synthesis | 18 | **0.85** | 1.00 | 1.00 | 1.00 | 1.00 |
| factual | 18 | 0.49 | **0.94** | 1.00 | 1.00 | 1.00 |
| conceptual | 18 | 0.17 | 0.40 | 0.65 | **0.81** | 0.97 |
| currency | 5 | 0.29 | 0.40 | 0.61 | **0.90** | 1.00 |

## What this means for the paper

**1. Two corrections to the first read of these results.**

*The type ordering is not noise.* The earlier write-up called the factual-over-synthesis gap
"well inside noise." The bootstrap says otherwise: **factual − synthesis = 0.72, CI [0.50, 0.94],
excluding zero**, and factual − conceptual = 0.50, CI [0.06, 1.06], also excluding zero. The
ordering survives question resampling. Note this is the direction **H1** predicts and the
opposite of **H2** — on this corpus, with this instrument, the wiki is *best* at fact lookup and
*worst* at synthesis. (Two of six contrasts, uncorrected for multiplicity; factual − synthesis
survives a Bonferroni threshold, factual − conceptual does not.)

*The ceiling does not preclude detecting a difference.* Also too pessimistic. Because saturation
drives variance down, paired comparisons are well powered: a 0.5-point drop in Arm A is detectable
overall at **power 0.96**, and the synthesis cell reaches 0.85 power at just 0.25 points. The
limitation is one-directional — the instrument can detect RAG being *worse*, but cannot detect
the wiki being *better* than it already scores, because nothing exceeds 10.

**2. So H2 is testable, but only as a one-sided claim.** "Wiki > RAG on synthesis" can be
supported if RAG scores meaningfully lower. What cannot be distinguished is "both arms are
excellent" from "the instrument saturates for both" — if Arm A also lands at 9+, the tie is
uninformative rather than a finding. Pre-committing to that interpretation before running Arm A
is worth doing.

**3. A 100% grounded rate is a broken signal, not a clean sweep.** `grounded` never fired false
across 59 answers, so it carries no information here and should not be reported as a result.

**4. The whole-corpus confound still stands.** The answerer received the entire 32k-char wiki in
a cached system block, so this arm is not testing *retrieval* — it is testing reading
comprehension over a corpus small enough to fit in the prompt. Arm A will be doing something
structurally different, and "wiki vs RAG" would partly be measuring "full context vs partial
context." This is the largest threat to the comparison and is independent of any judge issue.

**5. The earlier `score=5` results were a judge bug, now corroborated.** The previous smoke test
scored 10/10 questions at exactly 5 with rationales that described near-perfect answers. Those
same decision-trees questions score 9–10 here. Independently, the original score-first `Literal`
schema was tested against four answers of graded quality (excellent / partial / unsupported /
fabricated) and discriminated correctly, spread of 6, fabrication scored 1 — so the schema shape
is not the cause. The constant-5 is specific to `gpt-5-mini`'s structured output and still needs
reproducing locally. `diag_judge.py` does exactly that.

## Suggested next steps

- Run `diag_judge.py` locally against OpenAI to pin the `gpt-5-mini` constant-score bug, then
  re-grade this same answer set cross-provider — the answers are saved, so no re-answering needed.
- Harden the question set before Arm A. Ceiling effects usually mean the questions are too easy
  relative to the context provided. Options: questions the wiki genuinely cannot answer (to test
  refusal), questions requiring a specific number or code call rather than an explanation, or
  adversarial/contradiction items.
- Decide whether the wiki arm should retrieve rather than receive the whole corpus. Whole-corpus-
  in-context is a fair "wiki" condition only if Arm A also gets a comparable budget.
- Consider a finer-grained or paired instrument — per-claim citation accuracy, or blinded
  A/B preference between arms — since a 10-point holistic score is clearly too coarse here.

## Files

- `course_questions.json` — the filtered 59
- `course_eval_results.csv` / `.jsonl` — full answers, scores, rationales
- `filter_course_questions.py` — filter with per-question keep/drop rationale
- `wiki_qa_eval.py` — now supports `--judge-provider {openai,anthropic}`, defaults to openai
- `diag_judge.py` — OpenAI judge discrimination test (run this in VSCode)
- `bootstrap_analysis.py` — BCa intervals, pairwise contrasts, power simulation
- `bootstrap_figure.html` — interactive figure (ceiling, category means, contrasts)
