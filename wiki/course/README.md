# wiki/course/

Pilot domain for testing the llm-wiki pattern as a model for course delivery (see [[index]] for how this differs from the ATA business wiki elsewhere in this space).

Each page here is a **topic/concept page** for one week of DS-3021: it synthesizes the lecture slides, R→Python conversion notes, and lab notebooks for that topic into one readable page, and links back to the underlying raw files rather than restating their code verbatim. Students should be able to get the concept and the "why" from these pages, then open the linked notebook when they need exact, runnable code.

## Pilot scope

Three topics only, chosen for variety (technical/procedural, mathematical, discussion-based):

- [[course/knn]] — K-Nearest Neighbors classification
- [[course/decision-trees]] — CART decision trees
- [[course/ml-bias]] — ML bias and fairness

See [[course/pilot-plan]] for how this pilot is meant to be evaluated against a plain-RAG baseline.

## Ongoing sources

Beyond the initial bulk ingest of lecture slides/notebooks, two intake folders now feed these pages incrementally as the semester runs:

- `raw/course/DS-3021/classnotes/` — instructor or student lecture notes
- `raw/course/DS-3021/discussion-board/` — discussion board threads

See `WIKI.md`'s "Ongoing sources" section for the ingest steps. Short version: file the note in the right intake folder (anonymized if it's discussion board content — see that folder's README), figure out which topic page it touches, fold it in as a small addition rather than a rewrite, and link the raw file from that page's Sources list. This incremental loop is also the part of the pilot that tests whether the wiki actually compounds over time, versus the one-off bulk pass used to build the first three pages.

## Curation notes

Source files for this domain live in `raw/course/DS-3021/<topic>/`, copied from the full `DS-3021/` folder with the following excluded on purpose:
- Editor/OS junk: `.ipynb_checkpoints/`, Office lock files (`~$*`).
- Duplicate legacy slide decks — where multiple years' versions of the same lecture existed (e.g. three Decision Trees decks from 2021/2022/2024), only the most recently taught version was kept as canonical; older ones are noted on the relevant page but not ingested, to avoid the wiki treating superseded content as current.
- `xx_reinforcement-lab/assignment-submissions/` — actual student-identifiable work. Excluded from any student-facing wiki on principle; not part of this pilot regardless of topic.
