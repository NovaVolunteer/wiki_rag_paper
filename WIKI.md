# Wiki Schema & Directives

This is the schema file for this project's LLM wiki — read this before doing any ingest, lint, or query work here. It's the equivalent of a CLAUDE.md/AGENTS.md for this folder: it tells an agent how this knowledge base is organized and how to keep it healthy.


## The core idea

Most setups either dump everything in chat (nothing persists) or do RAG over a pile of raw documents (search finds fragments, not synthesis). A wiki does neither: raw sources go in untouched, and an agent continuously distills, links, and updates a small set of living pages. Answers get better over time because the *wiki* accumulates understanding, not just documents.

## Two folders, two rules

**`raw/`** — immutable. Whatever comes in (a transcript, an RFI, a schema, a plan) gets saved here exactly as received, named `YYYY-MM-DD-short-description.ext`. Never edited, never deleted, never summarized in place. This is the paper trail — if the wiki ever gets something wrong, `raw/` is how you check.

**`wiki/`** — living. Small, single-topic pages that get created and updated as new information comes in. This is what actually gets read to answer questions — nobody should need to dig through `raw/` to know what's going on.

## Atomicity

Each wiki page covers one thing: one decision, one meeting, one person, one piece of material. Don't let a page grow into a catch-all. If a page is covering two topics, split it. Small pages are easier to update correctly and easier to link precisely — a page that tries to be comprehensive becomes a page nobody trusts enough to update.

Suggested page types:
- `wiki/decisions/<date>-<slug>.md` — one decision, its rationale, what it superseded if anything
- `wiki/meetings/<date>-<slug>.md` — a distilled summary of a meeting (not the transcript — that's in `raw/`), with links to any decisions or materials it produced
- `wiki/people/<name>.md` — a stakeholder: role, contact, what they own, decisions they've made
- `wiki/materials/<category>/<slug>.md` — what a piece of material *means* for the project (an annotation page), linking back to the raw file it's about
- `wiki/course/<topic-slug>.md` — a course concept/topic page (pilot domain — see below), synthesizing that topic's lecture materials rather than restating them
- `wiki/status.md` — the one exception to atomicity: a single current-snapshot page, since "what's true right now" is inherently a synthesis. Update it in place, don't append.
- `wiki/index.md` — the home page. Every new page should be reachable from here, directly or through another page.

## Course domain (pilot)

This space also hosts a pilot testing the wiki pattern for course content delivery, kept deliberately separate from the ATA business pages above. Raw source material lives in `raw/course/DS-3021/<topic>/`; synthesized pages live in `wiki/course/<topic>.md`. See `wiki/course/README.md` for scope and curation notes, and `wiki/course/pilot-plan.md` for how this pilot is meant to be evaluated against a plain-RAG baseline — that's the actual point of this domain, not just organizing the material.

A `wiki/course/<topic>.md` page differs from a `wiki/materials/<category>/<slug>.md` page in one key way: materials pages annotate a single document ("what this RFI means for the project"), while course topic pages synthesize *several* source documents (slides + conversion doc + lab notebook) into one concept-level page, the way an encyclopedia entry would rather than a set of document summaries.

### Ongoing sources: class notes and discussion board

Two intake folders feed the course pages on an ongoing basis rather than a one-time bulk pass: `raw/course/DS-3021/classnotes/` and `raw/course/DS-3021/discussion-board/` (naming conventions in each folder's README). This is the part of the pilot that actually tests the framework's "compounding" claim — the three topic pages were built from a static bulk ingest, but class notes and discussion posts arrive continuously through the semester, which is the incremental ingest loop the framework is designed for.

**Ingest workflow for a new note or thread:**
1. File it in the appropriate `classnotes/` or `discussion-board/` folder using the naming convention there. If it's a discussion board export, anonymize student names first (see that folder's README) — never file identifiable student content that could later surface in a student-facing wiki page.
2. Read it and figure out which existing `wiki/course/<topic>.md` page(s) it touches. Most notes/threads will map to exactly one page; some may touch two.
3. Update the relevant topic page(s) — this is usually a small addition (a clarifying point, a common misconception the discussion surfaced, a question worth flagging), not a rewrite. If the note contradicts something already on the page, use the same `⚠️ Possible conflict` convention as everywhere else in this wiki rather than silently overwriting.
4. Add the new raw file to that topic page's Sources list.
5. If the note/thread doesn't fit any existing topic page (e.g. it's about a week not yet piloted), don't force it onto the wrong page — leave it filed in raw and note it as an ingest backlog item, the same way [[materials/notes/2025-2026-running-notes-backlog]] tracks backlog for the ATA domain.

## Linking

Use `[[relative/path/to/page]]` to reference another wiki page. When you create or update a page, link it from anything related — the decision that came out of a meeting, the person who owns a material, the index if it's a new category. A page with no incoming links is invisible; nobody will find it browsing the wiki, only by luck.

## Contradictions

If new information conflicts with an existing page, don't silently overwrite it. Add a flagged note at the top of the affected page (`⚠️ Possible conflict — see below`) with both pieces of information, and surface it to the user rather than deciding for them which one is right.

## Operations

- **Ingest** — new source comes in → file it in `raw/` → update or create the relevant `wiki/` pages → link them → update `index.md`/`status.md` if needed. Handled by the `wiki-ingest` skill.
- **Lint** — periodic health check for broken links, orphan pages, staleness, and unresolved conflicts. Handled by the `wiki-lint` skill.
- **Query** — no special skill needed. Read `wiki/index.md` first, then follow links to whatever's relevant, the way you'd navigate any wiki. Fall back to `raw/` only if the wiki doesn't have the answer yet — and if that happens, that's a sign something should be ingested that hasn't been.
