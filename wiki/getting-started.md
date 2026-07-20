# Getting Started

A quick guide for new contributors to this wiki.

## What this is

A living knowledge base for the ATA team's AI/ML projects. Raw source files (transcripts, RFIs, plans) are stored untouched; a small set of structured wiki pages distills what those sources *mean* — decisions made, meetings held, who owns what, and what the current project state is.

Start at [[index]] and follow links. You should be able to understand what's going on without ever opening a file in `raw/`.

## Two folders, two rules

**`raw/`** — never edit these. Drop new source files here exactly as received, named `YYYY-MM-DD-short-description.ext`. This is the paper trail.

**`wiki/`** — living pages. These get created and updated as new information comes in. This is what you read and maintain.

## Page types

| Type | Path | One page covers |
|---|---|---|
| Decisions | `wiki/decisions/<date>-<slug>.md` | One decision, its rationale, what it superseded |
| Meetings | `wiki/meetings/<date>-<slug>.md` | Distilled summary of a meeting (not the transcript) |
| People | `wiki/people/<name>.md` | Role, what they own, relevant links |
| Materials | `wiki/materials/<category>/<slug>.md` | What a document *means* for the project, linked to `raw/` |
| Status | `wiki/status.md` | Current snapshot — update in place, don't append |
| Index | `wiki/index.md` | Home page — every page must be reachable from here |

Keep pages small and single-topic. If a page is covering two things, split it.

## Adding a new meeting

1. Save the raw transcript/summary in `raw/meetings/` — don't edit it.
2. Create `wiki/meetings/<date>-<slug>.md` with a distilled summary, attendees, and action items.
3. If the meeting produced a decision, create `wiki/decisions/<date>-<slug>.md` and link to it from the meeting page.
4. Update `wiki/index.md` to link the new meeting page.
5. Update `wiki/status.md` if the current project state changed.

## Adding a new decision

1. Create `wiki/decisions/<date>-<slug>.md`.
2. State the decision clearly at the top, then the rationale and what it supersedes (if anything).
3. Link it from the meeting that produced it, the people who made it, and any materials it affects.
4. Add it to `wiki/index.md` under Decisions.

## Adding a new person

Create `wiki/people/<firstname-lastname>.md`. Include their role, what they own, and links to meetings they've attended. Link them from `wiki/index.md`.

## Linking

Use `[[relative/path/to/page]]` to link between wiki pages. Always link new pages from related pages — if nothing links to a page, it's effectively invisible.

## If something looks wrong

Check `raw/` to see what the wiki page was built from. If there's a conflict between two sources, add a `⚠️ Possible conflict — see below` note at the top of the affected page rather than silently overwriting it, and flag it for the team to resolve.

## Current projects

See [[status]] for the current state of all active workstreams, open blockers, and next actions.
