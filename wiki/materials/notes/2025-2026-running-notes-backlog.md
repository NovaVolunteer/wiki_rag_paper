# Material: Brian Wright's running notes log — ingest backlog

**Source:** `raw/meetings/2025-03-27-brian-wright-running-notes-log.txt`

## What it is

Not a single meeting — this is Brian Wright's personal running scratchpad of notes across many separate ATA calls and conversations, spanning roughly 2025-03-27 through 2026-01-15 (entries are not in strict chronological order within the file). It mixes fragments on dozens of topics at very different levels of completeness, from one-line ideas to fuller discussion recaps.

This ingest pass pulled out the two most substantive, recurring threads into their own pages:

- [[jts-dha-trauma-data]] — JTS/DHA battlefield trauma data initiative.
- [[swft-ai-security-risk-framework]] — draft AI-security risk framework for a SWFT response.

## Not yet ingested

The following topics appear in the raw log but haven't been distilled into their own wiki pages yet. Listed here so they aren't lost — worth a follow-up ingest pass if any of these are still active:

- **Edge computing / Low-SWaP-C deployment strategy** — running models at the edge without connectivity back to central compute; most developed in the 2025-12-12 entry.
- **Quantum computing** — inconsistent signal: dismissed as "not a thing" in the earliest entry (2025-03-27) but revisited seriously in later entries (2025-12-12, 2026-01-15). Worth clarifying current position with Brian/Jonathan.
- **Automated bias/fairness classification** — a recurring "make data AI safe" / bias-detection white-paper idea, mentioned for at least one existing client (2025-12-12, 2026-01-15).
- **MDA (Missile Defense Agency) object-tracking model sandbox** — a government-owned, independent evaluation environment for object-tracking models (2025-09-04 entry); separately, a data-sharing/object-tracking thread tied to the Air Force ("SOUP") appears earlier in the log without a clear date.
- **Two-stage YOLO + Textract form-onboarding technique** — YOLO to find form sections, then hand-labeled Random Forest (~10 examples per field) to onboard a new form type; directly relevant to the zero-day form extraction work ([[../../decisions/2026-06-05-omop-concept-extraction-pivot]]) but not yet cross-linked into that decision's history.
- **Classification-aggregation ("mosaic effect") risk** — raised by a contact named Scott Jack: combining several pieces of unclassified/lower-classification data can produce an output that should be classified higher than any individual input. Flagged in the notes as an open, unsolved problem (2025-12-12 entry).

## Why this page exists

Per this wiki's own rule: if the wiki doesn't have an answer and you have to fall back to `raw/`, that's a sign something should have been ingested. This page is the flag for that, so a future pass knows exactly what's still outstanding in this particular source file instead of re-reading the whole thing from scratch.
