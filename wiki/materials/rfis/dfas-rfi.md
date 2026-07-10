# Material: DFAS RFI

**Mentioned by:** [[../../people/jonathan-seale]]
**Source:** [[../../meetings/2026-06-05-zero-day-form-extraction]]

## What it is

ATA is responding to an RFI from DFAS (Defense Finance and Accounting Service, financial systems for DoD). As described in the 2026-06-05 meeting, DFAS's stated requirement is less like the highly structured extraction ATA does for JTS, and much more about **searchability** plus lighter metadata extraction — e.g., identifying cage codes or similar standard identifiers within documents, rather than mapping an entire document to a known schema.

## Why it's noted here

This requirement shape is conceptually similar to the [[../../decisions/2026-06-05-omop-concept-extraction-pivot]] approach for zero-day medical forms: look for specific, known bits of information wherever they appear, rather than trying to structure the whole document against a standard. Brian noted that (unlike medical data) some DFAS-related information is public through DoD auditing/documentation requirements, which could be useful for pre-training or testing extraction systems against real, non-protected examples.

## Status

No decision or output recorded yet beyond the RFI being in progress as of 2026-06-05 — flagging for a follow-up ingest once there's a response draft or outcome.
