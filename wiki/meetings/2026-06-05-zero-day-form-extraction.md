# 2026-06-05 — Zero Day Form Extraction Meeting

Attendees: [[../people/brian-wright]], [[../people/van]], [[../people/jonathan-seale]] (plus 1–2 unidentified participants briefly).

Raw sources: [`raw/meetings/2026-06-05-zero-day-form-extraction-meeting-transcript.txt`](../../raw/meetings/2026-06-05-zero-day-form-extraction-meeting-transcript.txt), [`raw/meetings/2026-06-05-zero-day-form-extraction-meeting-summary.txt`](../../raw/meetings/2026-06-05-zero-day-form-extraction-meeting-summary.txt)

## What happened

Main topic: how to extract structured data from a "zero day" form — a medical document ATA has never seen before and knows nothing about (could be a real form, a handwritten note, or a memo). The existing pipeline (VLM + Amazon Textract → key-value pairs → map to OMOP fields) only works when the form's structure is already known.

Van relayed a conversation with a colleague, Dave (who works on the OMOP conversion side), suggesting a paradigm shift: instead of trying to extract key-value pairs from an unknown structure, extract all text and scan it for a fixed set of known OMOP medical concepts (~300 of them — medications, procedures, conditions, measurements, demographics), regardless of where they appear or what the document looks like. This became [[../decisions/2026-06-05-omop-concept-extraction-pivot]].

Other threads discussed:
- Confidence scoring per extracted concept, and combining NLP confidence with positional/layout-based confidence (e.g., "this form type shouldn't have lab tests on it").
- AWS Comprehend Medical as an option for concept extraction — noted to be pre-LLM technology, falters on typos (e.g., medication name misspellings), and isn't tunable, unlike an LLM.
- The [[../materials/rfis/dfas-rfi]] was raised as a structurally similar problem: less about full structured extraction, more about searchability and pulling specific known fields (e.g., cage codes) out of financial documents.
- A long second-half discussion on knowledge management / long-term memory tooling for the team: Otter for transcription, Claude's cross-project memory vs. OpenAI's "Dreamer" persistent memory, and the tradeoffs between a markdown/YAML file tree, a vector database, and a graph/triple-store for storing project knowledge. This is the seed of the knowledge-graph direction picked up again in [[../meetings/2026-06-26-bi-weekly-ata]].

## Action items

- [ ] Brian Wright — sketch a system architecture for ATA knowledge retention.
- [ ] Van — pull together research/documents supporting the OMOP zero-day approach, then reconnect with Brian.
- [ ] Reconvene the following week (became the 2026-06-26 meeting).
