# Decision: extract OMOP concepts, not key-value pairs, from zero-day forms

**Date:** 2026-06-05
**Source:** [[../meetings/2026-06-05-zero-day-form-extraction]]
**Owners:** [[../people/van]], [[../people/brian-wright]]

## Decision

For "zero day" documents — medical forms/notes ATA has never seen and knows nothing about — stop trying to extract key-value pairs and map them into OMOP fields. Instead, extract all text (VLM/OCR) and scan it for a fixed set of ~300 known OMOP medical concepts (medications, procedures, conditions, measurements, demographics), wherever they appear in the text, with a confidence score per concept found.

## What this supersedes

The prior approach — VLM output combined with Amazon Textract, hunting for key-value pairs and organizing them into OMOP structure for review — remains the right approach for **known** forms (e.g., JTS forms) where field positions and meanings are already understood. This decision only applies to unstructured or previously-unseen documents, where there's no reliable structure to map against and no positional ground truth to validate an extracted value against.

## Rationale

Idea originated with Dave (via Van), who works on the OMOP conversion side: if a document's structure is unknown, ATA also doesn't know what any given piece of text *means* — trying to force it into key-value structure invites hallucination. Looking only for a defined, bounded set of medical concepts avoids that: either a concept is recognized with confidence, or it isn't extracted at all. No validation step is needed afterward the way it is for known-form extraction, because anything recognized as a valid OMOP concept is, by construction, valid data.

## Open threads

- Confidence could be strengthened by combining NLP confidence with positional/contextual signals (e.g., "this form type shouldn't have lab tests on it") if enough form-layout density is collected over time.
- AWS Comprehend Medical was discussed as one implementation option but is pre-LLM tech, not tunable, and falters on typos — an LLM-based (possibly locally fine-tuned) approach was seen as likely more robust.
- Conceptually similar to what [[../materials/rfis/dfas-rfi]] needs: search for specific known bits of information rather than fully structuring an unfamiliar document.
