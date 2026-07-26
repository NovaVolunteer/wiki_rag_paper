# Van

ATA team member driving the medical document-extraction work: zero-day form extraction, OMOP concept mapping, and the JTS (Joint Trauma System) initiative. Full name likely Van Duong (per vduong@ata-llc.com on the 2026-07-24 meeting invite — not otherwise confirmed).

## Owns

- Zero-day/unknown-form extraction approach — pushed the pivot from key-value extraction to OMOP concept extraction. See [[../decisions/2026-06-05-omop-concept-extraction-pivot]].
- Comparing ATA's OCR+YOLO extraction pipeline against a client's OpenAI/Palantir-based approach, to understand where the gap is between VLM-only extraction and ATA's traditional-OCR-plus-model pipeline. As of 2026-06-26, still investigating what the client's "extracted form values in two hours" claim actually means (text recognition vs. correctly-mapped values).
- Requested the LLM quality/drift tool be extended to visual LLMs (form-data interpretation) and medical-domain text. See [[../materials/plans/llm-quality-drift-tool]].
- JTS / DHA trauma-data work. See [[../materials/notes/jts-dha-trauma-data]].

## Meetings

- [[../meetings/2026-06-05-zero-day-form-extraction]]
- [[../meetings/2026-06-26-bi-weekly-ata]] (joined partway through)
- [[../meetings/2026-07-24-document-image-classification]]
