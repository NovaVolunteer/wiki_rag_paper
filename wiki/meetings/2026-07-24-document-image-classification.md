# 2026-07-24 — Document Image Classification

Attendees: [[../people/brian-wright]], [[../people/jonathan-seale]], [[../people/van]], [[../people/craig-parisot]] (ATA CEO/Founder).

Raw source: [`raw/meetings/2026-07-24-document-image-classification-meeting-summary.txt`](../../raw/meetings/2026-07-24-document-image-classification-meeting-summary.txt) (summary only, no transcript). Individual speaker turns in the source are anonymized ("Speaker 1"–"Speaker 5") and not mapped to attendee names except where Brian Wright is named directly.

## What happened

Follow-up to the pipeline work in [[2026-06-05-zero-day-form-extraction]] and [[2026-06-26-bi-weekly-ata]], this time focused on the **classification step** that runs before extraction: identifying which form type a given page is, upstream of both the known-form pipeline and the OMOP zero-day approach ([[../decisions/2026-06-05-omop-concept-extraction-pivot]]).

Current classifier: ~3,000 labeled training pages, 80% accuracy. The team discussed using an LLM as a secondary classifier to catch cases the primary model misses, and estimated this could bring effective per-page cost down from $0.50 to $0.04. Other threads:

- Skepticism about competitors' published accuracy claims, given lack of transparency on their training data.
- Options for generating more training data cheaply: synthetic/augmented forms, autoencoder/GAN-style augmentation, and low-cost human labeling (undergraduate students, Mechanical Turk) rather than more expensive labor.
- Confidence thresholds for routing low-confidence classifications to human review; Brian suggested using class entropy as an added signal.
- A hybrid approach — LLM used for secondary/ensemble classification and feature-space enhancement rather than replacing the existing YOLO-based classifier and field-extraction model outright.

No final decision was made; the team agreed to design experiments to test the LLM-assisted approach and reassess YOLO's effectiveness at a follow-up meeting.

## Action items

- [ ] Brian Wright — share the AWS Bedrock/local-LLM form-classification paper referenced in the meeting with the team.
- [ ] Brian Wright — consult DS Labs/UVA colleagues on computer-vision/data-augmentation techniques (synthetic data, autoencoder/GAN approaches) and low-cost labeling options.
- [ ] Brian Wright — add the missing attendee to the calendar invite for the Tuesday 1:00 PM follow-up.
- [ ] Brian Wright — send the calendar invite for the Tuesday 1:00 PM follow-up working session.
- [ ] Raise a separate discussion with leadership about budget for inexpensive labeling labor (grad students, Mechanical Turk) to grow the training set.
- [ ] Complete relabeling of the BA 4700 form pages and retrain the YOLO-based field-extraction model on the updated dataset.
