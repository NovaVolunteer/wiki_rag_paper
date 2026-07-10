# Material: JTS / DHA trauma data initiative

**Source:** `raw/meetings/2025-03-27-brian-wright-running-notes-log.txt`, entries dated approximately 2025-09-18, 2025-10-16, and 2026-01-15.
**Related people:** [[../../people/van]], [[../../people/jonathan-seale]]; client-side lead noted as "Dr. Stallings" (DHA).

## What it is

An ongoing thread around JTS (Joint Trauma System) — the DoD system that ingests battlefield trauma care data across service branches, starting at point of care (often still on paper forms) and consolidating it for aggregate analysis. Interest areas noted across several conversations:

- **Current-state pain point:** trauma data still often comes in on paper in the field, with digital/QR-code capture seen as a future improvement.
- **Self-consistency testing:** existing JTS form DA 4700 needs internal consistency checks — e.g., an injury noted on page 1 should match the treatment recorded on page 3.
- **Analytics/after-action layer:** interest in a data-science sandbox (R-based — the biomedical/stats community's preferred tool) sitting on top of JTS, used for analysis and after-action reporting rather than direct patient care. ATA has medical-background staff who could plausibly support this.
- **Synthetic data generation** for trauma scenarios was raised as a way to expand training data, with an explicit caution about distributional risk — e.g., over-representing one trauma type could bias downstream predictions (a note gives the example of a trauma type correlated with high blood pressure needing to preserve that ~90%-style conditional relationship rather than being generated independently).
- **Digitizing handwritten/scanned trauma forms**, including body-diagram annotation (marking injury location on a body outline), was raised as related to a possible SBIR proposal — directly related to the [[../../decisions/2026-06-05-omop-concept-extraction-pivot]] work.
- **Funding:** a possible STTR opportunity (roughly 4–5 FTE, longer-term) tied to Dr. Stallings' work — contingent on ATA finding someone internally with a strong medical-data research interest to lead/support it.

## Status

Exploratory / relationship-building as of the most recent note (2026-01-15) — no committed proposal or contract yet. Flagged here because it's a recurring thread across many months of notes; worth checking with Brian/Jonathan on current status before treating any of the above as still-accurate.
