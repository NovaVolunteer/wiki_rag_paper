# Material: SWFT response — AI security risk framework (draft)

**Source:** `raw/meetings/2025-03-27-brian-wright-running-notes-log.txt`, undated section (falls between entries dated 2025-06-06 and 2025-07-21, so approximately mid-2025 — exact date unclear).

## What it is

Draft thinking for a "SWFT" response (proposal/RFI response — exact program not specified in the notes) addressing how a vendor should articulate its use of AI in software delivery, prompted by a "shift left" expectation that developers clearly disclose where and how AI is used in the pipeline. Frames three use cases:

1. AI models developed and shipped as a discrete component of the software.
2. AI models used to help develop the software being proposed.
3. Both.

None of these is inherently risky in isolation, but the notes argue that *unclear articulation* of when/how AI is used creates downstream risk. Draft risk ranking (worst to best):

- Raw API calls with limited fine-tuning/data augmentation — **bad**
- Zero-shot prompting for specific, knowledge-dependent tasks — **bad**
- Large fine-tuned models asked to handle multiple tasks across the pipeline — **bad**
- Fine-tuned small/medium models for specific tasks — **better**
- No RAG/data-augmentation process — **probably not ideal**
- Highly-tuned, offline, small/medium models using RAG-style augmentation — **good**

Proposes a possible **"Milestone Zero"**: the developer must explicitly state how generative AI is being used and what evaluation measures guard against risk, potentially producing a risk score.

Other points raised for the same response:
- Lean into explainable AI (xAI) as a differentiator — simpler, fully-understood methods can outperform black-box generative approaches, cost less to maintain, and carry less risk, especially where the data landscape is already well established. Also notes "AI" isn't synonymous with transformer-based models.
- Suggests tracking a model's stability/performance as a function of dataset size as a way to estimate how much data a given task will need — drawing an analogy to statistical power analysis / sample-size estimation.

## Status

Draft/brainstorm notes only — no indication in the source that this was finalized or submitted. Flag to confirm with Jonathan/Brian whether this fed into an actual response and what became of it.
