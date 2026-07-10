# Material: LLM quality/drift test harness

**Owner:** [[../../people/brian-wright]]
**Status (as of 2026-06-26):** existing tool (general-purpose, running ~1 month); medical and visual-LLM variants requested but not yet built.

## What it is

A lightweight tool (simple click-app + SQL storage) that pings ~7–8 foundation models (spanning e.g. GPT-4o-class, Gemini, others) with fixed reasoning, coding, and math prompts, scores the responses with an LLM judge across several metrics (accuracy, speed, reliability, scored 0–10), and tracks scores over time using z-score-based drift detection to flag when a given model's quality on a fixed prompt set shifts. Motivated by reports that models can silently degrade or improve on specific task types as providers retrain/update them (e.g. a 2024 paper noting GPT-4o underperforming GPT-3.5 on math reasoning for a multi-month window before recovering).

## Requested extensions (from [[../../meetings/2026-06-26-bi-weekly-ata]])

- **Medical domain:** copy the repo, reformat quality metrics and test prompts to cover trauma, drug interventions, and basic diagnostic criteria — requested by [[../../people/van]].
- **Visual LLMs:** extend to image inputs, specifically for interpreting/recognizing form data (not image generation) — to support the zero-day form extraction work. See [[../../decisions/2026-06-05-omop-concept-extraction-pivot]].

## Why it matters to the team

Van wants ongoing assurance that ATA is "constantly using the best [model] for whatever it's doing," especially given plans to lean on LLMs in the zero-day extraction pipeline — a drift alert on a model ATA depends on could catch a quality regression before it shows up in production output.
