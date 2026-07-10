# 2026-06-26 — Bi-Weekly ATA

Attendees: [[../people/brian-wright]], [[../people/jonathan-seale]], [[../people/van]] (joined partway through).

Raw sources: [`raw/meetings/2026-06-26-bi-weekly-ata-meeting-transcript.txt`](../../raw/meetings/2026-06-26-bi-weekly-ata-meeting-transcript.txt), [`raw/meetings/2026-06-26-bi-weekly-ata-meeting-summary.txt`](../../raw/meetings/2026-06-26-bi-weekly-ata-meeting-summary.txt)

## What happened

**Knowledge assistant progress.** Brian reported progress on the ATA knowledge-assistant prototype: Otter transcripts are being tagged and vectorized into a Supabase database, hosted in a free-tier GitHub Codespace; a query front end is still pending. See [[../materials/plans/ata-knowledge-assistant]].

**Knowledge graph direction.** Jonathan reframed the goal: rather than just tracking conversations, structure captured knowledge into a small ontology of object types (requirements, implementation evidence, constraints, design decisions) connected as a graph, not just a flat vector search index. He raised a self-consistency concern — the graph should inform an LLM conversation rather than be treated as ground truth on its own. They discussed tying graph updates to a review step in the normal dev workflow (e.g., a bot commenting "here are new facts I think I learned" on a merge request for approval). This became [[../decisions/2026-06-26-knowledge-graph-over-pure-vector-search]]. Jonathan also mentioned an NSF phased funding program (name unclear in transcript — sounds like "Siber Sitter") as a possible funding source, and separately DARPA.

**LLM quality/drift tool.** Brian described his existing model-quality/drift-tracking tool (per-model reasoning/coding/math prompts, LLM-judge scoring, z-score drift alerts). Van asked for two extensions: a visual-LLM harness for form-data interpretation, and a medical-domain harness (trauma, drug interventions, basic diagnostic criteria). See [[../materials/plans/llm-quality-drift-tool]]. Van also gave an update on comparing ATA's OCR+YOLO pipeline against a client's OpenAI/Palantir-based extraction approach — still trying to determine whether the client's "extracted all form values in two hours" claim means correctly-mapped values or just recognized text.

**Pricing strategy.** Jonathan raised how AI-driven efficiency should affect ATA's bid pricing — risk of under-bidding (losing credibility if the government doesn't believe the claimed efficiency) vs. over-bidding (losing competitiveness). ATA doesn't yet have enough statistical volume on its own efficiency gains to price confidently (most work is hourly/level-of-effort, not per-deliverable, which makes gains hard to measure).

## Action items

- [ ] Brian Wright — design/implement the ATA knowledge-assistant prototype (Otter → vectorize → Supabase → front end, with monthly/quarterly/yearly summarization), architected so it can later support agents and a knowledge graph.
- [ ] Brian Wright — adapt the LLM quality/drift tool for a medical context (copy repo, reformat quality metrics and test prompts).
- [ ] Brian Wright — design an expanded medical test suite covering trauma, drug interventions, and basic diagnostic criteria; explore image-based visual-LLM evaluation.
- [ ] Brian Wright — explore adapting the quality-eval framework to visual LLMs for form-data extraction specifically.
- [ ] Brian Wright — research market data/literature on AI efficiency gains and cost reduction in software development, to inform ATA's pricing strategy.
- [ ] Continue researching funding opportunities (NSF, DARPA) for the knowledge-assistant project.
- [ ] Van — reconnect once he has a clearer understanding of the client's OpenAI-based form-extraction approach, to continue the zero-day conversation.
