# Status

Last updated: 2026-07-10

The one page in this wiki that's a snapshot instead of an atomic entry — update it in place, don't append. History lives in [[index|the decision and meeting pages]] instead.

## Current phase

Wiki just initialized from ATA team's raw meeting notes (2026-06-05 and 2026-06-26 meetings, plus a long-running personal notes log). Two active workstreams for the ATA team:

1. **Zero-day form extraction** — pivoted to OMOP-concept extraction for unknown documents ([[decisions/2026-06-05-omop-concept-extraction-pivot]]); [[people/van]] comparing this against a client's OpenAI/Palantir-based approach.
2. **ATA knowledge assistant** — [[people/brian-wright]] building an Otter → Supabase → query-front-end prototype, now aimed at a knowledge-graph structure rather than flat vector search ([[decisions/2026-06-26-knowledge-graph-over-pure-vector-search]]).

3. **SBIR Partnership** - currently working to determine best path forward to respond to the Darpa SBIR

## Open items and blockers

- No query front end yet for the knowledge assistant ([[materials/plans/ata-knowledge-assistant]]).
- Medical and visual-LLM variants of the drift-tracking tool not yet built ([[materials/plans/llm-quality-drift-tool]]).
- DFAS RFI response in progress, no outcome recorded yet ([[materials/rfis/dfas-rfi]]).
- ATA doesn't yet have enough data on its own AI-driven efficiency gains to price bids confidently — Brian researching market norms.
- Large backlog of un-ingested topics in Brian's older running notes log — see [[materials/notes/2025-2026-running-notes-backlog]] for the list (edge computing, quantum computing, bias/fairness classification, MDA object-tracking sandbox, classification-aggregation risk, etc.).
- This repo's git root is the entire home directory (`C:\Users\bw2zd`), not `llm-wiki` — flagged to the user 2026-07-10, not yet resolved.

## Next actions

- Brian: sketch/formalize the knowledge-assistant architecture (front end + knowledge-graph layer).
- Brian: adapt the LLM drift tool for medical use cases (trauma, drug interventions, diagnostic criteria) and explore a visual-LLM variant.
- Brian: research AI efficiency/cost-reduction market data for ATA's pricing strategy.
- Jonathan: continue chasing NSF/DARPA funding for the knowledge-assistant project.
- Van: report back once the client's OpenAI/Palantir extraction approach is better understood, to continue the zero-day conversation.
