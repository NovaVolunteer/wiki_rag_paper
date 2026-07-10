# Decision: structure the ATA knowledge assistant as a fact graph, not a flat vector store

**Date:** 2026-06-26
**Source:** [[../meetings/2026-06-26-bi-weekly-ata]]
**Owners:** [[../people/jonathan-seale]], [[../people/brian-wright]]

## Decision

The ATA team knowledge assistant (see [[../materials/plans/ata-knowledge-assistant]]) should organize captured knowledge as a graph of typed facts — design decisions, requirements, implementation evidence, constraints — rather than only as a flat, undifferentiated vector-search index over raw notes. Brian will prototype this against a single data stream first (meeting notes) before expanding to other sources (code review bot output, merge requests, project docs).

## What this refines

The original, looser plan was simply: Otter transcripts → vectorize → store in Supabase → query. That plan still stands as the ingestion mechanism (see [[../materials/plans/ata-knowledge-assistant]]) — this decision is about the *shape* of what gets stored and queried on top of it, not about replacing the pipeline.

## Rationale

Jonathan's concern: a pure vector search over an ever-growing pile of notes risks becoming self-inconsistent and hard to trust as "the answer" to a factual question. His proposed mitigation is to treat the graph as context that shapes an LLM conversation, not as a database you query for ground truth directly — a graph "fact" should never be assumed true on its own, only used as evidence the model can reason over.

Also discussed: tying graph growth to existing workflow touchpoints (e.g., a bot commenting proposed new facts on a merge request for a human to approve/deny) so maintaining the graph doesn't become a standalone burden on the team.

## Open threads

- No standard ontology defined yet for fact/node types — Jonathan estimated "half a dozen to a dozen" object types (design decision, requirement, constraint, evidence, etc.).
- Longer-term idea (unscoped): if multiple projects used a standardized ontology, their graphs could be compared (e.g., cosine similarity on graph structure) to find overlapping capability across ATA's siloed teams — flagged as possibly too ambitious for a first pass.
