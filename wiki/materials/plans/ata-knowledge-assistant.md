# Material: ATA knowledge assistant (prototype)

**Owner:** [[../../people/brian-wright]]
**Status (as of 2026-06-26):** in progress — ingestion pipeline exists, front end not yet built.

## What it is

A prototype tool to give the ATA team persistent, queryable memory of their own meetings and project conversations. Otter records/transcribes meetings; Brian tags and exports the long-form transcript (headings, action items, dates); notes are vectorized and stored in a Supabase database, hosted in a free-tier GitHub Codespace (~$10/month usage-based, no GPU). Goal is a lightweight query front end so the team can ask things like "what did we talk about in Q1 2025 for ATA?" and get a summarized answer, plus periodic (monthly/quarterly/yearly) auto-summaries.

Architecture is meant to later support:
- Multiple data streams beyond meeting notes (code review bot output, merge requests, project documentation).
- A knowledge-graph layer on top of the raw vectorized notes — see [[../../decisions/2026-06-26-knowledge-graph-over-pure-vector-search]] for the direction that decision took.

## Why

Jonathan wants a team assistant / "context tracker" that reduces the friction of cross-pollinating knowledge across ATA's small, siloed teams — both for real-time project execution and for longer-term reporting and past-performance write-ups. Jonathan is separately exploring NSF/DARPA funding to develop this further (see [[../../people/jonathan-seale]]).

## Status / next steps

- Action item from [[../../meetings/2026-06-26-bi-weekly-ata]]: build out the front end and formalize the architecture so it can support embedding agents and a knowledge graph.
- Jonathan is investigating hosting/tooling equivalents inside the team's existing Microsoft/OpenAI stack (SharePoint, Teams, GitHub-for-context) as an alternative or complement.
