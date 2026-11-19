#!/usr/bin/env python
"""
Generate the evaluation question set from the RAW course files only. The
generator never sees the compiled wiki, so the question set cannot be shaped
by the wiki's structure or wording.

Generator: an OpenAI model (default gpt-5.5), a different provider from the
answerer (Claude), so the answerer is not taking a test written by its own
model family.

  topic-page   6-7 per topic (45 total), from one topic's raw files
  cross-page   2 per topic pair (21 pairs -> 42) + 3 three-topic questions
  unanswerable in-scope questions the course material does not answer;
               each candidate is checked against the FULL raw corpus and
               dropped if any file answers it

Each question carries a reference answer and its gold raw files. Gold concept
pages are derived from the gold files' topic folders (raw/.../<topic>/ ->
course/<topic>.md).

    python generate_questions.py                 # -> course_questions_raw.json
    python generate_questions.py --out x.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from raw_corpus import TOPICS, build_raw_context, load_raw_corpus

load_dotenv()

DEFAULT_GEN_MODEL = "gpt-5.5"
TOPIC_COUNTS = {t: 7 if i < 3 else 6 for i, t in enumerate(TOPICS)}  # 3x7 + 4x6 = 45
PER_PAIR = 2
N_TRIPLES = 3
N_UNANSWERABLE_CANDIDATES = 10
SEED = 20261118

GUIDE = (
    "You are writing evaluation questions for an AI tutor in a university machine learning "
    "course (DS-3021). Write questions a real student in this course might ask. Each "
    "question must be answerable from the course files given (unless told otherwise), "
    "should require understanding rather than string matching, and must not be yes/no. "
    "Do not mention file names in the question, and do not say 'according to the slides'. "
    "Phrase each question naturally. Keep the reference answer to at most 150 words, "
    "stating only what the course files support. In gold_files, list the exact paths "
    "(as given after 'START DOCUMENT:') of every file the reference answer relies on."
)


class GenQ(BaseModel):
    question: str
    question_type: Literal["factual", "conceptual", "procedural", "comparison", "synthesis"]
    reference_answer: str
    gold_files: list[str]


class GenBatch(BaseModel):
    questions: list[GenQ]


class Unanswerable(BaseModel):
    question: str
    why_in_scope: str


class UnanswerableBatch(BaseModel):
    questions: list[Unanswerable]


class AnswerCheck(BaseModel):
    answerable: bool
    supporting_files: list[str]
    explanation: str


def parse(client: OpenAI, model: str, prompt: str, fmt):
    c = client.chat.completions.parse(model=model, messages=[{"role": "user", "content": prompt}],
                                      response_format=fmt)
    out = c.choices[0].message.parsed
    if out is None:
        raise RuntimeError(f"no parsed output (finish_reason={c.choices[0].finish_reason})")
    return out


def topic_of(path: str) -> str:
    return path.split("/")[3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--out", default="course_questions_raw.json")
    ap.add_argument("--max-workers", type=int, default=6)
    args = ap.parse_args()

    client = OpenAI()
    docs = load_raw_corpus()
    by_topic = {t: [d for d in docs if d["topic"] == t] for t in TOPICS}
    paths = {d["path"] for d in docs}
    full_ctx = build_raw_context(docs)
    rng = random.Random(SEED)

    jobs = []  # (class, topics, n)
    for t in TOPICS:
        jobs.append(("topic-page", [t], TOPIC_COUNTS[t]))
    for a, b in itertools.combinations(TOPICS, 2):
        jobs.append(("cross-page", [a, b], PER_PAIR))
    for tri in rng.sample(list(itertools.combinations(TOPICS, 3)), N_TRIPLES):
        jobs.append(("cross-page", list(tri), 1))

    def run(job):
        cls, topics, n = job
        ctx = build_raw_context([d for t in topics for d in by_topic[t]])
        if cls == "topic-page":
            task = (f"Write {n} distinct questions about the course topic '{topics[0]}', each answerable "
                    "from the files below. Cover different parts of the material (slides, notebooks, "
                    "labs, readings), not just one file.")
        else:
            task = (f"Write {n} distinct question(s) that REQUIRE material from ALL of these topics to "
                    f"answer well: {', '.join(topics)}. A question that one topic's files alone could "
                    "answer is not acceptable. Good questions compare, contrast, or connect methods, "
                    "assumptions, preprocessing, or evaluation across the topics, as the course teaches them.")
        batch = parse(client, args.model, f"{GUIDE}\n\nTASK: {task}\n\nCOURSE FILES:\n\n{ctx}", GenBatch)
        out = []
        for q in batch.questions[:n]:
            gold = [p for p in q.gold_files if p in paths]
            gold_topics = sorted({topic_of(p) for p in gold})
            if cls == "cross-page" and len(gold_topics) < 2:
                print(f"  dropped (gold files span <2 topics): {q.question[:80]}", file=sys.stderr)
                continue
            if not gold:
                print(f"  dropped (no valid gold files): {q.question[:80]}", file=sys.stderr)
                continue
            out.append({"question": q.question, "question_type": q.question_type,
                        "question_class": cls, "topics": topics, "reference_answer": q.reference_answer,
                        "gold_raw_docs": gold, "gold_pages": [f"course/{t}.md" for t in gold_topics]})
        print(f"  {cls} {topics}: {len(out)}/{n}", file=sys.stderr)
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        generated = [q for qs in pool.map(run, jobs) for q in qs]

    # Unanswerable controls: propose, then verify against the full corpus.
    cands = parse(client, args.model, (
        f"{GUIDE}\n\nTASK: Write {N_UNANSWERABLE_CANDIDATES} questions a student in this course could "
        "plausibly ask about these same topics, but that the course files below do NOT answer: e.g. a "
        "method variant, parameter, or practical detail the course never covers. Spread them across "
        f"topics. Leave nothing answerable from the files.\n\nCOURSE FILES:\n\n{full_ctx}"
    ), UnanswerableBatch).questions

    def check(u: Unanswerable):
        r = parse(client, args.model, (
            f"COURSE FILES:\n\n{full_ctx}\n\nQUESTION: {u.question}\n\nCould a tutor give a substantive, "
            "correct answer to this question using ONLY the course files above? Be strict: if any file "
            "addresses the core of the question, answerable is true."
        ), AnswerCheck)
        return u, r

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for u, r in pool.map(check, cands):
            if r.answerable:
                print(f"  unanswerable candidate rejected (answered by {r.supporting_files}): {u.question[:70]}",
                      file=sys.stderr)
                continue
            generated.append({"question": u.question, "question_type": "unanswerable",
                              "question_class": "unanswerable", "topics": [], "reference_answer":
                              "The course material does not cover this; a correct response says so.",
                              "gold_raw_docs": [], "gold_pages": [], "unanswerable_check": r.explanation})

    order = {"topic-page": 0, "cross-page": 1, "unanswerable": 2}
    generated.sort(key=lambda q: order[q["question_class"]])
    for i, q in enumerate(generated):
        q["id"] = i
        q["source_page"] = q["gold_pages"][0] if q["question_class"] == "topic-page" else f"({q['question_class']})"
        q["gold_annotation"] = f"generator:{args.model}"
    Path(args.out).write_text(json.dumps(generated, indent=2), encoding="utf-8")
    counts = {c: sum(q["question_class"] == c for q in generated) for c in order}
    print(f"Wrote {len(generated)} questions -> {args.out}  {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
