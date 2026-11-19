#!/usr/bin/env python
"""
Build the 100-question set for the full-course corpus (2026-11-19).

    50 RETAINED  a random half of the 7-topic question set
                 (results/archive_7topic_100q/course_questions_raw.json),
                 stratified by question class (23 topic-page, 22 cross-page,
                 5 unanswerable). Retained unanswerable controls are re-checked
                 against the EXPANDED raw corpus; any that the new material now
                 answers is replaced by a fresh one.
    50 NEW       generated with the same recipe and model as generate_questions.py,
                 from the raw files of the 9 new topics only (the generator never
                 sees the wiki):
                   22 topic-page   (3 each for the 4 largest new topics, 2 for the rest)
                   23 cross-page   21 random topic pairs + 2 random triples, each with
                                   at least one new topic (so new<->pilot links are tested)
                    5 unanswerable  about the new topics, verified against the full corpus

Every question keeps its reference answer and gold raw files; `origin` records
`retained:<old id>` or `new`. Output ids are 0..99 (retained first, then new).

    python expand_questions.py                  # -> course_questions_raw.json
    python expand_questions.py --dry-run        # show the plan, no API calls
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from generate_questions import (DEFAULT_GEN_MODEL, GUIDE, AnswerCheck, GenBatch, Unanswerable,
                                UnanswerableBatch, parse, topic_of)
from raw_corpus import NEW_TOPICS, PILOT_TOPICS, TOPICS, build_raw_context, load_raw_corpus

load_dotenv()

OLD_SET = Path("results/archive_7topic_100q/course_questions_raw.json")
SEED = 20261119
KEEP = {"topic-page": 23, "cross-page": 22, "unanswerable": 5}
NEW_TOPIC_COUNTS = {t: 3 if t in {"linear-regression", "data-prep", "ml-foundations", "python-basics"} else 2
                    for t in NEW_TOPICS}  # 4x3 + 5x2 = 22
N_NEW_PAIRS, N_NEW_TRIPLES = 21, 2
N_NEW_UNANSWERABLE = 5
SURPLUS = 3  # extra cross-page jobs, used only if some are dropped by validation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--out", default="course_questions_raw.json")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(SEED)

    # ---- retained half ------------------------------------------------------
    old = json.loads(OLD_SET.read_text(encoding="utf-8"))
    retained = []
    for cls, n in KEEP.items():
        pool = [q for q in old if q["question_class"] == cls]
        retained += sorted(rng.sample(pool, n), key=lambda q: q["id"])
    for q in retained:
        q["origin"] = f"retained:{q['id']}"

    # ---- generation plan for the new half -------------------------------------
    jobs = [("topic-page", [t], n) for t, n in NEW_TOPIC_COUNTS.items()]
    pairs = [p for p in itertools.combinations(TOPICS, 2) if set(p) & set(NEW_TOPICS)]
    triples = [t for t in itertools.combinations(TOPICS, 3) if set(t) & set(NEW_TOPICS)]
    cross_jobs = ([("cross-page", list(p), 1) for p in rng.sample(pairs, N_NEW_PAIRS + SURPLUS)]
                  + [("cross-page", list(t), 1) for t in rng.sample(triples, N_NEW_TRIPLES)])
    print(f"Retained {len(retained)}: " + ", ".join(f"{c}={sum(q['question_class'] == c for q in retained)}"
                                                    for c in KEEP), file=sys.stderr)
    print(f"New: {len(jobs)} topic-page jobs ({sum(n for *_, n in jobs)} q), {len(cross_jobs)} cross-page jobs "
          f"(target {N_NEW_PAIRS + N_NEW_TRIPLES}), {N_NEW_UNANSWERABLE} unanswerable", file=sys.stderr)
    for _, t, _ in cross_jobs:
        print(f"  cross: {' + '.join(t)}", file=sys.stderr)
    if args.dry_run:
        return

    # Full-corpus checks are ~225k tokens each; the org limit is 1M tokens/min, so run at most 2 at a
    # time and let the SDK back off on 429s.
    client = OpenAI(max_retries=10)
    check_workers = 2
    # Checkpoint each stage so a crash does not throw away (and re-bill) finished generation.
    ckpt_path = Path("results/expand_questions_checkpoint.json")
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}

    def save_ckpt():
        ckpt_path.write_text(json.dumps(ckpt, indent=1), encoding="utf-8")
    docs = load_raw_corpus()
    by_topic = {t: [d for d in docs if d["topic"] == t] for t in TOPICS}
    paths = {d["path"] for d in docs}
    full_ctx = build_raw_context(docs)

    def check(question: str) -> AnswerCheck:
        return parse(client, args.model, (
            f"COURSE FILES:\n\n{full_ctx}\n\nQUESTION: {question}\n\nCould a tutor give a substantive, "
            "correct answer to this question using ONLY the course files above? Be strict: if any file "
            "addresses the core of the question, answerable is true."
        ), AnswerCheck)

    # Retained unanswerable controls must still be unanswerable from the expanded corpus.
    kept_unans = [q for q in retained if q["question_class"] == "unanswerable"]
    if "retained_checks" not in ckpt:
        with concurrent.futures.ThreadPoolExecutor(max_workers=check_workers) as pool:
            ckpt["retained_checks"] = {str(q["id"]): r.model_dump() for q, r in
                                       zip(kept_unans, pool.map(lambda q: check(q["question"]), kept_unans))}
        save_ckpt()
    for q in kept_unans:
        r = AnswerCheck(**ckpt["retained_checks"][str(q["id"])])
        if r.answerable:
            print(f"  retained unanswerable now ANSWERED by {r.supporting_files}: {q['question'][:70]}",
                  file=sys.stderr)
            retained.remove(q)
        else:
            q["unanswerable_check_full_course"] = r.explanation
    n_unans_needed = N_NEW_UNANSWERABLE + (KEEP["unanswerable"] - sum(q["question_class"] == "unanswerable"
                                                                      for q in retained))

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
        try:
            batch = parse(client, args.model, f"{GUIDE}\n\nTASK: {task}\n\nCOURSE FILES:\n\n{ctx}", GenBatch)
        except Exception as e:  # noqa: BLE001
            print(f"  {cls} {topics}: generation failed ({e})", file=sys.stderr)
            return []
        out = []
        for q in batch.questions[:n]:
            gold = [p for p in q.gold_files if p in paths]
            gold_topics = sorted({topic_of(p) for p in gold})
            if not gold:
                print(f"  dropped (no valid gold files): {q.question[:80]}", file=sys.stderr)
                continue
            if cls == "cross-page" and len(gold_topics) < 2:
                print(f"  dropped (gold files span <2 topics): {q.question[:80]}", file=sys.stderr)
                continue
            out.append({"question": q.question, "question_type": q.question_type, "question_class": cls,
                        "topics": topics, "reference_answer": q.reference_answer, "gold_raw_docs": gold,
                        "gold_pages": [f"course/{t}.md" for t in gold_topics], "origin": "new"})
        print(f"  {cls} {topics}: {len(out)}/{n}", file=sys.stderr)
        return out

    if "topic_qs" not in ckpt:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            ckpt["topic_qs"] = [q for qs in pool.map(run, jobs) for q in qs]
            ckpt["cross_res"] = list(pool.map(run, cross_jobs))
        save_ckpt()
    topic_qs, cross_res = ckpt["topic_qs"], ckpt["cross_res"]
    # Keep the first N pairs that produced a valid question, then the triples.
    pair_res, triple_res = cross_res[: N_NEW_PAIRS + SURPLUS], cross_res[N_NEW_PAIRS + SURPLUS:]
    cross_qs = [qs[0] for qs in pair_res if qs][:N_NEW_PAIRS] + [qs[0] for qs in triple_res if qs]

    # New unanswerable controls, about the new topics, verified against the full corpus.
    new_ctx = build_raw_context([d for t in NEW_TOPICS for d in by_topic[t]])
    if "unans_cands" not in ckpt:
        ckpt["unans_cands"] = [u.model_dump() for u in parse(client, args.model, (
            f"{GUIDE}\n\nTASK: Write {2 * n_unans_needed} questions a student in this course could plausibly "
            "ask about the topics in these files, but that the course files do NOT answer: e.g. a method "
            "variant, parameter, or practical detail the course never covers. Spread them across topics. "
            f"Leave nothing answerable from the files.\n\nCOURSE FILES:\n\n{new_ctx}"
        ), UnanswerableBatch).questions]
        save_ckpt()
    cands = [Unanswerable(**u) for u in ckpt["unans_cands"]]
    unans_qs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=check_workers) as pool:
        for u, r in zip(cands, pool.map(lambda u: check(u.question), cands)):
            if r.answerable:
                print(f"  unanswerable candidate rejected (answered by {r.supporting_files}): {u.question[:70]}",
                      file=sys.stderr)
                continue
            if len(unans_qs) < n_unans_needed:
                unans_qs.append({"question": u.question, "question_type": "unanswerable",
                                 "question_class": "unanswerable", "topics": [], "reference_answer":
                                 "The course material does not cover this; a correct response says so.",
                                 "gold_raw_docs": [], "gold_pages": [], "unanswerable_check": r.explanation,
                                 "origin": "new"})

    new = topic_qs + cross_qs + unans_qs
    for q in new:
        q["gold_annotation"] = f"generator:{args.model}"
    order = {"topic-page": 0, "cross-page": 1, "unanswerable": 2}
    allq = sorted(retained, key=lambda q: order[q["question_class"]]) + \
        sorted(new, key=lambda q: order[q["question_class"]])
    for i, q in enumerate(allq):
        q["id"] = i
        q["source_page"] = q["gold_pages"][0] if q["question_class"] == "topic-page" else f"({q['question_class']})"
        q["review_status"] = "pending"
    Path(args.out).write_text(json.dumps(allq, indent=2), encoding="utf-8")
    counts = {(o, c): sum(q["question_class"] == c and q["origin"].startswith(o) for q in allq)
              for o in ("retained", "new") for c in order}
    print(f"Wrote {len(allq)} questions -> {args.out}  {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
