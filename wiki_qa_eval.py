#!/usr/bin/env python
"""
Generate a question set from wiki/, answer each question using the wiki as
context, and grade each answer with an LLM judge for groundedness/correctness.

Usage:
    python wiki_qa_eval.py                          # generate + answer + grade
    python wiki_qa_eval.py --limit 10                # quick smoke test
    python wiki_qa_eval.py --questions-file q.json    # reuse a saved question set
    python wiki_qa_eval.py --no-grade                 # just collect answers

Needs ANTHROPIC_API_KEY (question generation + answering) and OPENAI_API_KEY
(grading judge) in the environment or a .env file (project root or wiki/.env
both work — python-dotenv searches upward from the cwd). The judge uses a
different provider (OpenAI) than the answerer (Claude) on purpose, so the
same model isn't grading its own homework.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from pathlib import Path
from typing import Any, Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # only required when grading is enabled

load_dotenv()

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_GRADE_MODEL = "gpt-5-mini"


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------


class GeneratedQuestion(BaseModel):
    question: str
    question_type: Literal["factual", "conceptual", "synthesis", "currency"]


class QuestionBatch(BaseModel):
    questions: list[GeneratedQuestion]


class GradedAnswer(BaseModel):
    score: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    grounded: bool
    rationale: str


class Question(BaseModel):
    id: int
    question: str
    question_type: str
    source_page: str  # "(cross-page)" for synthesis questions spanning pages


# --------------------------------------------------------------------------
# Wiki loading
# --------------------------------------------------------------------------


def load_wiki_pages(wiki_dir: Path) -> list[tuple[str, str]]:
    pages = [
        (p.relative_to(wiki_dir).as_posix(), p.read_text(encoding="utf-8"))
        for p in sorted(wiki_dir.rglob("*.md"))
    ]
    if not pages:
        raise SystemExit(f"No .md files found under {wiki_dir}")
    return pages


def build_context(pages: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"--- START PAGE: wiki/{rel} ---\n{content}\n--- END PAGE: wiki/{rel} ---"
        for rel, content in pages
    )


# --------------------------------------------------------------------------
# Question generation
# --------------------------------------------------------------------------


def generate_page_questions(
    client: Anthropic, model: str, effort: str, rel_path: str, content: str, n: int
) -> list[QuestionBatch]:
    prompt = (
        "You are building an evaluation set to test a wiki-based Q&A system. "
        f"Here is one page from the wiki, at path `wiki/{rel_path}`:\n\n"
        f"---\n{content}\n---\n\n"
        f"Generate {n} distinct questions a real user of this wiki might ask, "
        "answerable starting from this page (a good answer may also need to "
        "follow a [[link]] to another page — that's fine, as long as this page "
        "is the natural starting point). Mix question types:\n"
        "- factual: a specific fact or detail stated on the page\n"
        "- conceptual: a why/how question requiring understanding, not just lookup\n"
        "- synthesis: a question that connects this page to something it links "
        "to or that links to it\n\n"
        "Avoid trivial, overly generic, or yes/no questions."
    )
    resp = client.messages.parse(
        model=model,
        max_tokens=4096,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
        output_format=QuestionBatch,
    )
    return resp.parsed_output.questions


def generate_cross_page_questions(
    client: Anthropic, model: str, effort: str, context: str, n: int
) -> list[QuestionBatch]:
    prompt = (
        "You are building an evaluation set to test a wiki-based Q&A system. "
        "Below is the FULL content of the wiki (every page).\n\n"
        f"{context}\n\n"
        f"Generate {n} distinct questions that specifically require connecting "
        "information from TWO OR MORE pages to answer well — a question a "
        "single page alone could not fully answer. Prefer questions that test "
        "whether the answerer actually followed the wiki's [[links]] between "
        "pages, not just questions about one page in isolation."
    )
    resp = client.messages.parse(
        model=model,
        max_tokens=4096,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
        output_format=QuestionBatch,
    )
    return resp.parsed_output.questions


def build_question_set(
    client: Anthropic,
    model: str,
    effort: str,
    pages: list[tuple[str, str]],
    context: str,
    per_page: int,
    cross_page: int,
) -> list[Question]:
    questions: list[Question] = []
    for rel_path, content in pages:
        print(f"  generating {per_page} questions for wiki/{rel_path} ...", file=sys.stderr)
        for gq in generate_page_questions(client, model, effort, rel_path, content, per_page):
            questions.append(
                Question(id=0, question=gq.question, question_type=gq.question_type, source_page=rel_path)
            )
    if cross_page > 0:
        print(f"  generating {cross_page} cross-page synthesis questions ...", file=sys.stderr)
        for gq in generate_cross_page_questions(client, model, effort, context, cross_page):
            questions.append(
                Question(id=0, question=gq.question, question_type=gq.question_type, source_page="(cross-page)")
            )
    for i, q in enumerate(questions):
        q.id = i
    return questions


# --------------------------------------------------------------------------
# Answering + grading
# --------------------------------------------------------------------------


def answer_question(client: Anthropic, model: str, effort: str, system_blocks: list[dict], question: Question) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        output_config={"effort": effort},
        system=system_blocks,
        messages=[{"role": "user", "content": question.question}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def grade_answer(
    openai_client: Any, model: str, context: str, question: Question, answer: str
) -> GradedAnswer:
    """Judge with a different provider than the one that answered, so the
    model isn't grading its own homework."""
    prompt = (
        "You are grading an AI-generated answer against source material for accuracy.\n\n"
        f"SOURCE MATERIAL (the wiki):\n{context}\n\n"
        f"QUESTION: {question.question}\n\n"
        f"ANSWER TO GRADE: {answer}\n\n"
        "Score the answer from 1 to 10:\n"
        "10 = fully correct and thoroughly grounded in the source material\n"
        "8-9 = correct, only trivial omissions\n"
        "6-7 = correct but missing meaningful detail or nuance\n"
        "4-5 = partially correct, or correct but poorly supported/cited\n"
        "2-3 = mostly incorrect or unsupported by the source material\n"
        "1 = wrong, fabricated, or contradicts the source material\n\n"
        "Set `grounded` to true only if every claim in the answer traces back "
        "to something actually stated in the source material (no fabrication)."
    )
    completion = openai_client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=GradedAnswer,
    )
    return completion.choices[0].message.parsed


def process_question(
    client: Anthropic,
    openai_client: Any,
    answer_model: str,
    grade_model: str,
    effort: str,
    system_blocks: list[dict],
    context: str,
    question: Question,
    do_grade: bool,
) -> dict:
    result = {
        "id": question.id,
        "question": question.question,
        "question_type": question.question_type,
        "source_page": question.source_page,
    }
    try:
        result["answer"] = answer_question(client, answer_model, effort, system_blocks, question)
    except Exception as e:  # noqa: BLE001 - log and keep going
        result["answer"] = ""
        result["error"] = f"answer failed: {e}"
        return result

    if do_grade:
        try:
            graded = grade_answer(openai_client, grade_model, context, question, result["answer"])
            result["score"] = graded.score
            result["grounded"] = graded.grounded
            result["rationale"] = graded.rationale
        except Exception as e:  # noqa: BLE001
            result["error"] = f"grading failed: {e}"
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wiki-dir", default="wiki", help="Path to the wiki directory (default: wiki)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model for question generation + answering (default: {DEFAULT_MODEL})")
    parser.add_argument("--grade-model", default=DEFAULT_GRADE_MODEL, help=f"OpenAI model for grading (default: {DEFAULT_GRADE_MODEL})")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--questions-per-page", type=int, default=15, help="Generated questions per wiki page (default: 15)")
    parser.add_argument("--cross-page-questions", type=int, default=25, help="Generated cross-page synthesis questions (default: 25)")
    parser.add_argument("--questions-file", default=None, help="Load questions from a JSON file instead of generating them")
    parser.add_argument("--save-questions", default="wiki_eval_questions.json", help="Where to save the generated question set")
    parser.add_argument("--output", default="wiki_eval_results", help="Output file basename (writes <output>.jsonl and <output>.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (for quick testing)")
    parser.add_argument("--max-workers", type=int, default=5, help="Concurrent API requests (default: 5)")
    parser.add_argument("--no-grade", action="store_true", help="Skip LLM-judge grading; just collect answers")
    args = parser.parse_args()

    do_grade = not args.no_grade

    client = Anthropic()
    openai_client = None
    if do_grade:
        if OpenAI is None:
            raise SystemExit("Grading needs the `openai` package: pip install openai (or pass --no-grade)")
        openai_client = OpenAI()  # reads OPENAI_API_KEY

    wiki_dir = Path(args.wiki_dir)
    pages = load_wiki_pages(wiki_dir)
    context = build_context(pages)
    print(f"Loaded {len(pages)} wiki pages ({len(context):,} chars).", file=sys.stderr)

    if args.questions_file:
        raw = json.loads(Path(args.questions_file).read_text(encoding="utf-8"))
        questions = [Question(**q) for q in raw]
        print(f"Loaded {len(questions)} questions from {args.questions_file}.", file=sys.stderr)
    else:
        print("Generating questions from the wiki ...", file=sys.stderr)
        questions = build_question_set(
            client, args.model, args.effort, pages, context, args.questions_per_page, args.cross_page_questions
        )
        Path(args.save_questions).write_text(
            json.dumps([q.model_dump() for q in questions], indent=2), encoding="utf-8"
        )
        print(f"Generated {len(questions)} questions -> {args.save_questions}", file=sys.stderr)

    if args.limit:
        questions = questions[: args.limit]

    system_blocks = [
        {
            "type": "text",
            "text": (
                "You are answering questions using ONLY the wiki content below. "
                "Cite the wiki page(s) that support your answer (e.g. "
                "`wiki/course/knn.md`). If the wiki doesn't contain the answer, "
                "say so explicitly rather than guessing or using outside "
                "knowledge.\n\n" + context
            ),
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]

    print(f"Answering {len(questions)} questions with {args.max_workers} workers ...", file=sys.stderr)
    jsonl_path = Path(f"{args.output}.jsonl")
    results: list[dict] = []
    write_lock = threading.Lock()

    def run_one(q: Question) -> dict:
        r = process_question(
            client, openai_client, args.model, args.grade_model, args.effort, system_blocks, context, q, do_grade
        )
        with write_lock:
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
        return r

    with jsonl_path.open("w", encoding="utf-8"):
        pass  # truncate/create

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, q) for q in questions]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(fut.result())
            if i % 10 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} done", file=sys.stderr)

    results.sort(key=lambda r: r["id"])

    csv_path = Path(f"{args.output}.csv")
    fieldnames = ["id", "question_type", "source_page", "question", "answer", "score", "grounded", "rationale", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nWrote {len(results)} results -> {jsonl_path} , {csv_path}", file=sys.stderr)

    if do_grade:
        scored = [r["score"] for r in results if "score" in r]
        errored = [r for r in results if "error" in r]
        if scored:
            avg = sum(scored) / len(scored)
            grounded_rate = sum(1 for r in results if r.get("grounded")) / len(scored)
            dist = {n: scored.count(n) for n in range(1, 11)}
            print(f"\n--- Summary ---", file=sys.stderr)
            print(f"Graded: {len(scored)}/{len(results)}", file=sys.stderr)
            print(f"Average score: {avg:.2f}/10", file=sys.stderr)
            print(f"Grounded rate: {grounded_rate:.0%}", file=sys.stderr)
            print(f"Score distribution: {dist}", file=sys.stderr)
            worst = sorted((r for r in results if "score" in r), key=lambda r: r["score"])[:5]
            print("\nLowest-scoring questions:", file=sys.stderr)
            for r in worst:
                print(f"  [{r['score']}] {r['question']}  ({r['source_page']})", file=sys.stderr)
        if errored:
            print(f"\n{len(errored)} questions hit an error — see {jsonl_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
