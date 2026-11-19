#!/usr/bin/env python
"""
Attach gold source pages to every question -> course_questions_gold.json.

Topic-page questions: gold = [source_page] (by construction; annotation "construction").
Cross-page questions: the answerer model, given every concept page, lists the
concept pages a complete answer requires (annotation "model"). These labels
should be human-checked; edit course_questions_gold.json directly and set
gold_annotation to "human" for any you verify or correct.

    python annotate_gold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from eval_conditions import NAV_PAGES
from wiki_qa_eval import DEFAULT_MODEL, build_context, load_wiki_pages

load_dotenv()


class GoldPages(BaseModel):
    gold_pages: list[str]
    rationale: str


def main() -> None:
    pages = dict(load_wiki_pages(Path("wiki")))
    concept = {p: t for p, t in pages.items() if p not in NAV_PAGES}
    context = build_context(list(concept.items()))
    questions = json.loads(Path("course_questions.json").read_text(encoding="utf-8"))
    client = Anthropic()
    out = []
    for q in questions:
        q = dict(q)
        if q["question_class"] == "topic-page":
            q["gold_pages"], q["gold_annotation"] = [q["source_page"]], "construction"
        else:
            resp = client.messages.parse(
                model=DEFAULT_MODEL,
                max_tokens=4000,
                output_config={"effort": "medium"},
                messages=[{"role": "user", "content": (
                    f"Below are the concept pages of a course wiki.\n\n{context}\n\n"
                    f"QUESTION: {q['question']}\n\n"
                    "List the concept pages (by path, exactly as shown after 'START PAGE: wiki/', "
                    "e.g. course/knn.md) whose content is REQUIRED for a complete, correct answer. "
                    "Include a page only if the answer would be missing something without it."
                )}],
                output_format=GoldPages,
            )
            gold = [p.removeprefix("wiki/") for p in resp.parsed_output.gold_pages]
            bad = [p for p in gold if p not in concept]
            if bad:
                print(f"  q{q['id']}: dropping unknown pages {bad}", file=sys.stderr)
            q["gold_pages"] = [p for p in gold if p in concept]
            q["gold_annotation"] = "model"
            q["gold_rationale"] = resp.parsed_output.rationale
            print(f"  q{q['id']}: {q['gold_pages']}", file=sys.stderr)
        out.append(q)
    Path("course_questions_gold.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} questions -> course_questions_gold.json", file=sys.stderr)


if __name__ == "__main__":
    main()
