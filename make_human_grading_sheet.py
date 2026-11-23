#!/usr/bin/env python
"""
Build a BLIND human-grading workbook to validate the LLM judge.

    python make_human_grading_sheet.py            # -> results/human_grading/grading_sheet.xlsx + key.csv
    python make_human_grading_sheet.py --force    # overwrite an existing sheet (destroys any grades in it!)

The grader sees only: question class, question, reference answer, gold source files, and the
response. Condition, budget, and every judge field are hidden in key.csv, which the grader
should NOT open until grading is done. Items are shuffled.

Sample (seeded, ~82 items):
  enriched  every response the judge marked non-grounded or scored < 9 in a grounded condition
            at the primary budget, plus up to 10 each from the k=5 A-ret / B-nav run
  c0        10 closed-book responses: 5 the judge called fabrication, 5 it called grounded
  random    10 per grounded condition (A-ret, A-full, B-nav, B-full) at the primary budget, drawn
            from responses the judge rated grounded with score >= 9, stratified by question class
The random stratum estimates agreement on typical responses; the enriched stratum checks the
judge's negative calls, which drive the paper's groundedness and fabrication findings.

Score the filled sheet with: python score_human_grading.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

SEED = 20261120
OUT_DIR = Path("results/human_grading")
QUESTIONS = Path("course_questions_raw.json")
RUNS = {"k10": Path("results"), "k5": Path("results/budget7500")}
GROUNDED_CONDS = ["A-ret", "A-full", "B-nav", "B-full"]
N_RANDOM_PER_COND = 10
N_K5_PER_COND = 10
N_C0_EACH = 5

GRADE_COLS = ["Response type", "Score (1-10)", "Grounded in course material?",
              "Says it goes beyond the course?", "Notes"]

INSTRUCTIONS = """HUMAN GRADING: validating the LLM judge

Each row is one tutor response to a course question. The condition that produced it is hidden.
Grade each response on its own, using the same rubric the LLM judge used (below).
Do NOT open key.csv until you have finished: it holds the hidden conditions and the judge's grades.

You do not have to grade every row. Partial sheets are scored on the rows you completed, but
grading rows in order (top to bottom) keeps the sample balanced.

STEP 1 - Response type
  answer   = gives a substantive answer, even a partial one or one with caveats
  refusal  = says the material does not contain the answer, that it cannot answer, or declines
  If you choose refusal, leave the other grade columns blank.

STEP 2 - only for answers
  Score (1-10), against the course material and the reference answer:
    10   fully correct and thoroughly grounded in the course material
    8-9  correct, only trivial omissions
    6-7  correct but missing meaningful detail or nuance
    4-5  partially correct, or correct but poorly supported
    2-3  mostly incorrect or unsupported by the course material
    1    wrong, fabricated, or contradicts the course material
    Use the full range.

  Grounded in course material? (yes/no)
    yes only if EVERY substantive claim is stated somewhere in the course material
    (slides, notebooks, readings, syllabus, wiki). A claim that is correct in general but
    not in the course material makes the answer NOT grounded. Citations in the response
    (raw/course/... paths or wiki pages) tell you where to check; gold source files are listed.

  Says it goes beyond the course? (yes/no)
    yes only if the response explicitly tells the student that some of what it says is not
    covered by the course material (e.g. "the course doesn't cover this, but in general...").

For unanswerable questions (class = unanswerable), the right response says the material does
not cover the question. If it does that and then offers only related course content, that is
a grounded answer.

Notes: anything that would explain a disagreement (e.g. "claim X is in the GUIDE reading").

Source files are under raw/course/ in the repository; the wiki is under wiki/.
"""


def load_rows(run: str, cond: str) -> list[dict]:
    p = RUNS[run] / f"{cond}.jsonl"
    return [json.loads(line) for line in p.open(encoding="utf-8")] if p.exists() else []


def flagged(r: dict) -> bool:
    return r.get("outcome") != "grounded_answer" or (r.get("score") or 0) < 9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sheet, key = OUT_DIR / "grading_sheet.xlsx", OUT_DIR / "key.csv"
    if sheet.exists() and not args.force:
        sys.exit(f"{sheet} exists (it may hold your grades). Use --force to overwrite.")
    rng = random.Random(SEED)
    qs = {q["id"]: q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}

    items: list[tuple[str, str, dict]] = []  # (stratum, run, row)
    for cond in GROUNDED_CONDS:
        rows = load_rows("k10", cond)
        items += [("enriched", "k10", r) for r in rows if flagged(r)]
        pool = [r for r in rows if not flagged(r)]
        by_cls: dict[str, list] = {}
        for r in pool:
            by_cls.setdefault(r["question_class"], []).append(r)
        # stratified by class, proportional: 45/45/10 -> about 4/4/2 of 10
        take = {"topic-page": 4, "cross-page": 4, "unanswerable": 2}
        for cls, n in take.items():
            items += [("random", "k10", r) for r in rng.sample(by_cls.get(cls, []), min(n, len(by_cls.get(cls, []))))]
    for cond in ["A-ret", "B-nav"]:
        rows = [r for r in load_rows("k5", cond) if flagged(r)]
        items += [("enriched", "k5", r) for r in rng.sample(rows, min(N_K5_PER_COND, len(rows)))]
    c0 = load_rows("k10", "C0")
    for outcome in ["fabrication", "grounded_answer"]:
        pool = [r for r in c0 if r.get("outcome") == outcome]
        items += [("c0", "k10", r) for r in rng.sample(pool, min(N_C0_EACH, len(pool)))]
    rng.shuffle(items)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ins = wb.active
    ins.title = "Instructions"
    for i, line in enumerate(INSTRUCTIONS.splitlines(), 1):
        ins.cell(row=i, column=1, value=line)
    ins.column_dimensions["A"].width = 110

    ws = wb.create_sheet("Grade")
    shown = ["Item", "Question class", "Question", "Reference answer", "Gold source files", "Response"]
    ws.append(shown + GRADE_COLS)
    head_fill = PatternFill("solid", fgColor="DDDDDD")
    grade_fill = PatternFill("solid", fgColor="FFF5CC")
    for c in ws[1]:
        c.font, c.fill = Font(bold=True), head_fill
    widths = [7, 13, 45, 50, 40, 90, 12, 10, 14, 14, 40]
    for col, w in zip("ABCDEFGHIJK", widths):
        ws.column_dimensions[col].width = w

    key_rows = []
    for n, (stratum, run, r) in enumerate(items, 1):
        q = qs[r["id"]]
        item = f"H{n:03d}"
        ws.append([item, q["question_class"], q["question"], q.get("reference_answer", ""),
                   "\n".join(q.get("gold_raw_docs", [])) or "(none: unanswerable)",
                   r["answer"]] + [None] * len(GRADE_COLS))
        key_rows.append({"item": item, "stratum": stratum, "run": run, "condition": r["condition"],
                         "id": r["id"], "question_class": r["question_class"],
                         "gold_in_context": r.get("gold_in_context"),
                         "judge_response_type": r.get("response_type"), "judge_score": r.get("score"),
                         "judge_grounded": r.get("grounded"), "judge_flags_missing": r.get("flags_missing"),
                         "judge_outcome": r.get("outcome"), "judge_rationale": r.get("rationale", "")})
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for c in row[6:]:
            c.fill = grade_fill
    ws.freeze_panes = "C2"
    last = len(items) + 1
    for rng_ref, opts in [(f"G2:G{last}", '"answer,refusal"'), (f"I2:I{last}", '"yes,no"'),
                          (f"J2:J{last}", '"yes,no"')]:
        dv = DataValidation(type="list", formula1=opts, allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(rng_ref)
    dv = DataValidation(type="whole", operator="between", formula1="1", formula2="10", allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"H2:H{last}")
    wb.save(sheet)

    with key.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)
    counts: dict[str, int] = {}
    for s, run, r in items:
        k = f"{s}:{r['condition']}{'@k5' if run == 'k5' else ''}"
        counts[k] = counts.get(k, 0) + 1
    print(f"Wrote {len(items)} items -> {sheet}\nKey (do not open while grading) -> {key}")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()
