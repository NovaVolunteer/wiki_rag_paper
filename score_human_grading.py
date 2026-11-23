#!/usr/bin/env python
"""
Compare human grades in results/human_grading/grading_sheet.xlsx with the LLM judge.

    python score_human_grading.py      # -> results/human_grading/AGREEMENT.md (also printed)

Only rows with a human "Response type" filled in are scored, so a partly graded sheet works.
Reports, overall and by stratum / condition:
  - response-type agreement (answer vs refusal)
  - grounded: % agreement and Cohen's kappa (over items both graders called answers)
  - score: human mean, judge mean, mean difference (judge - human), MAE, % within 1 point,
    Spearman rank correlation
  - outcome agreement (the five-way outcome, derived from the human grades with the same rules
    the pipeline applies to the judge's grades)
and lists every disagreement with the judge's rationale and the human note.

The "random" stratum is the unbiased estimate for typical responses; "enriched" and "c0"
over-sample the judge's negative calls and are reported separately.
"""

from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import load_workbook
from scipy.stats import spearmanr

DIR = Path("results/human_grading")


def yn(v) -> bool | None:
    s = str(v or "").strip().lower()
    return True if s in {"yes", "y", "true", "1"} else False if s in {"no", "n", "false", "0"} else None


def truthy(v) -> bool | None:
    s = str(v).strip().lower()
    return True if s == "true" else False if s == "false" else None


def outcome(rtype, grounded, flags, gold_in_ctx, condition) -> str | None:
    if rtype == "refusal":
        if condition == "C0":
            return "refusal"
        return "unjustified_refusal" if gold_in_ctx >= 0.999 else "justified_refusal"
    if rtype != "answer" or grounded is None:
        return None
    if grounded:
        return "grounded_answer"
    return "flagged_ungrounded" if flags else "fabrication"


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def stats(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    if not rows:
        return out
    out["type_agree"] = sum(r["h_type"] == r["j_type"] for r in rows) / len(rows)
    both = [r for r in rows if r["h_type"] == r["j_type"] == "answer"
            and r["h_grounded"] is not None and r["j_grounded"] is not None]
    g = [(r["h_grounded"], r["j_grounded"]) for r in both]
    out["n_grounded"] = len(g)
    if g:
        out["grounded_agree"] = sum(a == b for a, b in g) / len(g)
        out["kappa"] = kappa(g)
        out["h_grounded_rate"] = sum(a for a, _ in g) / len(g)
        out["j_grounded_rate"] = sum(b for _, b in g) / len(g)
    s = [(r["h_score"], r["j_score"]) for r in both if r["h_score"] is not None and r["j_score"] is not None]
    out["n_score"] = len(s)
    if s:
        h, j = [a for a, _ in s], [b for _, b in s]
        out["h_mean"], out["j_mean"] = sum(h) / len(h), sum(j) / len(j)
        out["diff"] = out["j_mean"] - out["h_mean"]
        out["mae"] = sum(abs(a - b) for a, b in s) / len(s)
        out["within1"] = sum(abs(a - b) <= 1 for a, b in s) / len(s)
        out["spearman"] = spearmanr(h, j).statistic if len(set(h)) > 1 and len(set(j)) > 1 else None
    o = [r for r in rows if r["h_outcome"] is not None]
    if o:
        out["outcome_agree"] = sum(r["h_outcome"] == r["j_outcome"] for r in o) / len(o)
    return out


def fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{100 * v:.0f}%" if pct else f"{v:.2f}"


def main() -> None:
    key = {r["item"]: r for r in csv.DictReader((DIR / "key.csv").open(encoding="utf-8"))}
    ws = load_workbook(DIR / "grading_sheet.xlsx", data_only=True)["Grade"]
    header = [c.value for c in ws[1]]
    col = {h: i for i, h in enumerate(header)}
    rows = []
    for vals in ws.iter_rows(min_row=2, values_only=True):
        item = vals[col["Item"]]
        h_type = str(vals[col["Response type"]] or "").strip().lower()
        if not item or h_type not in {"answer", "refusal"}:
            continue
        k = key[item]
        score = vals[col["Score (1-10)"]]
        r = {"item": item, "stratum": k["stratum"], "cond": k["condition"] + ("@k5" if k["run"] == "k5" else ""),
             "id": k["id"], "cls": k["question_class"], "h_type": h_type,
             "h_score": float(score) if h_type == "answer" and score not in (None, "") else None,
             "h_grounded": yn(vals[col["Grounded in course material?"]]) if h_type == "answer" else None,
             "h_flags": yn(vals[col["Says it goes beyond the course?"]]) if h_type == "answer" else None,
             "note": vals[col["Notes"]] or "", "j_type": k["judge_response_type"],
             "j_score": float(k["judge_score"]) if k["judge_score"] not in ("", "None") else None,
             "j_grounded": truthy(k["judge_grounded"]), "j_outcome": k["judge_outcome"],
             "rationale": k["judge_rationale"]}
        gic = float(k["gold_in_context"] or 0)
        r["h_outcome"] = outcome(h_type, r["h_grounded"], r["h_flags"], gic, k["condition"])
        rows.append(r)

    L = ["# Human vs LLM-judge agreement", "",
         f"Graded items: {len(rows)} of {len(key)}. Judge = gpt-5-mini. Grounded and score statistics are "
         "computed over items that both graders called answers.", ""]
    L += ["| Group | n | Type agree | Grounded agree | Cohen's kappa | Grounded (human / judge) | "
          "Score human / judge | Judge - human | MAE | Within 1 | Spearman | Outcome agree |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    groups = [("all", rows)]
    groups += [(f"stratum: {s}", [r for r in rows if r["stratum"] == s]) for s in ["random", "enriched", "c0"]]
    groups += [(f"condition: {c}", [r for r in rows if r["cond"] == c]) for c in sorted({r["cond"] for r in rows})]
    for name, rs in groups:
        s = stats(rs)
        if not s["n"]:
            continue
        L.append(f"| {name} | {s['n']} | {fmt(s.get('type_agree'), 1)} | {fmt(s.get('grounded_agree'), 1)} | "
                 f"{fmt(s.get('kappa'))} | {fmt(s.get('h_grounded_rate'), 1)} / {fmt(s.get('j_grounded_rate'), 1)} | "
                 f"{fmt(s.get('h_mean'))} / {fmt(s.get('j_mean'))} | {fmt(s.get('diff'))} | {fmt(s.get('mae'))} | "
                 f"{fmt(s.get('within1'), 1)} | {fmt(s.get('spearman'))} | {fmt(s.get('outcome_agree'), 1)} |")
    dis = [r for r in rows if r["h_type"] != r["j_type"] or (r["h_grounded"] is not None and r["j_grounded"] is not None
           and r["h_grounded"] != r["j_grounded"]) or (r["h_score"] is not None and r["j_score"] is not None
           and abs(r["h_score"] - r["j_score"]) >= 2)]
    L += ["", f"## Disagreements ({len(dis)})", "",
          "Response type differs, grounded differs, or scores differ by 2 or more.", ""]
    for r in sorted(dis, key=lambda r: r["item"]):
        L.append(f"- **{r['item']}** ({r['cond']}, Q{r['id']}, {r['cls']}): human {r['h_type']}/"
                 f"{r['h_score']}/grounded={r['h_grounded']} vs judge {r['j_type']}/{r['j_score']}/"
                 f"grounded={r['j_grounded']} ({r['j_outcome']}). Human note: {r['note'] or '(none)'}. "
                 f"Judge: {r['rationale'][:300]}")
    text = "\n".join(L) + "\n"
    (DIR / "AGREEMENT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
