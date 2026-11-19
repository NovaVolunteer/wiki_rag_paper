"""Paired bootstrap CIs for Arm A (vector RAG) vs Arm B (LLM-compiled wiki).

Both arms answered the same 59 questions (course_questions.json), so this
resamples QUESTION IDS with replacement -- the same resampled id set is used
for both arms on each draw, preserving the paired design -- and reports 95%
percentile CIs for each arm's mean score / grounded rate and for the
Wiki-minus-RAG difference, overall and split by question_class.

What this does and does not estimate
-------------------------------------
This resamples the 59 questions -- it estimates question-sampling
variability ("if a different question set had been drawn, how much would
the mean move?"). It does NOT estimate answerer or judge stochasticity
(the same question re-answered or the same answer re-graded may score
differently): that needs repeated RUNS, not resampling of a single run.
Both arms here were each run once.
"""
from __future__ import annotations

import json
import numpy as np

RNG = np.random.default_rng(20260903)
B = 20000

def load(path: str) -> dict[int, dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return {r["id"]: r for r in rows if "score" in r}

rag = load("rag_course_eval_results.jsonl")
wiki = load("course_eval_results.jsonl")
ids = sorted(set(rag) & set(wiki))
assert len(ids) == 59, f"expected 59 paired questions, got {len(ids)}"

def arrays(id_list, field):
    a = np.array([float(rag[i][field]) if field == "score" else float(bool(rag[i][field])) for i in id_list])
    b = np.array([float(wiki[i][field]) if field == "score" else float(bool(wiki[i][field])) for i in id_list])
    return a, b

def paired_ci(id_list, field):
    a, b = arrays(id_list, field)
    n = len(id_list)
    idx = RNG.integers(0, n, size=(B, n))
    boot_a = a[idx].mean(axis=1)
    boot_b = b[idx].mean(axis=1)
    boot_diff = boot_b - boot_a  # wiki minus RAG
    def pct(x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return float(lo), float(hi)
    return {
        "n": n,
        "rag_mean": float(a.mean()), "rag_ci": pct(boot_a),
        "wiki_mean": float(b.mean()), "wiki_ci": pct(boot_b),
        "diff_mean": float(b.mean() - a.mean()), "diff_ci": pct(boot_diff),
        "p_diff_gt_0": float((boot_diff > 0).mean()),
    }

def report(label, id_list):
    print(f"\n{'=' * 78}\n{label}  (n={len(id_list)})\n{'=' * 78}")
    for field, name, fmt in [("score", "Score (1-10)", "{:.2f}"), ("grounded", "Grounded rate", "{:.0%}")]:
        r = paired_ci(id_list, field)
        f = fmt.format
        print(f"{name}:")
        print(f"  RAG (A):  {f(r['rag_mean'])}  95% CI [{f(r['rag_ci'][0])}, {f(r['rag_ci'][1])}]")
        print(f"  Wiki (B): {f(r['wiki_mean'])}  95% CI [{f(r['wiki_ci'][0])}, {f(r['wiki_ci'][1])}]")
        print(f"  Wiki-RAG: {f(r['diff_mean'])}  95% CI [{f(r['diff_ci'][0])}, {f(r['diff_ci'][1])}]"
              f"  P(wiki>RAG)={r['p_diff_gt_0']:.3f}")

report("OVERALL", ids)

by_class: dict[str, list[int]] = {}
for i in ids:
    by_class.setdefault(wiki[i]["question_class"], []).append(i)
for cls, id_list in sorted(by_class.items()):
    report(f"CLASS = {cls}", id_list)

out = {}
out["overall"] = {f: paired_ci(ids, f) for f in ("score", "grounded")}
for cls, id_list in by_class.items():
    out[cls] = {f: paired_ci(id_list, f) for f in ("score", "grounded")}
json.dump(out, open("bootstrap_arms_out.json", "w"), indent=2)
print("\n-> bootstrap_arms_out.json")
