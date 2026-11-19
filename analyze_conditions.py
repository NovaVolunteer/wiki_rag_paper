#!/usr/bin/env python
"""
Summarize results/<condition>.jsonl for all five conditions.

Per condition x question class (all / topic-page / cross-page):
  n, answer rate, mean score over answered questions, mean score with refusals
  at the floor (1), grounded rate over answered, fabrication rate (over all),
  justified / unjustified refusal rates, gold-in-context rate (mean coverage
  and fraction fully covered), input tokens (mean, sd, min, max), material
  tokens (mpnet), units seen.

Paired bootstrap (resampling questions, B=20,000) 95% CIs for the planned
contrasts: B-nav vs A-ret (primary), B-full vs A-full, B-full vs B-nav, and
each grounded condition vs C0. Only questions completed in both conditions
are paired; n is reported.

Writes results/summary.json and results/RESULTS.md.

    python analyze_conditions.py
"""

from __future__ import annotations

import json
import statistics as st
from datetime import date
from pathlib import Path

import numpy as np

from eval_conditions import CONDITIONS, RESULTS_DIR

B = 20000
CLASSES = ["all", "topic-page", "cross-page", "unanswerable"]
CONTRASTS = [("B-nav", "A-ret"), ("B-full", "A-full"), ("B-full", "B-nav"),
             ("A-ret", "C0"), ("A-full", "C0"), ("B-nav", "C0"), ("B-full", "C0")]


def load(cond: str) -> dict[int, dict]:
    p = RESULTS_DIR / f"{cond}.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: r for r in rows if "error" not in r and "outcome" in r}


def floor_score(r: dict) -> float:
    return float(r["score"]) if r["response_type"] == "answer" else 1.0


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    ans = [r for r in rows if r["response_type"] == "answer"]
    oc = lambda k: sum(r["outcome"] == k for r in rows) / n  # noqa: E731
    toks = [r["input_tokens"] for r in rows]
    mat = [r.get("material_tokens", 0) for r in rows]
    cov = [r.get("gold_in_context", 0.0) for r in rows]
    units = [len(r.get("units_seen", [])) for r in rows]
    return {
        "n": n,
        "answer_rate": len(ans) / n,
        "score_answered": st.mean(r["score"] for r in ans) if ans else None,
        "score_floor": st.mean(floor_score(r) for r in rows),
        "grounded_rate_answered": (sum(bool(r["grounded"]) for r in ans) / len(ans)) if ans else None,
        "fabrication_rate": oc("fabrication"),
        "flagged_ungrounded_rate": oc("flagged_ungrounded"),
        "justified_refusal_rate": oc("justified_refusal"),
        "unjustified_refusal_rate": oc("unjustified_refusal"),
        "refusal_rate": 1 - len(ans) / n,
        "gold_coverage_mean": st.mean(cov),
        "gold_fully_in_context": sum(c >= 1.0 for c in cov) / n,
        "input_tokens": {"mean": st.mean(toks), "sd": st.pstdev(toks), "min": min(toks), "max": max(toks)},
        "material_tokens": {"mean": st.mean(mat), "sd": st.pstdev(mat), "min": min(mat), "max": max(mat)},
        "units_seen_mean": st.mean(units),
    }


def paired(a: dict[int, dict], b: dict[int, dict], cls: str, rng) -> dict | None:
    ids = sorted(i for i in set(a) & set(b) if cls == "all" or a[i]["question_class"] == cls)
    if len(ids) < 2:
        return None
    out = {"n": len(ids)}
    metrics = {
        "score_floor": floor_score,
        "fabrication": lambda r: float(r["outcome"] == "fabrication"),
        "answered": lambda r: float(r["response_type"] == "answer"),
    }
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    for name, f in metrics.items():
        d = np.array([f(a[i]) - f(b[i]) for i in ids])
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = {"diff": float(d.mean()), "ci": [float(lo), float(hi)]}
    return out


def fmt(x, pct=False, nd=2):
    if x is None:
        return "-"
    return f"{x:.0%}" if pct else f"{x:.{nd}f}"


def main() -> None:
    import argparse

    global RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR),
                    help="e.g. results/budget7500 for the k=5 sensitivity run (A-ret, B-nav only)")
    RESULTS_DIR = Path(ap.parse_args().results_dir)
    rng = np.random.default_rng(20261118)
    data = {c: load(c) for c in CONDITIONS}
    summary = {"conditions": {}, "contrasts": {}}
    for c, rows in data.items():
        summary["conditions"][c] = {
            cls: summarize([r for r in rows.values() if cls == "all" or r["question_class"] == cls])
            for cls in CLASSES
        }
    for a, b in CONTRASTS:
        if data[a] and data[b]:
            summary["contrasts"][f"{a} - {b}"] = {cls: paired(data[a], data[b], cls, rng) for cls in CLASSES}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    from eval_conditions import QUESTIONS_FILE

    n_q = len(json.loads(Path(QUESTIONS_FILE).read_text(encoding="utf-8")))
    L = [f"# Five-condition results\n", f"Generated {date.today().isoformat()} by `analyze_conditions.py` "
         "from `results/<condition>.jsonl`. Answerer and judge models are recorded per row.\n",
         "Completed rows per condition: " +
         ", ".join(f"{c} {len(data[c])}/{n_q}" for c in CONDITIONS) + "\n"]
    for cls in CLASSES:
        L.append(f"\n## Question class: {cls}\n")
        L.append("| Condition | n | Answer rate | Score (answered) | Score (refusal=1) | Grounded (answered) "
                 "| Fabrication | Flagged ungrounded | Justified ref. | Unjustified ref. | Gold fully in ctx | Input tok mean (sd) [min-max] | Units seen |")
        L.append("|" + "---|" * 13)
        for c in CONDITIONS:
            s = summary["conditions"][c][cls]
            if not s.get("n"):
                L.append(f"| {c} | 0 | not run |" + " |" * 10)
                continue
            t = s["input_tokens"]
            L.append(
                f"| {c} | {s['n']} | {fmt(s['answer_rate'], True)} | {fmt(s['score_answered'])} | "
                f"{fmt(s['score_floor'])} | {fmt(s['grounded_rate_answered'], True)} | "
                f"{fmt(s['fabrication_rate'], True)} | {fmt(s['flagged_ungrounded_rate'], True)} | "
                f"{fmt(s['justified_refusal_rate'], True)} | "
                f"{fmt(s['unjustified_refusal_rate'], True)} | {fmt(s['gold_fully_in_context'], True)} | "
                f"{t['mean']:,.0f} ({t['sd']:,.0f}) [{t['min']:,}-{t['max']:,}] | {s['units_seen_mean']:.1f} |"
            )
    L.append("\n## Paired contrasts (95% bootstrap CI, questions resampled)\n")
    L.append("| Contrast | Class | n | Score diff (refusal=1) | Fabrication diff | Answer-rate diff |")
    L.append("|---|---|---|---|---|---|")
    for name, by_cls in summary["contrasts"].items():
        for cls, r in by_cls.items():
            if not r:
                continue
            cell = lambda m: f"{r[m]['diff']:+.2f} [{r[m]['ci'][0]:+.2f}, {r[m]['ci'][1]:+.2f}]"  # noqa: E731
            L.append(f"| {name} | {cls} | {r['n']} | {cell('score_floor')} | {cell('fabrication')} | {cell('answered')} |")
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
