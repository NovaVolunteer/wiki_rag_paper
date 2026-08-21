"""Bootstrap error distributions for the wiki-arm scores.

What this does and does not estimate
------------------------------------
Resampling the 59 questions with replacement estimates ONE source of error:
question-sampling variability -- "if I had drawn a different set of questions from
the same generator, how much would the category mean move?"

It does NOT estimate:
  - judge stochasticity (same answer, re-graded, may score differently)
  - answerer stochasticity (same question, re-answered, may differ)
Both need repeated RUNS, not resampling of a single run. The prior paper's
"10 bootstrapped rounds" conflates these; worth separating in this one.

Outputs percentile and BCa CIs per category, plus bootstrap distributions of
pairwise differences (more informative than eyeballing CI overlap), and a
detectability floor for the Arm A comparison.
"""
from __future__ import annotations

import json
import itertools
import numpy as np
from scipy.stats import bootstrap

RNG = np.random.default_rng(20260810)
B = 20000

rows = [json.loads(l) for l in open("course_eval_results.jsonl", encoding="utf-8") if l.strip()]
rows = [r for r in rows if "score" in r]

def group(key):
    d: dict[str, list[int]] = {}
    for r in rows:
        d.setdefault(r[key], []).append(r["score"])
    return {k: np.array(v, dtype=float) for k, v in d.items()}


def ci(data: np.ndarray, stat=np.mean, method="BCa"):
    """Returns (point, lo, hi, se). Falls back to percentile if BCa degenerates
    (which it does when the resample distribution has zero variance)."""
    point = float(stat(data))
    if len(np.unique(data)) == 1:
        return point, point, point, 0.0
    try:
        res = bootstrap((data,), stat, n_resamples=B, method=method,
                        random_state=RNG, vectorized=False)
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        se = float(res.standard_error)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError
    except Exception:
        res = bootstrap((data,), stat, n_resamples=B, method="percentile",
                        random_state=RNG, vectorized=False)
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        se = float(res.standard_error)
    return point, lo, hi, se


def boot_means(data: np.ndarray, b=B) -> np.ndarray:
    idx = RNG.integers(0, len(data), size=(b, len(data)))
    return data[idx].mean(axis=1)


def report(title: str, groups: dict[str, np.ndarray]):
    print(f"\n{'=' * 86}\n{title}\n{'=' * 86}")
    print(f"{'category':26s} {'n':>3s} {'mean':>6s} {'95% CI (BCa)':>18s} {'SE':>6s} {'P(>=9)':>8s}")
    print("-" * 86)
    for k in sorted(groups, key=lambda k: -groups[k].mean()):
        v = groups[k]
        m, lo, hi, se = ci(v)
        p, plo, phi, _ = ci(v, lambda x: float(np.mean(np.asarray(x) >= 9)))
        print(f"{k:26s} {len(v):3d} {m:6.2f}  [{lo:5.2f}, {hi:5.2f}] {se:6.3f} "
              f"  {p:.2f} [{plo:.2f},{phi:.2f}]")


def pairwise(groups: dict[str, np.ndarray], label: str):
    print(f"\n--- pairwise differences in mean, {label} "
          f"(bootstrap, {B:,} resamples) ---")
    print(f"{'contrast':40s} {'diff':>6s} {'95% CI':>18s} {'P(diff>0)':>10s}")
    print("-" * 86)
    keys = sorted(groups, key=lambda k: -groups[k].mean())
    for a, b_ in itertools.combinations(keys, 2):
        da, db = boot_means(groups[a]), boot_means(groups[b_])
        diff = da - db
        lo, hi = np.percentile(diff, [2.5, 97.5])
        obs = groups[a].mean() - groups[b_].mean()
        pgt = float((diff > 0).mean())
        flag = "" if (lo <= 0 <= hi) else "  <-- CI excludes 0"
        print(f"{a + ' - ' + b_:40s} {obs:6.2f}  [{lo:5.2f}, {hi:5.2f}] {pgt:10.3f}{flag}")


all_scores = np.array([r["score"] for r in rows], dtype=float)
m, lo, hi, se = ci(all_scores)
p, plo, phi, _ = ci(all_scores, lambda x: float(np.mean(np.asarray(x) >= 9)))
print("=" * 86)
print(f"OVERALL  n={len(all_scores)}  mean={m:.2f}  95% BCa CI [{lo:.2f}, {hi:.2f}]  SE={se:.3f}")
print(f"         P(score>=9) = {p:.3f}  95% CI [{plo:.3f}, {phi:.3f}]")

by_type = group("question_type")
by_class = group("question_class")
report("BY QUESTION TYPE", by_type)
pairwise(by_type, "question type")
report("BY QUESTION CLASS", by_class)
pairwise(by_class, "question class")

# ---------------------------------------------------------------------------
# Detectability floor: how much worse would Arm A have to be to be separable?
# ---------------------------------------------------------------------------
print(f"\n{'=' * 86}\nDETECTABILITY FLOOR FOR ARM A\n{'=' * 86}")
print("Simulating: Arm A scores drawn from the wiki distribution shifted down by d,")
print("then testing whether a bootstrap CI on the paired difference excludes 0.")
print(f"{'cell':22s} {'n':>3s} " + " ".join(f"{d:>6.1f}" for d in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]))
print("-" * 86)
SIMS = 600
for label, v in [("overall", all_scores)] + [(f"type={k}", by_type[k]) for k in sorted(by_type)]:
    row = []
    for d in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        hits = 0
        for _ in range(SIMS):
            arm_b = RNG.choice(v, size=len(v), replace=True)
            arm_a = np.clip(RNG.choice(v, size=len(v), replace=True) - d, 1, 10)
            paired = arm_b - arm_a
            bs = paired[RNG.integers(0, len(paired), size=(2000, len(paired)))].mean(axis=1)
            l, h = np.percentile(bs, [2.5, 97.5])
            hits += (l > 0)
        row.append(hits / SIMS)
    print(f"{label:22s} {len(v):3d} " + " ".join(f"{x:6.2f}" for x in row))
print("\n(values = power: proportion of simulations where the 95% CI on the")
print(" wiki-minus-RAG difference excludes zero. 0.80 is the conventional target.)")

# Save for plotting
out = {
    "overall": {"n": len(all_scores), "mean": m, "lo": lo, "hi": hi},
    "by_type": {k: dict(zip(("n", "mean", "lo", "hi", "se"),
                            (len(v),) + ci(v))) for k, v in by_type.items()},
    "boot_type": {k: boot_means(v).tolist() for k, v in by_type.items()},
    "scores": all_scores.tolist(),
}
json.dump(out, open("bootstrap_out.json", "w"))
print("\n-> bootstrap_out.json")
