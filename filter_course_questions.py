"""Filter the 145-question set down to genuine classroom-QA questions.

Kept:
  - all questions sourced from the three DS-3021 topic pages
  - cross-page questions whose ANSWER is course subject matter

Dropped:
  - questions about wiki scaffolding (index / status / getting-started)
  - questions about the pilot itself (course/README, course/pilot-plan) -- these
    test wiki-governance knowledge, not course knowledge
  - cross-page questions about pilot design, ingest workflow, or ATA workstreams
"""
import json
from pathlib import Path

COURSE_PAGES = {"course/knn.md", "course/decision-trees.md", "course/ml-bias.md"}

# Cross-page questions kept as course content. Judgment calls flagged inline.
CROSS_PAGE_KEEP = {
    120,  # KNN scaling vs DT no-scaling -- algorithm properties
    121,  # confusion matrix -> Equality of Odds / Equal Opportunity
    122,  # class imbalance framed across KNN / DT / bias
    123,  # which weeks explain Amazon + Google bias cases
    124,  # sensitivity/specificity vs TPR formulas
    125,  # JUDGMENT CALL: canonical DT deck. Meta-flavoured, but it is the
          # canonical "currency" question type from pilot-plan and the subject
          # is course material versions -- kept as a currency probe.
    126,  # nested CV (KNN) vs hyperparameter search (DT)
    127,  # JUDGMENT CALL: framed via pilot plan, but the answer is the sklearn
          # variable-importance call for trees vs KNN permutation -- course content.
    134,  # R->Python conversion tables, missing-data inspection
    136,  # reweighting vs threshold adjustment, DT imbalance advice
    138,  # greediness of DT splitting vs KNN scaling requirement
    141,  # DS-3021 week ordering + pedagogical dependencies
    142,  # k=1 overfit moment vs max_depth/pruning
    144,  # DT vs KNN for interpretability and fairness auditing
}

# Explicitly dropped cross-page questions, for the record.
CROSS_PAGE_DROP = {
    128: "pilot page-selection rationale",
    129: "getting-started contributor guidance",
    130: "pilot 'known confound'",
    131: "wiki-maintenance flags (unfinished stub, un-ingested topics)",
    132: "ingest process for new material",
    133: "incremental-ingest / compounding hypothesis",
    135: "how the pilot differs from the rest of the wiki",
    137: "pilot plan's own currency claim",
    139: "where to log pilot results",
    140: "ATA vs course domain overlap",
    143: "ATA team workstreams",
}

qs = json.loads(Path("wiki_eval_questions.json").read_text(encoding="utf-8"))

kept, dropped = [], []
for q in qs:
    sp, qid = q["source_page"], q["id"]
    if sp in COURSE_PAGES:
        q["question_class"] = "topic-page"
        kept.append(q)
    elif sp == "(cross-page)" and qid in CROSS_PAGE_KEEP:
        q["question_class"] = "cross-page"
        kept.append(q)
    else:
        dropped.append(q)

Path("course_questions.json").write_text(json.dumps(kept, indent=2), encoding="utf-8")

import collections
print(f"kept    {len(kept):3d}")
print(f"dropped {len(dropped):3d}")
print("\nkept by source page:")
for k, v in collections.Counter(q["source_page"] for q in kept).most_common():
    print(f"  {v:3d}  {k}")
print("\nkept by question type:")
for k, v in sorted(collections.Counter(q["question_type"] for q in kept).items()):
    print(f"  {v:3d}  {k}")
print("\ndropped by source page:")
for k, v in collections.Counter(q["source_page"] for q in dropped).most_common():
    print(f"  {v:3d}  {k}")
print(f"\n-> course_questions.json")
