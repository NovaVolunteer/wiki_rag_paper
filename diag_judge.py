"""Diagnose why grade_answer returns a constant score."""
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()
MODEL = "gpt-5-mini"

CONTEXT = """--- START PAGE: wiki/course/knn.md ---
# KNN (K-Nearest Neighbors)
Week 6 of DS-3021. A distance-based classifier: to predict a new point's class,
look at its `k` nearest neighbors in feature space and take a majority vote.
Feature scaling is mandatory because classification depends on distance. The course
uses MinMax scaling. The lecture builds a chooseK() function testing odd k from 1 to 21.
In the bank-marketing example, k=1 came out best. The dataset has ~11.6% positive cases.
--- END PAGE: wiki/course/knn.md ---"""

QUESTION = "What k value did the in-class KNN example land on, and why is that notable?"

GOOD = ("k=1 came out best in the bank-marketing example (wiki/course/knn.md). That's "
        "notable because k=1 is the most overfit choice possible - every point is its own "
        "nearest neighbor - so best-on-sample accuracy isn't the same as best generalization.")

TERRIBLE = ("The example landed on k=47, chosen via the elbow method on a silhouette plot. "
            "This is notable because 47 is prime, which guarantees no tie votes in multiclass "
            "settings. The dataset was the Iris dataset with 60% positive cases, and the course "
            "recommends StandardScaler over MinMax scaling for this reason.")


# ---- Variant A: the current schema (score first, Literal enum) ----
class GradedAnswerA(BaseModel):
    score: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    grounded: bool
    rationale: str


# ---- Variant B: rationale first, then score (reason before committing) ----
class GradedAnswerB(BaseModel):
    rationale: str
    grounded: bool
    score: int = Field(ge=1, le=10)


PROMPT = (
    "You are grading an AI-generated answer against source material for accuracy.\n\n"
    "SOURCE MATERIAL (the wiki):\n{ctx}\n\n"
    "QUESTION: {q}\n\n"
    "ANSWER TO GRADE: {a}\n\n"
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


def run(schema, label, answer, aname):
    try:
        c = client.chat.completions.parse(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT.format(ctx=CONTEXT, q=QUESTION, a=answer)}],
            response_format=schema,
        )
        p = c.choices[0].message.parsed
        if p is None:
            print(f"{label:12s} {aname:9s} -> parsed=None  refusal={c.choices[0].message.refusal} "
                  f"finish={c.choices[0].finish_reason}")
            return
        print(f"{label:12s} {aname:9s} -> score={p.score:<3} grounded={str(p.grounded):5s} "
              f"| {p.rationale[:80]}")
    except Exception as e:
        print(f"{label:12s} {aname:9s} -> EXCEPTION {type(e).__name__}: {str(e)[:200]}")


print("=" * 100)
print("Does the judge discriminate between a correct answer and a fabricated one?")
print("=" * 100)
for schema, label in [(GradedAnswerA, "A: current"), (GradedAnswerB, "B: reason1st")]:
    for answer, aname in [(GOOD, "GOOD"), (TERRIBLE, "TERRIBLE")]:
        run(schema, label, answer, aname)
    print("-" * 100)

# ---- Also inspect what JSON schema each variant actually sends ----
print("\nJSON schema emitted for the current (Literal) score field:")
print(json.dumps(GradedAnswerA.model_json_schema()["properties"]["score"], indent=2)[:500])
