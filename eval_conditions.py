#!/usr/bin/env python
"""
Five-condition evaluation harness (paper Section 3).

    C0      closed book: question only, parametric knowledge
    A-ret   vector RAG (top-k mpnet chunks) over the RAW corpus
    A-full  the whole raw corpus in context (long-context control)
    B-nav   agent starts at wiki/index.md and reads pages with a read_page
            tool under a token budget matched to A-ret (k x chunk tokens)
    B-full  the whole compiled wiki in context (upper bound)

Held constant: answerer model + effort, question set, judge, shared answer
instructions. Every result row logs input tokens, units (chunks/pages) seen,
and whether the gold source was in context. The judge classifies each
response as an answer or a refusal BEFORE scoring; refusals get no 1-10
score and are labelled justified/unjustified from the logged gold coverage
(rule fixed in paper Section 3.5 before any of these runs).

Usage:
    python raw_corpus.py                                  # extract raw text once
    python annotate_gold.py                               # gold pages per question
    python eval_conditions.py --condition A-ret --build-index
    python eval_conditions.py --condition C0 [--limit 3]
    python eval_conditions.py --condition all
    python analyze_conditions.py                          # summary + CIs

Needs ANTHROPIC_API_KEY, OPENAI_API_KEY (judge) and PINECONE_API_KEY (A-ret,
unless --backend local) in the environment or .env.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from raw_corpus import build_raw_context, cited_raw_paths, load_raw_corpus
from wiki_qa_eval import DEFAULT_GRADE_MODEL, DEFAULT_MODEL, build_context, load_wiki_pages

load_dotenv()

CONDITIONS = ["C0", "A-ret", "A-full", "B-nav", "B-full"]
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_TOKENS = 1500
CHUNK_OVERLAP = 100
# Primary budget (decided 2026-11-19, before any full-course results): k=10, so index + ~3 median
# concept pages fit, and k is ~5% of the 203-chunk corpus (k=5 was ~6% of the 80-chunk pilot corpus).
# Sensitivity run: --top-k 5 --results-dir results/budget7500 --condition A-ret / B-nav.
TOP_K = 10
NAV_BUDGET = TOP_K * CHUNK_TOKENS  # B-nav budget matched to A-ret's retrieved-context budget
MAX_NAV_TURNS = 12
PINECONE_INDEX = "llm-wiki-rag"
PINECONE_NAMESPACE = "raw-v2"  # full course (16 topics, DS-3021 + MLBook); raw-v1 = the 7-topic pilot corpus
LOCAL_INDEX_PATH = "rag_raw_local_index.npz"
QUESTIONS_FILE = "course_questions_raw.json"
RESULTS_DIR = Path("results")

# Navigational / meta pages: not concept content. Never gold; B-nav may still read them.
NAV_PAGES = {"index.md", "getting-started.md", "status.md", "course/README.md", "course/pilot-plan.md"}

GROUNDED_INSTRUCTIONS = (
    "You are a tutor for a university machine learning course (DS-3021), answering a "
    "student's question. Answer using ONLY the course material {where}. Cite the "
    "file(s) that support each claim (e.g. `wiki/course/knn.md` or "
    "`raw/course/DS-3021/knn/...`). If the material does not contain the answer, say "
    "so explicitly rather than guessing or using outside knowledge."
)
CLOSED_BOOK_INSTRUCTIONS = (
    "You are a tutor for a university machine learning course (DS-3021), answering a "
    "student's question. No course material is available to you; answer from your own "
    "knowledge. If you cannot answer, say so explicitly rather than guessing."
)


# --------------------------------------------------------------------------
# Tokenizer (mpnet) - the unit for chunk sizes and the B-nav budget, so the
# A-ret and B-nav budgets are measured identically.
# --------------------------------------------------------------------------

_tok = None
_tok_lock = threading.Lock()


def tokenizer():
    global _tok
    with _tok_lock:
        if _tok is None:
            import transformers
            from transformers import AutoTokenizer

            transformers.logging.set_verbosity_error()  # silence >512-token length warnings
            _tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        return _tok


def n_tokens(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])


def offsets(text: str) -> list[tuple[int, int]]:
    return tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]


def truncate_to_tokens(text: str, budget: int) -> str:
    offs = offsets(text)
    if len(offs) <= budget:
        return text
    return text[: offs[budget - 1][1]] if budget > 0 else ""


# --------------------------------------------------------------------------
# Corpus + questions
# --------------------------------------------------------------------------


class Corpus:
    def __init__(self, wiki_dir: Path):
        self.wiki_pages = dict(load_wiki_pages(wiki_dir))  # rel path -> text
        self.wiki_context = build_context(list(self.wiki_pages.items()))
        self.raw_docs = load_raw_corpus()
        self.raw_context = build_raw_context(self.raw_docs)
        self.raw_paths = {d["path"] for d in self.raw_docs}
        # concept page -> raw docs (in the corpus) it cites
        self.page_raw = {
            p: cited_raw_paths(t) & self.raw_paths
            for p, t in self.wiki_pages.items()
            if p not in NAV_PAGES
        }


class Question(BaseModel):
    id: int
    question: str
    question_type: str
    source_page: str
    question_class: str
    gold_pages: list[str]
    gold_raw_docs: list[str] = []
    reference_answer: str = ""
    topics: list[str] = []
    gold_annotation: str = ""


def load_questions(path: str) -> list[Question]:
    if not Path(path).exists():
        raise SystemExit(f"{path} not found - run annotate_gold.py first")
    return [Question(**q) for q in json.loads(Path(path).read_text(encoding="utf-8"))]


# --------------------------------------------------------------------------
# A-ret index over raw documents (navigational files are never indexed)
# --------------------------------------------------------------------------


def chunk_raw_docs(docs: list[dict]) -> list[dict]:
    """Token windows (mpnet tokens) sliced from the ORIGINAL text via offsets,
    so the generator sees the source text verbatim (the pilot decoded token
    ids, which lowercased the text and broke spacing)."""
    stride = CHUNK_TOKENS - CHUNK_OVERLAP
    chunks = []
    for d in docs:
        offs = offsets(d["text"])
        start, idx = 0, 0
        while start < len(offs):
            window = offs[start : start + CHUNK_TOKENS]
            text = d["text"][window[0][0] : window[-1][1]]
            chunks.append(
                {"id": f"{d['path']}::chunk{idx}", "page": d["path"], "chunk_index": idx,
                 "text": text, "n_tokens": len(window)}
            )
            if start + CHUNK_TOKENS >= len(offs):
                break
            start += stride
            idx += 1
    return chunks


def make_index(backend: str):
    from rag_qa_eval import LocalIndex, PineconeIndex

    if backend == "pinecone":
        return PineconeIndex(PINECONE_INDEX, PINECONE_NAMESPACE, "aws", "us-east-1")
    return LocalIndex(LOCAL_INDEX_PATH)


_embed_lock = threading.Lock()


def embed(texts: list[str]) -> list[list[float]]:
    from rag_qa_eval import embed_texts

    with _embed_lock:
        return embed_texts(texts)


def build_index(corpus: Corpus, backend: str) -> None:
    chunks = chunk_raw_docs(corpus.raw_docs)
    sizes = [c["n_tokens"] for c in chunks]
    print(f"Raw corpus: {len(corpus.raw_docs)} docs -> {len(chunks)} chunks "
          f"(tokens min {min(sizes)}, max {max(sizes)}, total {sum(sizes)})", file=sys.stderr)
    index = make_index(backend)
    if backend == "pinecone":
        try:
            index.index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        except Exception:  # noqa: BLE001 - namespace may not exist yet
            pass
    index.upsert(chunks, embed([c["text"] for c in chunks]))
    Path("rag_raw_chunks.json").write_text(
        json.dumps([{k: c[k] for k in ("id", "page", "chunk_index", "n_tokens")} for c in chunks], indent=1),
        encoding="utf-8",
    )
    print(f"Upserted {len(chunks)} chunks ({backend}).", file=sys.stderr)


# --------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------


def total_input(usage) -> int:
    return (usage.input_tokens + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0))


def text_of(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text").strip()


class Answerer:
    def __init__(self, client: Anthropic, model: str, effort: str, corpus: Corpus, index=None):
        self.client, self.model, self.effort, self.c, self.index = client, model, effort, corpus, index

    def _call(self, system, messages, **kw):
        return self.client.messages.create(
            model=self.model, max_tokens=16000, output_config={"effort": self.effort},
            system=system, messages=messages, **kw,
        )

    def _single(self, system, q: Question, log: dict) -> str:
        resp = self._call(system, [{"role": "user", "content": q.question}])
        log.update(n_calls=1, input_tokens=total_input(resp.usage),
                   api_input_tokens=total_input(resp.usage), output_tokens=resp.usage.output_tokens,
                   cache_read_tokens=resp.usage.cache_read_input_tokens or 0, stop_reason=resp.stop_reason)
        return text_of(resp)

    def answer(self, condition: str, q: Question) -> tuple[str, dict]:
        log: dict[str, Any] = {}
        c = self.c
        if condition == "C0":
            ans = self._single(CLOSED_BOOK_INSTRUCTIONS, q, log)
            log.update(units_seen=[], material_tokens=0)
        elif condition in ("A-full", "B-full"):
            material = c.raw_context if condition == "A-full" else c.wiki_context
            where = "below (the raw course files)" if condition == "A-full" else "below (the course wiki)"
            system = [{"type": "text", "text": GROUNDED_INSTRUCTIONS.format(where=where) + "\n\n" + material,
                       "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
            ans = self._single(system, q, log)
            units = sorted(c.raw_paths) if condition == "A-full" else sorted(c.wiki_pages)
            log.update(units_seen=units, material_tokens=n_tokens(material))
        elif condition == "A-ret":
            hits = self.index.query(embed([q.question])[0], TOP_K)
            excerpts = "\n\n".join(
                f"--- START EXCERPT {i + 1}: {h['page']} (chunk {h['chunk_index']}) ---\n{h['text']}\n"
                f"--- END EXCERPT {i + 1} ---" for i, h in enumerate(hits)
            )
            system = GROUNDED_INSTRUCTIONS.format(where="in the retrieved excerpts below") + "\n\n" + excerpts
            ans = self._single(system, q, log)
            log.update(units_seen=[h["id"] for h in hits],
                       retrieved_excerpts=[{k: h[k] for k in ("page", "chunk_index", "text")} for h in hits],
                       retrieval_scores=[round(h["score"], 4) for h in hits],
                       material_tokens=sum(n_tokens(h["text"]) for h in hits))
        elif condition == "B-nav":
            ans = self._navigate(q, log)
        else:
            raise ValueError(condition)
        return ans, log

    # ---- B-nav ---------------------------------------------------------

    def _resolve(self, path: str) -> Optional[str]:
        p = path.strip().strip("`").removeprefix("[[").removesuffix("]]").strip()
        p = p.removeprefix("./").removeprefix("wiki/")
        if not p.endswith(".md"):
            p += ".md"
        if p in self.c.wiki_pages:
            return p
        matches = [k for k in self.c.wiki_pages if k.split("/")[-1] == p.split("/")[-1]]
        return matches[0] if len(matches) == 1 else None

    def _navigate(self, q: Question, log: dict) -> str:
        index_text = self.c.wiki_pages["index.md"]
        used = n_tokens(index_text)
        seen = ["index.md"]
        truncated: list[str] = []
        tools = [{
            "name": "read_page",
            "description": (
                "Read one page of the course wiki and return its full text. `path` is the page "
                "path as linked from the wiki, e.g. `course/knn.md` or `[[course/knn]]`. Each "
                "read uses part of your page-reading token budget; a page that would exceed "
                "the remaining budget is cut off."
            ),
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                             "required": ["path"], "additionalProperties": False},
        }]
        system = (
            GROUNDED_INSTRUCTIONS.format(where="in the wiki pages you read") +
            f"\n\nYou can read wiki pages with the read_page tool, following links from the index "
            f"below and from the pages you read. You have a budget of {NAV_BUDGET:,} tokens of page "
            f"text for this question; the index below already used {used:,}. Read only what you "
            f"need, then answer.\n\n--- START PAGE: wiki/index.md ---\n{index_text}\n--- END PAGE ---"
        )
        messages: list[dict] = [{"role": "user", "content": q.question}]
        api_in = out = calls = 0
        resp = None
        for turn in range(MAX_NAV_TURNS + 1):
            force_answer = used >= NAV_BUDGET or turn == MAX_NAV_TURNS
            resp = self._call(system, messages, tools=tools,
                              tool_choice={"type": "none"} if force_answer else {"type": "auto"})
            calls += 1
            api_in += total_input(resp.usage)
            out += resp.usage.output_tokens
            if resp.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                path = self._resolve(str(block.input.get("path", "")))
                if path is None:
                    content, is_err = f"No page `{block.input.get('path')}` exists in this wiki.", True
                elif path in seen:
                    content, is_err = f"You already read wiki/{path}.", False
                else:
                    remaining = NAV_BUDGET - used
                    page = self.c.wiki_pages[path]
                    text = truncate_to_tokens(page, max(remaining, 0))
                    used += n_tokens(text)
                    seen.append(path)
                    is_err = False
                    if text != page:
                        truncated.append(path)
                        text += "\n[... page cut off: page-reading budget exhausted]"
                    content = f"--- START PAGE: wiki/{path} ---\n{text}\n--- END PAGE ---"
                    if used >= NAV_BUDGET:
                        content += "\n\nBudget exhausted. Answer now from what you have read."
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": content, "is_error": is_err})
            messages.append({"role": "user", "content": results})
        log.update(n_calls=calls, input_tokens=total_input(resp.usage), api_input_tokens=api_in,
                   output_tokens=out, stop_reason=resp.stop_reason, units_seen=seen,
                   truncated_pages=truncated, material_tokens=used)
        return text_of(resp)


# --------------------------------------------------------------------------
# Gold coverage: was the needed source in the answerer's context?
# --------------------------------------------------------------------------


def gold_coverage(condition: str, q: Question, log: dict, c: Corpus) -> tuple[list[str], float]:
    """Fraction of the question's gold concept pages whose content was in
    context. B-*: the page itself was in context. A-*: any chunk/doc from a
    raw file that the gold page cites was in context."""
    gold = q.gold_pages
    if not gold or condition == "C0":
        return [], 0.0
    seen = log.get("units_seen", [])
    if condition in ("B-nav", "B-full"):
        covered = [g for g in gold if g in seen]
    else:
        seen_docs = {u.split("::chunk")[0] for u in seen}
        if q.gold_raw_docs:  # question generated from raw files: use its own gold files
            covered = [g for g in gold
                       if {d for d in q.gold_raw_docs if f"course/{d.split('/')[3]}.md" == g} & seen_docs]
        else:
            covered = [g for g in gold if c.page_raw.get(g, set()) & seen_docs]
    return covered, len(covered) / len(gold)


# --------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------


class Grade(BaseModel):
    response_type: Literal["answer", "refusal"]
    score: Optional[Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    grounded: Optional[bool]
    flags_missing: Optional[bool]
    rationale: str


def grade_prompt_prefix(c: Corpus, judge_context: str) -> str:
    # Identical across every call so the provider can cache it.
    if judge_context == "full":
        return (
            "You are grading an AI tutor's response to a student's question about a university "
            "machine learning course.\n\nCOURSE MATERIAL - the compiled course wiki, followed by "
            "the raw course files (slides, notebooks, readings) it was compiled from:\n\n"
            f"{c.wiki_context}\n\n{c.raw_context}\n\n"
        )
    return (
        "You are grading an AI tutor's response to a student's question about a university "
        "machine learning course.\n\nCOURSE MATERIAL - the compiled course wiki. Raw course "
        "files (slides, notebooks, readings) relevant to this question follow after it:\n\n"
        f"{c.wiki_context}\n\n"
    )


def judge_raw_material(c: Corpus, q: Question, answer: str, log: dict) -> str:
    """judge_context='gold': the raw material the judge needs for THIS response - the
    question's gold raw files, every raw file the response cites, and (A-ret) the
    retrieved excerpts the answerer saw. Lets the judge verify grounding without the
    whole raw corpus in context (the full course no longer fits the judge's window)."""
    docs = {d["path"]: d["text"] for d in c.raw_docs}
    paths = list(dict.fromkeys([*q.gold_raw_docs, *sorted(cited_raw_paths(answer) & c.raw_paths)]))
    parts = [f"--- START DOCUMENT: {p} ---\n{docs[p]}\n--- END DOCUMENT: {p} ---" for p in paths if p in docs]
    for ex in log.get("retrieved_excerpts", []):
        if ex["page"] not in paths:
            parts.append(f"--- START EXCERPT: {ex['page']} (chunk {ex['chunk_index']}) ---\n{ex['text']}\n"
                         "--- END EXCERPT ---")
    if not parts:
        return ""
    return "RAW COURSE FILES (gold sources and any raw files the response cites):\n\n" + "\n\n".join(parts) + "\n\n"


def grade_prompt_suffix(q: Question, answer: str, raw_material: str = "") -> str:
    ref = (f"REFERENCE ANSWER (written from the course material):\n{q.reference_answer}\n\n"
           if q.reference_answer else "")
    return (
        f"{raw_material}QUESTION: {q.question}\n\n{ref}RESPONSE TO GRADE:\n{answer}\n\n"
        "Step 1 - response_type. Set 'refusal' if the response does not give a substantive "
        "answer: it says the material does not contain the answer, that it cannot answer, or "
        "otherwise declines. Set 'answer' if it gives a substantive answer, even a partial one "
        "or one with caveats about missing information.\n\n"
        "Step 2 - only for 'answer' (for 'refusal' set score, grounded and flags_missing to "
        "null). Score the answer against the course material"
        + (" and the reference answer" if q.reference_answer else "") + " from 1 to 10:\n"
        "10 = fully correct and thoroughly grounded in the course material\n"
        "8-9 = correct, only trivial omissions\n"
        "6-7 = correct but missing meaningful detail or nuance\n"
        "4-5 = partially correct, or correct but poorly supported\n"
        "2-3 = mostly incorrect or unsupported by the course material\n"
        "1 = wrong, fabricated, or contradicts the course material\n"
        "Use the FULL range of the scale. Set `grounded` to true only if every substantive "
        "claim in the answer is supported by something actually stated in the course material "
        "above (no fabrication, no unsupported outside detail). Set `flags_missing` to true only "
        "if the response explicitly tells the student that the course material does not cover "
        "some or all of what it says (e.g. 'the course doesn't cover this, but in general...')."
    )


class Judge:
    def __init__(self, model: str, corpus: Corpus, judge_context: str = "gold"):
        from openai import OpenAI

        self.client, self.model, self.corpus, self.judge_context = OpenAI(), model, corpus, judge_context
        self.prefix = grade_prompt_prefix(corpus, judge_context)

    def grade(self, q: Question, answer: str, log: Optional[dict] = None) -> Grade:
        raw = "" if self.judge_context == "full" else judge_raw_material(self.corpus, q, answer, log or {})
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": self.prefix + grade_prompt_suffix(q, answer, raw)}],
            response_format=Grade,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(f"judge returned no parsed output (finish_reason="
                               f"{completion.choices[0].finish_reason})")
        if parsed.response_type == "refusal":
            parsed.score, parsed.grounded, parsed.flags_missing = None, None, None
        elif parsed.score is None or parsed.grounded is None or parsed.flags_missing is None:
            raise RuntimeError("judge classified an answer but left score/grounded empty")
        return parsed


def outcome(grade: Grade, coverage: float, condition: str) -> str:
    if grade.response_type == "answer":
        if grade.grounded:
            return "grounded_answer"
        # Ungrounded, but the response told the student it was going beyond the course material.
        return "flagged_ungrounded" if grade.flags_missing else "fabrication"
    if condition == "C0":
        return "refusal"
    # Justified when the answerer did not have all of the gold source in context.
    return "justified_refusal" if coverage < 1.0 else "unjustified_refusal"


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def run_condition(condition: str, args, corpus: Corpus, questions: list[Question], judge) -> None:
    index = make_index(args.backend) if condition == "A-ret" else None
    answerer = Answerer(Anthropic(max_retries=6), args.model, args.effort, corpus, index)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = RESULTS_DIR / f"{condition}.jsonl"
    done: dict[int, dict] = {}
    current = {q.id: q for q in questions}
    if args.resume and jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            q = current.get(r["id"])
            # Rerun rows that errored, or whose question / reference answer was edited since.
            if ("error" not in r and q is not None and r["question"] == q.question
                    and r.get("reference_answer", q.reference_answer) == q.reference_answer):
                done[r["id"]] = r
    todo = [q for q in questions if q.id not in done]
    print(f"[{condition}] {len(todo)} to run ({len(done)} already done)", file=sys.stderr)
    lock = threading.Lock()
    if not args.resume or not jsonl.exists():
        jsonl.write_text("", encoding="utf-8")
    else:  # rewrite without errored rows
        jsonl.write_text("".join(json.dumps(r) + "\n" for r in done.values()), encoding="utf-8")

    def one(q: Question) -> dict:
        r: dict[str, Any] = {"condition": condition, "id": q.id, "question_class": q.question_class,
                             "question_type": q.question_type, "source_page": q.source_page,
                             "gold_pages": q.gold_pages, "question": q.question,
                             "reference_answer": q.reference_answer,
                             "answer_model": args.model, "judge_model": args.grade_model,
                             "top_k": TOP_K, "nav_budget": NAV_BUDGET}
        try:
            t0 = time.time()
            ans, log = answerer.answer(condition, q)
            r["latency_s"] = round(time.time() - t0, 2)
            r["answer"] = ans
            r.update(log)
            covered, cov = gold_coverage(condition, q, log, corpus)
            r["gold_covered"], r["gold_in_context"] = covered, cov
            if judge is not None:
                g = judge.grade(q, ans, log)
                r["judge_context"] = judge.judge_context
                r.update(response_type=g.response_type, score=g.score, grounded=g.grounded,
                         flags_missing=g.flags_missing,
                         rationale=g.rationale, outcome=outcome(g, cov, condition))
        except Exception as e:  # noqa: BLE001 - log and keep going; --resume reruns it
            r["error"] = f"{type(e).__name__}: {e}"
        with lock:
            with jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
        return r

    # Run one question first so the cached context is written before the parallel batch.
    if todo:
        one(todo[0])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for i, _ in enumerate(pool.map(one, todo[1:]), 2):
            if i % 10 == 0 or i == len(todo):
                print(f"  [{condition}] {i}/{len(todo)}", file=sys.stderr)

    rows = sorted((json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines()), key=lambda r: r["id"])
    fields = ["condition", "id", "question_class", "source_page", "gold_pages", "question", "answer",
              "response_type", "outcome", "score", "grounded", "flags_missing", "gold_in_context", "input_tokens",
              "api_input_tokens", "material_tokens", "output_tokens", "n_calls", "units_seen",
              "latency_s", "rationale", "error"]
    with (RESULTS_DIR / f"{condition}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("; ".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items()})
    errs = [r for r in rows if "error" in r]
    print(f"[{condition}] wrote {len(rows)} rows -> {jsonl}" + (f"  ({len(errs)} errors; rerun with --resume)" if errs else ""),
          file=sys.stderr)


def main() -> None:
    global TOP_K, NAV_BUDGET, RESULTS_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, choices=CONDITIONS + ["all"])
    ap.add_argument("--build-index", action="store_true", help="(A-ret) chunk + embed + upsert the raw corpus first")
    ap.add_argument("--backend", default="pinecone", choices=["pinecone", "local"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--grade-model", default=DEFAULT_GRADE_MODEL)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--questions-file", default=QUESTIONS_FILE)
    ap.add_argument("--wiki-dir", default="wiki")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=TOP_K,
                    help=f"A-ret chunks per question; the B-nav budget is matched to top_k x {CHUNK_TOKENS} tokens")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--judge-context", default="gold", choices=["gold", "full"],
                    help="gold: wiki + the question's gold raw files + raw files the response cites "
                         "(+ A-ret excerpts); full: wiki + entire raw corpus (7-topic runs)")
    ap.add_argument("--resume", action="store_true", help="keep finished rows, rerun missing/errored ones")
    args = ap.parse_args()
    TOP_K, NAV_BUDGET, RESULTS_DIR = args.top_k, args.top_k * CHUNK_TOKENS, Path(args.results_dir)
    print(f"Budget: A-ret top_k={TOP_K}, B-nav {NAV_BUDGET:,} tokens -> {RESULTS_DIR}", file=sys.stderr)

    corpus = Corpus(Path(args.wiki_dir))
    print(f"Wiki: {len(corpus.wiki_pages)} pages ({n_tokens(corpus.wiki_context):,} tok); raw: "
          f"{len(corpus.raw_docs)} docs ({n_tokens(corpus.raw_context):,} tok)", file=sys.stderr)
    if args.build_index:
        build_index(corpus, args.backend)
    questions = load_questions(args.questions_file)
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        return
    judge = None if args.no_grade else Judge(args.grade_model, corpus, args.judge_context)
    for cond in (CONDITIONS if args.condition == "all" else [args.condition]):
        run_condition(cond, args, corpus, questions, judge)


if __name__ == "__main__":
    main()
