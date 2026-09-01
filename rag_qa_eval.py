#!/usr/bin/env python
"""
Arm A — Vector RAG pipeline over the wiki corpus.

The RAG-side counterpart to `wiki_qa_eval.py` (Arm B). Same corpus, same
question set, same answerer model and prompts, same judge — the ONLY thing
that differs is the data flow: instead of the whole wiki in a cached system
block, each question retrieves similarity-ranked chunks from a vector index
and the answerer sees only those.

Replicates the prior study's best configuration (paper §3.3, Arm A):
chunks of 1,500 tokens with 100-token overlap, embedded with
sentence-transformers/all-mpnet-base-v2 (768-d, cosine), indexed in Pinecone.

Usage:
    python rag_qa_eval.py --build-index                 # chunk + embed + upsert, then exit
    python rag_qa_eval.py --questions-file course_questions.json
    python rag_qa_eval.py --questions-file course_questions.json --limit 5   # smoke test
    python rag_qa_eval.py --backend local --build-index  # no Pinecone: exact-cosine local index
                                                         # (verification fallback — the paper
                                                         # config is the Pinecone backend)

Needs ANTHROPIC_API_KEY (answering), OPENAI_API_KEY (judge, unless
--judge-provider anthropic), and PINECONE_API_KEY (unless --backend local),
via the environment or .env.

Grading note: the judge grades against the FULL wiki, exactly as in Arm B —
the instrument is held constant across arms; only the answerer's context
changes. Retrieved chunk ids/scores are logged per question so retrieval
quality can be analyzed separately.

Embedding caveat (inherited from the prior study's configuration):
all-mpnet-base-v2 has a 384-word-piece sequence limit, so a 1,500-token chunk
is embedded from its first ~384 tokens while the full chunk text is what the
generator receives. On the current corpus most pages fit in a single chunk,
so in practice a chunk vector represents the head of a page.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# Reuse the Arm B harness so both arms share models, prompts, and the judge.
from wiki_qa_eval import (
    DEFAULT_ANTHROPIC_GRADE_MODEL,
    DEFAULT_GRADE_MODEL,
    DEFAULT_MODEL,
    Question,
    build_context,
    grade_answer,
    load_wiki_pages,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # only required when grading with the OpenAI judge

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 768
CHUNK_TOKENS = 1500
CHUNK_OVERLAP = 100
DEFAULT_INDEX_NAME = "llm-wiki-rag"
DEFAULT_TOP_K = 5
LOCAL_INDEX_PATH = "rag_local_index.npz"


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def chunk_pages(
    pages: list[tuple[str, str]], chunk_tokens: int, overlap: int
) -> list[dict]:
    """Split each wiki page into token-window chunks.

    Token boundaries come from the embedding model's own tokenizer, so the
    chunker and the embedder agree on what a token is. Chunks never cross
    page boundaries — source_page metadata stays exact for citation.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    if overlap >= chunk_tokens:
        raise SystemExit("--chunk-overlap must be smaller than --chunk-tokens")
    stride = chunk_tokens - overlap

    chunks: list[dict] = []
    for rel_path, content in pages:
        ids = tokenizer.encode(content, add_special_tokens=False)
        start = 0
        idx = 0
        while start < len(ids):
            window = ids[start : start + chunk_tokens]
            text = tokenizer.decode(window)
            chunks.append(
                {
                    "id": f"{rel_path}::chunk{idx}",
                    "page": rel_path,
                    "chunk_index": idx,
                    "text": text,
                    "n_tokens": len(window),
                }
            )
            if start + chunk_tokens >= len(ids):
                break
            start += stride
            idx += 1
    return chunks


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------


_embedder = None
_embedder_lock = threading.Lock()


def get_embedder():
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(EMBED_MODEL)
        return _embedder


def embed_texts(texts: list[str]) -> "list[list[float]]":
    model = get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


# --------------------------------------------------------------------------
# Vector index backends
# --------------------------------------------------------------------------


class PineconeIndex:
    """Pinecone serverless index (the paper's Arm A configuration)."""

    def __init__(self, index_name: str, namespace: str, cloud: str, region: str):
        from pinecone import Pinecone, ServerlessSpec

        self.pc = Pinecone()  # reads PINECONE_API_KEY
        self.namespace = namespace
        existing = {ix["name"] for ix in self.pc.list_indexes()}
        if index_name not in existing:
            print(f"Creating Pinecone index '{index_name}' ...", file=sys.stderr)
            self.pc.create_index(
                name=index_name,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self.index = self.pc.Index(index_name)

    def upsert(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        payload = [
            {
                "id": c["id"],
                "values": v,
                "metadata": {
                    "page": c["page"],
                    "chunk_index": c["chunk_index"],
                    "text": c["text"],
                },
            }
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(payload), 100):
            self.index.upsert(vectors=payload[i : i + 100], namespace=self.namespace)

    def query(self, vector: list[float], top_k: int) -> list[dict]:
        res = self.index.query(
            vector=vector, top_k=top_k, include_metadata=True, namespace=self.namespace
        )
        return [
            {
                "id": m["id"],
                "score": float(m["score"]),
                "page": m["metadata"]["page"],
                "chunk_index": int(m["metadata"]["chunk_index"]),
                "text": m["metadata"]["text"],
            }
            for m in res["matches"]
        ]

    def stats(self) -> str:
        s = self.index.describe_index_stats()
        return f"{s.get('total_vector_count', '?')} vectors"


class LocalIndex:
    """Exact-cosine fallback with the same interface, for environments without
    a Pinecone key. Retrieval math is identical (cosine over the same
    normalized mpnet vectors); Pinecone remains the paper configuration."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.chunks: list[dict] = []
        self.matrix = None
        if self.path.exists():
            import numpy as np

            data = np.load(self.path, allow_pickle=True)
            self.matrix = data["vectors"]
            self.chunks = json.loads(str(data["chunks"]))

    def upsert(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        import numpy as np

        self.chunks = chunks
        self.matrix = np.asarray(vectors, dtype="float32")
        np.savez_compressed(
            self.path, vectors=self.matrix, chunks=json.dumps(chunks)
        )

    def query(self, vector: list[float], top_k: int) -> list[dict]:
        import numpy as np

        if self.matrix is None:
            raise SystemExit(f"No local index at {self.path} — run --build-index first")
        q = np.asarray(vector, dtype="float32")
        sims = self.matrix @ q  # vectors are normalized -> dot == cosine
        order = np.argsort(-sims)[:top_k]
        return [
            {
                "id": self.chunks[i]["id"],
                "score": float(sims[i]),
                "page": self.chunks[i]["page"],
                "chunk_index": self.chunks[i]["chunk_index"],
                "text": self.chunks[i]["text"],
            }
            for i in order
        ]

    def stats(self) -> str:
        return f"{0 if self.matrix is None else len(self.matrix)} vectors ({self.path})"


def make_index(args) -> "PineconeIndex | LocalIndex":
    if args.backend == "pinecone":
        return PineconeIndex(args.index_name, args.namespace, args.cloud, args.region)
    return LocalIndex(args.local_index_path)


# --------------------------------------------------------------------------
# RAG answering
# --------------------------------------------------------------------------


def build_rag_system_prompt(retrieved: list[dict]) -> str:
    """Mirrors the Arm B system prompt, with retrieved excerpts in place of
    the full wiki."""
    excerpts = "\n\n".join(
        f"--- START EXCERPT {i + 1}: wiki/{r['page']} (chunk {r['chunk_index']}) ---\n"
        f"{r['text']}\n"
        f"--- END EXCERPT {i + 1} ---"
        for i, r in enumerate(retrieved)
    )
    return (
        "You are answering questions using ONLY the retrieved wiki excerpts "
        "below. Cite the wiki page(s) that support your answer (e.g. "
        "`wiki/course/knn.md`). If the excerpts don't contain the answer, "
        "say so explicitly rather than guessing or using outside "
        "knowledge.\n\n" + excerpts
    )


def answer_with_rag(
    client: Anthropic,
    model: str,
    effort: str,
    index,
    top_k: int,
    question: Question,
) -> tuple[str, list[dict]]:
    qvec = embed_texts([question.question])[0]
    retrieved = index.query(qvec, top_k)
    system_prompt = build_rag_system_prompt(retrieved)
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        output_config={"effort": effort},
        system=system_prompt,
        messages=[{"role": "user", "content": question.question}],
    )
    answer = "".join(b.text for b in resp.content if b.type == "text").strip()
    return answer, retrieved


def process_question(
    client: Anthropic,
    judge_client,
    judge_provider: str,
    answer_model: str,
    grade_model: str,
    effort: str,
    index,
    top_k: int,
    grade_context: str,
    question: Question,
    do_grade: bool,
) -> dict:
    result = {
        "id": question.id,
        "question": question.question,
        "question_type": question.question_type,
        "source_page": question.source_page,
        "question_class": question.question_class,
    }
    try:
        answer, retrieved = answer_with_rag(client, answer_model, effort, index, top_k, question)
        result["answer"] = answer
        result["retrieved"] = [
            {"id": r["id"], "page": r["page"], "score": round(r["score"], 4)}
            for r in retrieved
        ]
        # Did retrieval surface the question's source page at all?
        if question.source_page not in ("", "(cross-page)"):
            result["source_page_retrieved"] = any(
                r["page"] == question.source_page for r in retrieved
            )
    except Exception as e:  # noqa: BLE001 - log and keep going
        result["answer"] = ""
        result["error"] = f"answer failed: {e}"
        return result

    if do_grade:
        try:
            graded = grade_answer(
                judge_client, judge_provider, grade_model, grade_context, question, result["answer"]
            )
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
    parser.add_argument("--backend", default="pinecone", choices=["pinecone", "local"], help="Vector index backend (default: pinecone; 'local' is an exact-cosine fallback for keyless environments)")
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME, help=f"Pinecone index name (default: {DEFAULT_INDEX_NAME})")
    parser.add_argument("--namespace", default="", help="Pinecone namespace (default: '')")
    parser.add_argument("--cloud", default="aws", help="Pinecone serverless cloud (default: aws)")
    parser.add_argument("--region", default="us-east-1", help="Pinecone serverless region (default: us-east-1)")
    parser.add_argument("--local-index-path", default=LOCAL_INDEX_PATH, help=f"Path for the local backend's index file (default: {LOCAL_INDEX_PATH})")
    parser.add_argument("--build-index", action="store_true", help="Chunk + embed + upsert the wiki corpus, then exit (combine with --run to continue into the eval)")
    parser.add_argument("--run", action="store_true", help="With --build-index: continue into answering after the build")
    parser.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS, help=f"Chunk size in embedder tokens (default: {CHUNK_TOKENS}, per the prior study)")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help=f"Chunk overlap in tokens (default: {CHUNK_OVERLAP}, per the prior study)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Chunks retrieved per question (default: {DEFAULT_TOP_K}, the prior study's best text configuration)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Answerer model (default: {DEFAULT_MODEL})")
    parser.add_argument("--grade-model", default=None, help=f"Judge model (default: {DEFAULT_GRADE_MODEL} for openai, {DEFAULT_ANTHROPIC_GRADE_MODEL} for anthropic)")
    parser.add_argument("--judge-provider", default="openai", choices=["openai", "anthropic"], help="Which provider grades (same semantics as wiki_qa_eval.py)")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--questions-file", default="course_questions.json", help="Question set JSON (default: course_questions.json — the filtered 59 used for Arm B)")
    parser.add_argument("--output", default="rag_eval_results", help="Output basename (writes <output>.jsonl and <output>.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (for quick testing)")
    parser.add_argument("--max-workers", type=int, default=5, help="Concurrent API requests (default: 5)")
    parser.add_argument("--no-grade", action="store_true", help="Skip LLM-judge grading; just collect answers")
    args = parser.parse_args()

    wiki_dir = Path(args.wiki_dir)
    pages = load_wiki_pages(wiki_dir)
    print(f"Loaded {len(pages)} wiki pages.", file=sys.stderr)

    index = make_index(args)

    if args.build_index:
        chunks = chunk_pages(pages, args.chunk_tokens, args.chunk_overlap)
        sizes = [c["n_tokens"] for c in chunks]
        print(
            f"Chunked into {len(chunks)} chunks "
            f"(tokens: min {min(sizes)}, max {max(sizes)}, total {sum(sizes)}).",
            file=sys.stderr,
        )
        print(f"Embedding with {EMBED_MODEL} ...", file=sys.stderr)
        vectors = embed_texts([c["text"] for c in chunks])
        index.upsert(chunks, vectors)
        print(f"Upserted. Index now holds: {index.stats()}", file=sys.stderr)
        if not args.run:
            return

    do_grade = not args.no_grade
    grade_model = args.grade_model or (
        DEFAULT_GRADE_MODEL if args.judge_provider == "openai" else DEFAULT_ANTHROPIC_GRADE_MODEL
    )
    client = Anthropic()
    judge_client = None
    if do_grade:
        if args.judge_provider == "openai":
            if OpenAI is None:
                raise SystemExit("Grading needs the `openai` package: pip install openai (or pass --no-grade)")
            judge_client = OpenAI()
        else:
            judge_client = Anthropic()
            if grade_model == args.model:
                raise SystemExit(
                    f"Judge model and answerer model are both {grade_model}: the model would be "
                    "grading its own homework. Pass a different --grade-model."
                )
        print(f"Judge: {args.judge_provider}/{grade_model}  |  Answerer: anthropic/{args.model}", file=sys.stderr)

    raw = json.loads(Path(args.questions_file).read_text(encoding="utf-8"))
    questions = [Question(**q) for q in raw]
    print(f"Loaded {len(questions)} questions from {args.questions_file}.", file=sys.stderr)
    if args.limit:
        questions = questions[: args.limit]

    # The judge grades against the FULL wiki, identically to Arm B.
    grade_context = build_context(pages)

    print(
        f"Answering {len(questions)} questions (top-k={args.top_k}, backend={args.backend}) "
        f"with {args.max_workers} workers ...",
        file=sys.stderr,
    )
    jsonl_path = Path(f"{args.output}.jsonl")
    results: list[dict] = []
    write_lock = threading.Lock()

    def run_one(q: Question) -> dict:
        r = process_question(
            client, judge_client, args.judge_provider, args.model, grade_model,
            args.effort, index, args.top_k, grade_context, q, do_grade,
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
    fieldnames = [
        "id", "question_type", "question_class", "source_page", "question",
        "answer", "score", "grounded", "rationale",
        "retrieved_pages", "source_page_retrieved", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            if "retrieved" in r:
                row["retrieved_pages"] = "; ".join(
                    f"{x['page']}#{x['id'].rsplit('chunk', 1)[-1]}@{x['score']}" for x in r["retrieved"]
                )
            writer.writerow(row)

    print(f"\nWrote {len(results)} results -> {jsonl_path} , {csv_path}", file=sys.stderr)

    if do_grade:
        scored = [r["score"] for r in results if "score" in r]
        errored = [r for r in results if "error" in r]
        if scored:
            avg = sum(scored) / len(scored)
            grounded_rate = sum(1 for r in results if r.get("grounded")) / len(scored)
            dist = {n: scored.count(n) for n in range(1, 11)}
            hits = [r for r in results if "source_page_retrieved" in r]
            print("\n--- Summary ---", file=sys.stderr)
            print(f"Graded: {len(scored)}/{len(results)}", file=sys.stderr)
            print(f"Average score: {avg:.2f}/10", file=sys.stderr)
            print(f"Grounded rate: {grounded_rate:.0%}", file=sys.stderr)
            if hits:
                hit_rate = sum(1 for r in hits if r["source_page_retrieved"]) / len(hits)
                print(f"Source-page retrieval hit rate (topic-page qs): {hit_rate:.0%}", file=sys.stderr)
            print(f"Score distribution: {dist}", file=sys.stderr)
            worst = sorted((r for r in results if "score" in r), key=lambda r: r["score"])[:5]
            print("\nLowest-scoring questions:", file=sys.stderr)
            for r in worst:
                print(f"  [{r['score']}] {r['question']}  ({r['source_page']})", file=sys.stderr)
        if errored:
            print(f"\n{len(errored)} questions hit an error — see {jsonl_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
