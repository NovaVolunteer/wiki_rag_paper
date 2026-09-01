#!/usr/bin/env python
"""Plumbing smoke test for rag_qa_eval.py in a network-restricted sandbox.

HuggingFace and Pinecone are unreachable here, so the embedder/tokenizer are
mocked with deterministic stand-ins. Everything else is real: corpus loading,
chunk windowing, local index build/query, the answerer call, and the judge
call (anthropic provider). Run the real thing locally in VSCode.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import rag_qa_eval as rq

# ---- mock tokenizer: whitespace tokens, decode joins them back ----
class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split(" ")
    def decode(self, ids):
        return " ".join(ids)

# chunk_pages imports AutoTokenizer inside the function; patch via transformers
import transformers
transformers.AutoTokenizer.from_pretrained = classmethod(lambda cls, *a, **k: MockTokenizer())

# ---- mock embedder: deterministic hash-based unit vectors ----
def fake_embed(texts):
    out = []
    for t in texts:
        seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(rq.EMBED_DIM)
        v /= np.linalg.norm(v)
        out.append(v.tolist())
    return out

rq.embed_texts = fake_embed

# ---- 1. chunking ----
pages = rq.load_wiki_pages(Path("wiki"))
chunks = rq.chunk_pages(pages, chunk_tokens=200, overlap=20)
assert all(c["n_tokens"] <= 200 for c in chunks)
pagesets = {c["page"] for c in chunks}
assert len(pagesets) == len(pages), "every page should produce >=1 chunk"
# overlap check on a multi-chunk page
multi = [c for c in chunks if c["chunk_index"] > 0]
assert multi, "expected at least one multi-chunk page at 200 tokens"
for c in multi:
    prev = next(x for x in chunks if x["page"] == c["page"] and x["chunk_index"] == c["chunk_index"] - 1)
    tail = prev["text"].split(" ")[-20:]
    head = c["text"].split(" ")[:20]
    assert tail == head, "20-token overlap should match between consecutive chunks"
print(f"[1] chunking ok: {len(chunks)} chunks from {len(pages)} pages, overlap verified")

# ---- 2. local index build + query ----
class Args: pass
args = Args(); args.backend = "local"; args.local_index_path = "smoke_index.npz"
index = rq.make_index(args)
index.upsert(chunks, rq.embed_texts([c["text"] for c in chunks]))
hits = index.query(rq.embed_texts(["what is knn"])[0], 5)
assert len(hits) == 5 and all(set(h) >= {"id", "score", "page", "text"} for h in hits)
assert hits[0]["score"] >= hits[-1]["score"]
# self-retrieval sanity: a chunk's own text must retrieve itself at rank 1
self_hit = index.query(rq.embed_texts([chunks[3]["text"]])[0], 1)[0]
assert self_hit["id"] == chunks[3]["id"], "exact text should retrieve its own chunk first"
print(f"[2] local index ok: top hit {hits[0]['id']} @ {hits[0]['score']:.3f}, self-retrieval exact")

# ---- 3. persistence round-trip ----
index2 = rq.LocalIndex("smoke_index.npz")
hits2 = index2.query(rq.embed_texts(["what is knn"])[0], 5)
assert [h["id"] for h in hits2] == [h["id"] for h in hits]
print("[3] index persistence ok")

# ---- 4. real answer + judge on one question ----
qs = [rq.Question(**q) for q in json.loads(Path("course_questions.json").read_text())]
q = qs[0]
from anthropic import Anthropic
client = Anthropic()
answer, retrieved = rq.answer_with_rag(client, rq.DEFAULT_MODEL, "low", index, 5, q)
assert answer, "empty answer"
print(f"[4] answerer ok ({len(answer)} chars). retrieved: {[r['page'] for r in retrieved]}")

grade_context = rq.build_context(pages)
graded = rq.grade_answer(Anthropic(), "anthropic", rq.DEFAULT_ANTHROPIC_GRADE_MODEL, grade_context, q, answer)
print(f"[5] judge ok: score={graded.score} grounded={graded.grounded}")
print("\nALL SMOKE TESTS PASSED (embeddings mocked — run real build in VSCode)")
