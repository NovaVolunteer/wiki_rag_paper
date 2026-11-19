#!/usr/bin/env python
"""
Raw-corpus loader for the A-ret and A-full conditions.

The raw corpus is the source material behind the compiled concept pages:
every text-bearing file under raw/course/<source>/<topic>/ for each source in
SOURCES (the DS-3021 course folder and the MLBook online text) and each topic in
TOPICS. Each concept page in wiki/course/ cites its raw sources by path
(`raw/course/<source>/...`); those citations define the gold raw documents for
a question (see gold_raw_docs). Paths keep the topic as the 4th component
(raw/course/<source>/<topic>/<file>), which eval_conditions.py relies on.

Extraction (text only, matching the paper's text-only scope):
  .pdf   -> pypdf page text
  .pptx  -> python-pptx: slide text frames + tables + speaker notes
  .docx  -> python-docx paragraphs + tables
  .ipynb -> markdown + code cell sources (outputs dropped)
  .py/.md/.txt -> as-is
Skipped as non-text: .csv, .xlsx, images, Office lock files (~$*).

    python raw_corpus.py            # extract, cache to raw_corpus.json, print stats
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RAW_COURSE = Path("raw/course")
SOURCES = ["DS-3021", "MLBook"]
# The original 7 pilot topics (the 2026-11-18 run) ...
PILOT_TOPICS = [
    "knn",
    "decision-trees",
    "ml-eval-metrics",
    "dt-regression",
    "kmeans-clustering",
    "ensemble-methods",
    "ml-bias",
]
# ... plus the rest of the course, added by ingest_course_materials.py (2026-11-19).
NEW_TOPICS = [
    "course-overview",
    "dev-environment",
    "python-basics",
    "ml-foundations",
    "data-prep",
    "linear-regression",
    "logistic-regression",
    "pca",
    "final-project",
]
TOPICS = PILOT_TOPICS + NEW_TOPICS
TEXT_EXTS = {".pdf", ".pptx", ".docx", ".ipynb", ".py", ".md", ".txt"}
CACHE_PATH = Path("raw_corpus.json")
RAW_CITE_RE = re.compile(r"raw/course/(?:DS-3021|MLBook)/[^\s`)\"'\]]+")


def _pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(
        f"[page {i + 1}]\n{(p.extract_text() or '').strip()}" for i, p in enumerate(reader.pages)
    )


def _pptx(path: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(path))
    out = []
    for i, slide in enumerate(prs.slides, 1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                t = shape.text_frame.text.strip()
                if t:
                    parts.append(t)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"[speaker notes] {notes}")
        out.append(f"[slide {i}]\n" + "\n".join(parts))
    return "\n\n".join(out)


def _docx(path: Path) -> str:
    import docx

    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(parts)


def _ipynb(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if cell.get("cell_type") == "code":
            out.append(f"```python\n{src}\n```")
        else:
            out.append(src)
    return "\n\n".join(out)


def extract(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _pdf(path)
    if ext == ".pptx":
        return _pptx(path)
    if ext == ".docx":
        return _docx(path)
    if ext == ".ipynb":
        return _ipynb(path)
    return path.read_text(encoding="utf-8", errors="replace")


def build_raw_corpus() -> list[dict]:
    docs = []
    for topic in TOPICS:
        for source in SOURCES:
            folder = RAW_COURSE / source / topic
            if not folder.is_dir():
                continue
            for p in sorted(folder.iterdir()):
                if not p.is_file() or p.name.startswith("~$") or p.suffix.lower() not in TEXT_EXTS:
                    continue
                text = extract(p).strip()
                if text:
                    docs.append({"path": p.as_posix(), "topic": topic, "source": source, "text": text})
    return docs


def load_raw_corpus(rebuild: bool = False) -> list[dict]:
    if CACHE_PATH.exists() and not rebuild:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    docs = build_raw_corpus()
    CACHE_PATH.write_text(json.dumps(docs, indent=1), encoding="utf-8")
    return docs


def build_raw_context(docs: list[dict]) -> str:
    return "\n\n".join(
        f"--- START DOCUMENT: {d['path']} ---\n{d['text']}\n--- END DOCUMENT: {d['path']} ---"
        for d in docs
    )


def cited_raw_paths(page_text: str) -> set[str]:
    """Raw paths a wiki page cites (trailing punctuation stripped)."""
    return {m.rstrip(".,;:") for m in RAW_CITE_RE.findall(page_text)}


def gold_raw_docs(gold_pages: list[str], wiki_pages: dict[str, str], raw_paths: set[str]) -> set[str]:
    """Raw documents cited by the gold concept page(s) that are in the corpus."""
    out: set[str] = set()
    for gp in gold_pages:
        out |= cited_raw_paths(wiki_pages.get(gp, "")) & raw_paths
    return out


if __name__ == "__main__":
    docs = load_raw_corpus(rebuild=True)
    total = sum(len(d["text"]) for d in docs)
    print(f"{len(docs)} documents, {total:,} chars (~{total // 4:,} tokens)", file=sys.stderr)
    for d in docs:
        print(f"  {len(d['text']):>8,}  {d['path']}", file=sys.stderr)
