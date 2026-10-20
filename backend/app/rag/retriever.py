"""Lightweight RAG: TF-IDF over chunked knowledge-base markdown (no torch/embedding download).

Swap `search` for a vector store (FAISS/Chroma) without touching the agent.
"""
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

KB_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"


def _load_chunks(kb_dir: Path) -> list[dict]:
    chunks = []
    for f in sorted(kb_dir.glob("*.md")):
        text = f.read_text()
        title = text.splitlines()[0].lstrip("# ").strip()
        src = re.search(r"^Source: (.+)$", text, re.M)
        body = text.split("\n", 1)[1]
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for i, p in enumerate(paras):
            chunks.append({"id": f"{f.stem}#{i}", "doc": f.stem, "title": title,
                           "source": src.group(1) if src else "", "text": p})
    return chunks


class Retriever:
    def __init__(self, kb_dir: Path = KB_DIR):
        self.chunks = _load_chunks(kb_dir)
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([c["title"] + " " + c["text"] for c in self.chunks])

    def search(self, query: str, k: int = 4) -> list[dict]:
        scores = linear_kernel(self.vec.transform([query]), self.matrix).ravel()
        top = scores.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 3)} for i in top if scores[i] > 0]

    def by_doc(self, doc: str) -> list[dict]:
        return [c for c in self.chunks if c["doc"] == doc]
