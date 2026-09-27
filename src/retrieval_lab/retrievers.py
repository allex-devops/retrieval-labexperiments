"""Retrievers. Each one is `search(query, n) -> [Hit]`, so they stack: hybrid wraps two, rerank and rewrite wrap any."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import paths
from .bench import Hit
from .cache import DiskCache, make_key
from .corpus import Chunk
from .embedders import Embedder, free_torch_memory, torch_device


def top_n(scores: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(scores))
    if n == 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, n - 1)[:n]
    return idx[np.argsort(-scores[idx], kind="stable")]


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode())
        h.update(b"\0")
        h.update(c.text.encode())
        h.update(b"\1")
    return h.hexdigest()


class DenseRetriever:
    """Exact cosine search over a matrix of chunk vectors. The matrix is built once per (model, chunking)
    and saved, so every later experiment on the same chunks loads it in a second."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, name: str | None = None, matrix_dir: Path | None = None):
        self.chunks = chunks
        self.embedder = embedder
        self.name = name or f"dense:{embedder.name}"
        key = make_key(embedder.name, getattr(embedder, "model", None) or getattr(embedder, "hf_id", None), corpus_fingerprint(chunks))
        path = (matrix_dir or paths.CACHE_DIR / "matrices") / f"{key}.npy"
        if path.exists():
            self.matrix = np.load(path)
        else:
            self.matrix = embedder.encode([c.text for c in chunks], "document")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.matrix)

    def search(self, query: str, n: int) -> list[Hit]:
        q = self.embedder.encode([query], "query")[0]
        scores = self.matrix @ q
        return [Hit(self.chunks[i], float(scores[i])) for i in top_n(scores, n)]


STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how in is it its of on or that the their them "
    "these they this to was were what when where which who why will with".split()
)
TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


class BM25Retriever:
    """Okapi BM25 over the same chunks, with an inverted index so a query only touches chunks that share a term."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75, name: str = "bm25"):
        self.chunks = chunks
        self.name = name
        self.k1, self.b = k1, b
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = np.zeros(len(chunks), dtype=np.float32)
        for i, c in enumerate(chunks):
            tf = Counter(tokenize(c.text))
            self.lengths[i] = sum(tf.values())
            for term, count in tf.items():
                postings[term].append((i, count))
        self.avgdl = float(self.lengths.mean()) if len(chunks) else 0.0
        n = len(chunks)
        self.index = {
            t: (np.array([i for i, _ in p]), np.array([c for _, c in p], dtype=np.float32),
                math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)))
            for t, p in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self.index:
                continue
            ids, tf, idf = self.index[term]
            norm = self.k1 * (1 - self.b + self.b * self.lengths[ids] / self.avgdl)
            scores[ids] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def search(self, query: str, n: int) -> list[Hit]:
        scores = self.scores(query)
        return [Hit(self.chunks[i], float(scores[i])) for i in top_n(scores, n) if scores[i] > 0]


def rrf(rankings: list[list[Hit]], k: int = 60, weights: list[float] | None = None) -> list[Hit]:
    """Reciprocal rank fusion: score = sum of w / (k + rank). Needs only ranks, so it can merge
    BM25 and cosine scores that live on different scales."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = defaultdict(float)
    by_id: dict[str, Chunk] = {}
    for w, hits in zip(weights, rankings):
        for rank, h in enumerate(hits, 1):
            fused[h.chunk.id] += w / (k + rank)
            by_id[h.chunk.id] = h.chunk
    order = sorted(fused, key=lambda cid: (-fused[cid], cid))
    return [Hit(by_id[cid], fused[cid]) for cid in order]


class HybridRetriever:
    def __init__(self, retrievers: list, weights: list[float] | None = None, k: int = 60, depth: int = 100, name: str | None = None):
        self.retrievers = retrievers
        self.weights = weights
        self.k = k
        self.depth = depth
        self.name = name or "hybrid(" + "+".join(r.name for r in retrievers) + ")"

    def search(self, query: str, n: int) -> list[Hit]:
        depth = max(n, self.depth)
        return rrf([r.search(query, depth) for r in self.retrievers], self.k, self.weights)[:n]


RERANKERS = {
    "minilm-ce": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-reranker-m3": "BAAI/bge-reranker-v2-m3",
}


class CrossEncoderScorer:
    """Scores (query, passage) pairs with a cross-encoder. Scores are cached, since the same pairs recur across runs."""

    def __init__(self, short_name: str, batch_size: int = 16, max_length: int = 512):
        self.short_name = short_name
        self.hf_id = RERANKERS[short_name]
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self.cache = DiskCache(f"rerank-{short_name}")

    def __call__(self, query: str, chunks: list[Chunk]) -> list[float]:
        key = make_key(self.hf_id, query, [c.id for c in chunks], [hashlib.sha1(c.text.encode()).hexdigest() for c in chunks])
        hit = self.cache.get_json(key)
        if hit is not None:
            return hit
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.hf_id, device=torch_device(), max_length=self.max_length)
        scores = [float(s) for s in self._model.predict([(query, c.text) for c in chunks], batch_size=self.batch_size, show_progress_bar=False)]
        self.cache.set_json(key, scores)
        return scores

    def close(self) -> None:
        self._model = None
        free_torch_memory()


class RerankRetriever:
    """Take the base retriever's top `depth`, re-order them with a cross-encoder, keep the rest after."""

    def __init__(self, base, scorer, depth: int = 30, name: str | None = None):
        self.base = base
        self.scorer = scorer
        self.depth = depth
        self.name = name or f"{base.name}>rerank:{getattr(scorer, 'short_name', 'scorer')}@{depth}"

    def search(self, query: str, n: int) -> list[Hit]:
        hits = self.base.search(query, max(n, self.depth))
        head, tail = hits[: self.depth], hits[self.depth :]
        scores = self.scorer(query, [h.chunk for h in head])
        reranked = [Hit(h.chunk, s) for s, h in sorted(zip(scores, head), key=lambda p: -p[0])]
        return (reranked + tail)[:n]


class RewriteRetriever:
    """Rewrite the query with an LLM before searching.

    - multi: the original plus paraphrases, results fused with RRF
    - hyde: search with a hypothetical answer passage as well as the question, fused with RRF
    - translate: search with the query translated into the corpus language
    """

    STRATEGIES = ("multi", "hyde", "translate")

    def __init__(self, base, rewriter, strategy: str, depth: int = 100, name: str | None = None):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.base = base
        self.rewriter = rewriter
        self.strategy = strategy
        self.depth = depth
        self.name = name or f"{base.name}+rewrite:{strategy}"

    def search(self, query: str, n: int) -> list[Hit]:
        depth = max(n, self.depth)
        if self.strategy == "translate":
            return self.base.search(self.rewriter.to_english(query), n)
        if self.strategy == "multi":
            queries = [query, *self.rewriter.paraphrases(query)]
        else:
            queries = [query, self.rewriter.hypothetical_passage(query)]
        return rrf([self.base.search(q, depth) for q in queries])[:n]
