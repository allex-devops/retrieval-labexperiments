import hashlib
import re

import numpy as np
import pytest

from retrieval_lab.corpus import make_chunks

# six tiny "papers", two pages each, on clearly different topics
PAGES = [
    {"paper": "p1", "page": 1, "text": "Dense passage retrieval encodes questions and passages with dual encoders."},
    {"paper": "p1", "page": 2, "text": "Hard negatives mined with BM25 improve dual encoder training."},
    {"paper": "p2", "page": 1, "text": "Cross-encoder rerankers score each query passage pair jointly."},
    {"paper": "p2", "page": 2, "text": "Reranking the top thirty candidates costs latency but lifts precision."},
    {"paper": "p3", "page": 1, "text": "Chunk size controls how much context each retrieved passage carries."},
    {"paper": "p3", "page": 2, "text": "Overlapping windows stop sentences being cut at chunk boundaries."},
    {"paper": "p4", "page": 1, "text": "Multilingual embeddings map translations of a sentence close together."},
    {"paper": "p4", "page": 2, "text": "Swahili and Chinese queries retrieve English documents poorly without them."},
    {"paper": "p5", "page": 1, "text": "Query rewriting expands short queries into several paraphrases."},
    {"paper": "p5", "page": 2, "text": "Hypothetical document embeddings search with a generated answer passage."},
    {"paper": "p6", "page": 1, "text": "Reciprocal rank fusion merges lexical and vector rankings by rank alone."},
    {"paper": "p6", "page": 2, "text": "Hybrid search helps most on rare identifiers such as model names."},
]

QUESTIONS = [
    {"qid": "q1", "paper_id": "p2", "page": 1, "question": "How do cross-encoder rerankers score a query passage pair?", "answer": "jointly"},
    {"qid": "q2", "paper_id": "p3", "page": 2, "question": "Why use overlapping windows at chunk boundaries?", "answer": "sentences"},
    {"qid": "q3", "paper_id": "p6", "page": 1, "question": "How does reciprocal rank fusion merge rankings?", "answer": "by rank"},
    {"qid": "q4", "paper_id": "p5", "page": 2, "question": "What do hypothetical document embeddings search with?", "answer": "answer passage"},
]


class HashEmbedder:
    """Bag-of-words hashed into 128 buckets: deterministic, instant, and good enough to rank by word overlap."""

    name = "api"
    model = "hash-128"

    def __init__(self):
        self.calls = 0

    def encode(self, texts, kind):
        self.calls += 1
        m = np.zeros((len(texts), 128), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                if len(w) > 2:
                    m[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


@pytest.fixture(autouse=True)
def isolated_cache(request, tmp_path, monkeypatch):
    # offline tests never touch the real on-disk caches; live tests use them so they don't re-embed the corpus
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setattr("retrieval_lab.paths.CACHE_DIR", tmp_path / "cache")


@pytest.fixture
def pages():
    return [dict(p) for p in PAGES]


@pytest.fixture
def chunks(pages):
    return make_chunks(pages, size=800, overlap=120)


@pytest.fixture
def questions():
    return [dict(q) for q in QUESTIONS]
