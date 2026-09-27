import pytest

from retrieval_lab.bench import Hit
from retrieval_lab.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
    RewriteRetriever,
    rrf,
    tokenize,
)

from .conftest import HashEmbedder


def test_tokenize_drops_stopwords_and_single_characters():
    assert tokenize("What is the BM25 score of a Query?") == ["bm25", "score", "query"]


def test_bm25_ranks_the_chunk_with_the_rare_query_terms_first(chunks):
    bm25 = BM25Retriever(chunks)
    top = bm25.search("reciprocal rank fusion", 3)
    assert top[0].chunk.id == "p6:1:0"
    assert all(a.score >= b.score for a, b in zip(top, top[1:]))


def test_bm25_rare_terms_outweigh_common_ones(chunks):
    bm25 = BM25Retriever(chunks)
    # "chunk" appears on two pages, "swahili" on one: a single rare term should dominate
    assert bm25.index["swahili"][2] > bm25.index["chunk"][2]


def test_bm25_returns_nothing_rather_than_zero_score_noise(chunks):
    assert BM25Retriever(chunks).search("zzzz qqqq", 5) == []


def test_dense_ranks_by_cosine_and_caches_the_matrix(chunks):
    emb = HashEmbedder()
    dense = DenseRetriever(chunks, emb)
    hits = dense.search("cross-encoder rerankers score each query passage pair", 3)
    assert hits[0].chunk.id == "p2:1:0"
    calls = emb.calls
    again = DenseRetriever(chunks, emb)  # same model and chunks: loads the saved matrix
    assert emb.calls == calls and (again.matrix == dense.matrix).all()
    DenseRetriever(chunks[:5], emb)  # different chunks: must re-embed
    assert emb.calls == calls + 1


def _hits(chunks, ids):
    by_id = {c.id: c for c in chunks}
    return [Hit(by_id[i], 1.0) for i in ids]


def test_rrf_rewards_agreement_between_rankings(chunks):
    a = _hits(chunks, ["p1:1:0", "p2:1:0", "p3:1:0"])
    b = _hits(chunks, ["p2:1:0", "p3:1:0", "p1:1:0"])
    fused = rrf([a, b], k=60)
    assert fused[0].chunk.id == "p2:1:0"  # ranks 2 and 1 beat ranks 1 and 3
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    heavy = rrf([a, b], k=60, weights=[3.0, 1.0])
    assert heavy[0].chunk.id == "p1:1:0"


def test_hybrid_finds_what_either_side_finds(chunks):
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, HashEmbedder())
    hybrid = HybridRetriever([bm25, dense])
    ids = [h.chunk.id for h in hybrid.search("swahili queries", 5)]
    assert "p4:2:0" in ids[:2]
    assert hybrid.name == "hybrid(bm25+dense:api)"


class ReverseScorer:
    """Scores candidates in reverse of their input order, to make any reordering visible."""

    short_name = "reverse"

    def __call__(self, query, chunks):
        return [float(i) for i in range(len(chunks))]


def test_rerank_reorders_only_the_head_and_keeps_the_tail(chunks):
    class Fixed:
        name = "fixed"

        def search(self, q, n):
            return _hits(chunks, [c.id for c in chunks])[:n]

    rr = RerankRetriever(Fixed(), ReverseScorer(), depth=3)
    ids = [h.chunk.id for h in rr.search("q", 5)]
    order = [c.id for c in chunks]
    assert ids == [order[2], order[1], order[0], order[3], order[4]]


class FakeRewriter:
    def paraphrases(self, q):
        return ["reciprocal rank fusion"]

    def hypothetical_passage(self, q):
        return "Hypothetical document embeddings search with a generated answer passage."

    def to_english(self, q):
        return {"fusión de rangos": "reciprocal rank fusion"}.get(q, q)


def test_rewrite_strategies_change_what_gets_found(chunks):
    bm25 = BM25Retriever(chunks)
    assert bm25.search("fusión de rangos", 3) == []
    translated = RewriteRetriever(bm25, FakeRewriter(), "translate").search("fusión de rangos", 3)
    assert translated[0].chunk.id == "p6:1:0"
    multi = RewriteRetriever(bm25, FakeRewriter(), "multi").search("unrelated words", 3)
    assert multi[0].chunk.id == "p6:1:0"
    hyde = RewriteRetriever(bm25, FakeRewriter(), "hyde").search("what does it search with", 3)
    assert hyde[0].chunk.id == "p5:2:0"
    with pytest.raises(ValueError):
        RewriteRetriever(bm25, FakeRewriter(), "summarise")
