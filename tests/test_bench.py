from dataclasses import replace

import pytest

from retrieval_lab.bench import Hit, QueryResult, Run, compare, evaluate, load_runs, save_runs
from retrieval_lab.retrievers import BM25Retriever


class Scripted:
    """Returns a fixed list of chunk ids per question, to check the harness arithmetic exactly."""

    name = "scripted"

    def __init__(self, chunks, script):
        self.by_id = {c.id: c for c in chunks}
        self.script = script

    def search(self, query, n):
        return [Hit(self.by_id[i], 1.0) for i in self.script[query]][:n]


def test_page_paper_and_chunk_levels_are_scored_separately(chunks, questions):
    q = questions[0]  # labelled p2, page 1
    script = {q["question"]: ["p2:2:0", "p1:1:0", "p2:1:0"]}
    run = evaluate(Scripted(chunks, script), [q])
    r = run.results[0]
    assert r.papers == ["p2", "p1"]
    assert r.pages == ["p2#2", "p1#1", "p2#1"]
    assert r.chunk_pages == ["p2#2", "p1#1", "p2#1"]
    assert run.score("mrr", "paper") == 1.0  # right paper at rank 1
    assert run.score("mrr", "page") == pytest.approx(1 / 3)  # right page only at rank 3
    assert run.score("recall@1", "page") == 0.0 and run.score("recall@5", "page") == 1.0


def test_summary_and_language_split(chunks, questions):
    qs = questions + [{**questions[0], "qid": "q1@es", "lang": "es"}]
    run = evaluate(BM25Retriever(chunks), qs)
    s = run.summary()
    assert s["queries"] == 5 and set(s) == {"queries", "page", "paper", "latency_ms_p50"}
    langs = run.by_lang()
    assert sorted(langs) == ["en", "es"] and len(langs["es"].results) == 1


def test_bm25_solves_the_toy_benchmark(chunks, questions):
    run = evaluate(BM25Retriever(chunks), questions)
    assert run.score("recall@5", "page") == 1.0


def test_compare_pairs_by_query_id_not_position(chunks, questions):
    good = evaluate(BM25Retriever(chunks), questions)
    worse = Run("worse", [replace(r) for r in reversed(good.results)])
    for r in worse.results[:2]:
        r.pages = ["nowhere#1"]
    c = compare(good, worse, "mrr", "page")
    assert c["queries"] == 4 and c["diff"] == pytest.approx(-0.5)
    # the mean is order-independent; the interval is what exposes position-based pairing
    varied = Run("varied", [QueryResult(f"q{i}", "en", [], ["x#1"] * i + ["p#1"], "p", "p#1", 1.0) for i in range(6)])
    same = compare(varied, Run("shuffled", list(reversed(varied.results))))
    assert same["diff"] == 0.0 and same["lo"] == same["hi"] == 0.0 and same["p"] == 1.0


def test_runs_round_trip_through_json(tmp_path, chunks, questions):
    run = evaluate(BM25Retriever(chunks), questions, config={"k1": 1.2})
    save_runs(tmp_path / "x.json", [run], {"baseline": "bm25"})
    [back] = load_runs(tmp_path / "x.json")
    assert back.name == run.name and back.config == {"k1": 1.2} and back.summary() == run.summary()
