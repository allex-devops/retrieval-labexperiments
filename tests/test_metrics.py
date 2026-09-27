import math

import pytest

from retrieval_lab.metrics import dedupe, metric_values, ndcg_at_k, paired_bootstrap, recall_at_k, reciprocal_rank


def test_rank_metrics_on_a_known_ranking():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"c"}, 2) == 0.0
    assert recall_at_k(ranked, {"c"}, 3) == 1.0
    assert recall_at_k(ranked, {"a", "d", "z"}, 4) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0
    assert ndcg_at_k(ranked, {"a"}, 10) == 1.0
    assert ndcg_at_k(ranked, {"b"}, 10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(ranked, {"d"}, 3) == 0.0


def test_empty_relevant_set_scores_zero_not_an_error():
    assert recall_at_k(["a"], set(), 5) == 0.0
    assert ndcg_at_k(["a"], set(), 5) == 0.0


def test_metric_names():
    assert metric_values(["x", "y"], {"y"}, "mrr") == 0.5
    assert metric_values(["x", "y"], {"y"}, "recall@1") == 0.0
    assert metric_values(["x", "y"], {"y"}, "ndcg@2") == pytest.approx(1 / math.log2(3))
    with pytest.raises(ValueError):
        metric_values(["x"], {"x"}, "precision@3")


def test_dedupe_keeps_the_best_ranked_occurrence():
    assert dedupe(["p2", "p1", "p2", "p3", "p1"]) == ["p2", "p1", "p3"]


def test_bootstrap_identical_runs_show_no_difference():
    a = [0.1, 0.5, 1.0, 0.0, 0.3] * 10
    r = paired_bootstrap(a, a)
    assert r["diff"] == 0 and r["p"] == 1.0 and r["lo"] == r["hi"] == 0


def test_bootstrap_detects_a_consistent_improvement_and_is_reproducible():
    a = [0.2, 0.4, 0.1, 0.5, 0.3] * 20
    b = [x + 0.1 for x in a]
    r = paired_bootstrap(a, b)
    assert r["diff"] == pytest.approx(0.1) and r["lo"] > 0 and r["p"] < 0.05
    assert paired_bootstrap(a, b) == r  # fixed seed


def test_bootstrap_does_not_call_noise_significant():
    a = [1.0, 0.0] * 20
    b = [0.0, 1.0] * 19 + [1.0, 0.0]  # same mean, completely reshuffled per query
    r = paired_bootstrap(a, b)
    assert r["lo"] < 0 < r["hi"] and r["p"] > 0.5


def test_bootstrap_needs_paired_queries():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])
