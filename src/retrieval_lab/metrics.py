"""Ranking metrics over labelled queries, and a paired bootstrap for comparing two runs on the same queries."""
from __future__ import annotations

import math

import numpy as np


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, unit in enumerate(ranked, 1):
        if unit in relevant:
            return 1 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary relevance. 1.0 means every relevant unit is at the very top."""
    gain = sum(1 / math.log2(i + 1) for i, unit in enumerate(ranked[:k], 1) if unit in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return gain / ideal if ideal else 0.0


def dedupe(units: list[str]) -> list[str]:
    """Keep the first (best-ranked) occurrence of each unit: many chunks can come from one paper."""
    seen, out = set(), []
    for u in units:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def metric_values(ranked: list[str], relevant: set[str], metric: str) -> float:
    """`metric` is "mrr", "recall@K" or "ndcg@K"."""
    if metric == "mrr":
        return reciprocal_rank(ranked, relevant)
    name, _, k = metric.partition("@")
    if name == "recall":
        return recall_at_k(ranked, relevant, int(k))
    if name == "ndcg":
        return ndcg_at_k(ranked, relevant, int(k))
    raise ValueError(f"unknown metric {metric!r}")


def paired_bootstrap(a: list[float], b: list[float], n: int = 10_000, seed: int = 0) -> dict:
    """Is b better than a on the same queries? Resamples queries (not scores) so the pairing is kept.

    Returns the mean difference b - a, its 95% interval, and a two-sided p-value for "no difference".
    """
    if len(a) != len(b):
        raise ValueError("paired comparison needs the same queries in both runs")
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if len(diffs) == 0:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0}
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(n, len(diffs)))].mean(axis=1)
    observed = float(diffs.mean())
    # share of resamples on the other side of zero, doubled for a two-sided test
    if observed == 0:
        p = 1.0
    else:
        p = min(1.0, 2 * float(np.mean(means <= 0) if observed > 0 else np.mean(means >= 0)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": observed, "lo": float(lo), "hi": float(hi), "p": p}
