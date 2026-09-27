"""The one harness every experiment goes through: `evaluate(retriever, questions)`.

A retriever only has to return chunks, best first. The harness turns that into two rankings:
papers (did we find the right document?) and pages (did we find the passage the question was
written from?), and scores both with the same metrics, so every technique is compared the same way.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .corpus import Chunk, load_jsonl
from .metrics import dedupe, metric_values, paired_bootstrap
from .paths import BENCHMARK_DIR

QUESTIONS = BENCHMARK_DIR / "questions.jsonl"  # paraphrased: the main benchmark
EXTRACTIVE = BENCHMARK_DIR / "questions_extractive.jsonl"
METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")
KEEP = 20  # how many ranked papers/pages to store per query; enough for every metric above


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, n: int) -> list[Hit]: ...


@dataclass
class QueryResult:
    qid: str
    lang: str
    papers: list[str]
    pages: list[str]
    relevant_paper: str
    relevant_page: str
    ms: float
    chunk_pages: list[str] = field(default_factory=list)  # page of each of the top chunks, not deduplicated


@dataclass
class Run:
    name: str
    results: list[QueryResult]
    config: dict = field(default_factory=dict)

    def values(self, metric: str, level: str = "page") -> list[float]:
        """level "paper" or "page" ranks distinct units; "chunk" asks whether the labelled page is among
        the top chunks as retrieved, which is what an LLM would actually be given at top-k."""
        ranked = {"paper": lambda r: r.papers, "page": lambda r: r.pages, "chunk": lambda r: r.chunk_pages}[level]
        return [
            metric_values(ranked(r), {r.relevant_paper if level == "paper" else r.relevant_page}, metric)
            for r in self.results
        ]

    def score(self, metric: str, level: str = "page") -> float:
        vals = self.values(metric, level)
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict:
        out = {"queries": len(self.results)}
        for level in ("page", "paper"):
            out[level] = {m: round(self.score(m, level), 4) for m in METRICS}
        ms = [r.ms for r in self.results]
        out["latency_ms_p50"] = round(statistics.median(ms), 1) if ms else 0.0
        return out

    def by_lang(self) -> dict[str, "Run"]:
        langs: dict[str, list[QueryResult]] = {}
        for r in self.results:
            langs.setdefault(r.lang, []).append(r)
        return {lang: Run(f"{self.name} [{lang}]", rs, self.config) for lang, rs in langs.items()}

    def to_json(self) -> dict:
        return {"name": self.name, "config": self.config, "summary": self.summary(), "results": [asdict(r) for r in self.results]}

    @classmethod
    def from_json(cls, data: dict) -> "Run":
        return cls(data["name"], [QueryResult(**r) for r in data["results"]], data.get("config", {}))


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return load_jsonl(path)


def evaluate(retriever: Retriever, questions: list[dict], depth: int = 100, config: dict | None = None) -> Run:
    """Run every question through `retriever` and keep the top papers and pages for scoring.

    `depth` chunks are fetched per query so that, after collapsing chunks into papers, there are
    still enough distinct papers left for recall@10.
    """
    results = []
    for q in questions:
        started = time.perf_counter()
        hits = retriever.search(q["question"], depth)
        ms = (time.perf_counter() - started) * 1000
        results.append(
            QueryResult(
                qid=q["qid"],
                lang=q.get("lang", "en"),
                papers=dedupe([h.chunk.paper for h in hits])[:KEEP],
                pages=dedupe([h.chunk.page_id for h in hits])[:KEEP],
                relevant_paper=q["paper_id"],
                relevant_page=f"{q['paper_id']}#{q['page']}",
                ms=round(ms, 2),
                chunk_pages=[h.chunk.page_id for h in hits[:KEEP]],
            )
        )
    return Run(retriever.name, results, config or {})


def compare(baseline: Run, other: Run, metric: str = "mrr", level: str = "page") -> dict:
    """Paired comparison on the queries both runs share."""
    base = {r.qid: r for r in baseline.results}
    shared = [r.qid for r in other.results if r.qid in base]
    a = Run(baseline.name, [base[q] for q in shared]).values(metric, level)
    by_qid = {r.qid: r for r in other.results}
    b = Run(other.name, [by_qid[q] for q in shared]).values(metric, level)
    return {"metric": metric, "level": level, "queries": len(shared), **paired_bootstrap(a, b)}


def save_runs(path: Path, runs: list[Run], extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": [r.to_json() for r in runs], **(extra or {})}, indent=1, ensure_ascii=False))


def load_runs(path: Path) -> list[Run]:
    return [Run.from_json(r) for r in json.loads(path.read_text())["runs"]]
